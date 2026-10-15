#!/usr/bin/env python3
"""
Generate Stage 1 Pseudo Labels with Correct Configuration

This script generates pseudo labels from Stage 1 model with the optimal confidence threshold.
It uses the corrected inference approach and proper result access pattern.

Date: 2025-08-31
"""

import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path

# Add project root to path
# NOTE: this script was moved from repo root to `tools/pseudo_labels/`.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mmcv import Config
from mmdet3d.apis import init_model, inference_detector


def load_stage1_model(checkpoint_path: str):
    """Load Stage 1 model with correct 7-class architecture."""
    print("🏗️  Loading Stage 1 model...")
    
    # Load base config and modify for Stage 1
    base_config_path = project_root / "configs/tr3d/tr3d_scannet-3d-35class.py"
    config = Config.fromfile(str(base_config_path))
    
    # CRITICAL: Set correct number of classes for Stage 1
    config.model.head.n_classes = 7
    
    # Set optimal confidence threshold
    config.model.test_cfg.score_thr = 0.1  # Optimal threshold found from analysis
    config.model.test_cfg.nms_pre = 1000
    config.model.test_cfg.iou_thr = 0.5
    
    # Initialize model
    model = init_model(config, checkpoint_path, device='cuda:0')
    model.eval()
    
    print(f"✅ Model loaded with confidence threshold: {config.model.test_cfg.score_thr}")
    return model


def generate_scene_pseudo_labels(model, scene_info, stage1_nyu40_ids, stage1_names):
    """Generate pseudo labels for a single scene."""
    scene_id = scene_info['point_cloud']['lidar_idx']
    pts_path = project_root / "data/scannet" / scene_info['pts_path']
    
    if not pts_path.exists():
        print(f"❌ Point cloud not found: {pts_path}")
        return None
    
    print(f"🔍 Processing scene: {scene_id}")
    
    # Run inference
    with torch.no_grad():
        result = inference_detector(model, str(pts_path))
    
    # Extract detections using correct access pattern
    if result and len(result) > 0 and len(result[0]) > 0:
        res = result[0][0]  # Correct access: result[0][0]
        
        if isinstance(res, dict) and 'boxes_3d' in res:
            boxes_3d = res['boxes_3d']
            scores_3d = res['scores_3d']
            labels_3d = res['labels_3d']
            
            # Convert to numpy
            if hasattr(scores_3d, 'cpu'):
                scores_np = scores_3d.cpu().numpy()
                labels_np = labels_3d.cpu().numpy()
            else:
                scores_np = np.array(scores_3d)
                labels_np = np.array(labels_3d)
            
            if hasattr(boxes_3d, 'tensor'):
                boxes_np = boxes_3d.tensor.cpu().numpy()
            else:
                boxes_np = np.array(boxes_3d)
            
            print(f"  Raw detections: {len(scores_np)}")
            print(f"  Score range: [{scores_np.min():.3f}, {scores_np.max():.3f}]")
            
            # Filter for Stage 1 classes only (labels should already be 0-6)
            stage1_mask = labels_np < 7  # Should be all true for Stage 1 model
            
            final_boxes = boxes_np[stage1_mask]
            final_scores = scores_np[stage1_mask]
            final_labels = labels_np[stage1_mask]
            
            print(f"  Stage 1 detections: {len(final_scores)}")
            
            # Class distribution
            unique_labels, counts = np.unique(final_labels, return_counts=True)
            class_dist = {}
            for label, count in zip(unique_labels, counts):
                if label < len(stage1_names):
                    class_dist[stage1_names[label]] = count
            print(f"  Class distribution: {class_dist}")
            
            return {
                'scene_id': scene_id,
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels,
                'num_detections': len(final_scores),
                'class_distribution': class_dist
            }
    
    print(f"  ❌ No detections generated")
    return None


def main():
    """Main pseudo label generation function."""
    print("🚀 Generate Stage 1 Pseudo Labels")
    print("=" * 50)
    
    # Set environment
    os.environ['PYTHONPATH'] = str(project_root) + ":" + os.environ.get('PYTHONPATH', '')
    
    # Configuration
    checkpoint_path = "incremental_logs/frequency_finetuning_correct/seed_200_20250829_223859/checkpoints/stage_1/latest.pth"
    output_dir = Path("stage1_pseudo_labels")
    output_dir.mkdir(exist_ok=True)
    
    # Stage 1 class definitions
    stage1_nyu40_ids = [5, 8, 39, 23, 3, 7, 9]  
    stage1_names = ['chair', 'door', 'otherfurniture', 'books', 'cabinet', 'table', 'window']
    
    print(f"Stage 1 classes: {stage1_names}")
    
    # Load model
    model = load_stage1_model(checkpoint_path)
    
    # Load Stage 1 training scenes
    train_pkl = project_root / "data/scannet/scannet_infos_train_40class_corrected.pkl"
    with open(train_pkl, 'rb') as f:
        all_scenes = pickle.load(f)
    
    # Filter to Stage 1 scenes
    stage1_scenes = []
    for scene in all_scenes:
        if 'annos' in scene:
            gt_labels = scene['annos'].get('class', [])
            if any(label in stage1_nyu40_ids for label in gt_labels):
                stage1_scenes.append(scene)
    
    print(f"Found {len(stage1_scenes)} Stage 1 training scenes")
    
    # Generate pseudo labels for first 10 scenes as test
    num_test_scenes = min(10, len(stage1_scenes))
    pseudo_labels = {}
    total_detections = 0
    total_scenes_processed = 0
    all_class_counts = {name: 0 for name in stage1_names}
    
    for i in range(num_test_scenes):
        result = generate_scene_pseudo_labels(
            model, stage1_scenes[i], stage1_nyu40_ids, stage1_names
        )
        
        if result:
            pseudo_labels[result['scene_id']] = result
            total_detections += result['num_detections']
            total_scenes_processed += 1
            
            # Accumulate class counts
            for class_name, count in result['class_distribution'].items():
                all_class_counts[class_name] += count
    
    # Summary
    print(f"\n📊 Summary:")
    print(f"  Scenes processed: {total_scenes_processed}/{num_test_scenes}")
    print(f"  Total pseudo labels: {total_detections}")
    print(f"  Average per scene: {total_detections/max(total_scenes_processed, 1):.1f}")
    
    print(f"\n📋 Overall class distribution:")
    for class_name, count in all_class_counts.items():
        pct = 100 * count / max(total_detections, 1)
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    # Save results
    output_file = output_dir / "stage1_pseudo_labels_test.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(pseudo_labels, f)
    
    print(f"\n✅ Pseudo labels saved to: {output_file}")
    
    # Verification: Check chair representation
    chair_count = all_class_counts.get('chair', 0)
    chair_pct = 100 * chair_count / max(total_detections, 1)
    
    if chair_count > 0:
        print(f"✅ Chair detection SUCCESS: {chair_count} chairs ({chair_pct:.1f}%)")
    else:
        print(f"❌ Chair detection FAILED: No chairs detected!")
    
    return total_scenes_processed > 0


if __name__ == '__main__':
    main()
