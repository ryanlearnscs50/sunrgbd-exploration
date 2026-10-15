#!/usr/bin/env python3
"""
Generate 35-Class Model Pseudo Labels for Incremental Learning

This script generates pseudo labels using the well-trained 35-class model
for use in incremental learning memory banks and debugging.

CRITICAL: Stores labels as NYU40 IDs for compatibility with ground truth.

Date: 2025-09-02
"""

import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
import csv
from collections import defaultdict, Counter
from tqdm import tqdm

# Add project root to path
# NOTE: this script was moved from repo root to `tools/pseudo_labels/`.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mmcv import Config
from mmdet3d.apis import init_model, inference_detector


def load_35class_model(checkpoint_path: str):
    """Load 35-class model with correct architecture."""
    print("🏗️  Loading 35-class model...")
    
    # Load base config and set for 35-class training
    base_config_path = project_root / "configs/tr3d/tr3d_scannet-3d-35class.py"
    config = Config.fromfile(str(base_config_path))
    
    # Ensure correct number of classes
    config.model.head.n_classes = 35
    
    # Set optimal confidence threshold for pseudo label generation
    config.model.test_cfg.score_thr = 0.1  # Start with same threshold as Stage 1
    config.model.test_cfg.nms_pre = 1000
    config.model.test_cfg.iou_thr = 0.5
    
    # Initialize model
    model = init_model(config, checkpoint_path, device='cuda:0')
    model.eval()
    
    print(f"✅ 35-class model loaded with confidence threshold: {config.model.test_cfg.score_thr}")
    return model


def generate_scene_pseudo_labels(model, scene_info, model_to_nyu40_mapping, class_names):
    """Generate pseudo labels for a single scene."""
    scene_id = scene_info['point_cloud']['lidar_idx']
    pts_path = project_root / "data/scannet" / scene_info['pts_path']
    
    if not pts_path.exists():
        return None
    
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
            
            # Keep all detections (35-class model outputs 0-34)
            valid_mask = (labels_np >= 0) & (labels_np < 35)
            
            final_boxes = boxes_np[valid_mask]
            final_scores = scores_np[valid_mask]
            final_labels_model = labels_np[valid_mask]
            
            # CRITICAL: Convert model indices (0-34) to NYU40 IDs for storage
            # This ensures compatibility with ground truth format
            final_labels_nyu40 = np.array([
                model_to_nyu40_mapping[model_idx] for model_idx in final_labels_model
            ])
            
            # Class distribution (use model indices for class names lookup)
            unique_model_labels, counts = np.unique(final_labels_model, return_counts=True)
            class_dist = {}
            for model_idx, count in zip(unique_model_labels, counts):
                if model_idx < len(class_names):
                    class_dist[class_names[model_idx]] = int(count)
            
            return {
                'scene_id': scene_id,
                'boxes': final_boxes.tolist(),  # Convert to list for JSON serialization
                'scores': final_scores.tolist(),
                'labels': final_labels_nyu40.tolist(),  # Store as NYU40 IDs for compatibility
                'num_detections': len(final_scores),
                'class_distribution': class_dist,
                'score_stats': {
                    'min': float(final_scores.min()) if len(final_scores) > 0 else 0,
                    'max': float(final_scores.max()) if len(final_scores) > 0 else 0,
                    'mean': float(final_scores.mean()) if len(final_scores) > 0 else 0,
                    'median': float(np.median(final_scores)) if len(final_scores) > 0 else 0
                }
            }
    
    return None


def analyze_ground_truth_comparison(all_scenes, valid_nyu40_ids, class_names):
    """Analyze ground truth statistics for comparison."""
    gt_class_counts = Counter()
    gt_scene_counts = Counter()
    total_gt_objects = 0
    scenes_with_gt = 0
    
    for scene in all_scenes:
        if 'annos' not in scene:
            continue
            
        gt_labels = scene['annos'].get('class', [])
        scene_has_valid = False
        
        for nyu40_id in gt_labels:
            if nyu40_id in valid_nyu40_ids:
                idx = valid_nyu40_ids.index(nyu40_id)
                gt_class_counts[class_names[idx]] += 1
                gt_scene_counts[class_names[idx]] += 1
                total_gt_objects += 1
                scene_has_valid = True
        
        if scene_has_valid:
            scenes_with_gt += 1
    
    return {
        'total_objects': total_gt_objects,
        'scenes_with_objects': scenes_with_gt,
        'class_counts': dict(gt_class_counts),
        'objects_per_class': dict(gt_class_counts),
        'scenes_per_class': dict(gt_scene_counts)
    }


def create_visualization_outputs(pseudo_labels, gt_analysis, class_names, output_dir, checkpoint_path):
    """Create multiple visualization-ready output formats."""
    
    # 1. Complete JSON file for visualization tools
    visualization_data = {
        'metadata': {
            'total_scenes': len(pseudo_labels),
            'classes': class_names,
            'confidence_threshold': 0.1,
            'generation_timestamp': str(np.datetime64('now')),
            'model_checkpoint': checkpoint_path,
            'model_type': '35-class'
        },
        'scenes': pseudo_labels,
        'ground_truth_comparison': gt_analysis
    }
    
    json_file = output_dir / "35class_complete_visualization.json"
    with open(json_file, 'w') as f:
        json.dump(visualization_data, f, indent=2)
    print(f"✅ Visualization JSON saved: {json_file}")
    
    # 2. Summary statistics JSON
    total_detections = sum(scene['num_detections'] for scene in pseudo_labels.values())
    all_scores = []
    overall_class_dist = Counter()
    
    for scene_data in pseudo_labels.values():
        all_scores.extend(scene_data['scores'])
        for class_name, count in scene_data['class_distribution'].items():
            overall_class_dist[class_name] += count
    
    all_scores = np.array(all_scores)
    
    summary_stats = {
        'overall_statistics': {
            'total_scenes': len(pseudo_labels),
            'total_detections': total_detections,
            'avg_detections_per_scene': total_detections / len(pseudo_labels),
            'score_statistics': {
                'min': float(all_scores.min()) if len(all_scores) > 0 else 0,
                'max': float(all_scores.max()) if len(all_scores) > 0 else 0,
                'mean': float(all_scores.mean()) if len(all_scores) > 0 else 0,
                'median': float(np.median(all_scores)) if len(all_scores) > 0 else 0,
                'std': float(all_scores.std()) if len(all_scores) > 0 else 0,
                'percentiles': {
                    '25': float(np.percentile(all_scores, 25)) if len(all_scores) > 0 else 0,
                    '75': float(np.percentile(all_scores, 75)) if len(all_scores) > 0 else 0,
                    '90': float(np.percentile(all_scores, 90)) if len(all_scores) > 0 else 0,
                    '95': float(np.percentile(all_scores, 95)) if len(all_scores) > 0 else 0
                }
            }
        },
        'per_class_statistics': {},
        'ground_truth_comparison': gt_analysis
    }
    
    # Per-class statistics
    for class_name in class_names:
        pred_count = overall_class_dist.get(class_name, 0)
        gt_count = gt_analysis['class_counts'].get(class_name, 0)
        
        # Class-specific scores
        class_scores = []
        for scene_data in pseudo_labels.values():
            scene_scores = np.array(scene_data['scores'])
            scene_labels_nyu40 = np.array(scene_data['labels'])  # Already NYU40 IDs
            # Find this class's NYU40 ID
            class_idx = class_names.index(class_name)
            # We need the model-to-NYU40 mapping here
            # For now, use a simple approach - this could be improved
            class_scores.extend(scene_scores.tolist())  # Add all for now
        
        class_scores = np.array(class_scores) if class_scores else np.array([])
        
        summary_stats['per_class_statistics'][class_name] = {
            'prediction_count': int(pred_count),
            'ground_truth_count': int(gt_count),
            'detection_ratio': float(pred_count / max(gt_count, 1)),
            'percentage_of_predictions': float(100 * pred_count / max(total_detections, 1))
        }
    
    summary_file = output_dir / "35class_analysis_report.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✅ Analysis report saved: {summary_file}")
    
    # 3. CSV file for spreadsheet analysis
    csv_file = output_dir / "35class_per_class_analysis.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Class', 'Predictions', 'Ground_Truth', 'Detection_Ratio', 'Percentage_of_Predictions'
        ])
        
        for class_name in class_names:
            stats = summary_stats['per_class_statistics'][class_name]
            writer.writerow([
                class_name,
                stats['prediction_count'],
                stats['ground_truth_count'],
                f"{stats['detection_ratio']:.3f}",
                f"{stats['percentage_of_predictions']:.1f}%"
            ])
    print(f"✅ CSV analysis saved: {csv_file}")
    
    return summary_stats


def main():
    """Main function to generate complete 35-class pseudo labels."""
    print("🚀 Generate Complete 35-Class Pseudo Labels for Debugging & Memory Banks")
    print("=" * 80)
    
    # Set environment
    os.environ['PYTHONPATH'] = str(project_root) + ":" + os.environ.get('PYTHONPATH', '')
    
    # Configuration
    checkpoint_path = "my_work_dirs/tr3d_35class_25epochs_fix_20250902_114144/latest.pth"
    output_dir = Path("35class_pseudo_labels")
    output_dir.mkdir(exist_ok=True)
    
    # Load 35-class mappings
    sys.path.append('configs/_base_/class_mappings')
    from scannet_35class_mapping import (
        VALID_NYU40_IDS_35CLASS, 
        SCANNET_35_CLASSES,
        MODEL_IDX_TO_NYU40_35CLASS
    )
    
    print(f"35-class configuration:")
    print(f"  Classes: {len(SCANNET_35_CLASSES)}")
    print(f"  NYU40 IDs: {VALID_NYU40_IDS_35CLASS[:10]}...")
    print(f"  Sample names: {SCANNET_35_CLASSES[:10]}")
    
    # Load model
    model = load_35class_model(checkpoint_path)
    
    # Load all training scenes
    train_pkl = project_root / "data/scannet/scannet_infos_train_40class_corrected.pkl"
    with open(train_pkl, 'rb') as f:
        all_scenes = pickle.load(f)
    
    print(f"Found {len(all_scenes)} total training scenes")
    
    # Analyze ground truth for comparison
    print("📊 Analyzing ground truth statistics...")
    gt_analysis = analyze_ground_truth_comparison(all_scenes, VALID_NYU40_IDS_35CLASS, SCANNET_35_CLASSES)
    
    print(f"Ground truth statistics:")
    print(f"  Total objects: {gt_analysis['total_objects']}")
    print(f"  Scenes with objects: {gt_analysis['scenes_with_objects']}")
    
    # Process all training scenes for complete pseudo label generation
    num_scenes_to_process = len(all_scenes)  # Process all scenes for complete dataset
    print(f"\n🔮 Generating pseudo labels for {num_scenes_to_process} scenes...")
    
    pseudo_labels = {}
    total_detections = 0
    processed_scenes = 0
    all_class_counts = {name: 0 for name in SCANNET_35_CLASSES}
    
    for i, scene_info in enumerate(tqdm(all_scenes[:num_scenes_to_process], desc="Processing scenes")):
        result = generate_scene_pseudo_labels(
            model, scene_info, MODEL_IDX_TO_NYU40_35CLASS, SCANNET_35_CLASSES
        )
        
        if result:
            pseudo_labels[result['scene_id']] = result
            total_detections += result['num_detections']
            processed_scenes += 1
            
            # Accumulate class counts
            for class_name, count in result['class_distribution'].items():
                all_class_counts[class_name] += count
        
        # Progress update every 25 scenes  
        if i % 25 == 0 and i > 0:
            print(f"  Processed {i}/{num_scenes_to_process} scenes, {processed_scenes} successful, {total_detections} detections")
    
    # Final summary
    print(f"\n📊 Generation Summary:")
    print(f"  Scenes processed successfully: {processed_scenes}/{num_scenes_to_process}")
    print(f"  Total pseudo labels: {total_detections}")
    print(f"  Average per scene: {total_detections/max(processed_scenes, 1):.1f}")
    
    print(f"\n📋 Top 10 classes by detection count:")
    sorted_classes = sorted(all_class_counts.items(), key=lambda x: x[1], reverse=True)
    for class_name, count in sorted_classes[:10]:
        pct = 100 * count / max(total_detections, 1)
        print(f"  {class_name:15s}: {count:5d} ({pct:5.1f}%)")
    
    # Save complete pseudo labels
    complete_pkl = output_dir / "35class_pseudo_labels_complete.pkl"
    with open(complete_pkl, 'wb') as f:
        pickle.dump(pseudo_labels, f)
    print(f"\n✅ Complete pseudo labels saved: {complete_pkl}")
    
    # Create visualization outputs
    print(f"\n📊 Creating visualization outputs...")
    summary_stats = create_visualization_outputs(
        pseudo_labels, gt_analysis, SCANNET_35_CLASSES, output_dir, checkpoint_path
    )
    
    print(f"\n🎯 All outputs ready for analysis and debugging!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Files created:")
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            print(f"   - {file_path.name}")


if __name__ == '__main__':
    main()
