#!/usr/bin/env python3
"""
Generate Complete Pseudo Labels and Visualization Outputs

This script generates pseudo labels for ALL Stage 1 training scenes and creates 
visualization-ready outputs in multiple formats.

Date: 2025-08-31
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


def load_stage1_model(checkpoint_path: str):
    """Load Stage 1 model with correct 7-class architecture."""
    print("🏗️  Loading Stage 1 model...")
    
    # Load base config and modify for Stage 1
    base_config_path = project_root / "configs/tr3d/tr3d_scannet-3d-35class.py"
    config = Config.fromfile(str(base_config_path))
    
    # CRITICAL: Set correct number of classes for Stage 1
    config.model.head.n_classes = 7
    
    # Set optimal confidence threshold for pseudo label generation
    config.model.test_cfg.score_thr = 0.1  # Optimal threshold from analysis
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
            
            # Filter for Stage 1 classes only (labels should already be 0-6)
            stage1_mask = labels_np < 7  # Should be all true for Stage 1 model
            
            final_boxes = boxes_np[stage1_mask]
            final_scores = scores_np[stage1_mask]
            final_labels_model = labels_np[stage1_mask]
            
            # CRITICAL: Convert model indices (0-6) to NYU40 IDs for storage
            # This ensures compatibility with ground truth format
            final_labels_nyu40 = np.array([stage1_nyu40_ids[idx] for idx in final_labels_model])
            
            # Class distribution (use model indices for class names lookup)
            unique_model_labels, counts = np.unique(final_labels_model, return_counts=True)
            class_dist = {}
            for model_idx, count in zip(unique_model_labels, counts):
                if model_idx < len(stage1_names):
                    class_dist[stage1_names[model_idx]] = int(count)
            
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


def analyze_ground_truth_comparison(all_scenes, stage1_nyu40_ids, stage1_names):
    """Analyze ground truth statistics for comparison."""
    gt_class_counts = Counter()
    gt_scene_counts = Counter()
    total_gt_objects = 0
    scenes_with_gt = 0
    
    for scene in all_scenes:
        if 'annos' not in scene:
            continue
            
        gt_labels = scene['annos'].get('class', [])
        scene_has_stage1 = False
        
        for label in gt_labels:
            if label in stage1_nyu40_ids:
                idx = stage1_nyu40_ids.index(label)
                gt_class_counts[stage1_names[idx]] += 1
                gt_scene_counts[stage1_names[idx]] += 1
                total_gt_objects += 1
                scene_has_stage1 = True
        
        if scene_has_stage1:
            scenes_with_gt += 1
    
    return {
        'total_objects': total_gt_objects,
        'scenes_with_objects': scenes_with_gt,
        'class_counts': dict(gt_class_counts),
        'objects_per_class': dict(gt_class_counts),
        'scenes_per_class': dict(gt_scene_counts)
    }


def create_visualization_outputs(pseudo_labels, gt_analysis, stage1_names, output_dir):
    """Create multiple visualization-ready output formats."""
    
    # 1. Complete JSON file for visualization tools
    visualization_data = {
        'metadata': {
            'total_scenes': len(pseudo_labels),
            'stage1_classes': stage1_names,
            'confidence_threshold': 0.1,
            'generation_timestamp': str(np.datetime64('now')),
            'model_checkpoint': 'incremental_logs/frequency_finetuning_correct/seed_200_20250829_223859/checkpoints/stage_1/latest.pth'
        },
        'scenes': pseudo_labels,
        'ground_truth_comparison': gt_analysis
    }
    
    json_file = output_dir / "stage1_complete_visualization.json"
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
                'min': float(all_scores.min()),
                'max': float(all_scores.max()),
                'mean': float(all_scores.mean()),
                'median': float(np.median(all_scores)),
                'std': float(all_scores.std()),
                'percentiles': {
                    '25': float(np.percentile(all_scores, 25)),
                    '75': float(np.percentile(all_scores, 75)),
                    '90': float(np.percentile(all_scores, 90)),
                    '95': float(np.percentile(all_scores, 95))
                }
            }
        },
        'per_class_statistics': {},
        'ground_truth_comparison': gt_analysis
    }
    
    # Per-class statistics
    for class_name in stage1_names:
        pred_count = overall_class_dist.get(class_name, 0)
        gt_count = gt_analysis['class_counts'].get(class_name, 0)
        
        # Class-specific scores
        class_scores = []
        for scene_data in pseudo_labels.values():
            scene_scores = np.array(scene_data['scores'])
            scene_labels = np.array(scene_data['labels'])
            class_idx = stage1_names.index(class_name)
            # labels are NYU40 IDs; compare with the class NYU40 ID
            class_mask = scene_labels == stage1_nyu40_ids[class_idx]
            class_scores.extend(scene_scores[class_mask].tolist())
        
        class_scores = np.array(class_scores)
        
        summary_stats['per_class_statistics'][class_name] = {
            'prediction_count': int(pred_count),
            'ground_truth_count': int(gt_count),
            'detection_ratio': float(pred_count / max(gt_count, 1)),
            'percentage_of_predictions': float(100 * pred_count / max(total_detections, 1)),
            'score_statistics': {
                'count': len(class_scores),
                'mean': float(class_scores.mean()) if len(class_scores) > 0 else 0,
                'std': float(class_scores.std()) if len(class_scores) > 0 else 0,
                'min': float(class_scores.min()) if len(class_scores) > 0 else 0,
                'max': float(class_scores.max()) if len(class_scores) > 0 else 0
            }
        }
    
    summary_file = output_dir / "analysis_report.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    print(f"✅ Analysis report saved: {summary_file}")
    
    # 3. CSV file for spreadsheet analysis
    csv_file = output_dir / "per_class_analysis.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Class', 'Predictions', 'Ground_Truth', 'Detection_Ratio', 
            'Percentage_of_Predictions', 'Avg_Score', 'Score_Std', 'Min_Score', 'Max_Score'
        ])
        
        for class_name in stage1_names:
            stats = summary_stats['per_class_statistics'][class_name]
            writer.writerow([
                class_name,
                stats['prediction_count'],
                stats['ground_truth_count'],
                f"{stats['detection_ratio']:.3f}",
                f"{stats['percentage_of_predictions']:.1f}%",
                f"{stats['score_statistics']['mean']:.3f}",
                f"{stats['score_statistics']['std']:.3f}",
                f"{stats['score_statistics']['min']:.3f}",
                f"{stats['score_statistics']['max']:.3f}"
            ])
    print(f"✅ CSV analysis saved: {csv_file}")
    
    # 4. Confidence distribution JSON
    confidence_distributions = {}
    
    # Overall distribution
    hist, bin_edges = np.histogram(all_scores, bins=20)
    confidence_distributions['overall'] = {
        'histogram': hist.tolist(),
        'bin_edges': bin_edges.tolist()
    }
    
    # Per-class distributions
    for class_name in stage1_names:
        class_scores = []
        for scene_data in pseudo_labels.values():
            scene_scores = np.array(scene_data['scores'])
            scene_labels = np.array(scene_data['labels'])
            class_idx = stage1_names.index(class_name)
            class_mask = scene_labels == stage1_nyu40_ids[class_idx]
            class_scores.extend(scene_scores[class_mask].tolist())
        
        if class_scores:
            hist, bin_edges = np.histogram(class_scores, bins=20)
            confidence_distributions[class_name] = {
                'histogram': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'count': len(class_scores)
            }
    
    conf_file = output_dir / "confidence_distributions.json"
    with open(conf_file, 'w') as f:
        json.dump(confidence_distributions, f, indent=2)
    print(f"✅ Confidence distributions saved: {conf_file}")
    
    return summary_stats


def main():
    """Main function to generate complete pseudo labels and visualization outputs."""
    print("🚀 Generate Complete Stage 1 Pseudo Labels & Visualization Outputs")
    print("=" * 80)
    
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
    
    # Analyze ground truth for comparison
    print("📊 Analyzing ground truth statistics...")
    gt_analysis = analyze_ground_truth_comparison(stage1_scenes, stage1_nyu40_ids, stage1_names)
    
    print(f"Ground truth statistics:")
    print(f"  Total objects: {gt_analysis['total_objects']}")
    print(f"  Scenes with objects: {gt_analysis['scenes_with_objects']}")
    for class_name, count in gt_analysis['class_counts'].items():
        print(f"  {class_name}: {count} objects")
    
    # Generate pseudo labels for ALL Stage 1 training scenes
    num_scenes_to_process = len(stage1_scenes)  # Process all Stage 1 training scenes
    print(f"\n🔮 Generating pseudo labels for {num_scenes_to_process} scenes...")
    pseudo_labels = {}
    total_detections = 0
    processed_scenes = 0
    all_class_counts = {name: 0 for name in stage1_names}
    
    for i, scene_info in enumerate(tqdm(stage1_scenes[:num_scenes_to_process], desc="Processing scenes")):
        result = generate_scene_pseudo_labels(
            model, scene_info, stage1_nyu40_ids, stage1_names
        )
        
        if result:
            pseudo_labels[result['scene_id']] = result
            total_detections += result['num_detections']
            processed_scenes += 1
            
            # Accumulate class counts
            for class_name, count in result['class_distribution'].items():
                all_class_counts[class_name] += count
        
        # Progress update every 50 scenes  
        if i % 50 == 0 and i > 0:
            print(f"  Processed {i}/{num_scenes_to_process} scenes, {processed_scenes} successful, {total_detections} detections")
    
    # Final summary
    print(f"\n📊 Generation Summary:")
    print(f"  Scenes processed successfully: {processed_scenes}/{num_scenes_to_process}")
    print(f"  Total pseudo labels: {total_detections}")
    print(f"  Average per scene: {total_detections/max(processed_scenes, 1):.1f}")
    
    print(f"\n📋 Overall class distribution:")
    for class_name, count in all_class_counts.items():
        pct = 100 * count / max(total_detections, 1)
        gt_count = gt_analysis['class_counts'].get(class_name, 0)
        ratio = count / max(gt_count, 1)
        print(f"  {class_name:12s}: {count:5d} ({pct:5.1f}%) | GT: {gt_count:4d} | Ratio: {ratio:.2f}")
    
    # Save complete pseudo labels
    complete_pkl = output_dir / "stage1_pseudo_labels_complete.pkl"
    with open(complete_pkl, 'wb') as f:
        pickle.dump(pseudo_labels, f)
    print(f"\n✅ Complete pseudo labels saved: {complete_pkl}")
    
    # Create visualization outputs
    print(f"\n📊 Creating visualization outputs...")
    summary_stats = create_visualization_outputs(
        pseudo_labels, gt_analysis, stage1_names, output_dir
    )
    
    # Final chair analysis
    chair_stats = summary_stats['per_class_statistics']['chair']
    print(f"\n🪑 Chair Detection Final Analysis:")
    print(f"   Predictions: {chair_stats['prediction_count']}")
    print(f"   Ground truth: {chair_stats['ground_truth_count']}")
    print(f"   Detection ratio: {chair_stats['detection_ratio']:.3f}")
    print(f"   Percentage of all predictions: {chair_stats['percentage_of_predictions']:.1f}%")
    print(f"   Average confidence: {chair_stats['score_statistics']['mean']:.3f}")
    
    if chair_stats['prediction_count'] > 0:
        print(f"   ✅ Chair detection is working successfully!")
    else:
        print(f"   ❌ Chair detection failed!")
    
    print(f"\n🎯 All outputs ready for visualization!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Files created:")
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            print(f"   - {file_path.name}")


if __name__ == '__main__':
    main()
