#!/usr/bin/env python3
"""
Evaluate Pseudo Labels Against Ground Truth

This script evaluates the generated pseudo labels against ground truth annotations
to compute mAP and per-class AP on the training dataset. This measures the actual
quality of the pseudo labels after confidence filtering, not raw model output.

Date: 2025-08-31
"""

import os
import sys
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from collections import defaultdict, Counter
from tqdm import tqdm

# Add project root to path
# NOTE: this script was moved from repo root to `tools/pseudo_labels/`.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


def compute_bbox_iou_3d(box1, box2):
    """
    Compute IoU between two 3D bounding boxes.
    
    Args:
        box1, box2: [x, y, z, w, h, l, r] format (center coordinates + dimensions + rotation)
    
    Returns:
        IoU value between 0 and 1
    """
    # For simplicity, we'll use axis-aligned bounding box IoU
    # Convert to min/max format
    x1_min, y1_min, z1_min = box1[0] - box1[3]/2, box1[1] - box1[4]/2, box1[2] - box1[5]/2
    x1_max, y1_max, z1_max = box1[0] + box1[3]/2, box1[1] + box1[4]/2, box1[2] + box1[5]/2
    
    x2_min, y2_min, z2_min = box2[0] - box2[3]/2, box2[1] - box2[4]/2, box2[2] - box2[5]/2
    x2_max, y2_max, z2_max = box2[0] + box2[3]/2, box2[1] + box2[4]/2, box2[2] + box2[5]/2
    
    # Intersection
    inter_x_min, inter_x_max = max(x1_min, x2_min), min(x1_max, x2_max)
    inter_y_min, inter_y_max = max(y1_min, y2_min), min(y1_max, y2_max)
    inter_z_min, inter_z_max = max(z1_min, z2_min), min(z1_max, z2_max)
    
    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max or inter_z_min >= inter_z_max:
        return 0.0
    
    intersection = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min) * (inter_z_max - inter_z_min)
    
    # Union
    vol1 = box1[3] * box1[4] * box1[5]  # w * h * l
    vol2 = box2[3] * box2[4] * box2[5]  # w * h * l
    union = vol1 + vol2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_average_precision(precisions, recalls):
    """
    Compute Average Precision using the 11-point interpolation method.
    
    Args:
        precisions: List of precision values
        recalls: List of recall values
        
    Returns:
        Average Precision value
    """
    if len(precisions) == 0:
        return 0.0
    
    # Sort by recall
    sorted_pairs = sorted(zip(recalls, precisions))
    recalls_sorted = [r for r, p in sorted_pairs]
    precisions_sorted = [p for r, p in sorted_pairs]
    
    # 11-point interpolation
    ap = 0.0
    for recall_threshold in np.arange(0, 1.1, 0.1):
        # Find precisions for recalls >= threshold
        valid_precisions = [p for r, p in zip(recalls_sorted, precisions_sorted) if r >= recall_threshold]
        if len(valid_precisions) > 0:
            max_precision = max(valid_precisions)
        else:
            max_precision = 0.0
        ap += max_precision / 11.0
    
    return ap


def evaluate_single_class(pseudo_labels_class, gt_boxes_class, iou_threshold=0.25):
    """
    Evaluate single class predictions against ground truth.
    
    Args:
        pseudo_labels_class: List of (box, score) tuples for this class
        gt_boxes_class: List of ground truth boxes for this class
        iou_threshold: IoU threshold for positive detection
        
    Returns:
        Dictionary with precision, recall, AP, and statistics
    """
    if len(pseudo_labels_class) == 0:
        return {
            'ap': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'num_predictions': 0,
            'num_gt': len(gt_boxes_class),
            'true_positives': 0,
            'false_positives': 0
        }
    
    if len(gt_boxes_class) == 0:
        return {
            'ap': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'num_predictions': len(pseudo_labels_class),
            'num_gt': 0,
            'true_positives': 0,
            'false_positives': len(pseudo_labels_class)
        }
    
    # Sort predictions by confidence (descending)
    pseudo_labels_class.sort(key=lambda x: x[1], reverse=True)
    
    # Initialize arrays
    num_predictions = len(pseudo_labels_class)
    num_gt = len(gt_boxes_class)
    
    tp = np.zeros(num_predictions)
    fp = np.zeros(num_predictions)
    gt_matched = [False] * num_gt
    
    # For each prediction, find best matching ground truth
    for pred_idx, (pred_box, pred_score) in enumerate(pseudo_labels_class):
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes_class):
            if gt_matched[gt_idx]:
                continue
                
            iou = compute_bbox_iou_3d(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        # Check if prediction is positive
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    # Compute precision and recall curves
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recalls = tp_cumsum / num_gt
    
    # Compute AP
    ap = compute_average_precision(precisions.tolist(), recalls.tolist())
    
    # Overall statistics
    total_tp = int(tp_cumsum[-1])
    total_fp = int(fp_cumsum[-1])
    final_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    final_recall = total_tp / num_gt if num_gt > 0 else 0.0
    
    return {
        'ap': ap,
        'precision': final_precision,
        'recall': final_recall,
        'num_predictions': num_predictions,
        'num_gt': num_gt,
        'true_positives': total_tp,
        'false_positives': total_fp
    }


def load_ground_truth_data(train_pkl_path, stage1_nyu40_ids, stage1_names):
    """Load and organize ground truth data for Stage 1 scenes."""
    
    with open(train_pkl_path, 'rb') as f:
        all_scenes = pickle.load(f)
    
    print(f"📚 Loading ground truth from {len(all_scenes)} training scenes...")
    
    # Filter to Stage 1 scenes and organize by scene_id  
    gt_data = {}
    stage1_scene_count = 0
    
    for scene in all_scenes:
        if 'annos' not in scene:
            continue
            
        scene_id = scene['point_cloud']['lidar_idx']
        gt_labels = scene['annos'].get('class', [])
        
        # Check if scene has Stage 1 objects
        if not any(label in stage1_nyu40_ids for label in gt_labels):
            continue
            
        stage1_scene_count += 1
        
        # Extract Stage 1 objects
        gt_boxes = scene['annos'].get('gt_boxes_upright_depth', [])
        
        scene_gt_by_class = {name: [] for name in stage1_names}
        
        for i, label in enumerate(gt_labels):
            if label in stage1_nyu40_ids and i < len(gt_boxes):
                class_idx = stage1_nyu40_ids.index(label)
                class_name = stage1_names[class_idx]
                scene_gt_by_class[class_name].append(gt_boxes[i])
        
        gt_data[scene_id] = scene_gt_by_class
    
    print(f"✅ Loaded ground truth for {stage1_scene_count} Stage 1 scenes")
    return gt_data


def load_pseudo_labels(pseudo_labels_pkl, stage1_names):
    """Load and organize pseudo labels by class."""
    
    with open(pseudo_labels_pkl, 'rb') as f:
        pseudo_labels_data = pickle.load(f)
    
    print(f"🔮 Loading pseudo labels for {len(pseudo_labels_data)} scenes...")
    
    # Organize by class across all scenes
    pseudo_by_class = {name: [] for name in stage1_names}
    
    for scene_id, scene_data in pseudo_labels_data.items():
        boxes = np.array(scene_data['boxes'])
        scores = np.array(scene_data['scores'])
        labels = np.array(scene_data['labels'])
        
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if 0 <= label < len(stage1_names):
                class_name = stage1_names[label]
                pseudo_by_class[class_name].append((box, score))
    
    print(f"✅ Loaded pseudo labels organized by class")
    return pseudo_by_class


def evaluate_pseudo_labels_full(pseudo_labels_pkl, train_pkl_path, stage1_nyu40_ids, stage1_names, iou_thresholds=[0.25, 0.5]):
    """
    Full evaluation of pseudo labels against ground truth.
    
    Returns:
        Dictionary with per-class and overall evaluation results
    """
    print("🎯 Starting Full Pseudo Label Evaluation")
    print("=" * 60)
    
    # Load data
    gt_data = load_ground_truth_data(train_pkl_path, stage1_nyu40_ids, stage1_names)
    pseudo_by_class = load_pseudo_labels(pseudo_labels_pkl, stage1_names)
    
    # Flatten ground truth by class for evaluation
    gt_by_class = {name: [] for name in stage1_names}
    
    for scene_id, scene_gt in gt_data.items():
        for class_name, class_boxes in scene_gt.items():
            gt_by_class[class_name].extend(class_boxes)
    
    # Evaluate each class at different IoU thresholds
    results = {}
    
    for iou_threshold in iou_thresholds:
        print(f"\n📊 Evaluating at IoU threshold {iou_threshold}")
        
        class_results = {}
        total_ap = 0.0
        total_precision = 0.0  
        total_recall = 0.0
        valid_classes = 0
        
        for class_name in stage1_names:
            print(f"  Evaluating {class_name}...")
            
            result = evaluate_single_class(
                pseudo_by_class[class_name].copy(),  # Copy to avoid modifying original
                gt_by_class[class_name],
                iou_threshold=iou_threshold
            )
            
            class_results[class_name] = result
            
            # Accumulate for mAP
            total_ap += result['ap']
            total_precision += result['precision'] 
            total_recall += result['recall']
            valid_classes += 1
            
            print(f"    AP: {result['ap']:.4f}, P: {result['precision']:.4f}, R: {result['recall']:.4f}")
            print(f"    Pred: {result['num_predictions']}, GT: {result['num_gt']}, TP: {result['true_positives']}")
        
        # Overall metrics
        map_score = total_ap / valid_classes if valid_classes > 0 else 0.0
        avg_precision = total_precision / valid_classes if valid_classes > 0 else 0.0
        avg_recall = total_recall / valid_classes if valid_classes > 0 else 0.0
        
        results[f'iou_{iou_threshold}'] = {
            'mAP': map_score,
            'average_precision': avg_precision,
            'average_recall': avg_recall,
            'per_class': class_results
        }
        
        print(f"\n🎯 Overall Results (IoU {iou_threshold}):")
        print(f"  mAP: {map_score:.4f}")
        print(f"  Avg Precision: {avg_precision:.4f}")
        print(f"  Avg Recall: {avg_recall:.4f}")
    
    return results


def main():
    """Main evaluation function."""
    print("🎯 Evaluating Stage 1 Pseudo Labels Against Ground Truth")
    print("=" * 80)
    
    # Configuration
    pseudo_labels_pkl = "stage1_pseudo_labels/stage1_pseudo_labels_complete.pkl"
    train_pkl = "data/scannet/scannet_infos_train_40class_corrected.pkl"
    
    # Stage 1 class definitions
    stage1_nyu40_ids = [5, 8, 39, 23, 3, 7, 9]  
    stage1_names = ['chair', 'door', 'otherfurniture', 'books', 'cabinet', 'table', 'window']
    
    print(f"Stage 1 classes: {stage1_names}")
    print(f"Pseudo labels file: {pseudo_labels_pkl}")
    print(f"Training data file: {train_pkl}")
    
    # Run evaluation
    results = evaluate_pseudo_labels_full(
        pseudo_labels_pkl, train_pkl, stage1_nyu40_ids, stage1_names,
        iou_thresholds=[0.25, 0.5]
    )
    
    # Save results
    output_file = "stage1_pseudo_labels/evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Evaluation results saved to: {output_file}")
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📊 FINAL PSEUDO LABEL EVALUATION RESULTS")
    print("=" * 80)
    
    for iou_key in ['iou_0.25', 'iou_0.5']:
        if iou_key in results:
            iou_val = iou_key.split('_')[1]
            result = results[iou_key]
            
            print(f"\n🎯 IoU Threshold {iou_val}:")
            print(f"  Overall mAP: {result['mAP']:.4f}")
            print(f"  Average Precision: {result['average_precision']:.4f}")  
            print(f"  Average Recall: {result['average_recall']:.4f}")
            
            print(f"\n  Per-Class Results:")
            for class_name in stage1_names:
                if class_name in result['per_class']:
                    cls_result = result['per_class'][class_name]
                    print(f"    {class_name:12s}: AP={cls_result['ap']:.4f}, P={cls_result['precision']:.4f}, R={cls_result['recall']:.4f}")
                    print(f"    {' '*12}  Pred={cls_result['num_predictions']:5d}, GT={cls_result['num_gt']:4d}, TP={cls_result['true_positives']:4d}")


if __name__ == "__main__":
    main()
