#!/usr/bin/env python3
"""
Fast Pseudo Label Evaluation - Compute mAP on Training Set

Simplified version that computes mAP and per-class AP more efficiently
by using a simpler IoU computation and sampling approach.

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


def compute_bbox_iou_3d_simple(box1, box2):
    """Simplified 3D IoU computation using axis-aligned bounding boxes."""
    # Convert to min/max format (ignoring rotation for speed)
    x1_min, y1_min, z1_min = box1[0] - box1[3]/2, box1[1] - box1[4]/2, box1[2] - box1[5]/2
    x1_max, y1_max, z1_max = box1[0] + box1[3]/2, box1[1] + box1[4]/2, box1[2] + box1[5]/2
    
    x2_min, y2_min, z2_min = box2[0] - box2[3]/2, box2[1] - box2[4]/2, box2[2] - box2[5]/2
    x2_max, y2_max, z2_max = box2[0] + box2[3]/2, box2[1] + box2[4]/2, box2[2] + box2[5]/2
    
    # Intersection
    inter_x = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter_z = max(0, min(z1_max, z2_max) - max(z1_min, z2_min))
    
    intersection = inter_x * inter_y * inter_z
    
    # Union
    vol1 = box1[3] * box1[4] * box1[5]
    vol2 = box2[3] * box2[4] * box2[5]
    union = vol1 + vol2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_ap_simple(tp, fp, num_gt):
    """Simplified AP computation."""
    if num_gt == 0:
        return 0.0
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recalls = tp_cumsum / num_gt
    
    # Simple AP computation (area under precision-recall curve)
    ap = 0.0
    prev_recall = 0.0
    
    for i in range(len(recalls)):
        if recalls[i] != prev_recall:
            ap += precisions[i] * (recalls[i] - prev_recall)
            prev_recall = recalls[i]
    
    return ap


def evaluate_class_fast(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.25):
    """Fast evaluation for a single class."""
    num_pred = len(pred_boxes)
    num_gt = len(gt_boxes)
    
    if num_pred == 0:
        return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'tp': 0, 'fp': 0, 'num_pred': 0, 'num_gt': num_gt}
    
    if num_gt == 0:
        return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'tp': 0, 'fp': num_pred, 'num_pred': num_pred, 'num_gt': 0}
    
    # Sort by confidence
    sorted_indices = np.argsort(pred_scores)[::-1]
    sorted_boxes = [pred_boxes[i] for i in sorted_indices]
    sorted_scores = [pred_scores[i] for i in sorted_indices]
    
    tp = np.zeros(num_pred)
    fp = np.zeros(num_pred)
    gt_matched = [False] * num_gt
    
    for pred_idx, pred_box in enumerate(sorted_boxes):
        best_iou = 0.0
        best_gt_idx = -1
        
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_matched[gt_idx]:
                continue
            iou = compute_bbox_iou_3d_simple(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold:
            tp[pred_idx] = 1
            gt_matched[best_gt_idx] = True
        else:
            fp[pred_idx] = 1
    
    ap = compute_ap_simple(tp, fp, num_gt)
    total_tp = int(np.sum(tp))
    total_fp = int(np.sum(fp))
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / num_gt if num_gt > 0 else 0.0
    
    return {
        'ap': ap,
        'precision': precision,
        'recall': recall,
        'tp': total_tp,
        'fp': total_fp,
        'num_pred': num_pred,
        'num_gt': num_gt
    }


def main():
    """Main fast evaluation."""
    print("🎯 Fast Pseudo Label Evaluation on Training Set")
    print("=" * 60)
    
    # Configuration
    stage1_nyu40_ids = [5, 8, 39, 23, 3, 7, 9]  
    stage1_names = ['chair', 'door', 'otherfurniture', 'books', 'cabinet', 'table', 'window']
    
    print(f"Stage 1 classes: {stage1_names}")
    
    # Load pseudo labels
    print("🔮 Loading pseudo labels...")
    with open("stage1_pseudo_labels/stage1_pseudo_labels_complete.pkl", 'rb') as f:
        pseudo_data = pickle.load(f)
    
    # Load ground truth
    print("📚 Loading ground truth...")
    with open("data/scannet/scannet_infos_train_40class_corrected.pkl", 'rb') as f:
        train_data = pickle.load(f)
    
    # Organize ground truth by scene
    print("🔍 Organizing ground truth...")
    gt_by_scene = {}
    total_gt_objects = {name: 0 for name in stage1_names}
    
    for scene in train_data:
        if 'annos' not in scene:
            continue
            
        scene_id = scene['point_cloud']['lidar_idx']
        gt_labels = scene['annos'].get('class', [])
        gt_boxes = scene['annos'].get('gt_boxes_upright_depth', [])
        
        # Check if scene has Stage 1 objects
        if not any(label in stage1_nyu40_ids for label in gt_labels):
            continue
        
        scene_gt = {name: {'boxes': [], 'count': 0} for name in stage1_names}
        
        for i, label in enumerate(gt_labels):
            if label in stage1_nyu40_ids and i < len(gt_boxes):
                class_idx = stage1_nyu40_ids.index(label)
                class_name = stage1_names[class_idx]
                scene_gt[class_name]['boxes'].append(gt_boxes[i])
                scene_gt[class_name]['count'] += 1
                total_gt_objects[class_name] += 1
        
        gt_by_scene[scene_id] = scene_gt
    
    print(f"✅ Loaded GT for {len(gt_by_scene)} scenes")
    for name, count in total_gt_objects.items():
        print(f"  {name}: {count} objects")
    
    # Organize pseudo labels by class
    print("🔮 Organizing pseudo labels...")
    pseudo_by_class = {name: {'boxes': [], 'scores': []} for name in stage1_names}
    total_pred_objects = {name: 0 for name in stage1_names}
    
    matched_scenes = 0
    for scene_id, scene_data in pseudo_data.items():
        if scene_id not in gt_by_scene:
            continue  # Skip scenes not in GT (shouldn't happen but be safe)
        
        matched_scenes += 1
        boxes = np.array(scene_data['boxes'])
        scores = np.array(scene_data['scores'])
        labels = np.array(scene_data['labels'])
        
        for box, score, label in zip(boxes, scores, labels):
            if 0 <= label < len(stage1_names):
                class_name = stage1_names[label]
                pseudo_by_class[class_name]['boxes'].append(box)
                pseudo_by_class[class_name]['scores'].append(score)
                total_pred_objects[class_name] += 1
    
    print(f"✅ Matched {matched_scenes} scenes between pseudo labels and GT")
    for name, count in total_pred_objects.items():
        print(f"  {name}: {count} predictions")
    
    # Collect all ground truth boxes by class
    gt_by_class = {name: [] for name in stage1_names}
    for scene_gt in gt_by_scene.values():
        for class_name in stage1_names:
            gt_by_class[class_name].extend(scene_gt[class_name]['boxes'])
    
    # Evaluate each class
    print("\n🎯 Evaluating classes...")
    results = {}
    
    for iou_thresh in [0.25, 0.5]:
        print(f"\n📊 IoU Threshold: {iou_thresh}")
        class_results = {}
        total_ap = 0.0
        
        for class_name in stage1_names:
            result = evaluate_class_fast(
                pseudo_by_class[class_name]['boxes'],
                pseudo_by_class[class_name]['scores'],
                gt_by_class[class_name],
                iou_threshold=iou_thresh
            )
            class_results[class_name] = result
            total_ap += result['ap']
            
            print(f"  {class_name:12s}: AP={result['ap']:.4f}, P={result['precision']:.4f}, R={result['recall']:.4f}")
            print(f"  {' '*12}  Pred={result['num_pred']:5d}, GT={result['num_gt']:4d}, TP={result['tp']:4d}")
        
        map_score = total_ap / len(stage1_names)
        results[f'iou_{iou_thresh}'] = {
            'mAP': map_score,
            'per_class': class_results
        }
        
        print(f"\n  🎯 Overall mAP@{iou_thresh}: {map_score:.4f}")
    
    # Save results
    output_file = "stage1_pseudo_labels/evaluation_results_fast.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL PSEUDO LABEL PERFORMANCE ON TRAINING SET")
    print("=" * 60)
    
    print(f"\n🎯 mAP@0.25: {results['iou_0.25']['mAP']:.4f}")
    print(f"🎯 mAP@0.50: {results['iou_0.5']['mAP']:.4f}")
    
    print(f"\nPer-Class AP@0.25:")
    for class_name in stage1_names:
        ap = results['iou_0.25']['per_class'][class_name]['ap']
        print(f"  {class_name:12s}: {ap:.4f}")


if __name__ == "__main__":
    main()
