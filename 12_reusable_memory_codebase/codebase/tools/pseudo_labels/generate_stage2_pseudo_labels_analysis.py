#!/usr/bin/env python3
"""
Generate Stage 2 Pseudo Labels for Comprehensive Analysis

This script generates pseudo labels for Stage 2 training scenes using the Stage 1 model,
then performs comprehensive analysis including:
1. Threshold sweep analysis (confidence × IoU)
2. Per-scene and per-GT-object tracking
3. Matching analysis with Hungarian algorithm
4. Statistical evaluation and bug detection

Key differences from Stage 1 pseudo label generation:
- Targets Stage 2 scenes (containing classes 7-13)
- Stage 1 model can only predict classes 0-6
- Provides partial supervision for incremental learning replay

Date: 2025-09-03
"""

import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from collections import Counter, defaultdict
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Add project root to path
# NOTE: this script was moved from repo root to `tools/pseudo_labels/`.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mmcv import Config
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.core.bbox import BaseInstance3DBoxes

# Import class mappings
sys.path.append(str(project_root / 'configs' / '_base_' / 'class_mappings'))
from scannet_dynamic_head_mappings import (
    SCANNET_DYNAMIC_HEAD_CLASSES,
    VALID_NYU40_IDS_DYNAMIC_HEAD,
    DYNAMIC_HEAD_GCI_TO_NYU40,
    get_stage_definitions
)


def iou_3d_boxes(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Calculate 3D IoU between two sets of axis-aligned bounding boxes.
    
    Args:
        boxes1: (N, 7) array of boxes [x, y, z, dx, dy, dz, angle]
        boxes2: (M, 7) array of boxes [x, y, z, dx, dy, dz, angle]
        
    Returns:
        (N, M) array of IoU values
    """
    # Extract box parameters
    xyz1, dims1 = boxes1[:, :3], boxes1[:, 3:6]
    xyz2, dims2 = boxes2[:, :3], boxes2[:, 3:6]
    
    # Calculate min/max coordinates for each box
    min1 = xyz1 - dims1 / 2  # (N, 3)
    max1 = xyz1 + dims1 / 2  # (N, 3)
    min2 = xyz2 - dims2 / 2  # (M, 3)
    max2 = xyz2 + dims2 / 2  # (M, 3)
    
    # Broadcast for pairwise comparisons
    min1 = min1[:, None, :]  # (N, 1, 3)
    max1 = max1[:, None, :]  # (N, 1, 3)
    min2 = min2[None, :, :]  # (1, M, 3)
    max2 = max2[None, :, :]  # (1, M, 3)
    
    # Calculate intersection
    inter_min = np.maximum(min1, min2)  # (N, M, 3)
    inter_max = np.minimum(max1, max2)  # (N, M, 3)
    inter_dims = np.maximum(0, inter_max - inter_min)  # (N, M, 3)
    inter_vol = np.prod(inter_dims, axis=2)  # (N, M)
    
    # Calculate union
    vol1 = np.prod(dims1, axis=1)[:, None]  # (N, 1)
    vol2 = np.prod(dims2, axis=1)[None, :]  # (1, M)
    union_vol = vol1 + vol2 - inter_vol  # (N, M)
    
    # Calculate IoU, avoiding division by zero
    iou = np.where(union_vol > 0, inter_vol / union_vol, 0.0)
    return iou


def match_boxes_hungarian(pred_boxes: np.ndarray, pred_scores: np.ndarray, 
                         gt_boxes: np.ndarray, iou_threshold: float = 0.25) -> Dict:
    """Match predicted boxes to ground truth using Hungarian algorithm.
    
    Args:
        pred_boxes: (N, 7) predicted bounding boxes
        pred_scores: (N,) predicted confidence scores  
        gt_boxes: (M, 7) ground truth bounding boxes
        iou_threshold: Minimum IoU for considering a match
        
    Returns:
        Dictionary with matching results and statistics
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return {
            'matches': [],
            'unmatched_preds': list(range(len(pred_boxes))),
            'unmatched_gts': list(range(len(gt_boxes))),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'mean_iou': 0.0
        }
    
    # Calculate IoU matrix
    iou_matrix = iou_3d_boxes(pred_boxes, gt_boxes)  # (N, M)
    
    # Hungarian matching on cost matrix (1 - IoU)
    cost_matrix = 1 - iou_matrix
    pred_indices, gt_indices = linear_sum_assignment(cost_matrix)
    
    # Filter matches by IoU threshold
    matches = []
    matched_pred_ids = set()
    matched_gt_ids = set()
    ious = []
    
    for pred_idx, gt_idx in zip(pred_indices, gt_indices):
        iou_val = iou_matrix[pred_idx, gt_idx]
        if iou_val >= iou_threshold:
            matches.append({
                'pred_idx': int(pred_idx),
                'gt_idx': int(gt_idx),
                'iou': float(iou_val),
                'confidence': float(pred_scores[pred_idx])
            })
            matched_pred_ids.add(pred_idx)
            matched_gt_ids.add(gt_idx)
            ious.append(iou_val)
    
    # Find unmatched predictions and ground truths
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_pred_ids]
    unmatched_gts = [i for i in range(len(gt_boxes)) if i not in matched_gt_ids]
    
    # Calculate metrics
    tp = len(matches)
    fp = len(unmatched_preds)
    fn = len(unmatched_gts)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(ious) if ious else 0.0
    
    return {
        'matches': matches,
        'unmatched_preds': unmatched_preds,
        'unmatched_gts': unmatched_gts,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'mean_iou': mean_iou
    }


class Stage2PseudoLabelAnalyzer:
    """Comprehensive analyzer for Stage 2 pseudo labels."""
    
    def __init__(self, checkpoint_path: str, output_dir: Path):
        self.checkpoint_path = checkpoint_path
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Stage definitions
        stage_defs = get_stage_definitions('frequency')
        self.stage1_def = stage_defs[0]  # Stage 1: classes 0-6
        self.stage2_def = stage_defs[1]  # Stage 2: classes 7-13
        
        print(f"🎯 Stage 1 Classes (GCI 0-6): {self.stage1_def['class_names']}")
        print(f"🎯 Stage 2 Classes (GCI 7-13): {self.stage2_def['class_names']}")
        
        # Analysis configuration
        self.confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        self.iou_thresholds = [0.15, 0.25, 0.35, 0.5, 0.65, 0.75, 0.85, 0.9]
        
        # Load model
        self.model = self._load_stage1_model()
        
        # Statistics tracking
        self.stats = {
            'total_scenes_processed': 0,
            'scenes_with_detections': 0,
            'scenes_with_stage1_gt': 0,
            'total_predictions': 0,
            'total_stage1_gt_objects': 0,
            'class_counts': Counter(),
            'processing_errors': []
        }
    
    def _load_stage1_model(self):
        """Load Stage 1 model with correct 7-class architecture."""
        print("🏗️  Loading Stage 1 model...")
        
        # Load incremental learning config for Stage 1
        config_path = project_root / "configs/incremental/scannet/tr3d_dynamic_head_s5_pure_finetuning.py"
        config = Config.fromfile(str(config_path))
        
        # Ensure correct Stage 1 configuration
        config.model.head.n_classes = 7
        config.model.test_cfg.score_thr = 0.05  # Use low threshold, filter later
        config.model.test_cfg.nms_pre = 1000
        config.model.test_cfg.iou_thr = 0.5
        
        # Initialize model
        model = init_model(config, self.checkpoint_path, device='cuda:0')
        model.eval()
        
        print(f"✅ Stage 1 model loaded (7 classes)")
        return model
    
    def _extract_axis_align_matrix(self, scene_info: Dict) -> Optional[np.ndarray]:
        """Extract axis alignment matrix from scene annotation."""
        if 'annos' not in scene_info:
            return None
        
        axis_align_matrix = scene_info['annos'].get('axis_align_matrix', None)
        if axis_align_matrix is not None:
            return axis_align_matrix.astype(np.float32)
        return None
    
    def _apply_alignment_to_boxes(self, boxes_unaligned: np.ndarray, axis_align_matrix: np.ndarray) -> np.ndarray:
        """Apply axis alignment transformation to bounding boxes."""
        if axis_align_matrix is None:
            return boxes_unaligned
        
        # Extract rotation and translation from 4x4 matrix
        rot_mat = axis_align_matrix[:3, :3]
        trans_vec = axis_align_matrix[:3, 3]
        
        # Apply transformation to box centers
        centers = boxes_unaligned[:, :3]  # x, y, z
        centers_aligned = centers @ rot_mat.T + trans_vec
        
        # Keep original dimensions and angles (assuming upright boxes)
        boxes_aligned = boxes_unaligned.copy()
        boxes_aligned[:, :3] = centers_aligned
        
        return boxes_aligned
    
    def _get_stage2_scenes(self, all_scenes: List[Dict]) -> List[Dict]:
        """Get Stage 2 training scenes (scenes containing Stage 2 classes 7-13)."""
        stage2_nyu40_ids = set(self.stage2_def['nyu40_ids'])
        stage2_scenes = []
        
        for scene in all_scenes:
            if 'annos' not in scene or scene['annos']['gt_num'] == 0:
                continue
            
            gt_labels_nyu40 = scene['annos']['class']
            has_stage2_class = any(nyu40_id in stage2_nyu40_ids for nyu40_id in gt_labels_nyu40)
            
            if has_stage2_class:
                stage2_scenes.append(scene)
        
        print(f"📊 Found {len(stage2_scenes)} Stage 2 training scenes")
        return stage2_scenes
    
    def _extract_stage1_gt_objects(self, scene_info: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """Extract Stage 1 GT objects from a scene for analysis."""
        if 'annos' not in scene_info or scene_info['annos']['gt_num'] == 0:
            return np.array([]).reshape(0, 7), np.array([])
        
        gt_boxes = scene_info['annos']['gt_boxes_upright_depth'].astype(np.float32)
        gt_labels_nyu40 = scene_info['annos']['class']
        
        # Filter for Stage 1 classes only
        stage1_nyu40_ids = set(self.stage1_def['nyu40_ids'])
        stage1_mask = np.array([nyu40_id in stage1_nyu40_ids for nyu40_id in gt_labels_nyu40])
        
        if not stage1_mask.any():
            return np.array([]).reshape(0, 7), np.array([])
        
        stage1_gt_boxes = gt_boxes[stage1_mask]
        stage1_gt_labels = np.array(gt_labels_nyu40)[stage1_mask]
        
        return stage1_gt_boxes, stage1_gt_labels
    
    def generate_scene_analysis(self, scene_info: Dict) -> Optional[Dict]:
        """Generate pseudo labels and analysis for a single Stage 2 scene."""
        scene_id = scene_info['point_cloud']['lidar_idx']
        pts_path = project_root / "data/scannet" / scene_info['pts_path']
        
        if not pts_path.exists():
            self.stats['processing_errors'].append(f"Point cloud not found: {pts_path}")
            return None
        
        try:
            # Extract Stage 1 GT objects for comparison
            stage1_gt_boxes, stage1_gt_labels = self._extract_stage1_gt_objects(scene_info)
            
            # Run inference with Stage 1 model
            with torch.no_grad():
                result = inference_detector(self.model, str(pts_path))
            
            # Extract raw predictions
            if not (result and len(result) > 0 and len(result[0]) > 0):
                predictions = np.array([]).reshape(0, 7)
                scores = np.array([])
                labels = np.array([])
            else:
                res = result[0][0]
                if isinstance(res, dict) and 'boxes_3d' in res:
                    boxes_3d = res['boxes_3d']
                    scores_3d = res['scores_3d']
                    labels_3d = res['labels_3d']
                    
                    # Convert to numpy
                    if hasattr(scores_3d, 'cpu'):
                        scores = scores_3d.cpu().numpy()
                        labels = labels_3d.cpu().numpy()
                    else:
                        scores = np.array(scores_3d)
                        labels = np.array(labels_3d)
                    
                    if hasattr(boxes_3d, 'tensor'):
                        predictions = boxes_3d.tensor.cpu().numpy()
                    else:
                        predictions = np.array(boxes_3d)
                else:
                    predictions = np.array([]).reshape(0, 7)
                    scores = np.array([])
                    labels = np.array([])
            
            # CRITICAL FIX: Apply axis alignment transformation to match visualization
            # Both predictions and GT need to be transformed to ALIGNED coordinates
            # to match what is shown in the visualization app
            axis_align_matrix = self._extract_axis_align_matrix(scene_info)
            has_alignment = False
            
            if axis_align_matrix is not None and len(predictions) > 0:
                # Apply axis alignment transformation to predictions (same as visualization app)
                rot_mat = axis_align_matrix[:3, :3]
                trans_vec = axis_align_matrix[:3, 3]
                
                # Apply alignment to prediction centers
                for i in range(len(predictions)):
                    center = predictions[i, :3]
                    aligned_center = center @ rot_mat.T + trans_vec
                    predictions[i, :3] = aligned_center
                
                # Apply rotation to yaw angles if boxes have yaw component
                if predictions.shape[1] > 6:
                    # Extract rotation angle from rotation matrix
                    rotation_angle = np.arctan2(rot_mat[1, 0], rot_mat[0, 0])
                    # Apply the axis alignment rotation to the yaw
                    predictions[:, 6] += rotation_angle
                
                has_alignment = True
            
            # CRITICAL FIX: Convert predictions from bottom_center to gravity_center
            # Model predictions use bottom_center (Z at bottom of box) 
            # For proper alignment with GT and point clouds, we need gravity_center (Z at geometric center)
            if len(predictions) > 0:
                predictions[:, 2] += predictions[:, 5] * 0.5  # Z += height * 0.5
            
            # Convert prediction labels to NYU40 IDs for consistency
            if len(labels) > 0:
                nyu40_labels = np.array([DYNAMIC_HEAD_GCI_TO_NYU40[gci] for gci in labels])
            else:
                nyu40_labels = np.array([])
            
            # Update statistics
            self.stats['total_predictions'] += len(scores)
            self.stats['total_stage1_gt_objects'] += len(stage1_gt_boxes)
            if len(stage1_gt_boxes) > 0:
                self.stats['scenes_with_stage1_gt'] += 1
            
            for gci in labels:
                if gci < len(self.stage1_def['class_names']):
                    self.stats['class_counts'][self.stage1_def['class_names'][gci]] += 1
            
            return {
                'scene_id': scene_id,
                'predictions': {
                    'boxes': predictions.astype(np.float32),
                    'scores': scores.astype(np.float32),
                    'labels_gci': labels.astype(np.int64),
                    'labels_nyu40': nyu40_labels.astype(np.int64),
                },
                'stage1_ground_truth': {
                    'boxes': stage1_gt_boxes.astype(np.float32),
                    'labels_nyu40': stage1_gt_labels.astype(np.int64),
                },
                'axis_align_matrix': axis_align_matrix.astype(np.float32) if axis_align_matrix is not None else None,
                'has_alignment': has_alignment,  # True if axis alignment was applied to predictions
                'num_predictions': len(scores),
                'num_stage1_gt': len(stage1_gt_boxes)
            }
            
        except Exception as e:
            error_msg = f"Scene {scene_id}: {str(e)}"
            self.stats['processing_errors'].append(error_msg)
            print(f"❌ Error processing {scene_id}: {e}")
            return None
    
    def run_threshold_sweep_analysis(self, scene_data: Dict) -> Dict:
        """Run comprehensive threshold sweep analysis on scene data."""
        print("🔍 Running threshold sweep analysis...")
        
        results = {}
        total_combinations = len(self.confidence_thresholds) * len(self.iou_thresholds)
        
        with tqdm(total=total_combinations, desc="Threshold combinations") as pbar:
            for conf_thresh in self.confidence_thresholds:
                for iou_thresh in self.iou_thresholds:
                    key = f"conf_{conf_thresh:.2f}_iou_{iou_thresh:.2f}"
                    results[key] = self._analyze_single_threshold_combination(
                        scene_data, conf_thresh, iou_thresh
                    )
                    pbar.update(1)
        
        return results
    
    def _analyze_single_threshold_combination(self, scene_data: Dict, 
                                           conf_thresh: float, iou_thresh: float) -> Dict:
        """Analyze all scenes at a single confidence/IoU threshold combination."""
        total_matches = 0
        total_predictions = 0
        total_stage1_gt = 0
        scene_results = []
        
        for scene_id, scene_info in scene_data.items():
            predictions = scene_info['predictions']
            stage1_gt = scene_info['stage1_ground_truth']
            
            # Apply confidence threshold
            if len(predictions['scores']) > 0:
                conf_mask = predictions['scores'] >= conf_thresh
                filtered_boxes = predictions['boxes'][conf_mask]
                filtered_scores = predictions['scores'][conf_mask]
            else:
                filtered_boxes = predictions['boxes']
                filtered_scores = predictions['scores']
            
            # Perform matching
            matching_result = match_boxes_hungarian(
                filtered_boxes, filtered_scores, stage1_gt['boxes'], iou_thresh
            )
            
            scene_result = {
                'scene_id': scene_id,
                'num_predictions': len(filtered_boxes),
                'num_stage1_gt': len(stage1_gt['boxes']),
                'num_matches': len(matching_result['matches']),
                'precision': matching_result['precision'],
                'recall': matching_result['recall'],
                'f1': matching_result['f1'],
                'mean_iou': matching_result['mean_iou']
            }
            scene_results.append(scene_result)
            
            total_matches += len(matching_result['matches'])
            total_predictions += len(filtered_boxes)
            total_stage1_gt += len(stage1_gt['boxes'])
        
        # Calculate overall metrics
        overall_precision = total_matches / total_predictions if total_predictions > 0 else 0.0
        overall_recall = total_matches / total_stage1_gt if total_stage1_gt > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_recall / 
                     (overall_precision + overall_recall)) if (overall_precision + overall_recall) > 0 else 0.0
        
        return {
            'confidence_threshold': conf_thresh,
            'iou_threshold': iou_thresh,
            'overall_metrics': {
                'precision': overall_precision,
                'recall': overall_recall,
                'f1': overall_f1,
                'total_matches': total_matches,
                'total_predictions': total_predictions,
                'total_stage1_gt': total_stage1_gt
            },
            'scene_results': scene_results
        }
    
    def generate_per_object_analysis(self, scene_data: Dict) -> Dict:
        """Generate detailed per-GT-object analysis."""
        print("📊 Generating per-object analysis...")
        
        per_object_results = []
        
        for scene_id, scene_info in tqdm(scene_data.items(), desc="Analyzing objects"):
            predictions = scene_info['predictions']
            stage1_gt = scene_info['stage1_ground_truth']
            
            if len(stage1_gt['boxes']) == 0:
                continue
            
            # For each GT object, find best matching prediction across all thresholds
            for gt_idx, (gt_box, gt_label) in enumerate(zip(stage1_gt['boxes'], stage1_gt['labels_nyu40'])):
                if len(predictions['boxes']) == 0:
                    best_match = None
                    max_iou = 0.0
                else:
                    # Calculate IoU with all predictions
                    ious = iou_3d_boxes(gt_box.reshape(1, -1), predictions['boxes'])[0]
                    max_iou_idx = np.argmax(ious)
                    max_iou = ious[max_iou_idx]
                    
                    if max_iou > 0:
                        best_match = {
                            'pred_idx': int(max_iou_idx),
                            'confidence': float(predictions['scores'][max_iou_idx]),
                            'iou': float(max_iou),
                            'pred_label_gci': int(predictions['labels_gci'][max_iou_idx]),
                            'pred_label_nyu40': int(predictions['labels_nyu40'][max_iou_idx])
                        }
                    else:
                        best_match = None
                
                # Determine class name from NYU40 ID
                gt_class_name = None
                for class_name, nyu40_id in zip(self.stage1_def['class_names'], self.stage1_def['nyu40_ids']):
                    if nyu40_id == gt_label:
                        gt_class_name = class_name
                        break
                
                object_result = {
                    'scene_id': scene_id,
                    'gt_idx': gt_idx,
                    'gt_label_nyu40': int(gt_label),
                    'gt_class_name': gt_class_name,
                    'gt_box': gt_box.tolist(),
                    'best_match': best_match,
                    'max_iou': float(max_iou)
                }
                per_object_results.append(object_result)
        
        return {
            'per_object_results': per_object_results,
            'summary': self._summarize_per_object_results(per_object_results)
        }
    
    def _summarize_per_object_results(self, per_object_results: List[Dict]) -> Dict:
        """Summarize per-object analysis results."""
        class_summaries = defaultdict(lambda: {
            'total_objects': 0,
            'detected_objects': 0,
            'total_iou': 0.0,
            'confidence_scores': []
        })
        
        for obj_result in per_object_results:
            class_name = obj_result['gt_class_name']
            if class_name is None:
                continue
            
            class_summaries[class_name]['total_objects'] += 1
            
            if obj_result['best_match'] is not None:
                class_summaries[class_name]['detected_objects'] += 1
                class_summaries[class_name]['total_iou'] += obj_result['max_iou']
                class_summaries[class_name]['confidence_scores'].append(
                    obj_result['best_match']['confidence']
                )
        
        # Calculate final statistics
        final_summary = {}
        for class_name, stats in class_summaries.items():
            detection_rate = stats['detected_objects'] / stats['total_objects']
            avg_iou = stats['total_iou'] / stats['detected_objects'] if stats['detected_objects'] > 0 else 0.0
            avg_confidence = np.mean(stats['confidence_scores']) if stats['confidence_scores'] else 0.0
            
            final_summary[class_name] = {
                'total_objects': stats['total_objects'],
                'detected_objects': stats['detected_objects'],
                'detection_rate': detection_rate,
                'average_iou': avg_iou,
                'average_confidence': avg_confidence,
                'confidence_std': np.std(stats['confidence_scores']) if len(stats['confidence_scores']) > 1 else 0.0
            }
        
        return final_summary
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        else:
            return obj

    def save_analysis_results(self, scene_data: Dict, threshold_results: Dict, 
                            per_object_results: Dict):
        """Save all analysis results to files."""
        print("💾 Saving analysis results...")
        
        # Create subdirectories
        (self.output_dir / "pseudo_labels").mkdir(exist_ok=True)
        (self.output_dir / "analysis").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        
        # 1. Save pseudo labels in training-compatible format
        training_format = {}
        raw_format = {}
        
        for scene_id, scene_info in scene_data.items():
            predictions = scene_info['predictions']
            
            # Training format (NYU40 IDs, like original pipeline)
            training_format[scene_id] = {
                'scene_id': scene_id,
                'boxes': predictions['boxes'],
                'scores': predictions['scores'],
                'labels': predictions['labels_nyu40'],  # NYU40 format
                'num_detections': predictions['boxes'].shape[0]
            }
            
            # Raw format with extra analysis data
            raw_format[scene_id] = scene_info
        
        with open(self.output_dir / "pseudo_labels/stage2_pseudo_labels.pkl", 'wb') as f:
            pickle.dump(training_format, f)
        
        with open(self.output_dir / "pseudo_labels/stage2_pseudo_labels_raw.pkl", 'wb') as f:
            pickle.dump(raw_format, f)
        
        # 2. Save threshold sweep results (convert numpy types)
        threshold_results_json = self._convert_numpy_types(threshold_results)
        with open(self.output_dir / "analysis/iou_sweep_8x9_matrix.json", 'w') as f:
            json.dump(threshold_results_json, f, indent=2)
        
        # 3. Save per-object analysis (convert numpy types)
        per_object_results_json = self._convert_numpy_types(per_object_results)
        with open(self.output_dir / "analysis/per_object_analysis.json", 'w') as f:
            json.dump(per_object_results_json, f, indent=2)
        
        # 4. Save overall statistics
        overall_stats = {
            'generation_info': {
                'checkpoint': self.checkpoint_path,
                'target_stage': 'Stage 2',
                'stage1_classes': self.stage1_def['class_names'],
                'stage2_classes': self.stage2_def['class_names'],
                'confidence_thresholds': self.confidence_thresholds,
                'iou_thresholds': self.iou_thresholds,
                'generation_timestamp': str(np.datetime64('now'))
            },
            'processing_statistics': self.stats,
            'summary': self._generate_overall_summary(scene_data, threshold_results)
        }
        
        overall_stats_json = self._convert_numpy_types(overall_stats)
        with open(self.output_dir / "analysis/overall_statistics.json", 'w') as f:
            json.dump(overall_stats_json, f, indent=2)
        
        # 5. Find optimal thresholds
        optimal_thresholds = self._find_optimal_thresholds(threshold_results)
        optimal_thresholds_json = self._convert_numpy_types(optimal_thresholds)
        with open(self.output_dir / "analysis/optimal_thresholds.json", 'w') as f:
            json.dump(optimal_thresholds_json, f, indent=2)
        
        print(f"✅ Results saved to {self.output_dir}")
        
    def _generate_overall_summary(self, scene_data: Dict, threshold_results: Dict) -> Dict:
        """Generate overall summary statistics."""
        total_scenes = len(scene_data)
        scenes_with_predictions = sum(1 for s in scene_data.values() if s['num_predictions'] > 0)
        scenes_with_stage1_gt = sum(1 for s in scene_data.values() if s['num_stage1_gt'] > 0)
        
        return {
            'scene_statistics': {
                'total_stage2_scenes': total_scenes,
                'scenes_with_predictions': scenes_with_predictions,
                'scenes_with_stage1_gt': scenes_with_stage1_gt,
                'prediction_rate': scenes_with_predictions / total_scenes if total_scenes > 0 else 0.0
            },
            'prediction_statistics': {
                'total_predictions': self.stats['total_predictions'],
                'total_stage1_gt_objects': self.stats['total_stage1_gt_objects'],
                'predictions_per_scene': self.stats['total_predictions'] / total_scenes if total_scenes > 0 else 0.0,
                'stage1_gt_per_scene': self.stats['total_stage1_gt_objects'] / scenes_with_stage1_gt if scenes_with_stage1_gt > 0 else 0.0
            },
            'class_distribution': dict(self.stats['class_counts'])
        }
    
    def _find_optimal_thresholds(self, threshold_results: Dict) -> Dict:
        """Find optimal threshold combinations based on different criteria."""
        best_f1 = {'key': None, 'f1': 0.0}
        best_precision = {'key': None, 'precision': 0.0}
        best_recall = {'key': None, 'recall': 0.0}
        
        for key, result in threshold_results.items():
            metrics = result['overall_metrics']
            
            if metrics['f1'] > best_f1['f1']:
                best_f1 = {'key': key, 'f1': metrics['f1'], 'conf': result['confidence_threshold'], 'iou': result['iou_threshold']}
            
            if metrics['precision'] > best_precision['precision']:
                best_precision = {'key': key, 'precision': metrics['precision'], 'conf': result['confidence_threshold'], 'iou': result['iou_threshold']}
            
            if metrics['recall'] > best_recall['recall']:
                best_recall = {'key': key, 'recall': metrics['recall'], 'conf': result['confidence_threshold'], 'iou': result['iou_threshold']}
        
        return {
            'best_f1_score': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'recommended_for_training': best_f1  # F1 balance is usually best for training
        }
    
    def run_full_analysis(self) -> Path:
        """Run the complete Stage 2 pseudo label analysis pipeline."""
        print("🚀 Starting Stage 2 Pseudo Label Analysis")
        print("=" * 60)
        
        # Load training data and get Stage 2 scenes
        train_pkl = project_root / "data/scannet/scannet_infos_train_40class_corrected.pkl"
        with open(train_pkl, 'rb') as f:
            all_scenes = pickle.load(f)
        
        stage2_scenes = self._get_stage2_scenes(all_scenes)
        
        # Generate pseudo labels and extract GT for all scenes
        print("🔍 Generating pseudo labels for Stage 2 scenes...")
        scene_data = {}
        
        for scene_info in tqdm(stage2_scenes, desc="Processing scenes"):
            scene_id = scene_info['point_cloud']['lidar_idx']
            
            result = self.generate_scene_analysis(scene_info)
            if result is not None:
                scene_data[scene_id] = result
                self.stats['scenes_with_detections'] += 1
            
            self.stats['total_scenes_processed'] += 1
        
        print(f"✅ Generated analysis for {len(scene_data)}/{len(stage2_scenes)} scenes")
        
        if not scene_data:
            print("❌ No scenes with valid data found!")
            return self.output_dir
        
        # Run threshold sweep analysis
        threshold_results = self.run_threshold_sweep_analysis(scene_data)
        
        # Generate per-object analysis
        per_object_results = self.generate_per_object_analysis(scene_data)
        
        # Save all results
        self.save_analysis_results(scene_data, threshold_results, per_object_results)
        
        # Print summary
        print("\n" + "=" * 60)
        print("🎉 STAGE 2 PSEUDO LABEL ANALYSIS COMPLETE")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"📊 Analyzed {len(scene_data)} scenes")
        print(f"🔢 Total predictions: {self.stats['total_predictions']}")
        print(f"🎯 Total Stage 1 GT objects: {self.stats['total_stage1_gt_objects']}")
        print(f"🔍 Threshold combinations: {len(self.confidence_thresholds)} × {len(self.iou_thresholds)} = {len(threshold_results)}")
        
        if self.stats['processing_errors']:
            print(f"⚠️ Processing errors: {len(self.stats['processing_errors'])}")
        
        return self.output_dir


def main():
    """Main execution function."""
    print("🚀 Stage 2 Pseudo Label Generation and Analysis")
    print("=" * 60)
    
    # Configuration
    checkpoint_path = "stage_1_checkpoints/epoch_12.pth"
    output_dir = Path("stage2_pseudo_labels_analysis")
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for path in Path(".").glob("**/stage_1_checkpoints/*.pth"):
            print(f"  {path}")
        return
    
    # Run analysis
    analyzer = Stage2PseudoLabelAnalyzer(checkpoint_path, output_dir)
    result_dir = analyzer.run_full_analysis()
    
    print(f"\n🔍 Next steps:")
    print(f"1. Review results in: {result_dir}")
    print(f"2. Check optimal thresholds in: {result_dir}/analysis/optimal_thresholds.json")
    print(f"3. Use visualization tool to inspect specific scenes")
    print(f"4. Validate pseudo labels format for training pipeline")


if __name__ == "__main__":
    main()
