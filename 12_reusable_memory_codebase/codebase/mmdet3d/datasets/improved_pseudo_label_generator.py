"""
Improved Pseudo Label Generator for Incremental Learning

This module provides an enhanced pseudo label generator that fixes critical issues:
1. Proper 3D IoU-based NMS instead of distance fallback
2. Correct model loading for dynamic head architectures  
3. Stage-aware class filtering and validation
4. Comprehensive quality metrics and validation

Date: 2025-08-31
"""

import numpy as np
import torch
import copy
import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from mmdet3d.core.bbox import Box3DMode, Coord3DMode
from mmdet3d.core.bbox.structures import DepthInstance3DBoxes
from mmcv import Config
from mmdet3d.apis import init_model


class ImprovedPseudoLabelGenerator:
    """Enhanced pseudo label generator with proper 3D NMS and model loading."""
    
    def __init__(self,
                 confidence_threshold: float = 0.7,
                 nms_threshold: float = 0.3,
                 cache_dir: Optional[str] = None,
                 max_pseudo_per_scene: int = 50,
                 debug_mode: bool = True):
        """Initialize the improved pseudo label generator.
        
        Args:
            confidence_threshold: Minimum confidence for pseudo labels
            nms_threshold: IoU threshold for 3D NMS
            cache_dir: Directory to cache pseudo labels
            max_pseudo_per_scene: Maximum pseudo labels per scene
            debug_mode: Enable detailed logging
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.cache_dir = cache_dir
        self.max_pseudo_per_scene = max_pseudo_per_scene
        self.debug_mode = debug_mode
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        if debug_mode:
            self.logger.setLevel(logging.DEBUG)
        
        # Create cache directory if specified
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            
        # Statistics tracking
        self.stats = {
            'total_predictions': 0,
            'filtered_by_confidence': 0,
            'filtered_by_nms': 0,
            'final_pseudo_labels': 0,
            'scenes_processed': 0,
            'average_confidence': 0.0,
            'class_distribution': {}
        }
    
    def load_stage_model(self, checkpoint_path: str, stage_classes: List[int], 
                        project_root: Optional[str] = None) -> torch.nn.Module:
        """
        Load model with correct architecture for the given stage.
        
        Args:
            checkpoint_path: Path to checkpoint file
            stage_classes: List of class indices for this stage
            project_root: Path to project root (for config loading)
            
        Returns:
            Loaded and configured model
        """
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent.absolute()
        else:
            project_root = Path(project_root)
            
        self.logger.info(f"Loading model for stage with {len(stage_classes)} classes")
        
        # Load base config and modify for this stage
        base_config_path = project_root / "configs/tr3d/tr3d_scannet-3d-35class.py"
        if not base_config_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_config_path}")
            
        config = Config.fromfile(str(base_config_path))
        
        # CRITICAL: Modify for current stage classes
        config.model.head.n_classes = len(stage_classes)
        
        self.logger.info(f"Modified config for {len(stage_classes)} classes")
        
        # Initialize model with correct architecture
        model = init_model(config, checkpoint_path, device='cuda:0')
        model.eval()
        
        # Validate model architecture matches expected
        if hasattr(model, 'head') and hasattr(model.head, 'n_classes'):
            actual_classes = model.head.n_classes
            if actual_classes != len(stage_classes):
                raise ValueError(f"Model architecture mismatch: {actual_classes} vs {len(stage_classes)} expected")
                
        self.logger.info(f"✅ Model loaded successfully with {len(stage_classes)} classes")
        
        return model
    
    def calculate_3d_iou_vectorized(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """
        Calculate 3D IoU between two sets of axis-aligned boxes.
        
        Args:
            boxes1: Array of shape (N, 6) with format [x, y, z, w, h, d]
            boxes2: Array of shape (M, 6) with format [x, y, z, w, h, d]
            
        Returns:
            IoU matrix of shape (N, M)
        """
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
            
        # Expand dimensions for broadcasting
        boxes1 = boxes1[:, None, :]  # (N, 1, 6)
        boxes2 = boxes2[None, :, :]  # (1, M, 6)
        
        # Extract centers and dimensions
        centers1, dims1 = boxes1[..., :3], boxes1[..., 3:6]
        centers2, dims2 = boxes2[..., :3], boxes2[..., 3:6]
        
        # Calculate box bounds
        min1 = centers1 - dims1/2  # (N, 1, 3)
        max1 = centers1 + dims1/2  # (N, 1, 3)
        min2 = centers2 - dims2/2  # (1, M, 3)
        max2 = centers2 + dims2/2  # (1, M, 3)
        
        # Calculate intersection bounds
        inter_min = np.maximum(min1, min2)  # (N, M, 3)
        inter_max = np.minimum(max1, max2)  # (N, M, 3)
        
        # Check if intersection exists
        valid_inter = np.all(inter_min < inter_max, axis=2)  # (N, M)
        
        # Calculate intersection volume
        inter_dims = np.maximum(0, inter_max - inter_min)  # (N, M, 3)
        inter_vol = np.prod(inter_dims, axis=2)  # (N, M)
        inter_vol = np.where(valid_inter, inter_vol, 0)
        
        # Calculate individual volumes
        vol1 = np.prod(dims1, axis=2)  # (N, 1)
        vol2 = np.prod(dims2, axis=2)  # (1, M)
        
        # Calculate union volume
        union_vol = vol1 + vol2 - inter_vol  # (N, M)
        
        # Calculate IoU
        iou = np.where(union_vol > 0, inter_vol / union_vol, 0)
        
        return iou
    
    def apply_proper_3d_nms(self, boxes: np.ndarray, scores: np.ndarray, 
                           nms_threshold: float) -> np.ndarray:
        """
        Apply proper 3D NMS using IoU calculation.
        
        Args:
            boxes: Array of 3D boxes (N, 6) [x, y, z, w, h, d]
            scores: Array of confidence scores (N,)
            nms_threshold: IoU threshold for suppression
            
        Returns:
            Array of indices of boxes to keep
        """
        if len(boxes) == 0:
            return np.array([], dtype=np.int64)
        
        if len(boxes) == 1:
            return np.array([0], dtype=np.int64)
        
        try:
            # Sort by scores (descending)
            sorted_indices = np.argsort(scores)[::-1]
            
            keep = []
            suppressed = set()
            
            for idx in sorted_indices:
                if idx in suppressed:
                    continue
                    
                keep.append(idx)
                
                # Calculate IoU with remaining boxes
                remaining_indices = [i for i in sorted_indices if i != idx and i not in suppressed]
                if not remaining_indices:
                    continue
                
                current_box = boxes[idx:idx+1]  # (1, 6)
                remaining_boxes = boxes[remaining_indices]  # (M, 6)
                
                # Calculate IoU
                iou_matrix = self.calculate_3d_iou_vectorized(current_box, remaining_boxes)  # (1, M)
                iou_scores = iou_matrix[0]  # (M,)
                
                # Suppress overlapping boxes
                for i, remaining_idx in enumerate(remaining_indices):
                    if iou_scores[i] > nms_threshold:
                        suppressed.add(remaining_idx)
            
            self.logger.debug(f"3D NMS: kept {len(keep)}/{len(boxes)} boxes")
            return np.array(keep, dtype=np.int64)
            
        except Exception as e:
            self.logger.warning(f"Proper 3D NMS failed, using fallback: {e}")
            return self._simple_3d_nms_fallback(boxes, scores, nms_threshold)
    
    def _simple_3d_nms_fallback(self, boxes: np.ndarray, scores: np.ndarray, 
                               nms_threshold: float) -> np.ndarray:
        """Fallback NMS using center distance."""
        if len(boxes) <= 1:
            return np.arange(len(boxes), dtype=np.int64)
        
        # Sort by confidence score (descending)
        sorted_indices = np.argsort(scores)[::-1]
        
        keep = []
        while len(sorted_indices) > 0:
            # Take the highest scoring box
            current_idx = sorted_indices[0]
            keep.append(current_idx)
            
            if len(sorted_indices) == 1:
                break
            
            # Calculate distances to remaining boxes
            current_center = boxes[current_idx][:3]  # x, y, z
            remaining_indices = sorted_indices[1:]
            remaining_centers = boxes[remaining_indices][:, :3]
            
            # Calculate Euclidean distances
            distances = np.linalg.norm(remaining_centers - current_center[np.newaxis, :], axis=1)
            
            # Keep boxes that are far enough away
            current_size = np.mean(boxes[current_idx][3:6])  # w, h, d
            distance_threshold = current_size * nms_threshold * 2  # Heuristic
            
            keep_mask = distances > distance_threshold
            sorted_indices = remaining_indices[keep_mask]
        
        return np.array(keep, dtype=np.int64)
    
    def generate_pseudo_labels_for_scenes(self, model: torch.nn.Module, 
                                        scenes_data: List[Dict],
                                        stage_classes: List[int],
                                        scene_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate pseudo labels for a list of scenes.
        
        Args:
            model: Trained model for generating predictions
            scenes_data: List of scene data dictionaries
            stage_classes: List of class indices for current stage
            scene_ids: Optional list of scene IDs (inferred if not provided)
            
        Returns:
            Dictionary mapping scene_id to pseudo label data
        """
        model.eval()
        pseudo_labels = {}
        
        self.logger.info(f"Generating pseudo labels for {len(scenes_data)} scenes")
        self.logger.info(f"Stage classes: {stage_classes}")
        
        with torch.no_grad():
            for i, scene_data in enumerate(scenes_data):
                scene_id = scene_ids[i] if scene_ids else scene_data.get('sample_idx', f'scene_{i}')
                
                try:
                    # Generate predictions for this scene
                    predictions = self._generate_scene_predictions(model, scene_data, scene_id)
                    
                    if predictions is None:
                        continue
                    
                    # Process predictions to create pseudo labels
                    pseudo_labels_scene = self._process_scene_predictions(
                        predictions, stage_classes, scene_id
                    )
                    
                    if pseudo_labels_scene and pseudo_labels_scene['gt_num'] > 0:
                        pseudo_labels[scene_id] = pseudo_labels_scene
                        self.logger.debug(f"Generated {pseudo_labels_scene['gt_num']} pseudo labels for {scene_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error processing scene {scene_id}: {e}")
                    continue
                
                if self.debug_mode and (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(scenes_data)} scenes")
        
        self.stats['scenes_processed'] = len(scenes_data)
        self._update_final_statistics(pseudo_labels)
        self._log_statistics()
        
        return pseudo_labels
    
    def _generate_scene_predictions(self, model: torch.nn.Module, scene_data: Dict, 
                                  scene_id: str) -> Optional[Dict]:
        """Generate model predictions for a single scene."""
        # Resolve ScanNet point clouds relative to the repo root.
        repo_root = Path(__file__).resolve().parents[2]
        scannet_root = repo_root / 'data' / 'scannet'
        if 'pts_path' in scene_data:
            pts_file = str(scannet_root / scene_data['pts_path'])
        else:
            self.logger.warning(f"No pts_path in scene data for {scene_id}, trying fallback")
            pts_file = str(scannet_root / 'points' / f'{scene_id}.bin')

        if not os.path.exists(pts_file):
            self.logger.warning(f"Point cloud file not found: {pts_file}")
            return None
            
        self.logger.debug(f"Loading point cloud from: {pts_file}")
            
        try:
            # Run inference using mmdet3d API
            from mmdet3d.apis import inference_detector
            results = inference_detector(model, pts_file)
            
            # Handle inference results structure
            # Results is a tuple: (predictions_list, metadata)
            # predictions_list[0] contains the actual predictions dict
            
            if len(results) > 0 and len(results[0]) > 0:
                result = results[0][0]  # First element of tuple, first element of list
                self.logger.debug(f"Result type: {type(result)}")
                
                if isinstance(result, dict) and 'boxes_3d' in result:
                    boxes_3d = result['boxes_3d']
                    scores_3d = result['scores_3d'] 
                    labels_3d = result['labels_3d']
                    
                    # Log basic info
                    num_predictions = len(scores_3d) if hasattr(scores_3d, '__len__') else 0
                    self.logger.debug(f"Raw predictions for {scene_id}: {num_predictions} detections")
                    
                    if num_predictions > 0:
                        # Log score statistics for debugging
                        if hasattr(scores_3d, 'detach'):
                            scores_np = scores_3d.detach().cpu().numpy()
                            self.logger.debug(f"Score range: [{scores_np.min():.3f}, {scores_np.max():.3f}], mean: {scores_np.mean():.3f}")
                    
                    return {
                        'boxes_3d': boxes_3d,
                        'scores_3d': scores_3d,
                        'labels_3d': labels_3d
                    }
                else:
                    self.logger.warning(f"Unexpected result format for scene {scene_id}: {type(result)}")
                    if isinstance(result, dict):
                        self.logger.warning(f"Available keys: {list(result.keys())}")
                    return None
            else:
                self.logger.warning(f"No valid predictions for scene {scene_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Inference failed for scene {scene_id}: {e}")
            return None
    
    def _process_scene_predictions(self, predictions: Dict[str, Any],
                                 stage_classes: List[int],
                                 scene_id: str) -> Optional[Dict[str, Any]]:
        """Process raw model predictions into filtered pseudo labels."""
        if not predictions or 'scores_3d' not in predictions:
            return None
        
        # Extract prediction components
        boxes_3d = predictions.get('boxes_3d')
        scores_3d = predictions['scores_3d']
        labels_3d = predictions.get('labels_3d')
        
        if boxes_3d is None or labels_3d is None:
            return None
        
        # Convert to numpy for processing
        if isinstance(scores_3d, torch.Tensor):
            scores_3d = scores_3d.cpu().numpy()
        if isinstance(labels_3d, torch.Tensor):
            labels_3d = labels_3d.cpu().numpy()
        if hasattr(boxes_3d, 'tensor'):
            boxes_np = boxes_3d.tensor.cpu().numpy()
        elif isinstance(boxes_3d, torch.Tensor):
            boxes_np = boxes_3d.cpu().numpy()
        else:
            boxes_np = np.array(boxes_3d)
        
        self.stats['total_predictions'] += len(scores_3d)
        
        # Step 1: Filter by confidence threshold
        confidence_mask = scores_3d >= self.confidence_threshold
        if not confidence_mask.any():
            return None
        
        filtered_scores = scores_3d[confidence_mask]
        filtered_labels = labels_3d[confidence_mask]
        filtered_boxes = boxes_np[confidence_mask]
        
        self.stats['filtered_by_confidence'] += (len(scores_3d) - confidence_mask.sum())
        
        # Step 2: Filter by stage classes only (map from model indices to class indices)
        stage_class_mask = np.isin(filtered_labels, stage_classes)
        if not stage_class_mask.any():
            return None
        
        filtered_scores = filtered_scores[stage_class_mask]
        filtered_labels = filtered_labels[stage_class_mask]
        filtered_boxes = filtered_boxes[stage_class_mask]
        
        # Step 3: Apply proper 3D NMS
        try:
            # Ensure boxes are in correct format [x, y, z, w, h, d]
            if filtered_boxes.shape[1] > 6:
                filtered_boxes = filtered_boxes[:, :6]  # Remove yaw if present
                
            nms_indices = self.apply_proper_3d_nms(
                filtered_boxes, filtered_scores, self.nms_threshold
            )
            
            if len(nms_indices) == 0:
                return None
            
            final_scores = filtered_scores[nms_indices]
            final_labels = filtered_labels[nms_indices]
            final_boxes = filtered_boxes[nms_indices]
            
            nms_filtered_count = len(filtered_scores) - len(nms_indices)
            self.stats['filtered_by_nms'] += nms_filtered_count
            
        except Exception as e:
            self.logger.error(f"NMS failed for scene {scene_id}: {e}")
            return None
        
        # Step 4: Limit number of pseudo labels per scene
        if len(final_scores) > self.max_pseudo_per_scene:
            # Sort by confidence and keep top predictions
            sort_indices = np.argsort(final_scores)[::-1][:self.max_pseudo_per_scene]
            final_scores = final_scores[sort_indices]
            final_labels = final_labels[sort_indices]
            final_boxes = final_boxes[sort_indices]
        
        self.stats['final_pseudo_labels'] += len(final_scores)
        
        # Convert to expected format
        pseudo_labels = {
            'gt_boxes_upright_depth': final_boxes,
            'class': final_labels.astype(np.int64),
            'gt_num': len(final_boxes),
            'confidence_scores': final_scores,
            'is_pseudo': True,
            'generated_from_scene': scene_id,
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold
        }
        
        if self.debug_mode:
            self.logger.debug(f"Scene {scene_id}: Generated {len(final_boxes)} pseudo labels")
            self.logger.debug(f"  Classes: {np.unique(final_labels)}")
            self.logger.debug(f"  Confidence range: [{final_scores.min():.3f}, {final_scores.max():.3f}]")
        
        return pseudo_labels
    
    def _update_final_statistics(self, pseudo_labels: Dict[str, Any]):
        """Update final statistics after processing all scenes."""
        if not pseudo_labels:
            return
            
        all_confidences = []
        class_counts = {}
        
        for scene_id, labels in pseudo_labels.items():
            confidences = labels.get('confidence_scores', [])
            classes = labels.get('class', [])
            
            all_confidences.extend(confidences)
            
            for cls in classes:
                class_counts[int(cls)] = class_counts.get(int(cls), 0) + 1
        
        if all_confidences:
            self.stats['average_confidence'] = np.mean(all_confidences)
            
        self.stats['class_distribution'] = class_counts
    
    def _log_statistics(self):
        """Log comprehensive generation statistics."""
        total = self.stats['total_predictions']
        if total == 0:
            self.logger.info("No predictions generated")
            return
        
        self.logger.info("Pseudo label generation statistics:")
        self.logger.info(f"  Scenes processed: {self.stats['scenes_processed']}")
        self.logger.info(f"  Total predictions: {total}")
        self.logger.info(f"  Filtered by confidence: {self.stats['filtered_by_confidence']} "
                        f"({100 * self.stats['filtered_by_confidence'] / total:.1f}%)")
        self.logger.info(f"  Filtered by NMS: {self.stats['filtered_by_nms']} "
                        f"({100 * self.stats['filtered_by_nms'] / total:.1f}%)")
        self.logger.info(f"  Final pseudo labels: {self.stats['final_pseudo_labels']} "
                        f"({100 * self.stats['final_pseudo_labels'] / total:.1f}%)")
        
        if self.stats['final_pseudo_labels'] > 0:
            self.logger.info(f"  Average confidence: {self.stats['average_confidence']:.3f}")
            self.logger.info(f"  Class distribution: {self.stats['class_distribution']}")
    
    def save_pseudo_labels(self, pseudo_labels: Dict[str, Any], save_path: str):
        """Save pseudo labels to disk in pickle format."""
        try:
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(pseudo_labels, f)
            self.logger.info(f"Saved pseudo labels for {len(pseudo_labels)} scenes to {save_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save pseudo labels: {e}")
            return False
    
    def export_statistics(self, export_path: str):
        """Export generation statistics to JSON file."""
        try:
            stats_with_config = {
                'generation_config': {
                    'confidence_threshold': self.confidence_threshold,
                    'nms_threshold': self.nms_threshold,
                    'max_pseudo_per_scene': self.max_pseudo_per_scene,
                    'debug_mode': self.debug_mode
                },
                'statistics': self.stats,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(export_path, 'w') as f:
                json.dump(stats_with_config, f, indent=2, default=str)
            
            self.logger.info(f"Statistics exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export statistics: {e}")
            return False


# Test function
def test_improved_generator():
    """Test the improved pseudo label generator."""
    print("Testing ImprovedPseudoLabelGenerator...")
    
    # Create test data
    generator = ImprovedPseudoLabelGenerator(
        confidence_threshold=0.7,
        nms_threshold=0.3,
        debug_mode=True
    )
    
    # Test 3D IoU calculation
    boxes1 = np.array([[0, 0, 0, 2, 2, 2], [3, 3, 3, 1, 1, 1]])
    boxes2 = np.array([[0, 0, 0, 2, 2, 2], [0.5, 0.5, 0.5, 1, 1, 1]])
    
    iou_matrix = generator.calculate_3d_iou_vectorized(boxes1, boxes2)
    print(f"IoU matrix shape: {iou_matrix.shape}")
    print(f"IoU values: {iou_matrix}")
    
    # Test NMS
    boxes = np.array([
        [0, 0, 0, 2, 2, 2],
        [0.1, 0.1, 0.1, 2, 2, 2],  # Overlapping
        [5, 5, 5, 1, 1, 1]  # Separate
    ])
    scores = np.array([0.9, 0.8, 0.7])
    
    keep = generator.apply_proper_3d_nms(boxes, scores, 0.5)
    print(f"NMS keep indices: {keep}")
    
    print("✅ ImprovedPseudoLabelGenerator test completed successfully!")


if __name__ == '__main__':
    test_improved_generator()
