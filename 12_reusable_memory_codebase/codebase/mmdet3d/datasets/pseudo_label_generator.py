"""
Pseudo Label Generator for Incremental Learning

This module generates pseudo labels for replay scenes to enhance
incremental learning performance. It uses confidence-based filtering
and 3D NMS to ensure high-quality pseudo labels.
"""

import numpy as np
import torch
import copy
import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from mmdet3d.core.bbox import Box3DMode, Coord3DMode
from mmdet3d.core.bbox.structures import DepthInstance3DBoxes
from mmdet3d.datasets.pseudo_label_utils import (
    pairwise_aligned_iou,
    nms_indices_iou,
)
import logging


class PseudoLabelGenerator:
    """Generate pseudo labels for scenes using trained model predictions.
    
    This class handles the generation, filtering, and caching of pseudo labels
    for incremental learning scenarios. It applies confidence thresholding
    and 3D NMS to ensure high-quality pseudo labels.
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.7,
                 nms_threshold: float = 0.3,
                 cache_dir: Optional[str] = None,
                 max_pseudo_per_scene: int = 50,
                 debug_mode: bool = True,
                 class_thresholds: Optional[Dict[int, float]] = None):
        """Initialize the pseudo label generator.
        
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
        # Optional per-class confidence thresholds (model class idx -> thr)
        self.class_thresholds = class_thresholds or None
        
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
            'final_pseudo_labels': 0
        }
        
        # Visualization data tracking
        self.viz_data = {
            'stage_statistics': {},
            'scene_samples': {},
            'confidence_histograms': {},
            'class_distributions': {}
        }
        self.max_viz_scenes_per_stage = 15  # Limit scenes for visualization
    
    def generate_pseudo_labels(self,
                             model,
                             data_loader,
                             stage_id: int,
                             seen_classes: List[int],
                             device: str = 'cuda') -> Dict[str, Any]:
        """Generate pseudo labels for all scenes in the data loader.
        
        Args:
            model: Trained model for generating predictions
            data_loader: Data loader containing scenes
            stage_id: Current stage ID for caching
            seen_classes: List of class indices seen up to current stage
            device: Device to run inference on
            
        Returns:
            Dictionary mapping scene_id to pseudo label data
        """
        model.eval()
        pseudo_labels = {}
        
        self.logger.info(f"Generating pseudo labels for stage {stage_id}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"Seen classes: {seen_classes}")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_loader):
                # Extract scene information
                scene_ids = self._extract_scene_ids(batch_data)
                
                # Run inference
                predictions = model(return_loss=False, rescale=True, **batch_data)
                
                # Process each scene in the batch
                for scene_idx, scene_id in enumerate(scene_ids):
                    scene_predictions = self._extract_scene_predictions(
                        predictions, scene_idx
                    )
                    
                    # Filter and process predictions
                    pseudo_labels_scene = self._process_scene_predictions(
                        scene_predictions, seen_classes, scene_id, generated_stage_id=stage_id
                    )
                    
                    if pseudo_labels_scene:
                        pseudo_labels[scene_id] = pseudo_labels_scene
                
                if self.debug_mode and batch_idx % 10 == 0:
                    self.logger.debug(f"Processed {batch_idx + 1} batches")
        
        # Cache results if cache directory is provided
        if self.cache_dir:
            cache_file = os.path.join(
                self.cache_dir, f"pseudo_labels_stage_{stage_id}.json"
            )
            self._cache_pseudo_labels(pseudo_labels, cache_file)
        
        self._log_statistics()
        
        # Collect visualization data for this stage
        self._collect_visualization_data(pseudo_labels, stage_id, seen_classes)
        
        return pseudo_labels
    
    def _extract_scene_ids(self, batch_data: Dict) -> List[str]:
        """Extract scene IDs from batch data."""
        scene_ids = []
        
        # Try different possible locations for scene IDs
        if 'img_metas' in batch_data:
            for img_meta in batch_data['img_metas']:
                if isinstance(img_meta, list):
                    img_meta = img_meta[0]  # Handle nested structure
                
                # Try various fields where scene ID might be stored
                scene_id = None
                for field in ['sample_idx', 'scene_id', 'lidar_idx']:
                    if field in img_meta:
                        scene_id = str(img_meta[field])
                        break
                
                if scene_id is None and 'pts_filename' in img_meta:
                    # Extract from filename as fallback
                    scene_id = os.path.basename(img_meta['pts_filename']).replace('.bin', '')
                
                scene_ids.append(scene_id or f"unknown_{len(scene_ids)}")
        
        return scene_ids
    
    def _extract_scene_predictions(self, 
                                 predictions: List[Dict], 
                                 scene_idx: int) -> Dict[str, Any]:
        """Extract predictions for a specific scene from batch predictions."""
        if scene_idx >= len(predictions):
            return {}
        
        scene_pred = predictions[scene_idx]
        
        # Extract key prediction components
        result = {}
        if 'boxes_3d' in scene_pred:
            result['boxes_3d'] = scene_pred['boxes_3d']
        if 'scores_3d' in scene_pred:
            result['scores_3d'] = scene_pred['scores_3d']
        if 'labels_3d' in scene_pred:
            result['labels_3d'] = scene_pred['labels_3d']
            
        return result
    
    def _process_scene_predictions(self,
                                 predictions: Dict[str, Any],
                                 seen_classes: List[int],
                                 scene_id: str,
                                 generated_stage_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Process predictions for a single scene to generate pseudo labels.
        
        Args:
            predictions: Raw model predictions for the scene
            seen_classes: Classes seen up to current stage
            scene_id: Scene identifier for debugging
            
        Returns:
            Processed pseudo labels or None if no valid labels
        """
        if not predictions or 'scores_3d' not in predictions:
            return None
        
        # Extract prediction tensors
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
        else:
            boxes_np = np.array(boxes_3d)
        
        self.stats['total_predictions'] += len(scores_3d)
        
        # Step 1: Filter by confidence threshold (supports per-class overrides)
        if self.class_thresholds:
            # Build per-prediction thresholds based on predicted class
            # Default to global threshold when class not in overrides
            thr_arr = np.full_like(scores_3d, fill_value=self.confidence_threshold, dtype=np.float32)
            # Vectorized mapping: for each unique label, apply its specific threshold
            for cls_id in np.unique(labels_3d):
                if int(cls_id) in self.class_thresholds:
                    thr_arr[labels_3d == cls_id] = float(self.class_thresholds[int(cls_id)])
            confidence_mask = scores_3d >= thr_arr
        else:
            confidence_mask = scores_3d >= self.confidence_threshold
        if not confidence_mask.any():
            return None
        
        filtered_scores = scores_3d[confidence_mask]
        filtered_labels = labels_3d[confidence_mask]
        filtered_boxes = boxes_np[confidence_mask]
        
        self.stats['filtered_by_confidence'] += (len(scores_3d) - confidence_mask.sum())
        
        # Step 2: Filter by seen classes only
        seen_class_mask = np.isin(filtered_labels, seen_classes)
        if not seen_class_mask.any():
            return None
        
        filtered_scores = filtered_scores[seen_class_mask]
        filtered_labels = filtered_labels[seen_class_mask]
        filtered_boxes = filtered_boxes[seen_class_mask]
        
        # Step 3: Apply 3D NMS to remove overlapping detections
        try:
            # Apply per-class IoU NMS for better suppression stability
            kept_global: List[int] = []
            for cls in np.unique(filtered_labels):
                cls_idx = np.where(filtered_labels == cls)[0]
                if cls_idx.size == 0:
                    continue
                cls_boxes = filtered_boxes[cls_idx]
                cls_scores = filtered_scores[cls_idx]
                keep_local = self._apply_3d_nms(
                    cls_boxes, cls_scores, self.nms_threshold
                )
                if keep_local.size:
                    kept_global.extend(cls_idx[keep_local].tolist())
            kept_global = np.array(sorted(set(kept_global)), dtype=np.int64)
            if kept_global.size == 0:
                return None
            final_scores = filtered_scores[kept_global]
            final_labels = filtered_labels[kept_global]
            final_boxes = filtered_boxes[kept_global]
            nms_filtered_count = len(filtered_scores) - len(kept_global)
            self.stats['filtered_by_nms'] += nms_filtered_count
        except Exception as e:
            self.logger.error(f"3D NMS failed for scene {scene_id}: {e}")
            return None
        
        # Step 4: Limit number of pseudo labels per scene
        if len(final_scores) > self.max_pseudo_per_scene:
            # Sort by confidence and keep top predictions
            sort_indices = np.argsort(final_scores)[::-1][:self.max_pseudo_per_scene]
            final_scores = final_scores[sort_indices]
            final_labels = final_labels[sort_indices]
            final_boxes = final_boxes[sort_indices]
        
        self.stats['final_pseudo_labels'] += len(final_scores)
        
        # Convert back to expected format
        pseudo_labels = {
            'gt_boxes_upright_depth': final_boxes,
            'class': final_labels.astype(np.int64),
            'gt_num': len(final_boxes),
            'confidence_scores': final_scores,
            'is_pseudo': True,
            'generated_scene': scene_id,
            'generated_stage': generated_stage_id,
            'confidence_threshold': self.confidence_threshold
        }
        
        if self.debug_mode:
            self.logger.debug(f"Scene {scene_id}: Generated {len(final_boxes)} pseudo labels")
            for i, (score, label) in enumerate(zip(final_scores, final_labels)):
                self.logger.debug(f"  Label {i}: class={label}, confidence={score:.3f}")
        
        return pseudo_labels
    
    def _apply_3d_nms(self, 
                     boxes: np.ndarray, 
                     scores: np.ndarray, 
                     nms_threshold: float) -> np.ndarray:
        """Apply axis-aligned IoU NMS; fallback to distance-NMS if needed."""
        try:
            return nms_indices_iou(boxes, scores, iou_thr=nms_threshold)
        except Exception as e:
            self.logger.warning(f"IoU NMS failed, fallback to simple NMS: {e}")
            return self._simple_3d_nms(boxes, scores, nms_threshold)
    
    def _simple_3d_nms(self, 
                      boxes: np.ndarray, 
                      scores: np.ndarray, 
                      nms_threshold: float) -> np.ndarray:
        """Simple 3D NMS implementation as fallback.
        
        This uses center distance instead of IoU for simplicity and robustness.
        """
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
            remaining_centers = boxes[remaining_indices][:, :3]  # Fix: use correct slicing
            
            # Calculate Euclidean distances
            distances = np.linalg.norm(remaining_centers - current_center[np.newaxis, :], axis=1)
            
            # Keep boxes that are far enough away
            # Use a distance threshold based on average box size
            current_size = np.mean(boxes[current_idx][3:6])  # w, h, d
            distance_threshold = current_size * nms_threshold * 2  # Heuristic
            
            keep_mask = distances > distance_threshold
            sorted_indices = remaining_indices[keep_mask]
        
        return np.array(keep, dtype=np.int64)
    
    def _cache_pseudo_labels(self, pseudo_labels: Dict, cache_file: str):
        """Cache pseudo labels to disk for reuse."""
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_labels = {}
            for scene_id, labels in pseudo_labels.items():
                serializable_scene = {}
                for key, value in labels.items():
                    if isinstance(value, np.ndarray):
                        serializable_scene[key] = value.tolist()
                    else:
                        serializable_scene[key] = value
                serializable_labels[scene_id] = serializable_scene
            
            with open(cache_file, 'w') as f:
                json.dump({
                    'pseudo_labels': serializable_labels,
                    'stats': self.stats,
                    'config': {
                        'confidence_threshold': self.confidence_threshold,
                        'nms_threshold': self.nms_threshold,
                        'max_pseudo_per_scene': self.max_pseudo_per_scene
                    }
                }, f, indent=2)
            
            self.logger.info(f"Cached pseudo labels to {cache_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to cache pseudo labels: {e}")
    
    def load_cached_pseudo_labels(self, cache_file: str) -> Optional[Dict[str, Any]]:
        """Load pseudo labels from cache if available."""
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            pseudo_labels = {}
            for scene_id, labels in data['pseudo_labels'].items():
                converted_scene = {}
                for key, value in labels.items():
                    if key in ['gt_boxes_upright_depth', 'class', 'confidence_scores']:
                        converted_scene[key] = np.array(value)
                    else:
                        converted_scene[key] = value
                pseudo_labels[scene_id] = converted_scene
            
            self.logger.info(f"Loaded cached pseudo labels from {cache_file}")
            return pseudo_labels
            
        except Exception as e:
            self.logger.error(f"Failed to load cached pseudo labels: {e}")
            return None
    
    def _log_statistics(self):
        """Log generation statistics."""
        total = self.stats['total_predictions']
        if total == 0:
            return
        
        self.logger.info("Pseudo label generation statistics:")
        self.logger.info(f"  Total predictions: {total}")
        self.logger.info(f"  Filtered by confidence: {self.stats['filtered_by_confidence']} "
                        f"({100 * self.stats['filtered_by_confidence'] / total:.1f}%)")
        self.logger.info(f"  Filtered by NMS: {self.stats['filtered_by_nms']} "
                        f"({100 * self.stats['filtered_by_nms'] / total:.1f}%)")
        self.logger.info(f"  Final pseudo labels: {self.stats['final_pseudo_labels']} "
                        f"({100 * self.stats['final_pseudo_labels'] / total:.1f}%)")
    
    def _collect_visualization_data(self, pseudo_labels: Dict[str, Any], stage_id: int, seen_classes: List[int]):
        """Collect data for visualization with smart sampling."""
        if not pseudo_labels:
            return
        
        # Collect stage-level statistics
        stage_stats = {
            'stage_id': stage_id,
            'total_scenes': len(pseudo_labels),
            'total_pseudo_labels': sum(data['gt_num'] for data in pseudo_labels.values()),
            'seen_classes': seen_classes.copy(),
            'confidence_threshold': self.confidence_threshold,
            'nms_threshold': self.nms_threshold,
            'stats': self.stats.copy()
        }
        
        # Collect confidence distribution
        all_confidences = []
        class_counts = {}
        
        # Sample scenes for detailed visualization (limit to prevent huge files)
        scene_ids = list(pseudo_labels.keys())
        np.random.seed(42)  # Reproducible sampling
        
        # Smart sampling: get diverse scenes (high, medium, low pseudo label counts)
        scene_pseudo_counts = [(sid, pseudo_labels[sid]['gt_num']) for sid in scene_ids]
        scene_pseudo_counts.sort(key=lambda x: x[1])  # Sort by pseudo label count
        
        # Sample from different quantiles
        n_scenes = min(len(scene_ids), self.max_viz_scenes_per_stage)
        if n_scenes > 0:
            # Get scenes from different quantiles for diversity
            indices = np.linspace(0, len(scene_pseudo_counts) - 1, n_scenes, dtype=int)
            sampled_scenes = [scene_pseudo_counts[i][0] for i in indices]
        else:
            sampled_scenes = []
        
        # Collect detailed data for sampled scenes
        scene_samples = {}
        for scene_id in sampled_scenes:
            scene_data = pseudo_labels[scene_id]
            
            # Collect confidence scores for this scene
            confidences = scene_data.get('confidence_scores', [])
            if len(confidences) > 0:
                all_confidences.extend(confidences)
                
                # Collect class counts
                classes = scene_data.get('class', [])
                for cls in classes:
                    class_counts[int(cls)] = class_counts.get(int(cls), 0) + 1
                
                # Store scene sample data
                scene_samples[scene_id] = {
                    'pseudo_labels': [
                        {
                            'box': box.tolist() if hasattr(box, 'tolist') else box,
                            'class': int(cls),
                            'confidence': float(conf)
                        }
                        for box, cls, conf in zip(
                            scene_data['gt_boxes_upright_depth'],
                            scene_data['class'],
                            scene_data['confidence_scores']
                        )
                    ],
                    'scene_statistics': {
                        'total_pseudo': int(scene_data['gt_num']),
                        'avg_confidence': float(np.mean(confidences)),
                        'min_confidence': float(np.min(confidences)),
                        'max_confidence': float(np.max(confidences)),
                        'confidence_std': float(np.std(confidences))
                    }
                }
        
        # Create confidence histogram
        if all_confidences:
            hist, bin_edges = np.histogram(all_confidences, bins=20, range=(0, 1))
            confidence_histogram = {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist(),
                'total_samples': len(all_confidences)
            }
        else:
            confidence_histogram = {'counts': [], 'bin_edges': [], 'total_samples': 0}
        
        # Store visualization data
        self.viz_data['stage_statistics'][stage_id] = stage_stats
        self.viz_data['scene_samples'][stage_id] = scene_samples
        self.viz_data['confidence_histograms'][stage_id] = confidence_histogram
        self.viz_data['class_distributions'][stage_id] = class_counts
        
        if self.debug_mode:
            self.logger.debug(f"Collected visualization data for stage {stage_id}:")
            self.logger.debug(f"  Sampled {len(scene_samples)} scenes for visualization")
            self.logger.debug(f"  Confidence samples: {len(all_confidences)}")
            self.logger.debug(f"  Class distribution: {class_counts}")
    
    def export_visualization_data(self, export_path: str):
        """Export collected visualization data to JSON file."""
        try:
            # Create export directory if needed
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Prepare final export data
            export_data = {
                'metadata': {
                    'generator_config': {
                        'confidence_threshold': self.confidence_threshold,
                        'nms_threshold': self.nms_threshold,
                        'max_pseudo_per_scene': self.max_pseudo_per_scene,
                        'debug_mode': self.debug_mode
                    },
                    'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                    'total_stages': len(self.viz_data['stage_statistics'])
                },
                'visualization_data': self.viz_data
            }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            self.logger.info(f"Visualization data exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to export visualization data: {e}")
            return False
