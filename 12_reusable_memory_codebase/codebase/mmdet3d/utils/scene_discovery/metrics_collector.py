"""
Utilities for collecting and analyzing per-scene training metrics.

This module provides tools to collect metrics during training and analyze
them to determine which scenes are most valuable for incremental learning.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import logging


class SceneMetricsCollector:
    """
    Collects and analyzes per-scene metrics during training.
    
    This class provides utilities for:
    - Collecting per-scene losses during training
    - Computing gradient norms for individual scenes
    - Analyzing scene utility for memory bank selection
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the metrics collector.
        
        Args:
            logger: Logger instance for debugging output
        """
        self.scene_losses = defaultdict(list)
        self.scene_gradients = defaultdict(list)
        self.scene_features = defaultdict(list)
        self.logger = logger or logging.getLogger(__name__)
    
    def collect_batch_metrics(self, 
                            batch_data: Dict,
                            losses: Dict[str, Union[torch.Tensor, float]], 
                            model: torch.nn.Module,
                            iteration: int,
                            collect_gradients: bool = False):
        """
        Collect metrics from a training batch.
        
        Args:
            batch_data: Batch data containing img_metas with scene information
            losses: Loss dictionary from forward pass
            model: The model being trained
            iteration: Current training iteration
            collect_gradients: Whether to compute and store gradients
        """
        img_metas = batch_data.get('img_metas', [])
        
        if not img_metas:
            self.logger.warning("No img_metas found in batch data")
            return
        
        for i, img_meta in enumerate(img_metas):
            scene_id = img_meta.get('scene_id', f'scene_{i}')
            
            # Collect loss metrics
            scene_metrics = {
                'iteration': iteration,
                'bbox_loss': self._extract_loss_for_sample(losses.get('bbox_loss'), i),
                'cls_loss': self._extract_loss_for_sample(losses.get('cls_loss'), i),
                'total_loss': self._extract_loss_for_sample(losses.get('loss'), i),
                'lr': losses.get('lr', 0.0) if isinstance(losses.get('lr'), (int, float)) else 0.0
            }
            
            self.scene_losses[scene_id].append(scene_metrics)
            
            # Optionally collect gradients (expensive)
            if collect_gradients:
                grad_norm = self._compute_gradient_norm_for_scene(
                    losses.get('loss', torch.tensor(0.0)), model, i
                )
                self.scene_gradients[scene_id].append({
                    'iteration': iteration,
                    'gradient_norm': grad_norm
                })
    
    def _extract_loss_for_sample(self, 
                                loss_tensor: Optional[Union[torch.Tensor, float]], 
                                index: int) -> float:
        """
        Extract loss value for a specific sample from batch loss.
        
        Args:
            loss_tensor: Loss tensor (could be scalar, 1D, or higher dimensional)
            index: Index of the sample in the batch
            
        Returns:
            Float loss value for the sample
        """
        if loss_tensor is None:
            return 0.0
        
        if isinstance(loss_tensor, (int, float)):
            return float(loss_tensor)
        
        if not torch.is_tensor(loss_tensor):
            return 0.0
        
        try:
            if loss_tensor.dim() == 0:  # Scalar loss (shared across batch)
                return loss_tensor.item()
            elif loss_tensor.dim() == 1:  # Per-sample loss
                if index < len(loss_tensor):
                    return loss_tensor[index].item()
                else:
                    return loss_tensor.mean().item()
            else:
                # Higher dimensional - take mean
                return loss_tensor.mean().item()
        except Exception as e:
            self.logger.warning(f"Error extracting loss for sample {index}: {e}")
            return 0.0
    
    def _compute_gradient_norm_for_scene(self, 
                                       loss: torch.Tensor, 
                                       model: torch.nn.Module,
                                       scene_index: int) -> float:
        """
        Compute gradient norm for a specific scene.
        
        Args:
            loss: Loss tensor
            model: The model
            scene_index: Index of scene in batch (currently not used)
            
        Returns:
            Gradient norm as float
        """
        if not torch.is_tensor(loss) or not loss.requires_grad:
            return 0.0
        
        try:
            # Compute gradients
            gradients = torch.autograd.grad(
                loss, 
                model.parameters(),
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            
            # Compute total gradient norm
            total_norm = 0.0
            for grad in gradients:
                if grad is not None:
                    total_norm += grad.data.norm(2).item() ** 2
            
            return np.sqrt(total_norm)
            
        except Exception as e:
            self.logger.warning(f"Error computing gradient norm: {e}")
            return 0.0
    
    def get_scene_utility_scores(self, 
                               weight_cls: float = 2.0,
                               weight_bbox: float = 1.0) -> Dict[str, float]:
        """
        Compute utility scores for each scene based on collected losses.
        
        Lower loss generally indicates higher utility for memory bank.
        
        Args:
            weight_cls: Weight for classification loss (higher = more important)
            weight_bbox: Weight for bbox regression loss
            
        Returns:
            Dictionary mapping scene_id to utility score
        """
        utility_scores = {}
        
        for scene_id, loss_history in self.scene_losses.items():
            if not loss_history:
                continue
                
            # Compute average losses over all iterations
            avg_bbox_loss = np.mean([m['bbox_loss'] for m in loss_history])
            avg_cls_loss = np.mean([m['cls_loss'] for m in loss_history])
            
            # Utility score: inverse of weighted loss
            # Lower loss = higher utility for preventing forgetting
            weighted_loss = weight_cls * avg_cls_loss + weight_bbox * avg_bbox_loss
            utility_score = 1.0 / (1.0 + weighted_loss)
            
            utility_scores[scene_id] = utility_score
        
        return utility_scores
    
    def get_scene_loss_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get comprehensive loss statistics for each scene.
        
        Returns:
            Dictionary with scene_id -> statistics mapping
        """
        stats = {}
        
        for scene_id, loss_history in self.scene_losses.items():
            if not loss_history:
                continue
            
            # Extract loss arrays
            bbox_losses = [m['bbox_loss'] for m in loss_history]
            cls_losses = [m['cls_loss'] for m in loss_history]
            total_losses = [m['total_loss'] for m in loss_history]
            
            # Compute statistics
            stats[scene_id] = {
                'num_iterations': len(loss_history),
                'bbox_loss_mean': np.mean(bbox_losses),
                'bbox_loss_std': np.std(bbox_losses),
                'bbox_loss_min': np.min(bbox_losses),
                'bbox_loss_max': np.max(bbox_losses),
                'cls_loss_mean': np.mean(cls_losses),
                'cls_loss_std': np.std(cls_losses),
                'cls_loss_min': np.min(cls_losses),
                'cls_loss_max': np.max(cls_losses),
                'total_loss_mean': np.mean(total_losses),
                'total_loss_std': np.std(total_losses),
                'loss_trend': self._compute_loss_trend(total_losses)
            }
        
        return stats
    
    def _compute_loss_trend(self, losses: List[float]) -> float:
        """
        Compute the trend in loss values (positive = increasing, negative = decreasing).
        
        Args:
            losses: List of loss values over time
            
        Returns:
            Slope of linear fit (trend)
        """
        if len(losses) < 2:
            return 0.0
        
        try:
            x = np.arange(len(losses))
            y = np.array(losses)
            
            # Linear regression to get slope
            slope, _ = np.polyfit(x, y, 1)
            return float(slope)
        except Exception:
            return 0.0
    
    def get_top_scenes_by_utility(self, 
                                 top_k: int = 50,
                                 min_iterations: int = 1) -> List[Tuple[str, float]]:
        """
        Get top-k scenes by utility score.
        
        Args:
            top_k: Number of top scenes to return
            min_iterations: Minimum iterations a scene must have been seen
            
        Returns:
            List of (scene_id, utility_score) tuples, sorted by utility
        """
        utility_scores = self.get_scene_utility_scores()
        
        # Filter by minimum iterations
        filtered_scores = {
            scene_id: score 
            for scene_id, score in utility_scores.items()
            if len(self.scene_losses[scene_id]) >= min_iterations
        }
        
        # Sort by utility score (descending)
        sorted_scenes = sorted(
            filtered_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_scenes[:top_k]
    
    def get_scene_gradient_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get gradient statistics for each scene (if gradients were collected).
        
        Returns:
            Dictionary with scene_id -> gradient statistics mapping
        """
        stats = {}
        
        for scene_id, grad_history in self.scene_gradients.items():
            if not grad_history:
                continue
            
            grad_norms = [g['gradient_norm'] for g in grad_history]
            
            stats[scene_id] = {
                'num_iterations': len(grad_history),
                'grad_norm_mean': np.mean(grad_norms),
                'grad_norm_std': np.std(grad_norms),
                'grad_norm_min': np.min(grad_norms),
                'grad_norm_max': np.max(grad_norms)
            }
        
        return stats
    
    def clear_metrics(self):
        """Clear all collected metrics."""
        self.scene_losses.clear()
        self.scene_gradients.clear()
        self.scene_features.clear()
        self.logger.info("All scene metrics cleared")
    
    def save_metrics_summary(self, filepath: str):
        """
        Save a summary of collected metrics to file.
        
        Args:
            filepath: Path to save the summary JSON file
        """
        import json
        
        summary = {
            'total_scenes': len(self.scene_losses),
            'scenes_with_gradients': len(self.scene_gradients),
            'utility_scores': self.get_scene_utility_scores(),
            'loss_statistics': self.get_scene_loss_statistics()
        }
        
        if self.scene_gradients:
            summary['gradient_statistics'] = self.get_scene_gradient_statistics()
        
        try:
            with open(filepath, 'w') as f:
                json.dump(summary, f, indent=2, default=self._json_serialize)
            
            self.logger.info(f"Metrics summary saved to: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving metrics summary: {e}")
    
    def _json_serialize(self, obj):
        """Handle non-serializable objects for JSON."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif torch.is_tensor(obj):
            return obj.detach().cpu().numpy().tolist()
        else:
            return str(obj)