"""
Custom MMCV hook for collecting per-scene training metrics.
Can be enabled/disabled via configuration without affecting normal training.

This hook collects per-scene losses, gradients, and features during training
to support scene discovery experiments for incremental learning.
"""

import os
import json
import time
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

import torch
import numpy as np
from mmcv.runner import HOOKS, Hook
from mmcv.utils import get_logger


@HOOKS.register_module()
class SceneMetricsHook(Hook):
    """
    Collects per-scene training metrics during incremental learning experiments.
    
    This hook is designed to support scene discovery experiments where we need
    to understand which scenes are most valuable for preventing catastrophic
    forgetting. All metrics collection can be disabled via configuration.
    
    Key Features:
    - Per-scene loss tracking
    - Optional gradient norm computation
    - Optional feature statistics
    - Configurable enable/disable
    - Automatic cleanup and saving
    """
    
    def __init__(self,
                 enabled: bool = True,
                 collect_losses: bool = True,
                 collect_gradients: bool = False,
                 collect_features: bool = False,
                 save_interval: int = 100,
                 output_dir: Optional[str] = None,
                 max_scenes_in_memory: int = 10000):
        """
        Initialize the scene metrics collection hook.
        
        Args:
            enabled: Master switch to enable/disable all metrics collection
            collect_losses: Whether to collect per-scene loss values
            collect_gradients: Whether to compute and store gradient norms (expensive)
            collect_features: Whether to collect feature statistics (expensive)
            save_interval: Save metrics every N iterations (0 = save only at end)
            output_dir: Directory to save metrics (None = work_dir/scene_metrics)
            max_scenes_in_memory: Maximum number of scene metrics to keep in memory
        """
        self.enabled = enabled
        self.collect_losses = collect_losses
        self.collect_gradients = collect_gradients
        self.collect_features = collect_features
        self.save_interval = save_interval
        self.output_dir = output_dir
        self.max_scenes_in_memory = max_scenes_in_memory
        
        # Early return if disabled
        if not self.enabled:
            self._disabled_init()
            return
            
        # Initialize storage
        self.scene_metrics = defaultdict(list)
        self.scene_gradient_norms = defaultdict(list)
        self.scene_feature_stats = defaultdict(list)
        
        # Tracking variables
        self.iteration_count = 0
        self.last_save_time = time.time()
        
        # Gradient collection state
        self.gradient_hooks = []
        self.current_grad_norms = {}
        
        # Logger
        self.logger = get_logger('SceneMetricsHook', log_level=logging.INFO)
        
        if self.enabled:
            self.logger.info("SceneMetricsHook initialized:")
            self.logger.info(f"  - Collect losses: {collect_losses}")
            self.logger.info(f"  - Collect gradients: {collect_gradients}")
            self.logger.info(f"  - Collect features: {collect_features}")
            self.logger.info(f"  - Save interval: {save_interval}")
    
    def _disabled_init(self):
        """Initialize in disabled state."""
        self.scene_metrics = {}
        self.scene_gradient_norms = {}
        self.scene_feature_stats = {}
        self.iteration_count = 0
        self.gradient_hooks = []
        self.current_grad_norms = {}
        self.logger = get_logger('SceneMetricsHook', log_level=logging.INFO)
        self.logger.info("SceneMetricsHook disabled - no metrics will be collected")
    
    def before_run(self, runner):
        """Setup before training begins."""
        if not self.enabled:
            return
            
        # Set output directory
        if self.output_dir is None:
            self.output_dir = os.path.join(runner.work_dir, 'scene_metrics')
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"Scene metrics will be saved to: {self.output_dir}")
    
    def before_train_iter(self, runner):
        """Setup before each training iteration."""
        if not self.enabled:
            return
            
        # Register gradient hooks if needed
        if self.collect_gradients and hasattr(runner.model, 'module'):
            self._register_gradient_hooks(runner.model.module)
    
    def after_train_iter(self, runner):
        """Collect metrics after each training iteration."""
        if not self.enabled:
            return
            
        try:
            # Collect per-scene metrics from this iteration
            self._collect_iteration_metrics(runner)
            
            # Update iteration counter
            self.iteration_count += 1
            
            # Save periodically
            if (self.save_interval > 0 and 
                self.iteration_count % self.save_interval == 0):
                self._save_metrics_to_file()
            
            # Memory management
            self._manage_memory()
            
        except Exception as e:
            self.logger.error(f"Error collecting scene metrics: {e}")
        
        finally:
            # Always cleanup gradient hooks
            self._cleanup_gradient_hooks()
    
    def after_run(self, runner):
        """Cleanup and save final metrics."""
        if not self.enabled:
            return
            
        try:
            # Save final metrics
            self._save_metrics_to_file()
            
            # Save summary statistics
            self._save_summary_statistics()
            
            self.logger.info("Scene metrics collection completed")
            self.logger.info(f"Total scenes tracked: {len(self.scene_metrics)}")
            self.logger.info(f"Total iterations: {self.iteration_count}")
            
        except Exception as e:
            self.logger.error(f"Error saving final metrics: {e}")
    
    def _collect_iteration_metrics(self, runner):
        """Collect metrics from the current training iteration."""
        # Get batch data
        batch_data = runner.data_batch
        img_metas = batch_data.get('img_metas', [])
        
        if not img_metas:
            return
        
        # Get losses from runner's log buffer
        if hasattr(runner, 'log_buffer') and runner.log_buffer.ready:
            losses = runner.log_buffer.output
            
            # Collect per-scene losses
            if self.collect_losses:
                self._collect_scene_losses(img_metas, losses)
        
        # Collect gradient norms if enabled
        if self.collect_gradients and self.current_grad_norms:
            self._collect_scene_gradients(img_metas)
        
        # Collect feature statistics if enabled
        if self.collect_features:
            self._collect_scene_features(img_metas, runner.model)
    
    def _collect_scene_losses(self, img_metas: List[Dict], losses: Dict[str, float]):
        """Collect per-scene loss values."""
        for i, img_meta in enumerate(img_metas):
            scene_id = img_meta.get('scene_id', f'batch_{i}')
            
            # Store loss metrics for this scene
            scene_loss_data = {
                'iteration': self.iteration_count,
                'timestamp': time.time(),
                'bbox_loss': losses.get('bbox_loss', 0.0),
                'cls_loss': losses.get('cls_loss', 0.0),
                'total_loss': losses.get('loss', 0.0),
                'lr': losses.get('lr', 0.0)
            }
            
            self.scene_metrics[scene_id].append(scene_loss_data)
    
    def _collect_scene_gradients(self, img_metas: List[Dict]):
        """Collect gradient norms for scenes."""
        for i, img_meta in enumerate(img_metas):
            scene_id = img_meta.get('scene_id', f'batch_{i}')
            
            # Get gradient norm for this scene (if available)
            grad_norm = self.current_grad_norms.get(i, 0.0)
            
            gradient_data = {
                'iteration': self.iteration_count,
                'timestamp': time.time(),
                'gradient_norm': grad_norm
            }
            
            self.scene_gradient_norms[scene_id].append(gradient_data)
        
        # Clear current gradient norms
        self.current_grad_norms.clear()
    
    def _collect_scene_features(self, img_metas: List[Dict], model):
        """Collect feature statistics for scenes."""
        # This is expensive and optional
        # Could collect things like:
        # - Feature magnitudes
        # - Feature diversity
        # - Activation patterns
        pass
    
    def _register_gradient_hooks(self, model):
        """Register hooks to capture gradient norms."""
        if hasattr(model, 'head') and hasattr(model.head, 'cls_conv'):
            # Hook on classification layer
            handle = model.head.cls_conv.register_backward_hook(
                self._gradient_hook_fn
            )
            self.gradient_hooks.append(handle)
    
    def _gradient_hook_fn(self, module, grad_input, grad_output):
        """Hook function to capture gradient norms."""
        if grad_output[0] is not None:
            # Compute gradient norm
            grad_norm = grad_output[0].norm().item()
            
            # Store gradient norm (indexed by position in batch)
            # This is a simplification - in practice you'd need to map
            # gradients back to specific scenes in the batch
            batch_size = grad_output[0].size(0) if len(grad_output[0].shape) > 0 else 1
            
            for i in range(batch_size):
                self.current_grad_norms[i] = grad_norm
    
    def _cleanup_gradient_hooks(self):
        """Remove all gradient hooks."""
        for handle in self.gradient_hooks:
            handle.remove()
        self.gradient_hooks.clear()
    
    def _manage_memory(self):
        """Manage memory usage by limiting stored metrics."""
        if len(self.scene_metrics) > self.max_scenes_in_memory:
            # Remove oldest entries
            scenes_to_remove = list(self.scene_metrics.keys())[:100]  # Remove 100 oldest
            
            for scene_id in scenes_to_remove:
                if scene_id in self.scene_metrics:
                    del self.scene_metrics[scene_id]
                if scene_id in self.scene_gradient_norms:
                    del self.scene_gradient_norms[scene_id]
                if scene_id in self.scene_feature_stats:
                    del self.scene_feature_stats[scene_id]
            
            self.logger.warning(f"Removed metrics for {len(scenes_to_remove)} scenes to manage memory")
    
    def _save_metrics_to_file(self):
        """Save collected metrics to JSON files."""
        if not self.output_dir:
            return
            
        timestamp = int(time.time())
        
        # Save loss metrics
        if self.collect_losses and self.scene_metrics:
            losses_file = os.path.join(
                self.output_dir, 
                f'scene_losses_iter_{self.iteration_count}_{timestamp}.json'
            )
            self._save_json_file(dict(self.scene_metrics), losses_file)
        
        # Save gradient metrics
        if self.collect_gradients and self.scene_gradient_norms:
            gradients_file = os.path.join(
                self.output_dir,
                f'scene_gradients_iter_{self.iteration_count}_{timestamp}.json'
            )
            self._save_json_file(dict(self.scene_gradient_norms), gradients_file)
        
        # Update last save time
        self.last_save_time = time.time()
    
    def _save_summary_statistics(self):
        """Save summary statistics about collected metrics."""
        if not self.output_dir:
            return
            
        summary = {
            'total_iterations': self.iteration_count,
            'total_scenes': len(self.scene_metrics),
            'collection_settings': {
                'collect_losses': self.collect_losses,
                'collect_gradients': self.collect_gradients,
                'collect_features': self.collect_features,
                'save_interval': self.save_interval
            },
            'scene_statistics': {}
        }
        
        # Add per-scene statistics
        for scene_id, metrics in self.scene_metrics.items():
            if metrics:
                losses = [m['total_loss'] for m in metrics]
                summary['scene_statistics'][scene_id] = {
                    'iterations_seen': len(metrics),
                    'avg_loss': np.mean(losses),
                    'std_loss': np.std(losses),
                    'min_loss': np.min(losses),
                    'max_loss': np.max(losses)
                }
        
        summary_file = os.path.join(self.output_dir, 'metrics_summary.json')
        self._save_json_file(summary, summary_file)
        
        self.logger.info(f"Metrics summary saved to: {summary_file}")
    
    def _save_json_file(self, data: Dict, filepath: str):
        """Save data to JSON file with error handling."""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=self._json_default)
        except Exception as e:
            self.logger.error(f"Failed to save {filepath}: {e}")
    
    def _json_default(self, obj):
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
    
    def get_scene_utility_scores(self) -> Dict[str, float]:
        """
        Compute utility scores for each scene based on collected metrics.
        
        Returns:
            Dictionary mapping scene_id to utility score
        """
        if not self.enabled or not self.scene_metrics:
            return {}
        
        utility_scores = {}
        
        for scene_id, metrics in self.scene_metrics.items():
            if not metrics:
                continue
                
            # Compute average losses
            avg_total_loss = np.mean([m['total_loss'] for m in metrics])
            avg_cls_loss = np.mean([m['cls_loss'] for m in metrics])
            avg_bbox_loss = np.mean([m['bbox_loss'] for m in metrics])
            
            # Utility score (lower loss = higher utility)
            # Emphasize classification loss for old class retention
            utility_score = 1.0 / (1.0 + 2.0 * avg_cls_loss + avg_bbox_loss)
            utility_scores[scene_id] = utility_score
        
        return utility_scores
    
    def clear_metrics(self):
        """Clear all collected metrics (useful for memory management)."""
        if not self.enabled:
            return
            
        self.scene_metrics.clear()
        self.scene_gradient_norms.clear()
        self.scene_feature_stats.clear()
        self.current_grad_norms.clear()
        
        self.logger.info("All scene metrics cleared")