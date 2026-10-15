"""
Gradient alignment computation for scene selection.

This module implements gradient alignment scoring to identify which scenes
have gradients that align well with validation data, indicating they would
be useful for preventing forgetting of old classes.
"""

import os
import logging
from typing import Dict, List, Tuple, Optional, Any
import json

import torch
import torch.nn.functional as F
import numpy as np
from mmcv import Config

from .data_utils import SceneDataLoader


class GradientAlignmentScorer:
    """
    Computes gradient alignment between candidate scenes and validation data.
    
    The idea is that scenes whose gradients align well with validation gradients
    on old classes are likely to help prevent catastrophic forgetting.
    """
    
    def __init__(self, 
                 model: torch.nn.Module, 
                 config: Config,
                 device: str = 'cuda:0',
                 cache_gradients: bool = True):
        """
        Initialize gradient alignment scorer.
        
        Args:
            model: Trained model (typically Stage 1 checkpoint)
            config: Training configuration  
            device: Device for computation
            cache_gradients: Whether to cache computed gradients
        """
        self.model = model
        self.config = config
        self.device = device
        self.cache_gradients = cache_gradients
        self.logger = logging.getLogger(__name__)
        
        # Initialize data loaders
        self.train_loader = SceneDataLoader(config, 'train', device)
        self.val_loader = SceneDataLoader(config, 'val', device)
        
        # Gradient cache
        self.gradient_cache = {} if cache_gradients else None
        self.validation_gradient_cache = {}
        
        # Ensure model is in eval mode
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(True)
        
        self.logger.info("GradientAlignmentScorer initialized")
        self.logger.info(f"  Device: {device}")
        self.logger.info(f"  Cache gradients: {cache_gradients}")
    
    def compute_validation_gradient(self, 
                                  target_classes: List[int],
                                  validation_batch_size: int = 8,
                                  cache_key: Optional[str] = None) -> Optional[torch.Tensor]:
        """
        Compute gradient on validation batch for specific classes.
        
        Args:
            target_classes: Classes to focus on (e.g., [0,1,2,3,4,5,6] for Stage 1)
            validation_batch_size: Size of validation batch
            cache_key: Optional key for caching (defaults to str(target_classes))
            
        Returns:
            Flattened gradient tensor for validation data
        """
        if cache_key is None:
            cache_key = str(sorted(target_classes))
        
        # Check cache first
        if self.validation_gradient_cache and cache_key in self.validation_gradient_cache:
            self.logger.debug(f"Using cached validation gradient for classes {target_classes}")
            return self.validation_gradient_cache[cache_key]
        
        self.logger.info(f"Computing validation gradient for classes {target_classes}")
        
        # Get validation batch with target classes
        validation_batch = self.val_loader.get_validation_batch_for_classes(
            target_classes, validation_batch_size
        )
        
        if validation_batch is None:
            self.logger.warning("Could not create validation batch from validation set")
            self.logger.info("Attempting fallback: using training scenes as pseudo-validation")
            
            # Fallback: Use training scenes as pseudo-validation
            validation_batch = self.train_loader.get_validation_batch_for_classes(
                target_classes, validation_batch_size
            )
            
            if validation_batch is None:
                self.logger.warning("Could not create validation batch from training set either")
                self.logger.info("Returning zero gradient as final fallback")
                return self._get_zero_gradient()
            else:
                self.logger.info("Successfully created pseudo-validation batch from training data")
        
        try:
            # Move batch to device
            validation_batch = self._move_batch_to_device(validation_batch)
            
            # Filter ground truth to target classes only
            filtered_batch = self._filter_batch_to_classes(validation_batch, target_classes)
            
            if filtered_batch is None:
                self.logger.warning("No objects found in validation batch after filtering")
                self.logger.info("Returning zero gradient due to no target class objects")
                return self._get_zero_gradient()
            
            # Compute gradient
            val_gradient = self._compute_gradient_for_batch(filtered_batch)
            
            # Cache the result
            if self.validation_gradient_cache is not None:
                self.validation_gradient_cache[cache_key] = val_gradient
            
            self.logger.info(f"Computed validation gradient: {val_gradient.shape} parameters")
            return val_gradient
            
        except Exception as e:
            self.logger.error(f"Error computing validation gradient: {e}")
            return None
    
    def compute_scene_gradient(self, 
                             scene_id: str,
                             target_classes: List[int]) -> Optional[torch.Tensor]:
        """
        Compute gradient for a specific candidate scene.
        
        Args:
            scene_id: Scene identifier
            target_classes: Classes to focus on
            
        Returns:
            Flattened gradient tensor for the scene
        """
        cache_key = f"{scene_id}_{sorted(target_classes)}"
        
        # Check cache first
        if self.gradient_cache and cache_key in self.gradient_cache:
            self.logger.debug(f"Using cached gradient for scene {scene_id}")
            return self.gradient_cache[cache_key]
        
        # Load scene data
        scene_data = self.train_loader.get_scene_by_id(scene_id)
        if scene_data is None:
            self.logger.warning(f"Could not load scene {scene_id}")
            return None
        
        try:
            # Move to device and create batch
            scene_batch = self._move_batch_to_device({'batch_key': [scene_data]})
            scene_batch = self._prepare_single_scene_batch(scene_data)
            
            # Filter to target classes
            filtered_batch = self._filter_batch_to_classes(scene_batch, target_classes)
            
            if filtered_batch is None:
                self.logger.debug(f"No target class objects in scene {scene_id}")
                # Return zero gradient
                return self._get_zero_gradient()
            
            # Compute gradient
            scene_gradient = self._compute_gradient_for_batch(filtered_batch)
            
            # Cache the result
            if self.gradient_cache is not None:
                self.gradient_cache[cache_key] = scene_gradient
            
            return scene_gradient
            
        except Exception as e:
            self.logger.warning(f"Error computing gradient for scene {scene_id}: {e}")
            return self._get_zero_gradient()
    
    def compute_alignment_score(self, 
                              scene_id: str,
                              validation_gradient: torch.Tensor,
                              target_classes: List[int]) -> float:
        """
        Compute alignment score between scene and validation gradients.
        
        Args:
            scene_id: Scene identifier
            validation_gradient: Pre-computed validation gradient
            target_classes: Target classes for filtering
            
        Returns:
            Cosine similarity score [-1, 1]
        """
        scene_gradient = self.compute_scene_gradient(scene_id, target_classes)
        
        if scene_gradient is None or validation_gradient is None:
            return 0.0
        
        try:
            # Compute cosine similarity
            alignment = F.cosine_similarity(
                validation_gradient.unsqueeze(0),
                scene_gradient.unsqueeze(0),
                dim=1
            )
            
            return alignment.item()
            
        except Exception as e:
            self.logger.warning(f"Error computing alignment for scene {scene_id}: {e}")
            return 0.0
    
    def compute_alignment_scores_batch(self,
                                     scene_ids: List[str],
                                     target_classes: List[int],
                                     validation_batch_size: int = 8) -> Dict[str, float]:
        """
        Compute alignment scores for multiple scenes efficiently.
        
        Args:
            scene_ids: List of scene identifiers
            target_classes: Target classes for gradient computation
            validation_batch_size: Size of validation batch
            
        Returns:
            Dictionary mapping scene_id -> alignment_score
        """
        self.logger.info(f"Computing alignment scores for {len(scene_ids)} scenes")
        
        # Compute validation gradient once
        validation_gradient = self.compute_validation_gradient(
            target_classes, validation_batch_size
        )
        
        if validation_gradient is None:
            self.logger.error("Could not compute validation gradient")
            return {scene_id: 0.0 for scene_id in scene_ids}
        
        # Compute alignment for each scene
        alignment_scores = {}
        for i, scene_id in enumerate(scene_ids):
            if i % 50 == 0:
                self.logger.info(f"  Processing scene {i+1}/{len(scene_ids)}")
            
            score = self.compute_alignment_score(scene_id, validation_gradient, target_classes)
            alignment_scores[scene_id] = score
        
        self.logger.info(f"Computed alignment scores for {len(alignment_scores)} scenes")
        return alignment_scores
    
    def _compute_gradient_for_batch(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute gradient for a batch and return flattened tensor."""
        self.model.eval()  # Ensure eval mode
        self.model.zero_grad()  # Clear existing gradients
        
        # Forward pass
        points = batch['points']
        gt_bboxes = batch['gt_bboxes_3d']
        gt_labels = batch['gt_labels_3d']
        img_metas = batch['img_metas']
        
        # Extract features
        features = self.model.extract_feats(points)
        
        # Compute loss
        losses = self.model.head.forward_train(
            features, gt_bboxes, gt_labels, img_metas
        )
        
        # Focus on classification loss for old class retention
        loss = losses['cls_loss']
        
        # Compute gradients w.r.t. classification layer parameters
        gradients = torch.autograd.grad(
            loss,
            self.model.head.cls_conv.parameters(),
            retain_graph=False,
            create_graph=False,
            allow_unused=True
        )
        
        # Flatten and concatenate all gradients
        grad_list = []
        for grad in gradients:
            if grad is not None:
                grad_list.append(grad.flatten())
        
        if not grad_list:
            # Return zero gradient if no gradients computed
            return self._get_zero_gradient()
        
        return torch.cat(grad_list)
    
    def _filter_batch_to_classes(self, 
                                batch: Dict[str, Any], 
                                target_classes: List[int]) -> Optional[Dict[str, Any]]:
        """Filter batch to only include target classes."""
        gt_bboxes = batch.get('gt_bboxes_3d', [])
        gt_labels = batch.get('gt_labels_3d', [])
        
        if not gt_bboxes or not gt_labels:
            return None
        
        filtered_bboxes = []
        filtered_labels = []
        has_target_objects = False
        
        for bboxes, labels in zip(gt_bboxes, gt_labels):
            if isinstance(labels, torch.Tensor):
                # Create mask for target classes
                target_tensor = torch.tensor(target_classes, device=labels.device)
                mask = torch.isin(labels, target_tensor)
                
                if mask.any():
                    has_target_objects = True
                    filtered_bboxes.append(bboxes[mask])
                    filtered_labels.append(labels[mask])
                else:
                    # Add empty tensors to maintain batch structure
                    filtered_bboxes.append(bboxes[:0])  # Empty tensor with correct type
                    filtered_labels.append(labels[:0])
            else:
                # Handle non-tensor case
                filtered_bboxes.append(bboxes)
                filtered_labels.append(labels)
        
        if not has_target_objects:
            return None
        
        # Create filtered batch
        filtered_batch = batch.copy()
        filtered_batch['gt_bboxes_3d'] = filtered_bboxes
        filtered_batch['gt_labels_3d'] = filtered_labels
        
        return filtered_batch
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch data to the specified device."""
        device_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, list):
                device_list = []
                for item in value:
                    if isinstance(item, torch.Tensor):
                        device_list.append(item.to(self.device))
                    else:
                        device_list.append(item)
                device_batch[key] = device_list
            else:
                device_batch[key] = value
        
        return device_batch
    
    def _prepare_single_scene_batch(self, scene_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert single scene data to batch format."""
        batch = {}
        
        for key, value in scene_data.items():
            if key in ['points', 'gt_bboxes_3d', 'gt_labels_3d', 'img_metas']:
                if isinstance(value, list):
                    batch[key] = value
                else:
                    batch[key] = [value]
            else:
                batch[key] = value
        
        return self._move_batch_to_device(batch)
    
    def _get_zero_gradient(self) -> torch.Tensor:
        """Get zero gradient tensor with correct shape."""
        # Get parameter count from classification layer
        total_params = sum(p.numel() for p in self.model.head.cls_conv.parameters())
        return torch.zeros(total_params, device=self.device)
    
    def save_alignment_results(self, 
                             alignment_scores: Dict[str, float],
                             target_classes: List[int],
                             output_path: str):
        """
        Save alignment scores to file.
        
        Args:
            alignment_scores: Dictionary of scene_id -> alignment_score
            target_classes: Classes used for alignment computation
            output_path: Path to save results
        """
        results = {
            'alignment_scores': alignment_scores,
            'target_classes': target_classes,
            'num_scenes': len(alignment_scores),
            'statistics': {
                'mean_alignment': np.mean(list(alignment_scores.values())),
                'std_alignment': np.std(list(alignment_scores.values())),
                'min_alignment': min(alignment_scores.values()) if alignment_scores else 0.0,
                'max_alignment': max(alignment_scores.values()) if alignment_scores else 0.0,
                'positive_alignments': sum(1 for s in alignment_scores.values() if s > 0),
                'negative_alignments': sum(1 for s in alignment_scores.values() if s < 0)
            }
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            self.logger.info(f"Alignment results saved to: {output_path}")
            self.logger.info(f"Statistics: mean={results['statistics']['mean_alignment']:.4f}, "
                           f"std={results['statistics']['std_alignment']:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error saving alignment results: {e}")
    
    def get_top_aligned_scenes(self, 
                             alignment_scores: Dict[str, float],
                             top_k: int = 50,
                             min_alignment: float = 0.0) -> List[Tuple[str, float]]:
        """
        Get top-k scenes by alignment score.
        
        Args:
            alignment_scores: Dictionary of scene_id -> alignment_score
            top_k: Number of top scenes to return
            min_alignment: Minimum alignment threshold
            
        Returns:
            List of (scene_id, alignment_score) tuples, sorted by score
        """
        # Filter by minimum alignment
        filtered_scores = {
            scene_id: score 
            for scene_id, score in alignment_scores.items() 
            if score >= min_alignment
        }
        
        # Sort by alignment score (descending)
        sorted_scenes = sorted(
            filtered_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_scenes[:top_k]
    
    def clear_caches(self):
        """Clear all cached gradients."""
        if self.gradient_cache:
            self.gradient_cache.clear()
        if self.validation_gradient_cache:
            self.validation_gradient_cache.clear()
        
        # Also clear data loader caches
        self.train_loader.clear_cache()
        self.val_loader.clear_cache()
        
        self.logger.info("All gradient caches cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics about gradient caches."""
        return {
            'scene_gradients_cached': len(self.gradient_cache) if self.gradient_cache else 0,
            'validation_gradients_cached': len(self.validation_gradient_cache),
            'train_scenes_cached': len(self.train_loader.scene_cache),
            'val_scenes_cached': len(self.val_loader.scene_cache),
            'caching_enabled': self.cache_gradients
        }