"""
Data utilities for scene discovery experiments.

This module provides utilities for loading individual scenes, creating
batches for gradient computation, and managing scene data efficiently
during discovery experiments.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from collections import defaultdict

import torch
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet.datasets import build_dataloader


class SceneDataLoader:
    """
    Utility class for loading individual scenes from ScanNet dataset.
    
    This class provides efficient access to specific scenes by scene_id
    for gradient alignment computation and other scene-specific analysis.
    """
    
    def __init__(self, 
                 config: Config,
                 dataset_split: str = 'train',
                 device: str = 'cuda:0'):
        """
        Initialize the scene data loader.
        
        Args:
            config: Training configuration
            dataset_split: 'train' or 'val' dataset
            device: Device for data loading
        """
        self.config = config
        self.dataset_split = dataset_split
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Build dataset
        if dataset_split == 'train':
            dataset_config = config.data.train
        elif dataset_split == 'val':
            dataset_config = config.data.val
        else:
            raise ValueError(f"Unknown dataset split: {dataset_split}")
        
        # Create a deep copy to avoid modifying the original config
        from copy import deepcopy
        dataset_config = deepcopy(dataset_config)
        
        # For discovery mode, use standard dataset (not incremental) to avoid stage warnings
        if hasattr(dataset_config, 'type') and 'Incremental' in str(dataset_config.type):
            # Convert to standard dataset for scene discovery
            dataset_config.type = 'ScanNet35ClassBinFileDataset'
            # Remove incremental-specific parameters
            for key in ['stage_id', 'use_sequential_gci', 'use_pseudo_labels']:
                if hasattr(dataset_config, key):
                    delattr(dataset_config, key)
        
        # Ensure the dataset has a type field
        if not hasattr(dataset_config, 'type'):
            dataset_config.type = 'ScanNet35ClassBinFileDataset'
        
        self.dataset = build_dataset(dataset_config)
        self.logger.info(f"Loaded {dataset_split} dataset with {len(self.dataset)} scenes")
        
        # Create scene ID mapping
        self.scene_id_to_index = {}
        self.index_to_scene_id = {}
        self._build_scene_mapping()
        
        # Cache for loaded scenes
        self.scene_cache = {}
        self.max_cache_size = 50  # Limit cache to save memory
    
    def _build_scene_mapping(self):
        """Build mapping between scene IDs and dataset indices."""
        self.logger.info("Building scene ID mapping...")
        
        for i in range(len(self.dataset)):
            try:
                data_info = self.dataset.get_data_info(i)
                
                # Check if data_info is None or invalid
                if data_info is None:
                    self.logger.debug(f"Scene {i}: data_info is None, using fallback scene_id")
                    scene_id = f'scene_{i:06d}'
                else:
                    # First try to get scene_id from the proper ScanNet location (point_cloud['lidar_idx'])
                    scene_id = None
                    if 'point_cloud' in data_info and 'lidar_idx' in data_info['point_cloud']:
                        scene_id = data_info['point_cloud']['lidar_idx']
                    elif 'scene_id' in data_info:
                        scene_id = data_info['scene_id']
                    
                    if scene_id is None:
                        # Try to extract from file path
                        pts_path = data_info.get('pts_path', '')
                        if 'scene' in pts_path:
                            # Extract scene_id from path like "scene0000_00.bin"
                            scene_name = os.path.basename(pts_path).split('.')[0]
                            scene_id = scene_name
                        else:
                            scene_id = f'scene_{i:06d}'
                
                self.scene_id_to_index[scene_id] = i
                self.index_to_scene_id[i] = scene_id
                
            except Exception as e:
                self.logger.debug(f"Error processing scene {i}: {e}")
                # Use fallback scene_id and continue
                fallback_scene_id = f'scene_{i:06d}'
                self.scene_id_to_index[fallback_scene_id] = i
                self.index_to_scene_id[i] = fallback_scene_id
        
        self.logger.info(f"Built mapping for {len(self.scene_id_to_index)} scenes")
    
    def get_scene_metadata_by_id(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        Get scene metadata only (without loading full data) by scene ID.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene metadata dictionary or None if not found
        """
        # Get dataset index
        if scene_id not in self.scene_id_to_index:
            self.logger.warning(f"Scene ID not found: {scene_id}")
            return None
        
        index = self.scene_id_to_index[scene_id]
        
        try:
            # Get metadata only (much faster than full data loading)
            data_info = self.dataset.get_data_info(index)
            if data_info is None:
                return None
            
            # Add scene_id to metadata
            data_info['scene_id'] = scene_id
            data_info['dataset_index'] = index
            
            return data_info
            
        except Exception as e:
            self.logger.error(f"Error loading metadata for scene {scene_id} (index {index}): {e}")
            return None

    def get_scene_by_id(self, scene_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a specific scene by scene ID.
        
        Args:
            scene_id: Scene identifier
            
        Returns:
            Scene data dictionary or None if not found
        """
        # Check cache first
        if scene_id in self.scene_cache:
            return self.scene_cache[scene_id]
        
        # Get dataset index
        if scene_id not in self.scene_id_to_index:
            self.logger.warning(f"Scene ID not found: {scene_id}")
            return None
        
        index = self.scene_id_to_index[scene_id]
        
        try:
            # Load scene data (this triggers full data loading pipeline)
            scene_data = self.dataset[index]
            
            # Add scene_id to metadata if not present
            if 'img_metas' in scene_data:
                from mmcv.parallel import DataContainer
                img_metas = scene_data['img_metas']
                
                # Handle DataContainer wrapper
                if isinstance(img_metas, DataContainer):
                    img_metas = img_metas.data
                
                if isinstance(img_metas, list):
                    for meta in img_metas:
                        if isinstance(meta, dict) and 'scene_id' not in meta:
                            meta['scene_id'] = scene_id
                elif isinstance(img_metas, dict):
                    if 'scene_id' not in img_metas:
                        img_metas['scene_id'] = scene_id
            
            # Cache the scene (with size limit)
            if len(self.scene_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.scene_cache))
                del self.scene_cache[oldest_key]
            
            self.scene_cache[scene_id] = scene_data
            return scene_data
            
        except Exception as e:
            self.logger.error(f"Error loading scene {scene_id} (index {index}): {e}")
            return None
    
    def get_scenes_by_ids(self, scene_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Load multiple scenes by scene IDs.
        
        Args:
            scene_ids: List of scene identifiers
            
        Returns:
            List of scene data dictionaries (may contain None for failed loads)
        """
        scenes = []
        for scene_id in scene_ids:
            scene_data = self.get_scene_by_id(scene_id)
            scenes.append(scene_data)
        
        return scenes
    
    def create_scene_batch(self, scene_ids: List[str]) -> Optional[Dict[str, Any]]:
        """
        Create a batch from specific scenes for gradient computation.
        
        Args:
            scene_ids: List of scene identifiers
            
        Returns:
            Batch dictionary suitable for model forward pass
        """
        scenes = self.get_scenes_by_ids(scene_ids)
        
        # Filter out None scenes
        valid_scenes = [s for s in scenes if s is not None]
        if not valid_scenes:
            self.logger.warning("No valid scenes found for batch creation")
            return None
        
        try:
            # Create batch by collating scenes
            batch = self._collate_scenes(valid_scenes)
            return batch
            
        except Exception as e:
            self.logger.error(f"Error creating batch from scenes: {e}")
            return None
    
    def _collate_scenes(self, scenes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate multiple scenes into a single batch.
        
        Args:
            scenes: List of scene data dictionaries
            
        Returns:
            Batch dictionary
        """
        if not scenes:
            return {}
        
        # Initialize batch
        batch = defaultdict(list)
        
        # Collect data from all scenes
        for scene in scenes:
            for key, value in scene.items():
                batch[key].append(value)
        
        # Handle special keys that need specific collation
        collated_batch = {}
        
        for key, values in batch.items():
            if key == 'points':
                # Points need to be combined into a list
                collated_batch[key] = values
            elif key in ['gt_bboxes_3d', 'gt_labels_3d']:
                # Ground truth data as lists
                collated_batch[key] = values
            elif key == 'img_metas':
                # Metadata as list
                collated_batch[key] = values
            else:
                # Try to handle other keys appropriately
                if isinstance(values[0], torch.Tensor):
                    try:
                        collated_batch[key] = torch.stack(values)
                    except:
                        collated_batch[key] = values
                else:
                    collated_batch[key] = values
        
        return collated_batch
    
    def get_scene_ids_by_classes(self, target_classes: List[int]) -> List[str]:
        """
        Get scene IDs that contain objects from target classes.
        
        Args:
            target_classes: List of class indices to look for
            
        Returns:
            List of scene IDs containing target classes
        """
        matching_scenes = []
        
        self.logger.info(f"Searching for scenes with classes: {target_classes}")
        self.logger.info(f"Checking {len(self.scene_id_to_index)} scenes...")
        
        processed_count = 0
        for scene_id in self.scene_id_to_index.keys():
            processed_count += 1
            if processed_count % 100 == 0:
                self.logger.info(f"  Processed {processed_count}/{len(self.scene_id_to_index)} scenes, found {len(matching_scenes)} matches")
            
            # Use metadata-only access for much faster filtering
            scene_metadata = self.get_scene_metadata_by_id(scene_id)
            if scene_metadata is None:
                continue
            
            # Check if scene contains target classes from metadata
            # The gt_labels are often stored in the data_info  
            has_target_class = False
            
            # Try different common locations for ground truth labels
            if 'instances' in scene_metadata:
                # Check instances field
                instances = scene_metadata['instances']
                if isinstance(instances, list) and instances:
                    for instance in instances:
                        if 'bbox_label_3d' in instance:
                            label = instance['bbox_label_3d']
                            if label in target_classes:
                                has_target_class = True
                                break
                        elif 'label' in instance:
                            label = instance['label']  
                            if label in target_classes:
                                has_target_class = True
                                break
            
            # If not found in instances, try direct gt_labels
            elif 'gt_labels_3d' in scene_metadata:
                import numpy as np
                gt_labels = scene_metadata['gt_labels_3d']
                if isinstance(gt_labels, (list, np.ndarray)):
                    unique_labels = np.unique(gt_labels)
                    if any(cls in unique_labels for cls in target_classes):
                        has_target_class = True
            
            # If still not found, we might need to fall back to full data loading for some scenes
            elif not has_target_class:
                try:
                    scene_data = self.get_scene_by_id(scene_id)
                    if scene_data is not None:
                        gt_labels = scene_data.get('gt_labels_3d')
                        if gt_labels is not None:
                            from mmcv.parallel import DataContainer
                            if isinstance(gt_labels, DataContainer):
                                gt_labels = gt_labels.data
                            
                            if isinstance(gt_labels, list):
                                labels = gt_labels[0] if gt_labels else torch.tensor([])
                            else:
                                labels = gt_labels
                            
                            if isinstance(labels, DataContainer):
                                labels = labels.data
                            
                            if isinstance(labels, torch.Tensor):
                                unique_labels = labels.unique().cpu().numpy()
                                if any(cls in unique_labels for cls in target_classes):
                                    has_target_class = True
                except Exception as e:
                    self.logger.debug(f"Error checking scene {scene_id} with full loading: {e}")
                    continue
            
            if has_target_class:
                matching_scenes.append(scene_id)
        
        self.logger.info(f"Found {len(matching_scenes)} scenes with target classes")
        return matching_scenes
    
    def get_validation_batch_for_classes(self, 
                                       target_classes: List[int],
                                       batch_size: int = 8) -> Optional[Dict[str, Any]]:
        """
        Create a validation batch containing target classes.
        
        Args:
            target_classes: Classes to include in batch
            batch_size: Number of scenes in batch
            
        Returns:
            Validation batch for gradient computation
        """
        # Find scenes with target classes
        matching_scene_ids = self.get_scene_ids_by_classes(target_classes)
        
        if not matching_scene_ids:
            self.logger.warning(f"No scenes found with classes {target_classes}")
            return None
        
        # Select scenes for batch
        selected_scenes = matching_scene_ids[:batch_size]
        
        # Create batch
        batch = self.create_scene_batch(selected_scenes)
        
        if batch is not None:
            self.logger.info(f"Created validation batch with {len(selected_scenes)} scenes")
        
        return batch
    
    def clear_cache(self):
        """Clear the scene cache to free memory."""
        self.scene_cache.clear()
        self.logger.info("Scene cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the loaded dataset."""
        return {
            'total_scenes': len(self.dataset),
            'mapped_scenes': len(self.scene_id_to_index),
            'cached_scenes': len(self.scene_cache),
            'dataset_split': self.dataset_split
        }


class SceneFilterDataset:
    """
    Wrapper around a dataset to filter by specific scene IDs.
    
    This is useful for creating datasets with only specific scenes
    for gradient alignment experiments.
    """
    
    def __init__(self, 
                 base_dataset,
                 scene_ids: List[str],
                 scene_data_loader: SceneDataLoader):
        """
        Initialize filtered dataset.
        
        Args:
            base_dataset: Original dataset
            scene_ids: List of scene IDs to include
            scene_data_loader: Scene data loader for mapping
        """
        self.base_dataset = base_dataset
        self.scene_ids = scene_ids
        self.scene_data_loader = scene_data_loader
        
        # Build index mapping
        self.indices = []
        for scene_id in scene_ids:
            if scene_id in scene_data_loader.scene_id_to_index:
                self.indices.append(scene_data_loader.scene_id_to_index[scene_id])
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Created filtered dataset with {len(self.indices)} scenes")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.base_dataset[real_idx]
    
    def get_data_info(self, idx):
        real_idx = self.indices[idx]
        return self.base_dataset.get_data_info(real_idx)
    
    def evaluate(self, *args, **kwargs):
        """Forward evaluate to base dataset."""
        return self.base_dataset.evaluate(*args, **kwargs)