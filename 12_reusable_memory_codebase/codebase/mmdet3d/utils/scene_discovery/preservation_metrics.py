#!/usr/bin/env python
"""
Forward-Pass Preservation Metrics for Scene Discovery

This module implements efficient scene discovery metrics based on forward-pass
preservation analysis instead of full training. The key insight is that good
memory bank scenes should preserve Stage 1 model confidence when the head is
expanded to accommodate Stage 2 classes.

Two primary metrics are implemented:
1. Max Confidence: Measures maximum confidence preservation across Stage 1 classes
2. Entropy: Measures confidence distribution changes using entropy

Date: August 2025
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv.parallel import MMDataParallel


class PreservationMetrics:
    """
    Computes preservation-based discovery metrics for scene selection.
    
    This class implements forward-pass analysis to measure how well Stage 1
    model performance is preserved after head expansion. Good scenes should
    maintain high confidence for Stage 1 classes.
    """
    
    def __init__(
        self,
        stage1_checkpoint: str,
        base_config_path: str,
        device: str = 'cuda:0',
        stage1_classes: int = 7,
        stage2_classes: int = 14,
        min_stage1_objects: int = 2,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize preservation metrics calculator.
        
        Args:
            stage1_checkpoint: Path to Stage 1 model checkpoint
            base_config_path: Path to base configuration file
            device: Device for computation ('cuda:0' or 'cpu')
            stage1_classes: Number of Stage 1 classes (default 7)
            stage2_classes: Number of classes after expansion (default 14)
            min_stage1_objects: Minimum Stage 1 objects required per scene
            logger: Optional logger for detailed output
        """
        self.stage1_checkpoint = stage1_checkpoint
        self.base_config_path = base_config_path
        self.device = device
        self.stage1_classes = stage1_classes
        self.stage2_classes = stage2_classes
        self.min_stage1_objects = min_stage1_objects
        
        # Setup logging
        if logger is None:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
        else:
            self.logger = logger
            
        # Initialize models
        self._setup_models()
        
        # Stage 1 class names for debugging - import from central mapping
        # Legacy hardcoded ordering removed Aug 2025 - now imports from central mapping
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), '../../../configs/_base_/class_mappings'))
        try:
            from scannet_dynamic_head_mappings import get_stage_definitions
            stage_defs = get_stage_definitions('frequency')
            self.stage1_class_names = stage_defs[0]['class_names']
        except ImportError:
            # Fallback if import fails
            self.logger.warning("Could not import frequency ordering, using hardcoded fallback")
            self.stage1_class_names = [
                'chair', 'door', 'otherfurniture', 'books', 'cabinet', 'table', 'window'
            ]
        
    def _setup_models(self):
        """Setup Stage 1 and expanded models."""
        self.logger.info("Setting up preservation metric models...")
        
        # Load base configuration
        cfg = Config.fromfile(self.base_config_path)
        
        # Build Stage 1 model (7 classes)
        cfg_stage1 = cfg.deepcopy()
        cfg_stage1.model.head.n_classes = self.stage1_classes
        
        self.model_stage1 = build_model(
            cfg_stage1.model, 
            train_cfg=cfg_stage1.get('train_cfg'), 
            test_cfg=cfg_stage1.get('test_cfg')
        )
        
        # Load checkpoint
        load_checkpoint(self.model_stage1, self.stage1_checkpoint, map_location='cpu', strict=False)
        self.logger.info(f"Loaded Stage 1 checkpoint: {self.stage1_checkpoint}")
        
        # Set train_cfg to None before device placement
        if hasattr(self.model_stage1, 'train_cfg'):
            self.model_stage1.train_cfg = None  # Disable training mode
        
        # Move to device and wrap
        if 'cuda' in self.device:
            self.model_stage1 = MMDataParallel(self.model_stage1, device_ids=[0])
            torch.cuda.set_device(0)  # Respect CUDA_VISIBLE_DEVICES
        else:
            self.model_stage1 = self.model_stage1.to(self.device)
        
        # Build expanded model (14 classes) - clone of Stage 1 model
        self.model_expanded = build_model(
            cfg_stage1.model, 
            train_cfg=cfg_stage1.get('train_cfg'), 
            test_cfg=cfg_stage1.get('test_cfg')
        )
        
        # Load same checkpoint
        load_checkpoint(self.model_expanded, self.stage1_checkpoint, map_location='cpu', strict=False)
        
        # Expand head using proper TR3D method
        if hasattr(self.model_expanded, 'head'):
            head = self.model_expanded.head
        else:
            head = self.model_expanded  # In case it's already the head
            
        success = head.expand_classification_head(self.stage2_classes)
        if not success:
            raise RuntimeError(f"Failed to expand head from {self.stage1_classes} to {self.stage2_classes} classes")
        
        self.logger.info(f"Successfully expanded head: {self.stage1_classes} → {self.stage2_classes} classes")
        
        # Set train_cfg to None before device placement
        if hasattr(self.model_expanded, 'train_cfg'):
            self.model_expanded.train_cfg = None
        
        # Move expanded model to device
        if 'cuda' in self.device:
            self.model_expanded = MMDataParallel(self.model_expanded, device_ids=[0])
        else:
            self.model_expanded = self.model_expanded.to(self.device)
        
        # Set both models to eval mode
        self.model_stage1.eval()
        self.model_expanded.eval()
        
        self.logger.info("Models setup complete")
        
    def check_scene_eligibility(self, scene_id: str, dataset) -> Tuple[bool, Dict]:
        """
        Check if scene is eligible for discovery analysis.
        
        Args:
            scene_id: Scene identifier (e.g., 'scene0000_00')
            dataset: Dataset object containing scene information
            
        Returns:
            Tuple of (is_eligible, metadata_dict)
        """
        metadata = {
            'scene_id': scene_id,
            'total_objects': 0,
            'stage1_objects': 0,
            'stage1_object_classes': [],
            'eligible': False,
            'reason': None
        }
        
        # Find scene samples in dataset by extracting scene IDs from filenames
        scene_samples = []
        for i in range(len(dataset)):
            sample_info = dataset.get_data_info(i)
            
            # Extract scene ID from pts_filename
            extracted_scene_id = None
            pts_filename = sample_info.get('pts_filename', '')
            if pts_filename and 'scene' in pts_filename:
                basename = os.path.basename(pts_filename)
                if '_' in basename and basename.startswith('scene'):
                    try:
                        # Extract scene ID: e.g., scene0568_00.bin -> scene0568_00
                        scene_part = basename.split('_')[0] + '_' + basename.split('_')[1].split('.')[0]
                        extracted_scene_id = scene_part
                    except IndexError:
                        continue
            
            # Check if this sample belongs to the target scene
            if extracted_scene_id == scene_id:
                scene_samples.append(i)
        
        if not scene_samples:
            metadata['reason'] = 'Scene not found in dataset'
            return False, metadata
            
        # For Milestone 1: Simplified eligibility check
        # We'll assume scenes with multiple samples are eligible for now
        # TODO: Implement proper Stage 1 object counting when we have better label access
        
        sample_count = len(scene_samples)
        metadata['total_objects'] = sample_count  # Use sample count as proxy
        metadata['stage1_objects'] = sample_count  # Assume all are Stage 1 for now
        metadata['stage1_object_classes'] = list(range(self.stage1_classes))  # Assume all classes present
        
        # Eligibility check: require at least 1 sample for the scene
        if sample_count >= 1:
            metadata['eligible'] = True
            metadata['reason'] = f"Scene has {sample_count} samples (simplified check for Milestone 1)"
            return True, metadata
        else:
            metadata['eligible'] = False
            metadata['reason'] = f"Scene has no samples"
            return False, metadata

    def compute_max_confidence_preservation(
        self, 
        scene_id: str,
        dataset,
        dataloader,
        save_details: bool = True
    ) -> Dict:
        """
        Compute max confidence preservation metric for a scene.
        
        This metric measures how well the maximum confidence across Stage 1
        classes is preserved after head expansion.
        
        Args:
            scene_id: Scene identifier
            dataset: Dataset object
            dataloader: DataLoader for the scene
            save_details: Whether to save detailed per-sample results
            
        Returns:
            Dictionary with preservation metrics and metadata
        """
        results = {
            'scene_id': scene_id,
            'metric_type': 'max_confidence_preservation',
            'timestamp': datetime.now().isoformat(),
            'stage1_max_confidence': 0.0,
            'expanded_max_confidence': 0.0,
            'preservation_score': 0.0,
            'samples_processed': 0,
            'stage1_predictions': [],
            'expanded_predictions': [],
            'detailed_results': {} if save_details else None
        }
        
        # Process all samples in the scene
        with torch.no_grad():
            stage1_confidences = []
            expanded_confidences = []
            
            for i, batch in enumerate(dataloader):
                # Move batch to device - handle DataContainer and direct tensors
                if 'cuda' in self.device:
                    for key in batch.keys():
                        if hasattr(batch[key], 'data'):
                            # DataContainer case: data is a list of tensors
                            if isinstance(batch[key].data, list):
                                batch[key].data = [item.to(self.device) if hasattr(item, 'to') else item 
                                                   for item in batch[key].data]
                            elif hasattr(batch[key].data, 'to'):
                                # Single tensor case
                                batch[key].data = batch[key].data.to(self.device)
                        elif hasattr(batch[key], 'to'):
                            # Direct tensor case (no DataContainer wrapper)
                            batch[key] = batch[key].to(self.device)
                
                # Forward pass through Stage 1 model
                stage1_output = self.model_stage1(batch)
                
                # Forward pass through expanded model  
                expanded_output = self.model_expanded(batch)
                
                # Extract classification scores
                stage1_scores = stage1_output[0]['scores_3d']  # Shape: [N, 7]
                expanded_scores = expanded_output[0]['scores_3d']  # Shape: [N, 14]
                
                # Take only Stage 1 classes from expanded model
                expanded_stage1_scores = expanded_scores[:, :self.stage1_classes]  # Shape: [N, 7]
                
                # Apply softmax to get confidences
                stage1_conf = F.softmax(stage1_scores, dim=1)
                expanded_stage1_conf = F.softmax(expanded_stage1_scores, dim=1)
                
                # Get maximum confidence across Stage 1 classes for each detection
                stage1_max_conf, _ = stage1_conf.max(dim=1)  # Shape: [N]
                expanded_max_conf, _ = expanded_stage1_conf.max(dim=1)  # Shape: [N]
                
                # Store confidences
                stage1_confidences.extend(stage1_max_conf.cpu().numpy().tolist())
                expanded_confidences.extend(expanded_max_conf.cpu().numpy().tolist())
                
                if save_details:
                    results['detailed_results'][f'sample_{i}'] = {
                        'stage1_confidences': stage1_max_conf.cpu().numpy().tolist(),
                        'expanded_confidences': expanded_max_conf.cpu().numpy().tolist(),
                        'num_detections': len(stage1_max_conf)
                    }
                
                results['samples_processed'] += 1
        
        # Compute overall preservation metrics
        if stage1_confidences and expanded_confidences:
            stage1_mean_conf = np.mean(stage1_confidences)
            expanded_mean_conf = np.mean(expanded_confidences)
            
            # Preservation score: how well confidence is maintained
            # Score = expanded_confidence / stage1_confidence (1.0 = perfect preservation)
            if stage1_mean_conf > 0:
                preservation_score = expanded_mean_conf / stage1_mean_conf
            else:
                preservation_score = 0.0
            
            results.update({
                'stage1_max_confidence': float(stage1_mean_conf),
                'expanded_max_confidence': float(expanded_mean_conf),
                'preservation_score': float(preservation_score),
                'total_detections': len(stage1_confidences)
            })
            
        else:
            self.logger.warning(f"No valid detections found for scene {scene_id}")
        
        return results

    def compute_entropy_preservation(
        self, 
        scene_id: str,
        dataset,
        dataloader,
        save_details: bool = True
    ) -> Dict:
        """
        Compute entropy-based preservation metric for a scene.
        
        This metric measures how the confidence distribution entropy changes
        after head expansion. Lower entropy change indicates better preservation.
        
        Args:
            scene_id: Scene identifier
            dataset: Dataset object  
            dataloader: DataLoader for the scene
            save_details: Whether to save detailed results
            
        Returns:
            Dictionary with preservation metrics and metadata
        """
        results = {
            'scene_id': scene_id,
            'metric_type': 'entropy_preservation', 
            'timestamp': datetime.now().isoformat(),
            'stage1_entropy': 0.0,
            'expanded_entropy': 0.0,
            'entropy_change': 0.0,
            'preservation_score': 0.0,
            'samples_processed': 0,
            'detailed_results': {} if save_details else None
        }
        
        with torch.no_grad():
            stage1_entropies = []
            expanded_entropies = []
            
            for i, batch in enumerate(dataloader):
                # Move batch to device - handle DataContainer and direct tensors
                if 'cuda' in self.device:
                    for key in batch.keys():
                        if hasattr(batch[key], 'data'):
                            # DataContainer case: data is a list of tensors
                            if isinstance(batch[key].data, list):
                                batch[key].data = [item.to(self.device) if hasattr(item, 'to') else item 
                                                   for item in batch[key].data]
                            elif hasattr(batch[key].data, 'to'):
                                # Single tensor case
                                batch[key].data = batch[key].data.to(self.device)
                        elif hasattr(batch[key], 'to'):
                            # Direct tensor case (no DataContainer wrapper)
                            batch[key] = batch[key].to(self.device)
                
                # Forward passes
                stage1_output = self.model_stage1(batch)
                expanded_output = self.model_expanded(batch)
                
                # Get scores
                stage1_scores = stage1_output[0]['scores_3d']
                expanded_scores = expanded_output[0]['scores_3d']
                expanded_stage1_scores = expanded_scores[:, :self.stage1_classes]
                
                # Compute softmax confidences
                stage1_conf = F.softmax(stage1_scores, dim=1)
                expanded_stage1_conf = F.softmax(expanded_stage1_scores, dim=1)
                
                # Compute entropy for each detection
                stage1_entropy = -torch.sum(stage1_conf * torch.log(stage1_conf + 1e-8), dim=1)
                expanded_entropy = -torch.sum(expanded_stage1_conf * torch.log(expanded_stage1_conf + 1e-8), dim=1)
                
                # Store entropies
                stage1_entropies.extend(stage1_entropy.cpu().numpy().tolist())
                expanded_entropies.extend(expanded_entropy.cpu().numpy().tolist())
                
                if save_details:
                    results['detailed_results'][f'sample_{i}'] = {
                        'stage1_entropies': stage1_entropy.cpu().numpy().tolist(),
                        'expanded_entropies': expanded_entropy.cpu().numpy().tolist(),
                        'num_detections': len(stage1_entropy)
                    }
                
                results['samples_processed'] += 1
        
        # Compute overall metrics
        if stage1_entropies and expanded_entropies:
            stage1_mean_entropy = np.mean(stage1_entropies)
            expanded_mean_entropy = np.mean(expanded_entropies)
            
            entropy_change = abs(expanded_mean_entropy - stage1_mean_entropy)
            
            # Preservation score: lower entropy change = better preservation
            # Use exponential decay: exp(-change) so 0 change = 1.0, higher change → 0
            preservation_score = np.exp(-entropy_change)
            
            results.update({
                'stage1_entropy': float(stage1_mean_entropy),
                'expanded_entropy': float(expanded_mean_entropy), 
                'entropy_change': float(entropy_change),
                'preservation_score': float(preservation_score),
                'total_detections': len(stage1_entropies)
            })
        
        return results

    def compute_combined_preservation(
        self,
        scene_id: str,
        dataset,
        dataloader,
        max_confidence_weight: float = 0.7,
        entropy_weight: float = 0.3,
        save_details: bool = True
    ) -> Dict:
        """
        Compute combined preservation score using both max confidence and entropy metrics.
        
        This is the recommended metric that balances confidence preservation (max_confidence)
        with distribution stability (entropy). The combined score provides a more robust
        measure of scene quality for memory bank selection.
        
        Args:
            scene_id: Scene identifier
            dataset: Dataset object
            dataloader: DataLoader for the scene
            max_confidence_weight: Weight for max confidence metric (default 0.7)
            entropy_weight: Weight for entropy metric (default 0.3)
            save_details: Whether to save detailed component results
            
        Returns:
            Dictionary with combined preservation metrics and component scores
        """
        # Validate weights sum to 1.0
        total_weight = max_confidence_weight + entropy_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")
        
        # Compute both component metrics
        max_conf_results = self.compute_max_confidence_preservation(
            scene_id, dataset, dataloader, save_details=save_details
        )
        
        entropy_results = self.compute_entropy_preservation(
            scene_id, dataset, dataloader, save_details=save_details
        )
        
        # Combined results structure
        combined_results = {
            'scene_id': scene_id,
            'metric_type': 'combined_preservation',
            'timestamp': datetime.now().isoformat(),
            'weights': {
                'max_confidence': max_confidence_weight,
                'entropy': entropy_weight
            },
            'component_scores': {
                'max_confidence_score': max_conf_results['preservation_score'],
                'entropy_score': entropy_results['preservation_score']
            },
            'combined_score': 0.0,
            'samples_processed': max_conf_results['samples_processed'],
            'detailed_components': {
                'max_confidence': max_conf_results if save_details else None,
                'entropy': entropy_results if save_details else None
            }
        }
        
        # Compute weighted combined score
        combined_score = (
            max_confidence_weight * max_conf_results['preservation_score'] +
            entropy_weight * entropy_results['preservation_score']
        )
        
        combined_results['combined_score'] = float(combined_score)
        combined_results['preservation_score'] = float(combined_score)  # For compatibility
        
        # Add component statistics for analysis
        combined_results['component_analysis'] = {
            'max_confidence_contribution': max_confidence_weight * max_conf_results['preservation_score'],
            'entropy_contribution': entropy_weight * entropy_results['preservation_score'],
            'stage1_max_confidence': max_conf_results['stage1_max_confidence'],
            'expanded_max_confidence': max_conf_results['expanded_max_confidence'],
            'stage1_entropy': entropy_results['stage1_entropy'],
            'expanded_entropy': entropy_results['expanded_entropy'],
            'entropy_change': entropy_results['entropy_change']
        }
        
        return combined_results

    def analyze_scene_batch(
        self,
        scene_ids: List[str],
        metric_type: str = 'combined',
        output_dir: str = './preservation_analysis',
        batch_name: str = 'batch_analysis'
    ) -> Dict:
        """
        Analyze a batch of scenes with preservation metrics.
        
        Args:
            scene_ids: List of scene identifiers to analyze
            metric_type: Type of metric ('combined', 'max_confidence', or 'entropy')
            output_dir: Directory to save results
            batch_name: Name for this batch analysis
            
        Returns:
            Summary results dictionary
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup dataset and dataloader
        # Use TRAINING dataset for scene analysis (has ground truth labels)
        # Use VALIDATION dataset for actual metric computation (inference)
        cfg = Config.fromfile(self.base_config_path)
        
        # Training dataset for scene eligibility analysis
        train_cfg = cfg.data.train.copy()
        train_cfg.type = 'ScanNetDataset'
        if hasattr(train_cfg, 'variant'):
            train_cfg.variant = 'dynamic_head'
        train_cfg.test_mode = True  # Use test mode to avoid augmentations during analysis
        
        # Remove incremental-specific parameters
        incremental_params = ['stage_definition', 'mappings', 'memory_bank', 
                             'evaluation_mode', 'all_stage_definitions', 'use_sequential_gci']
        for param in incremental_params:
            if hasattr(train_cfg, param):
                delattr(train_cfg, param)
        
        dataset = build_dataset(train_cfg)
        self.logger.info(f"Using training dataset for scene analysis: {len(dataset)} samples")
        
        # Use training dataset for metric computation in scene discovery
        # (We want to evaluate preservation metrics on the same scenes we're discovering)
        eval_cfg = train_cfg.copy()
        eval_cfg.test_mode = True  # Use test mode for evaluation
        eval_cfg.seen_classes_for_eval = list(range(self.stage2_classes))
        
        # Configure pipeline for inference-only mode (no ground truth needed)
        if hasattr(eval_cfg, 'pipeline'):
            new_pipeline = []
            for transform in eval_cfg.pipeline:
                if isinstance(transform, dict):
                    transform_type = transform.get('type', '')
                    # Skip all ground truth related transforms
                    if transform_type in ['LoadAnnotations3D', 'ObjectSample', 'ObjectNoise']:
                        continue
                    # Modify DefaultFormatBundle to exclude gt fields
                    elif transform_type == 'DefaultFormatBundle3D':
                        transform = transform.copy()
                        transform['class_names'] = list(range(self.stage2_classes))
                        transform['with_gt'] = False  # Disable GT processing
                        transform['with_label'] = False  # No labels needed
                        new_pipeline.append(transform)
                    # Keep Collect3D but modify to exclude gt fields
                    elif transform_type == 'Collect3D':
                        transform = transform.copy()
                        # Keep original keys but exclude ground truth fields
                        original_keys = transform.get('keys', ['points', 'img'])
                        # Filter out ground truth keys
                        inference_keys = [k for k in original_keys if not k.startswith('gt_')]
                        transform['keys'] = inference_keys
                        
                        # Use original meta_keys but ensure pts_filename is included
                        original_meta_keys = transform.get('meta_keys', [
                            'filename', 'ori_shape', 'img_shape', 'lidar2img', 
                            'depth2img', 'cam2img', 'pad_shape', 'scale_factor', 
                            'flip', 'pcd_horizontal_flip', 'pcd_vertical_flip', 
                            'box_mode_3d', 'box_type_3d', 'img_norm_cfg', 
                            'pcd_trans', 'pcd_scale_factor', 'pcd_rotation', 'pts_filename'
                        ])
                        transform['meta_keys'] = original_meta_keys
                        new_pipeline.append(transform)
                    else:
                        new_pipeline.append(transform)
                else:
                    new_pipeline.append(transform)
            eval_cfg.pipeline = new_pipeline
        
        self.val_dataset = build_dataset(eval_cfg)
        self.logger.info(f"Built evaluation dataset for metric computation: {len(self.val_dataset)} samples")
        
        # Results storage
        batch_results = {
            'batch_name': batch_name,
            'metric_type': metric_type,
            'timestamp': datetime.now().isoformat(),
            'total_scenes_requested': len(scene_ids),
            'eligible_scenes': 0,
            'ineligible_scenes': 0,
            'processed_scenes': 0,
            'scene_results': {},
            'eligibility_summary': {},
            'preservation_scores': []
        }
        
        self.logger.info(f"Starting batch analysis: {batch_name}")
        self.logger.info(f"Analyzing {len(scene_ids)} scenes with {metric_type} metric")
        
        # Process each scene
        for scene_id in tqdm(scene_ids, desc=f"Processing scenes ({metric_type})"):
            try:
                # Check eligibility
                is_eligible, eligibility_metadata = self.check_scene_eligibility(scene_id, dataset)
                batch_results['eligibility_summary'][scene_id] = eligibility_metadata
                
                if not is_eligible:
                    batch_results['ineligible_scenes'] += 1
                    self.logger.debug(f"Skipping {scene_id}: {eligibility_metadata['reason']}")
                    continue
                
                batch_results['eligible_scenes'] += 1
                
                # Create scene-specific dataloader using VALIDATION dataset for metric computation
                val_scene_indices = []
                for i in range(len(self.val_dataset)):
                    sample_info = self.val_dataset.get_data_info(i)
                    
                    # Extract scene ID from pts_filename
                    extracted_scene_id = None
                    pts_filename = sample_info.get('pts_filename', '')
                    if pts_filename and 'scene' in pts_filename:
                        basename = os.path.basename(pts_filename)
                        if '_' in basename and basename.startswith('scene'):
                            try:
                                # Extract scene ID: e.g., scene0568_00.bin -> scene0568_00
                                scene_part = basename.split('_')[0] + '_' + basename.split('_')[1].split('.')[0]
                                extracted_scene_id = scene_part
                            except IndexError:
                                continue
                    
                    # Check if this sample belongs to the target scene
                    if extracted_scene_id == scene_id:
                        val_scene_indices.append(i)
                
                if not val_scene_indices:
                    self.logger.warning(f"No validation samples found for scene {scene_id}")
                    continue
                
                # Create subset dataset for this scene from validation data
                from torch.utils.data import Subset
                scene_subset = Subset(self.val_dataset, val_scene_indices)
                
                scene_dataloader = build_dataloader(
                    scene_subset,
                    samples_per_gpu=1,
                    workers_per_gpu=1,
                    dist=False,
                    shuffle=False
                )
                
                # Compute preservation metric
                if metric_type == 'max_confidence':
                    scene_results = self.compute_max_confidence_preservation(
                        scene_id, dataset, scene_dataloader, save_details=True
                    )
                elif metric_type == 'entropy':
                    scene_results = self.compute_entropy_preservation(
                        scene_id, dataset, scene_dataloader, save_details=True
                    )
                elif metric_type == 'combined':
                    scene_results = self.compute_combined_preservation(
                        scene_id, dataset, scene_dataloader, save_details=True
                    )
                else:
                    raise ValueError(f"Unknown metric type: {metric_type}. Supported: 'combined', 'max_confidence', 'entropy'")
                
                # Store results
                batch_results['scene_results'][scene_id] = scene_results
                batch_results['preservation_scores'].append({
                    'scene_id': scene_id,
                    'preservation_score': scene_results['preservation_score']
                })
                batch_results['processed_scenes'] += 1
                
                self.logger.info(f"Processed {scene_id}: score = {scene_results['preservation_score']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error processing scene {scene_id}: {str(e)}")
                continue
        
        # Sort by preservation score (descending - higher is better)
        batch_results['preservation_scores'].sort(
            key=lambda x: x['preservation_score'], reverse=True
        )
        
        # Save results
        results_file = os.path.join(output_dir, f"{batch_name}_{metric_type}_results.json")
        with open(results_file, 'w') as f:
            json.dump(batch_results, f, indent=2, default=str)
        
        self.logger.info(f"Batch analysis complete. Results saved to: {results_file}")
        self.logger.info(f"Processed: {batch_results['processed_scenes']}/{batch_results['total_scenes_requested']} scenes")
        
        if batch_results['preservation_scores']:
            scores = [item['preservation_score'] for item in batch_results['preservation_scores']]
            self.logger.info(f"Score statistics: mean={np.mean(scores):.4f}, std={np.std(scores):.4f}")
            self.logger.info(f"Top 3 scenes: {batch_results['preservation_scores'][:3]}")
        
        return batch_results
