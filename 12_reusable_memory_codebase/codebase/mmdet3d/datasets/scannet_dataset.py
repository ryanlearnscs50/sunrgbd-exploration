# Copyright (c) OpenMMLab. All rights reserved.
"""
ScanNet Dataset for 3D Object Detection

This module provides a unified ScanNet dataset class that supports both:
- 18-class training (traditional subset)
- 35-class training (40-class data with 5 ignored classes)

Uses externalized class mapping configurations for maintainability.
"""

import tempfile
import warnings
from os import path as osp

import torch
import numpy as np

from mmdet3d.core import (
    instance_seg_eval, instance_seg_eval_v2, show_result_v2, show_seg_result)
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmseg.datasets import DATASETS as SEG_DATASETS
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .custom_3d_seg import Custom3DSegDataset
from .pipelines import Compose

# Import class mapping configurations
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(project_root, 'configs', '_base_', 'class_mappings'))
from scannet_18class_mapping import (
    SCANNET_18_CLASSES,
    SCANNET_18_NYU40_IDS,
    NYU40_TO_18CLASS_MODEL_IDX,
    MODEL_IDX_TO_NYU40_18CLASS,
    MODEL_IDX_TO_NAME_18CLASS
)
from scannet_dynamic_head_mappings import (
    SCANNET_DYNAMIC_HEAD_CLASSES,
    NYU40_IDS_DYNAMIC_HEAD,
    VALID_NYU40_IDS_DYNAMIC_HEAD,
    IGNORED_NYU40_IDS_DYNAMIC_HEAD,
    NYU40_TO_DYNAMIC_HEAD_GCI,
    DYNAMIC_HEAD_GCI_TO_NYU40,
    DYNAMIC_HEAD_GCI_TO_NAME
)
from scannet_35class_mapping import (
    SCANNET_35_CLASSES,
    VALID_NYU40_IDS_35CLASS,
    IGNORED_NYU40_IDS_35CLASS,
    NYU40_TO_35CLASS_MODEL_IDX,
    MODEL_IDX_TO_NYU40_35CLASS,
    MODEL_IDX_TO_NAME_35CLASS
)


@DATASETS.register_module()
class ScanNetDataset(Custom3DDataset):
    r"""Unified ScanNet Dataset for 3D Object Detection.
    
    Supports multiple training modes via 'variant' parameter:
    - variant='dynamic_head': Dynamic head incremental learning (DEFAULT)
    - variant='18': Traditional 18-class subset
    
    The dynamic_head variant is the recommended approach for all new development.
    It uses frequency-based class ordering for stage-wise incremental learning.
    
    Uses standard TR3D .bin file approach for proper data loading.
    """
    
    # Default to dynamic_head mode for incremental learning
    CLASSES = tuple(SCANNET_DYNAMIC_HEAD_CLASSES)
    VALID_CLASS_IDS = tuple(VALID_NYU40_IDS_DYNAMIC_HEAD)
    IGNORED_CLASS_IDS = tuple(IGNORED_NYU40_IDS_DYNAMIC_HEAD)
    
    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 variant='dynamic_head',  # 'dynamic_head' (default), '18' for 18-class
                 modality=dict(use_camera=False, use_depth=True),
                 box_type_3d='Depth',
                 filter_empty_gt=True,
                 test_mode=False,
                 seen_classes_for_eval=None,  # For incremental learning evaluation
                 current_stage_classes=None,  # Current stage classes for dynamic analysis
                 stage_definitions=None,  # Full stage definitions for dynamic analysis
                 **kwargs):
        
        self.variant = variant
        self.seen_classes_for_eval = seen_classes_for_eval  # Store for evaluation filtering
        self.current_stage_classes = current_stage_classes  # Store current stage classes
        self.stage_definitions = stage_definitions  # Store stage definitions for dynamic analysis
        
        # Set up class mappings based on variant
        if variant == '18':
            self.CLASSES = tuple(SCANNET_18_CLASSES)
            self.VALID_CLASS_IDS = tuple(SCANNET_18_NYU40_IDS)
            self.IGNORED_CLASS_IDS = tuple()
            self.nyu40_to_model_id = NYU40_TO_18CLASS_MODEL_IDX.copy()
            self.model_id_to_nyu40 = MODEL_IDX_TO_NYU40_18CLASS.copy()
            self.model_id_to_name = MODEL_IDX_TO_NAME_18CLASS.copy()
        elif variant == '35':
            self.CLASSES = tuple(SCANNET_35_CLASSES)
            self.VALID_CLASS_IDS = tuple(VALID_NYU40_IDS_35CLASS)
            self.IGNORED_CLASS_IDS = tuple(IGNORED_NYU40_IDS_35CLASS)
            self.nyu40_to_model_id = NYU40_TO_35CLASS_MODEL_IDX.copy()
            self.model_id_to_nyu40 = MODEL_IDX_TO_NYU40_35CLASS.copy()
            self.model_id_to_name = MODEL_IDX_TO_NAME_35CLASS.copy()
        else:  # variant == 'dynamic_head' (default)
            self.CLASSES = tuple(SCANNET_DYNAMIC_HEAD_CLASSES)
            self.VALID_CLASS_IDS = tuple(VALID_NYU40_IDS_DYNAMIC_HEAD)
            self.IGNORED_CLASS_IDS = tuple(IGNORED_NYU40_IDS_DYNAMIC_HEAD)
            self.nyu40_to_model_id = NYU40_TO_DYNAMIC_HEAD_GCI.copy()
            self.model_id_to_nyu40 = DYNAMIC_HEAD_GCI_TO_NYU40.copy()
            self.model_id_to_name = DYNAMIC_HEAD_GCI_TO_NAME.copy()
        
        # Use variant-specific classes if not provided
        if classes is None:
            classes = self.CLASSES
            
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert 'use_camera' in self.modality and \
               'use_depth' in self.modality
        assert self.modality['use_camera'] or self.modality['use_depth']

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - file_name (str): Filename of point clouds.
                - img_prefix (str, optional): Prefix of image files.
                - img_info (dict, optional): Image info.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        sample_idx = info['point_cloud']['lidar_idx']
        pts_filename = osp.join(self.data_root, info['pts_path'])
        input_dict = dict(sample_idx=sample_idx)

        if self.modality['use_depth']:
            input_dict['pts_filename'] = pts_filename
            input_dict['file_name'] = pts_filename

        if self.modality['use_camera']:
            img_info = []
            for img_path in info['img_paths']:
                img_info.append(
                    dict(filename=osp.join(self.data_root, img_path)))
            intrinsic = info['intrinsics']
            axis_align_matrix = self._get_axis_align_matrix(info)
            depth2img = []
            for extrinsic in info['extrinsics']:
                depth2img.append(
                    intrinsic @ np.linalg.inv(axis_align_matrix @ extrinsic))

            input_dict['img_prefix'] = None
            input_dict['img_info'] = img_info
            input_dict['depth2img'] = depth2img

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and ~(annos['gt_labels_3d'] != -1).any():
                return None
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`DepthInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - pts_instance_mask_path (str): Path of instance masks.
                - pts_semantic_mask_path (str): Path of semantic masks.
                - axis_align_matrix (np.ndarray): Transformation matrix for
                    global scene alignment.
        """
        # Use index to get the annos, thus the evalhook could also use this api
        info = self.data_infos[index]
        if info['annos']['gt_num'] != 0:
            gt_bboxes_3d = info['annos']['gt_boxes_upright_depth'].astype(
                np.float32)  # k, 6
            gt_labels_3d_raw = info['annos']['class'].astype(np.int64)
            
            # Check if we're using 18-class variant with pre-mapped indices
            # The 18-class annotation files already have model indices (0-17), not NYU40 IDs
            if self.variant == '18':
                # For 18-class variant, the annotations are already in model index format (0-17)
                # No filtering or remapping needed - use annotations as-is
                gt_labels_3d = gt_labels_3d_raw
                # Keep all bboxes since no filtering is needed
                # gt_bboxes_3d is already correct
            else:
                # For other variants (35-class, dynamic_head), perform NYU40 to model index mapping
                gt_labels_3d_nyu40 = gt_labels_3d_raw
                
                # Filter out ignored classes and map to model indices
                valid_mask = np.isin(gt_labels_3d_nyu40, list(self.VALID_CLASS_IDS))
                gt_bboxes_3d = gt_bboxes_3d[valid_mask]
                gt_labels_3d_nyu40 = gt_labels_3d_nyu40[valid_mask]
                
                # Map NYU40 IDs to model class indices (0-34)
                gt_labels_3d = np.array([
                    self.nyu40_to_model_id[nyu40_id] for nyu40_id in gt_labels_3d_nyu40
                ], dtype=np.int64)
            
            # CRITICAL VALIDATION: Check if any labels exceed expected range for incremental learning
            if hasattr(self, 'stage_classes') and len(gt_labels_3d) > 0:
                max_label = gt_labels_3d.max()
                if max_label >= len(self.CLASSES):
                    print(f"❌ CRITICAL ERROR: Found label {max_label} but model only has {len(self.CLASSES)} classes")
                    print(f"   Labels in this sample: {gt_labels_3d}")
                    print(f"   Raw labels: {gt_labels_3d_raw}")
                    print(f"   Variant: {self.variant}")
                    print(f"   Stage classes: {getattr(self, 'stage_classes', 'not set')}")
                    raise ValueError(f"Label {max_label} exceeds model class count {len(self.CLASSES)}")
        else:
            gt_bboxes_3d = np.zeros((0, 6), dtype=np.float32)
            gt_labels_3d = np.zeros((0, ), dtype=np.int64)

        # to target box structure
        gt_bboxes_3d = DepthInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            with_yaw=False,
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        # Use 40-class mask paths (since we're using 40-class data with ignored classes)
        pts_instance_mask_path = osp.join(self.data_root,
                                          info['pts_instance_mask_path'])
        pts_semantic_mask_path = osp.join(self.data_root,
                                          info['pts_semantic_mask_path'])

        axis_align_matrix = self._get_axis_align_matrix(info)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            pts_instance_mask_path=pts_instance_mask_path,
            pts_semantic_mask_path=pts_semantic_mask_path,
            axis_align_matrix=axis_align_matrix)
        return anns_results

    def prepare_test_data(self, index):
        """Prepare data for testing.

        We should take axis_align_matrix from self.data_infos since we need
            to align point clouds.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        # take the axis_align_matrix from data_infos
        input_dict['ann_info'] = dict(
            axis_align_matrix=self._get_axis_align_matrix(
                self.data_infos[index]))
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    @staticmethod
    def _get_axis_align_matrix(info):
        """Get axis_align_matrix from info. If not exist, return identity mat.

        Args:
            info (dict): one data info term.

        Returns:
            np.ndarray: 4x4 transformation matrix.
        """
        if 'axis_align_matrix' in info['annos'].keys():
            return info['annos']['axis_align_matrix'].astype(np.float32)
        else:
            warnings.warn(
                'axis_align_matrix is not found in ScanNet data info, please '
                'use new pre-process scripts to re-generate ScanNet data')
            return np.eye(4).astype(np.float32)

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='DEPTH',
                shift_height=False,
                load_dim=6,
                use_dim=[0, 1, 2, 3, 4, 5]),
            dict(type='GlobalAlignment', rotation_axis=2),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def evaluate(self,
                 results,
                 metric=None,
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 classwise=False):
        """Evaluate.

        Evaluation in indoor protocol with proper class ID mapping.
        
        CRITICAL FIX: This method ensures ground truth NYU40 IDs are mapped to 
        model indices (0-34) before evaluation, matching the prediction format.
        
        Args:
            results (list[dict]): List of results.
            metric (str | list[str], optional): Metrics to be evaluated.
                Defaults to None.
            iou_thr (list[float]): AP IoU thresholds. Defaults to (0.25, 0.5).
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
            classwise (bool, optional): Whether to return classwise results.
                Default: False.

        Returns:
            dict: Evaluation results.
        """
        # Helper function to handle logging when logger might be None
        def safe_log(message, level='info'):
            if logger is not None:
                if level == 'warning':
                    logger.warning(message)
                elif level == 'error':
                    logger.error(message)
                else:  # default to info
                    logger.info(message)
            else:
                print(message)
        
        from mmdet3d.core.evaluation import indoor_eval
        assert isinstance(results, list), f'Expect results to be list, got {type(results)}.'
        assert len(results) > 0, 'Expect length of results > 0.'
        assert len(results) == len(self.data_infos)
        assert isinstance(results[0], dict), f'Expect elements in results to be dict, got {type(results[0])}.'
        
        # CRITICAL FIX: Create corrected ground truth annotations with mapped class IDs
        gt_annos_corrected = []
        
        # Check if we're using 18-class variant with pre-mapped indices
        # The 18-class annotation files already have model indices (0-17), not NYU40 IDs
        if self.variant == '18':
            # For 18-class variant, the annotations are already in model index format (0-17)
            # No remapping needed - just use the annotations as-is
            gt_annos_corrected = [info['annos'].copy() for info in self.data_infos]
        else:
            # For other variants (35-class, dynamic_head), perform NYU40 to model index mapping
            for info in self.data_infos:
                gt_anno = info['annos'].copy()
                if gt_anno['gt_num'] != 0:
                    # Get original NYU40 class IDs
                    gt_labels_nyu40 = gt_anno['class'].astype(np.int64)
                    
                    # Filter valid classes and map to model indices
                    valid_mask = np.isin(gt_labels_nyu40, list(self.VALID_CLASS_IDS))
                    
                    if np.any(valid_mask):
                        # Apply filtering to all annotation arrays
                        gt_anno_corrected = gt_anno.copy()
                        gt_anno_corrected['gt_boxes_upright_depth'] = gt_anno['gt_boxes_upright_depth'][valid_mask]
                        
                        # Map NYU40 IDs to model indices (0-34) for evaluation
                        gt_labels_mapped = np.array([
                            self.nyu40_to_model_id[nyu40_id] 
                            for nyu40_id in gt_labels_nyu40[valid_mask]
                        ], dtype=np.int64)
                        gt_anno_corrected['class'] = gt_labels_mapped
                        gt_anno_corrected['gt_num'] = len(gt_labels_mapped)
                        
                        # Debug messages removed - class mapping working correctly
                    else:
                        # No valid objects in this scene
                        gt_anno_corrected = gt_anno.copy()
                        gt_anno_corrected['gt_boxes_upright_depth'] = np.array([], dtype=np.float32).reshape(0, 6)
                        gt_anno_corrected['class'] = np.array([], dtype=np.int64)
                        gt_anno_corrected['gt_num'] = 0
                else:
                    gt_anno_corrected = gt_anno
                
                gt_annos_corrected.append(gt_anno_corrected)
        
        # Check if class-agnostic evaluation is requested
        if hasattr(self, '_class_agnostic_mode') and self._class_agnostic_mode:
            # CLASS-AGNOSTIC MODE: Convert all GT labels to class 0 and use single class mapping
            safe_log("🎯 APPLYING CLASS-AGNOSTIC EVALUATION MODE")
            
            # Convert all GT labels to class 0 for pure localization assessment
            for gt_anno in gt_annos_corrected:
                if gt_anno['gt_num'] > 0:
                    gt_anno['class'] = np.zeros_like(gt_anno['class'], dtype=np.int64)
            
            # Convert all predictions to class 0 (should already be done, but ensure consistency)
            total_predictions = 0
            for result in results:
                if 'labels_3d' in result:
                    result['labels_3d'] = torch.zeros_like(result['labels_3d'])
                    total_predictions += len(result['labels_3d'])
            
            # Create single-class mapping for evaluation
            label2cat = {0: 'object'}
            
            safe_log(f"✅ Class-agnostic conversion complete:")
            safe_log(f"   All GT and predictions converted to single 'object' class")
            safe_log(f"   This evaluates pure localization performance (ignoring classification)")
            
        else:
            # STANDARD MODE: Create label mapping for evaluation (model index -> class name)  
            # INCREMENTAL LEARNING FIX: Only create label2cat for seen classes during incremental learning
            if hasattr(self, 'seen_classes_for_eval') and self.seen_classes_for_eval is not None:
                # For incremental learning, only include seen classes in label2cat
                # Do not cap by len(self.CLASSES) to avoid truncation when configs override classes
                label2cat = {i: self.model_id_to_name.get(i, f'class_{i}') for i in self.seen_classes_for_eval}
            else:
                # For standard training, include all classes
                label2cat = {i: self.model_id_to_name[i] for i in range(len(self.CLASSES))}
        
        # Log incremental learning evaluation details if applicable (skip for class-agnostic mode)
        if (hasattr(self, 'seen_classes_for_eval') and self.seen_classes_for_eval is not None and 
            not (hasattr(self, '_class_agnostic_mode') and self._class_agnostic_mode)):
            safe_log(f"🎯 INCREMENTAL EVALUATION: Filtering to {len(self.seen_classes_for_eval)} seen classes: {self.seen_classes_for_eval}")
            
            # VERIFICATION: Count ground truth objects by class before filtering
            # For incremental learning, only show GT counts for seen classes
            gt_class_counts = {}
            total_gt_objects = 0
            seen_gt_objects = 0
            for gt_anno in gt_annos_corrected:
                if gt_anno['gt_num'] > 0:
                    for class_id in gt_anno['class']:
                        gt_class_counts[class_id] = gt_class_counts.get(class_id, 0) + 1
                        total_gt_objects += 1
                        if class_id in self.seen_classes_for_eval:
                            seen_gt_objects += 1
            
            # Only show GT counts for seen classes in incremental learning
            seen_gt_counts = {cls: gt_class_counts.get(cls, 0) for cls in self.seen_classes_for_eval if cls in gt_class_counts}
            
            safe_log(f"📊 GROUND TRUTH VERIFICATION (Incremental Learning):")
            safe_log(f"   Total GT objects in full dataset: {total_gt_objects}")
            safe_log(f"   GT objects from seen classes only: {seen_gt_objects}")
            safe_log(f"   GT per seen class: {dict(sorted(seen_gt_counts.items()))}")
            
            # Check if previous stage classes have ground truth
            seen_classes_with_gt = [cls for cls in self.seen_classes_for_eval if cls in gt_class_counts]
            seen_classes_no_gt = [cls for cls in self.seen_classes_for_eval if cls not in gt_class_counts]
            
            if seen_classes_no_gt:
                safe_log(f"⚠️  Seen classes with NO ground truth: {seen_classes_no_gt}", level='warning')
            if seen_classes_with_gt:
                safe_log(f"✅ Seen classes WITH ground truth: {seen_classes_with_gt}")
                
                # DYNAMIC PREVIOUS STAGES DETECTION
                if self.current_stage_classes is not None:
                    # Get previous stage classes (seen classes that are not in current stage)
                    prev_stage_classes = [cls for cls in seen_classes_with_gt 
                                        if cls not in self.current_stage_classes]
                    if prev_stage_classes:
                        prev_stage_gt_counts = {cls: gt_class_counts[cls] for cls in prev_stage_classes}
                        safe_log(f"📈 Previous stage classes (dynamic): {sorted(prev_stage_classes)}")
                        safe_log(f"📈 Previous stage GT counts: {prev_stage_gt_counts}")
                else:
                    # Fallback: assume previous stages are lower-numbered classes
                    # This preserves backward compatibility when stage info is not available
                    if len(seen_classes_with_gt) > 12:  # Only if we have enough classes to suggest multiple stages
                        # Heuristic: assume roughly equal stage sizes
                        stage_size = max(12, len(self.seen_classes_for_eval) // 3)  # Estimate stage size
                        prev_stage_threshold = min(seen_classes_with_gt) + stage_size
                        prev_stage_classes = [cls for cls in seen_classes_with_gt if cls < prev_stage_threshold]
                        if prev_stage_classes:
                            prev_stage_gt_counts = {cls: gt_class_counts[cls] for cls in prev_stage_classes}
                            safe_log(f"📈 Previous stage classes (heuristic, threshold<{prev_stage_threshold}): {sorted(prev_stage_classes)}")
                            safe_log(f"📈 Previous stage GT counts: {prev_stage_gt_counts}")
            
            # label2cat already filtered to seen classes, just log it
            safe_log(f"   Evaluation will show {len(label2cat)} classes: {list(label2cat.values())}")
            
            # VERIFICATION: Count predictions by class to diagnose catastrophic forgetting
            pred_class_counts = {}
            total_predictions = 0
            high_confidence_preds = {}
            
            for result in results:
                if 'labels_3d' in result and len(result['labels_3d']) > 0:
                    labels = result['labels_3d'].cpu().numpy() if hasattr(result['labels_3d'], 'cpu') else result['labels_3d']
                    scores = result['scores_3d'].cpu().numpy() if hasattr(result['scores_3d'], 'cpu') else result['scores_3d']
                    
                    for label, score in zip(labels, scores):
                        pred_class_counts[label] = pred_class_counts.get(label, 0) + 1
                        total_predictions += 1
                        
                        # Count high-confidence predictions (>0.5)
                        if score > 0.5:
                            high_confidence_preds[label] = high_confidence_preds.get(label, 0) + 1
            
            safe_log(f"📊 PREDICTION VERIFICATION:")
            safe_log(f"   Total predictions: {total_predictions}")
            safe_log(f"   Predictions per class: {dict(sorted(pred_class_counts.items()))}")
            if high_confidence_preds:
                safe_log(f"   High-confidence predictions (>0.5): {dict(sorted(high_confidence_preds.items()))}")
            
            # DYNAMIC CATASTROPHIC FORGETTING ANALYSIS
            if seen_classes_with_gt:
                # Determine previous stage classes dynamically
                prev_stage_classes = []
                analysis_method = "none"
                
                if self.current_stage_classes is not None:
                    # Method 1: Use explicit current stage info
                    prev_stage_classes = [cls for cls in seen_classes_with_gt 
                                        if cls not in self.current_stage_classes]
                    analysis_method = "explicit stage info"
                elif len(seen_classes_with_gt) > 12:
                    # Method 2: Heuristic approach for backward compatibility
                    stage_size = max(12, len(self.seen_classes_for_eval) // 3)
                    prev_stage_threshold = min(seen_classes_with_gt) + stage_size
                    prev_stage_classes = [cls for cls in seen_classes_with_gt if cls < prev_stage_threshold]
                    analysis_method = f"heuristic (threshold<{prev_stage_threshold})"
                
                if prev_stage_classes:
                    prev_stage_pred_counts = {cls: pred_class_counts.get(cls, 0) for cls in prev_stage_classes}
                    prev_stage_gt_counts_subset = {cls: gt_class_counts.get(cls, 0) for cls in prev_stage_classes}
                    
                    safe_log(f"🔍 CATASTROPHIC FORGETTING ANALYSIS (method: {analysis_method}):")
                    safe_log(f"   Previous stage classes: {sorted(prev_stage_classes)}")
                    safe_log(f"   Previous stage GT counts: {prev_stage_gt_counts_subset}")
                    safe_log(f"   Previous stage prediction counts: {prev_stage_pred_counts}")
                    
                    zero_pred_classes = [cls for cls in prev_stage_classes if pred_class_counts.get(cls, 0) == 0 and gt_class_counts.get(cls, 0) > 0]
                    if zero_pred_classes:
                        safe_log(f"⚠️  Classes with GT but ZERO predictions (catastrophic forgetting): {zero_pred_classes}", level='warning')
                    else:
                        safe_log(f"✅ All previous stage classes have predictions")
                else:
                    safe_log(f"ℹ️  No previous stage classes detected for catastrophic forgetting analysis")
        
        # INCREMENTAL LEARNING FIX: Filter annotations to only include seen classes
        # This prevents KeyError when ground truth contains unseen classes
        if (hasattr(self, 'seen_classes_for_eval') and self.seen_classes_for_eval is not None):
            safe_log(f"🎯 FILTERING GT AND PREDICTIONS: Keeping only {len(self.seen_classes_for_eval)} seen classes")
            
            # Filter ground truth annotations to only seen classes
            filtered_gt_annos = []
            for gt_anno in gt_annos_corrected:
                if 'class' in gt_anno and gt_anno['gt_num'] > 0:
                    labels = gt_anno['class']
                    seen_mask = np.isin(labels, self.seen_classes_for_eval)
                    
                    if seen_mask.any():
                        filtered_anno = {}
                        for key, value in gt_anno.items():
                            if key == 'class':
                                filtered_anno[key] = labels[seen_mask]
                            elif key == 'gt_bboxes_3d':
                                # Handle 3D bounding boxes
                                if hasattr(value, '__getitem__') and len(value) == len(labels):
                                    filtered_anno[key] = value[seen_mask]
                                else:
                                    filtered_anno[key] = value
                            elif isinstance(value, np.ndarray) and len(value) == len(labels):
                                # Filter arrays that match the number of objects
                                filtered_anno[key] = value[seen_mask]
                            else:
                                # Keep metadata unchanged
                                filtered_anno[key] = value
                        
                        # Update gt_num to reflect filtered objects
                        filtered_anno['gt_num'] = seen_mask.sum()
                        filtered_gt_annos.append(filtered_anno)
                    else:
                        # Scene has no objects from seen classes - add empty annotation
                        empty_anno = gt_anno.copy()
                        empty_anno['gt_num'] = 0
                        empty_anno['class'] = np.array([], dtype=labels.dtype)
                        if 'gt_bboxes_3d' in empty_anno:
                            empty_anno['gt_bboxes_3d'] = np.array([]).reshape(0, *gt_anno['gt_bboxes_3d'].shape[1:])
                        filtered_gt_annos.append(empty_anno)
                else:
                    # Keep scenes with no annotations as-is
                    filtered_gt_annos.append(gt_anno)
            
            # Filter predictions to only seen classes
            filtered_results = []
            for result in results:
                if 'labels_3d' in result and len(result['labels_3d']) > 0:
                    labels = result['labels_3d']
                    if hasattr(labels, 'cpu'):
                        labels = labels.cpu().numpy()
                    elif hasattr(labels, 'numpy'):
                        labels = labels.numpy()
                    
                    seen_mask = np.isin(labels, self.seen_classes_for_eval)
                    
                    if seen_mask.any():
                        filtered_result = {}
                        for key, value in result.items():
                            if key == 'labels_3d':
                                filtered_result[key] = result[key][seen_mask]
                            elif key in ['scores_3d', 'boxes_3d']:
                                # Handle detection results
                                filtered_result[key] = value[seen_mask]
                            elif hasattr(value, '__len__') and len(value) == len(labels):
                                # Filter arrays that match the number of detections
                                filtered_result[key] = value[seen_mask]
                            else:
                                # Keep metadata unchanged
                                filtered_result[key] = value
                        filtered_results.append(filtered_result)
                    else:
                        # No predictions from seen classes - add empty result
                        empty_result = {key: value for key, value in result.items() 
                                     if key not in ['labels_3d', 'scores_3d', 'boxes_3d']}
                        empty_result['labels_3d'] = result['labels_3d'][[]]  # Empty tensor of same type
                        empty_result['scores_3d'] = result['scores_3d'][[]]
                        empty_result['boxes_3d'] = result['boxes_3d'][[]]
                        filtered_results.append(empty_result)
                else:
                    # Keep results with no predictions as-is
                    filtered_results.append(result)
            
            # Use filtered annotations for incremental evaluation
            gt_annos_for_eval = filtered_gt_annos
            results_for_eval = filtered_results
            safe_log(f"✅ Filtering complete - using {len(gt_annos_for_eval)} GT annotations and {len(results_for_eval)} predictions")
        else:
            # Joint training: use original annotations
            gt_annos_for_eval = gt_annos_corrected
            results_for_eval = results
        
        # Call indoor_eval with appropriate annotations
        ret_dict = indoor_eval(
            gt_annos_for_eval,
            results_for_eval,
            iou_thr,
            label2cat,
            logger=logger,
            box_type_3d=self.box_type_3d,
            box_mode_3d=self.box_mode_3d)
        
        if show:
            self.show(results, out_dir, pipeline=pipeline)
        
        return ret_dict

    def show(self, results, out_dir, show=True, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Visualize the results online.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._build_default_pipeline()
        for i, result in enumerate(results):
            data_info = self.data_infos[i]
            pts_path = data_info['pts_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points', load_annos=True).numpy()
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d']
            gt_bboxes = gt_bboxes.corners.numpy() if len(gt_bboxes) else None
            gt_labels = self.get_ann_info(i)['gt_labels_3d']
            pred_bboxes = result['boxes_3d']
            pred_bboxes = pred_bboxes.corners.numpy() if len(pred_bboxes) else None
            pred_labels = result['labels_3d']
            show_result_v2(points, gt_bboxes, gt_labels,
                           pred_bboxes, pred_labels, out_dir, file_name)

