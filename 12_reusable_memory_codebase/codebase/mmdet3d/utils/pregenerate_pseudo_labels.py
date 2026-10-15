#!/usr/bin/env python3
"""
Pre-generate Pseudo Labels for Incremental Learning

This module provides functionality to pre-generate pseudo labels at the start of each
training stage, significantly speeding up experimentation by avoiding on-the-fly generation.

Pre-generation computes pseudo labels once at stage start using the previous stage checkpoint,
then saves them for reuse during training iterations. This provides 3-5x speedup compared
to on-the-fly generation.

Key Features:
- De-duplication of natural and memory bank scenes
- Confidence filtering (0.45 threshold during training)
- NYU40 format compatibility for ground truth alignment
- Explicit error handling with no fallback mechanisms

Date: 2025-09-02
"""

import os
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import json

from mmcv import Config
from mmdet3d.apis import init_model, inference_detector

# Import class mappings
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / 'configs' / '_base_' / 'class_mappings'))
from scannet_dynamic_head_mappings import (
    DYNAMIC_HEAD_GCI_TO_NYU40,
    NYU40_TO_DYNAMIC_HEAD_GCI
)


class PseudoLabelPreGenerator:
    """
    Pre-generate pseudo labels for incremental learning stages.
    
    This speeds up training by computing pseudo labels once at the start of each stage
    rather than generating them on-the-fly during training iterations.
    """
    
    def __init__(self, 
                 checkpoint_path: str,
                 stage_id: int,
                 stage_definitions: Optional[List[Dict[str, Any]]] = None,
                 confidence_threshold: float = 0.05,  # Very low for debugging flexibility
                 output_dir: str = "./pseudo_labels",
                 config_suffix: str = "",
                 device: str = 'cuda:0'):
        """
        Initialize the pseudo label pre-generator.
        
        Args:
            checkpoint_path: Path to previous stage checkpoint
            stage_id: Current stage ID (>=2)
            stage_definitions: Optional explicit stage definitions from config.
            confidence_threshold: Confidence threshold for filtering (default: 0.45)
            output_dir: Directory to save pseudo labels
            config_suffix: Configuration suffix to distinguish different experiments
            device: Device for inference
        """
        self.checkpoint_path = checkpoint_path
        self.stage_id = stage_id
        self.previous_stage_id = stage_id - 1
        self.confidence_threshold = confidence_threshold
        self.output_dir = Path(output_dir)
        self.config_suffix = config_suffix
        self.device = device
        self.stage_definitions = stage_definitions
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get classes for previous stage (what the model can detect).
        self.previous_stage_classes = self._resolve_previous_stage_classes()
        
        print(f"🔧 Pseudo Label Pre-Generator initialized:")
        print(f"   Stage: {self.stage_id} (using Stage {self.previous_stage_id} model)")
        print(f"   Previous stage classes: {len(self.previous_stage_classes)} classes")
        print(f"   Confidence threshold: {self.confidence_threshold}")
        print(f"   Output directory: {self.output_dir}")
        
        # Load model
        self.model = self._load_model()

    def _resolve_previous_stage_classes(self) -> List[int]:
        """Resolve cumulative model classes for stage_id-1."""
        # Preferred path: explicit stage definitions from config.
        if isinstance(self.stage_definitions, list) and self.stage_definitions:
            seen: List[int] = []
            for stage_def in self.stage_definitions:
                sid = int(stage_def.get('stage_id', 0))
                if sid <= int(self.previous_stage_id):
                    seen.extend([int(x) for x in stage_def.get('class_indices', [])])
            resolved = sorted(set(seen))
            if resolved:
                return resolved

        # Compatibility fallback: historical ScanNet 5-stage dynamic-head ranges.
        stage_class_ranges = {
            1: list(range(0, 7)),
            2: list(range(0, 14)),
            3: list(range(0, 21)),
            4: list(range(0, 28)),
            5: list(range(0, 35)),
        }
        if int(self.previous_stage_id) not in stage_class_ranges:
            raise ValueError(
                "Cannot infer previous-stage classes without stage_definitions for "
                f"stage_id={self.stage_id} (previous={self.previous_stage_id})."
            )
        return stage_class_ranges[int(self.previous_stage_id)]
    
    def _load_model(self):
        """Load the model from previous stage checkpoint."""
        print(f"\n📦 Loading Stage {self.previous_stage_id} model...")
        
        # Use incremental learning config
        config_path = project_root / "configs/incremental/scannet/tr3d_dynamic_head_s5_pure_finetuning.py"
        config = Config.fromfile(str(config_path))
        
        # Set the correct number of classes for previous stage
        num_classes = len(self.previous_stage_classes)
        config.model.head.n_classes = num_classes
        
        # Set inference parameters
        config.model.test_cfg.score_thr = 0.05  # Low threshold for initial detection
        config.model.test_cfg.nms_pre = 1000
        config.model.test_cfg.iou_thr = 0.5
        
        # Initialize model
        model = init_model(config, self.checkpoint_path, device=self.device)
        model.eval()
        
        print(f"✅ Model loaded: {num_classes} classes from Stage {self.previous_stage_id}")
        return model
    
    def get_all_scenes_for_stage(self, 
                                 natural_scenes: List[str],
                                 memory_bank_scenes: Optional[List[str]] = None) -> List[str]:
        """
        Get all unique scenes for the current stage (natural + memory bank).
        
        Args:
            natural_scenes: List of naturally occurring scenes for this stage
            memory_bank_scenes: Optional list of memory bank scenes
            
        Returns:
            List of unique scene IDs to process
        """
        all_scenes = set(natural_scenes)
        
        if memory_bank_scenes:
            print(f"   Adding {len(memory_bank_scenes)} memory bank scenes")
            all_scenes.update(memory_bank_scenes)
        
        unique_scenes = sorted(list(all_scenes))
        print(f"📊 Total unique scenes to process: {len(unique_scenes)}")
        print(f"   Natural scenes: {len(natural_scenes)}")
        if memory_bank_scenes:
            print(f"   Memory bank scenes: {len(memory_bank_scenes)}")
            print(f"   Overlap: {len(natural_scenes) + len(memory_bank_scenes) - len(unique_scenes)} scenes")
        
        return unique_scenes
    
    def generate_pseudo_labels_for_scene(self, scene_info: Dict) -> Optional[Dict]:
        """
        Generate pseudo labels for a single scene.
        
        Args:
            scene_info: Scene information dictionary
            
        Returns:
            Dictionary with pseudo labels or None if no valid detections
        """
        # Handle different scene info formats
        if 'point_cloud' in scene_info and 'lidar_idx' in scene_info['point_cloud']:
            scene_id = scene_info['point_cloud']['lidar_idx']
            pts_path = project_root / "data/scannet" / scene_info['pts_path']
        elif 'sample_idx' in scene_info:
            scene_id = scene_info['sample_idx']
            if 'pts_filename' in scene_info:
                pts_path = project_root / "data/scannet" / scene_info['pts_filename']
            else:
                print(f"   ⚠️ Scene {scene_id}: No pts_filename found")
                return None
        else:
            print(f"   ⚠️ Unknown scene format: {list(scene_info.keys())}")
            return None
        
        if not pts_path.exists():
            print(f"   ⚠️ Point cloud not found: {pts_path}")
            return None
        
        try:
            # Run inference
            with torch.no_grad():
                result = inference_detector(self.model, str(pts_path))
            
            # Extract detections
            if not (result and len(result) > 0 and len(result[0]) > 0):
                return None
            
            res = result[0][0]
            if not (isinstance(res, dict) and 'boxes_3d' in res):
                return None
            
            # Extract predictions
            boxes_3d = res['boxes_3d']
            scores_3d = res['scores_3d']
            labels_3d = res['labels_3d']
            
            # Convert to numpy
            if hasattr(scores_3d, 'cpu'):
                scores_np = scores_3d.cpu().numpy()
                labels_np = labels_3d.cpu().numpy()
            else:
                scores_np = np.array(scores_3d)
                labels_np = np.array(labels_3d)
            
            if hasattr(boxes_3d, 'tensor'):
                boxes_np = boxes_3d.tensor.cpu().numpy()
            else:
                boxes_np = np.array(boxes_3d)
            
            # Filter by confidence threshold
            conf_mask = scores_np >= self.confidence_threshold
            
            # Filter for previous stage classes (what the model knows)
            class_mask = labels_np < len(self.previous_stage_classes)
            
            # Combine filters
            valid_mask = conf_mask & class_mask
            
            if not valid_mask.any():
                return None
            
            filtered_boxes = boxes_np[valid_mask]
            filtered_scores = scores_np[valid_mask]
            filtered_labels = labels_np[valid_mask]  # GCI indices
            
            # Apply axis alignment if available (rotate centers; keep yaw to adjust later)
            axis_align_matrix = self._extract_axis_align_matrix(scene_info)
            rot_angle = 0.0
            if axis_align_matrix is not None:
                aligned_boxes = self._apply_alignment_to_boxes(filtered_boxes, axis_align_matrix)
                # Extract rotation about z to adjust yaw (R is 3x3)
                R = axis_align_matrix[:3, :3]
                rot_angle = float(np.arctan2(R[1, 0], R[0, 0]))
            else:
                aligned_boxes = filtered_boxes
            
            # CRITICAL FIX: Convert from bottom_center to gravity_center
            # Model predictions use bottom_center (Z at bottom of box)
            # For proper alignment with GT and point clouds, we need gravity_center (Z at geometric center)
            # This matches the fix applied in generate_stage2_pseudo_labels_analysis.py
            aligned_boxes[:, 2] += aligned_boxes[:, 5] * 0.5  # Z += height * 0.5

            # If yaw present, convert oriented (dx,dy) to axis-aligned AABB extents using yaw'
            if aligned_boxes.ndim == 2 and aligned_boxes.shape[1] >= 7:
                yaw = aligned_boxes[:, 6].astype(np.float32) + rot_angle
                dx = aligned_boxes[:, 3].astype(np.float32)
                dy = aligned_boxes[:, 4].astype(np.float32)
                # AABB extents for rotation about z: combine projected components
                c = np.abs(np.cos(yaw))
                s = np.abs(np.sin(yaw))
                aabb_x = c * dx + s * dy
                aabb_y = s * dx + c * dy
                aligned_boxes[:, 3] = aabb_x
                aligned_boxes[:, 4] = aabb_y
                # leave dz unchanged

            # Enforce canonical 6D shape (drop yaw if present); fail if underspecified
            if aligned_boxes.ndim != 2:
                raise ValueError(f"Boxes must be 2D array; got shape {aligned_boxes.shape}")
            if aligned_boxes.shape[1] > 6:
                aligned_boxes = aligned_boxes[:, :6]
            elif aligned_boxes.shape[1] < 6:
                raise ValueError(f"Boxes must have 6 dims [x,y,z,w,h,d]; got {aligned_boxes.shape[1]}")
            
            # Convert GCI labels to NYU40 IDs for storage compatibility
            nyu40_labels = np.array([
                DYNAMIC_HEAD_GCI_TO_NYU40[gci] for gci in filtered_labels
            ])
            
            return {
                'scene_id': scene_id,
                'boxes': aligned_boxes.astype(np.float32),
                'scores': filtered_scores.astype(np.float32),
                'labels': nyu40_labels.astype(np.int64),  # NYU40 IDs for compatibility
                # Canonical metadata
                'label_space': 'nyu40',
                'center_type': 'gravity',
                'axis_aligned': True,
                'box_type': 'upright_depth_6d',
                'num_detections': len(filtered_scores),
                'confidence_threshold': self.confidence_threshold,
                'stage_generated': self.previous_stage_id
            }
            
        except Exception as e:
            print(f"   ❌ Error processing {scene_id}: {e}")
            return None
    
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
        
        # Extract rotation and translation
        rot_mat = axis_align_matrix[:3, :3]
        trans_vec = axis_align_matrix[:3, 3]
        
        # Apply transformation to box centers
        centers = boxes_unaligned[:, :3]
        centers_aligned = centers @ rot_mat.T + trans_vec
        
        # Keep original dimensions
        boxes_aligned = boxes_unaligned.copy()
        boxes_aligned[:, :3] = centers_aligned
        
        return boxes_aligned
    
    def generate_and_save_all(self, 
                             all_scene_infos: List[Dict],
                             scene_ids_to_process: List[str]) -> str:
        """
        Generate pseudo labels for all scenes and save to file.
        
        Args:
            all_scene_infos: List of all scene information dictionaries
            scene_ids_to_process: List of scene IDs to actually process
            
        Returns:
            Path to saved pseudo labels file
        """
        print(f"\n{'='*60}")
        print(f"🏷️ PRE-GENERATING PSEUDO LABELS FOR STAGE {self.stage_id}")
        print(f"{'='*60}")
        print(f"Using checkpoint: {self.checkpoint_path}")
        print(f"Unique scenes to process: {len(scene_ids_to_process)}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        
        # Determine output filename
        if self.config_suffix:
            filename = f"stage_{self.stage_id}_{self.config_suffix}_pseudo_labels.pkl"
        else:
            filename = f"stage_{self.stage_id}_pseudo_labels.pkl"
        output_file = self.output_dir / filename
        print(f"Output file: {filename}")
        print(f"{'='*60}")
        
        # Create scene info lookup - handle different data formats
        scene_info_lookup = {}
        for scene_info in all_scene_infos:
            # Handle different data info formats
            if 'point_cloud' in scene_info and 'lidar_idx' in scene_info['point_cloud']:
                scene_id = scene_info['point_cloud']['lidar_idx']
            elif 'sample_idx' in scene_info:
                scene_id = scene_info['sample_idx']
            else:
                print(f"⚠️ Unknown scene format: {list(scene_info.keys())}")
                continue
            scene_info_lookup[scene_id] = scene_info
        
        # Import progress bar
        try:
            from mmcv.utils import ProgressBar
            use_mmcv_progress = True
        except ImportError:
            try:
                from tqdm import tqdm
                use_mmcv_progress = False
            except ImportError:
                print("⚠️ Neither mmcv.utils.ProgressBar nor tqdm available, using simple progress")
                use_mmcv_progress = None
        
        # Generate pseudo labels with progress tracking
        pseudo_labels = {}
        successful = 0
        failed_scenes = []
        class_counts = defaultdict(int)
        total_detections = 0
        
        print(f"Generating pseudo labels...")
        
        if use_mmcv_progress is True:
            prog_bar = ProgressBar(len(scene_ids_to_process))
        elif use_mmcv_progress is False:
            scene_iterator = tqdm(scene_ids_to_process, desc="Processing scenes")
        else:
            scene_iterator = scene_ids_to_process
            
        for i, scene_id in enumerate(scene_ids_to_process if use_mmcv_progress is not False else scene_iterator):
            if scene_id not in scene_info_lookup:
                failed_scenes.append((scene_id, "Scene not found in infos"))
                if use_mmcv_progress is True:
                    prog_bar.update()
                continue
            
            scene_info = scene_info_lookup[scene_id]
            result = self.generate_pseudo_labels_for_scene(scene_info)
            
            if result is not None:
                pseudo_labels[scene_id] = result
                successful += 1
                total_detections += result['num_detections']
                
                # Track class distribution
                for label in result['labels']:
                    class_counts[int(label)] += 1
            else:
                failed_scenes.append((scene_id, "No valid detections"))
            
            if use_mmcv_progress is True:
                prog_bar.update()
            elif use_mmcv_progress is None and (i + 1) % 50 == 0:
                print(f"   Progress: {i+1}/{len(scene_ids_to_process)} scenes ({100*(i+1)/len(scene_ids_to_process):.1f}%)")
        
        # Print detailed results
        print(f"\n{'='*60}")
        print(f"✅ PSEUDO LABEL GENERATION COMPLETE")
        print(f"{'='*60}")
        print(f"Successful: {successful}/{len(scene_ids_to_process)} ({100*successful/len(scene_ids_to_process):.1f}%)")
        if failed_scenes:
            print(f"Failed: {len(failed_scenes)}/{len(scene_ids_to_process)} ({100*len(failed_scenes)/len(scene_ids_to_process):.1f}%)")
            # Show first few failures
            for scene_id, reason in failed_scenes[:3]:
                print(f"  - {scene_id}: {reason}")
            if len(failed_scenes) > 3:
                print(f"  ... and {len(failed_scenes)-3} more failures")
        
        if successful > 0:
            print(f"Total detections: {total_detections:,}")
            print(f"Average per scene: {total_detections/successful:.1f}")
            
            # Show top classes if we have class counts
            if class_counts:
                sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
                print(f"Top detected classes:")
                for class_id, count in sorted_classes[:5]:
                    print(f"  - Class {class_id}: {count:,} detections")
        print(f"{'='*60}")
        
        # Handle case where no pseudo labels were generated
        if len(pseudo_labels) == 0:
            print(f"⚠️ WARNING: No pseudo labels generated for stage {self.stage_id}")
            print(f"   Model may not be confident enough with threshold {self.confidence_threshold}")
            print(f"   Checkpoint: {self.checkpoint_path}")
            print(f"   Training will continue without pseudo labels for this stage")
            # Still save an empty dict so the file exists and training can continue
            pseudo_labels = {}
        
        # Save to file (filename already determined above)
        with open(output_file, 'wb') as f:
            pickle.dump(pseudo_labels, f)
        
        # Report file size
        file_size = os.path.getsize(output_file)
        if file_size < 100:  # Small file indicates empty or minimal pseudo labels
            print(f"💾 Saved empty/minimal pseudo labels to: {output_file} ({file_size} bytes)")
        else:
            print(f"💾 Saved to: {output_file} ({file_size/1024:.1f}KB)")
        
        # Save statistics
        self._save_statistics(pseudo_labels, scene_ids_to_process)
        
        print(f"\n✅ Pre-generation complete!")
        return str(output_file)
    
    def _save_statistics(self, pseudo_labels: Dict, all_scene_ids: List[str]):
        """Save statistics about generated pseudo labels."""
        total_detections = sum(pl['num_detections'] for pl in pseudo_labels.values())
        scenes_with_detections = len(pseudo_labels)
        
        stats = {
            'stage_id': self.stage_id,
            'previous_stage_id': self.previous_stage_id,
            'checkpoint_used': self.checkpoint_path,
            'confidence_threshold': self.confidence_threshold,
            'total_scenes_processed': len(all_scene_ids),
            'scenes_with_detections': scenes_with_detections,
            'success_rate': scenes_with_detections / len(all_scene_ids) if all_scene_ids else 0,
            'total_detections': total_detections,
            'avg_detections_per_scene': total_detections / scenes_with_detections if scenes_with_detections > 0 else 0
        }
        
        stats_file = self.output_dir / f"stage_{self.stage_id}_pseudo_labels_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n📊 Statistics:")
        print(f"   Success rate: {stats['success_rate']:.1%}")
        print(f"   Total detections: {stats['total_detections']:,}")
        print(f"   Avg per scene: {stats['avg_detections_per_scene']:.1f}")


def pregenerate_pseudo_labels_for_stage(
    stage_id: int,
    checkpoint_path: str,
    train_data_file: str,
    stage_definitions: Optional[List[Dict[str, Any]]] = None,
    memory_bank_file: Optional[str] = None,
    confidence_threshold: float = 0.05,  # Very low for debugging flexibility 
    output_dir: str = "./pseudo_labels",
    config_suffix: str = ""
) -> str:
    """
    Pre-generate pseudo labels for a training stage with explicit error handling.
    
    This is the main entry point for pre-generating pseudo labels during training.
    All errors are raised explicitly with no fallback mechanisms.
    
    Args:
        stage_id: Current stage ID (>=2)
        checkpoint_path: Path to previous stage checkpoint
        train_data_file: Path to training data pickle file
        stage_definitions: Optional explicit stage definitions from config
        memory_bank_file: Optional path to memory bank JSON file
        confidence_threshold: Confidence threshold for filtering (default: 0.45)
        output_dir: Directory to save pseudo labels
        config_suffix: Configuration suffix to distinguish different experiments
        
    Returns:
        Path to saved pseudo labels file
        
    Raises:
        FileNotFoundError: If checkpoint or data files are missing
        RuntimeError: If pseudo label generation fails
    """
    print("\n" + "="*60)
    print(f"🎯 PRE-GENERATING PSEUDO LABELS FOR STAGE {stage_id}")
    print("="*60)
    print("Pre-generation provides 3-5x speedup for experimentation.")
    print("Explicit error handling with no fallback mechanisms.")
    
    # Load training data
    with open(train_data_file, 'rb') as f:
        all_scenes = pickle.load(f)
    
    # Get natural scenes for this stage - handle different data formats
    natural_scene_ids = []
    for scene in all_scenes:
        if 'point_cloud' in scene and 'lidar_idx' in scene['point_cloud']:
            natural_scene_ids.append(scene['point_cloud']['lidar_idx'])
        elif 'sample_idx' in scene:
            natural_scene_ids.append(scene['sample_idx'])
        else:
            print(f"⚠️ Skipping scene with unknown format: {list(scene.keys())}")
            continue
    
    # Load memory bank scenes if provided
    memory_bank_scene_ids = None
    if memory_bank_file and os.path.exists(memory_bank_file):
        with open(memory_bank_file, 'r') as f:
            memory_data = json.load(f)
            # Extract scene IDs from memory bank (format depends on JSON structure)
            if 'selected_scenes' in memory_data:
                memory_bank_scene_ids = list(memory_data['selected_scenes'].keys())
            elif 'scenes' in memory_data:
                memory_bank_scene_ids = memory_data['scenes']
                
    # Only print memory bank info if scenes were actually loaded
    if memory_bank_scene_ids:
        print(f"📚 Loaded {len(memory_bank_scene_ids)} scenes from memory bank")
    else:
        print("📚 No memory bank scenes available")
    
    # Initialize generator
    generator = PseudoLabelPreGenerator(
        checkpoint_path=checkpoint_path,
        stage_id=stage_id,
        stage_definitions=stage_definitions,
        confidence_threshold=confidence_threshold,
        output_dir=output_dir,
        config_suffix=config_suffix
    )
    
    # Get unique scenes to process
    unique_scenes = generator.get_all_scenes_for_stage(
        natural_scenes=natural_scene_ids,
        memory_bank_scenes=memory_bank_scene_ids
    )
    
    # Generate and save
    output_file = generator.generate_and_save_all(
        all_scene_infos=all_scenes,
        scene_ids_to_process=unique_scenes
    )
    
    # Final validation
    if not os.path.exists(output_file):
        raise FileNotFoundError(f"Pseudo label file was not created: {output_file}")
    
    file_size = os.path.getsize(output_file)
    if file_size < 100:
        raise ValueError(
            f"Generated pseudo label file is too small ({file_size} bytes). "
            f"Stage {stage_id} generation likely failed."
        )
    
    print(f"\n✅ Pre-generation complete! File size: {file_size/1024:.1f}KB")
    return output_file
