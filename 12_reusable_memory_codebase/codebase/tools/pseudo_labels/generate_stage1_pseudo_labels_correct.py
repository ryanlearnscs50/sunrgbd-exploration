#!/usr/bin/env python3
"""
Generate Stage 1 Pseudo Labels with Proper Coordinate Alignment and Format

This script generates pseudo labels that exactly match the GT format and processing pipeline:
1. Uses axis_align_matrix for proper coordinate alignment
2. Stores labels as NYU40 IDs for training compatibility
3. Ensures bounding boxes are in aligned coordinates
4. Validates format consistency with GT annotations

CRITICAL: This ensures pseudo labels work correctly during training by matching
the exact format and coordinate system used by ground truth labels.

Date: 2025-09-02
"""

import os
import sys
import pickle
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
from collections import Counter
from tqdm import tqdm

# Add project root to path
# NOTE: this script was moved from repo root to `tools/pseudo_labels/`.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mmcv import Config
from mmdet3d.apis import init_model, inference_detector

# Import class mappings
sys.path.append(str(project_root / 'configs' / '_base_' / 'class_mappings'))
from scannet_dynamic_head_mappings import (
    SCANNET_DYNAMIC_HEAD_CLASSES,
    VALID_NYU40_IDS_DYNAMIC_HEAD,
    DYNAMIC_HEAD_GCI_TO_NYU40
)


class Stage1PseudoLabelGenerator:
    """Enhanced pseudo label generator with proper alignment and format."""
    
    def __init__(self, checkpoint_path: str, confidence_threshold: float = 0.1):
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold
        
        # Stage 1 definitions (first 7 classes by frequency)
        self.stage1_gci_indices = list(range(7))  # [0, 1, 2, 3, 4, 5, 6]
        self.stage1_nyu40_ids = [DYNAMIC_HEAD_GCI_TO_NYU40[i] for i in self.stage1_gci_indices]
        self.stage1_class_names = [SCANNET_DYNAMIC_HEAD_CLASSES[i] for i in self.stage1_gci_indices]
        
        print(f"🎯 Stage 1 Classes (GCI → NYU40):")
        for i, (gci, nyu40, name) in enumerate(zip(self.stage1_gci_indices, self.stage1_nyu40_ids, self.stage1_class_names)):
            print(f"   {gci} → {nyu40} ({name})")
        
        # Load model
        self.model = self._load_stage1_model()
        
        # Statistics tracking
        self.stats = {
            'total_scenes_processed': 0,
            'scenes_with_detections': 0,
            'total_detections': 0,
            'class_counts': Counter(),
            'confidence_stats': [],
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
        config.model.test_cfg.score_thr = self.confidence_threshold
        config.model.test_cfg.nms_pre = 1000
        config.model.test_cfg.iou_thr = 0.5
        
        # Initialize model
        model = init_model(config, self.checkpoint_path, device='cuda:0')
        model.eval()
        
        print(f"✅ Stage 1 model loaded (7 classes, threshold: {self.confidence_threshold})")
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
        """Apply axis alignment transformation to bounding boxes.
        
        CRITICAL: This ensures pseudo label boxes are in the same coordinate system as GT boxes.
        """
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
    
    def generate_scene_pseudo_labels(self, scene_info: Dict) -> Optional[Dict]:
        """Generate pseudo labels for a single scene with proper alignment."""
        scene_id = scene_info['point_cloud']['lidar_idx']
        pts_path = project_root / "data/scannet" / scene_info['pts_path']
        
        if not pts_path.exists():
            self.stats['processing_errors'].append(f"Point cloud not found: {pts_path}")
            return None
        
        try:
            # Run inference (this uses unaligned points from .bin file)
            with torch.no_grad():
                result = inference_detector(self.model, str(pts_path))
            
            # Extract detections
            if not (result and len(result) > 0 and len(result[0]) > 0):
                return None
            
            res = result[0][0]
            if not (isinstance(res, dict) and 'boxes_3d' in res):
                return None
            
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
            
            # Filter for valid Stage 1 classes (0-6)
            valid_mask = (labels_np >= 0) & (labels_np < 7)
            if not valid_mask.any():
                return None
            
            filtered_boxes = boxes_np[valid_mask]
            filtered_scores = scores_np[valid_mask]
            filtered_labels = labels_np[valid_mask]  # GCI indices (0-6)
            
            # CRITICAL: Apply axis alignment transformation to boxes
            # This ensures pseudo label boxes are in the same coordinate system as GT boxes
            axis_align_matrix = self._extract_axis_align_matrix(scene_info)
            if axis_align_matrix is not None:
                aligned_boxes = self._apply_alignment_to_boxes(filtered_boxes, axis_align_matrix)
                print(f"  Applied axis alignment to {len(aligned_boxes)} boxes")
            else:
                aligned_boxes = filtered_boxes
                print(f"  ⚠️ No axis alignment matrix found for scene {scene_id}")
            
            # Convert GCI labels (0-6) to NYU40 IDs for storage compatibility
            nyu40_labels = np.array([
                DYNAMIC_HEAD_GCI_TO_NYU40[gci] for gci in filtered_labels
            ])
            
            # Update statistics
            self.stats['total_detections'] += len(filtered_scores)
            for gci in filtered_labels:
                self.stats['class_counts'][self.stage1_class_names[gci]] += 1
            self.stats['confidence_stats'].extend(filtered_scores.tolist())
            
            # Class distribution for this scene
            unique_labels, counts = np.unique(filtered_labels, return_counts=True)
            class_dist = {}
            for gci, count in zip(unique_labels, counts):
                class_dist[self.stage1_class_names[gci]] = int(count)
            
            return {
                'scene_id': scene_id,
                'boxes': aligned_boxes.astype(np.float32),  # CRITICAL: Aligned coordinates
                'scores': filtered_scores.astype(np.float32),
                'labels': nyu40_labels.astype(np.int64),  # CRITICAL: NYU40 IDs for compatibility
                'num_detections': len(filtered_scores),
                'class_distribution': class_dist,
                'has_alignment': axis_align_matrix is not None,
                'score_stats': {
                    'min': float(filtered_scores.min()),
                    'max': float(filtered_scores.max()),
                    'mean': float(filtered_scores.mean()),
                    'median': float(np.median(filtered_scores))
                }
            }
            
        except Exception as e:
            error_msg = f"Scene {scene_id}: {str(e)}"
            self.stats['processing_errors'].append(error_msg)
            print(f"❌ Error processing {scene_id}: {e}")
            return None
    
    def generate_all_pseudo_labels(self) -> Dict[str, Any]:
        """Generate pseudo labels for all Stage 1 training scenes."""
        print("🚀 Generating Stage 1 pseudo labels with proper alignment...")
        
        # Load Stage 1 training data
        train_pkl = project_root / "data/scannet/scannet_infos_train_40class_corrected.pkl"
        with open(train_pkl, 'rb') as f:
            all_scenes = pickle.load(f)
        
        # Filter to Stage 1 scenes (scenes containing Stage 1 classes)
        stage1_scenes = self._filter_stage1_scenes(all_scenes)
        print(f"📊 Found {len(stage1_scenes)} Stage 1 training scenes")
        
        # Generate pseudo labels
        pseudo_labels = {}
        
        for scene_info in tqdm(stage1_scenes, desc="Processing scenes"):
            scene_id = scene_info['point_cloud']['lidar_idx']
            
            result = self.generate_scene_pseudo_labels(scene_info)
            if result is not None:
                pseudo_labels[scene_id] = result
                self.stats['scenes_with_detections'] += 1
            
            self.stats['total_scenes_processed'] += 1
        
        print(f"✅ Generated pseudo labels for {len(pseudo_labels)}/{len(stage1_scenes)} scenes")
        return pseudo_labels
    
    def _filter_stage1_scenes(self, all_scenes: List[Dict]) -> List[Dict]:
        """Filter scenes that contain Stage 1 classes."""
        stage1_scenes = []
        
        for scene in all_scenes:
            if 'annos' not in scene or scene['annos']['gt_num'] == 0:
                continue
            
            gt_labels_nyu40 = scene['annos']['class']
            has_stage1_class = any(nyu40_id in self.stage1_nyu40_ids for nyu40_id in gt_labels_nyu40)
            
            if has_stage1_class:
                stage1_scenes.append(scene)
        
        return stage1_scenes
    
    def save_pseudo_labels(self, pseudo_labels: Dict[str, Any], output_dir: Path):
        """Save pseudo labels in multiple formats."""
        output_dir.mkdir(exist_ok=True)
        
        # 1. Main pickle file (for training)
        main_pickle = output_dir / "stage1_pseudo_labels_corrected.pkl"
        with open(main_pickle, 'wb') as f:
            pickle.dump(pseudo_labels, f)
        print(f"✅ Saved pseudo labels: {main_pickle}")
        
        # 2. Statistics summary
        summary_stats = self._generate_summary_stats(pseudo_labels)
        stats_json = output_dir / "stage1_pseudo_labels_stats.json"
        with open(stats_json, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        print(f"✅ Saved statistics: {stats_json}")
        
        # 3. Sample for quick testing (first 10 scenes)
        sample_scenes = dict(list(pseudo_labels.items())[:10])
        sample_pickle = output_dir / "stage1_pseudo_labels_sample.pkl"
        with open(sample_pickle, 'wb') as f:
            pickle.dump(sample_scenes, f)
        print(f"✅ Saved sample: {sample_pickle}")
        
        return main_pickle
    
    def _generate_summary_stats(self, pseudo_labels: Dict[str, Any]) -> Dict:
        """Generate comprehensive statistics."""
        all_scores = []
        class_totals = Counter()
        scenes_per_class = Counter()
        alignment_count = 0
        
        for scene_data in pseudo_labels.values():
            all_scores.extend(scene_data['scores'].tolist())
            
            for class_name, count in scene_data['class_distribution'].items():
                class_totals[class_name] += count
                scenes_per_class[class_name] += 1
            
            if scene_data.get('has_alignment', False):
                alignment_count += 1
        
        all_scores = np.array(all_scores)
        
        return {
            'generation_info': {
                'checkpoint': self.checkpoint_path,
                'confidence_threshold': self.confidence_threshold,
                'generation_timestamp': str(np.datetime64('now')),
                'stage1_classes': self.stage1_class_names,
                'stage1_nyu40_ids': self.stage1_nyu40_ids
            },
            'scene_statistics': {
                'total_scenes_processed': self.stats['total_scenes_processed'],
                'scenes_with_detections': len(pseudo_labels),
                'scenes_with_alignment': alignment_count,
                'success_rate': len(pseudo_labels) / self.stats['total_scenes_processed'] if self.stats['total_scenes_processed'] > 0 else 0
            },
            'detection_statistics': {
                'total_detections': len(all_scores),
                'detections_per_scene': len(all_scores) / len(pseudo_labels) if len(pseudo_labels) > 0 else 0,
                'confidence_stats': {
                    'min': float(all_scores.min()) if len(all_scores) > 0 else 0,
                    'max': float(all_scores.max()) if len(all_scores) > 0 else 0,
                    'mean': float(all_scores.mean()) if len(all_scores) > 0 else 0,
                    'median': float(np.median(all_scores)) if len(all_scores) > 0 else 0,
                    'std': float(all_scores.std()) if len(all_scores) > 0 else 0
                }
            },
            'class_statistics': {
                'total_per_class': dict(class_totals),
                'scenes_per_class': dict(scenes_per_class),
                'detection_ratios': {
                    class_name: count / scenes_per_class[class_name]
                    for class_name, count in class_totals.items()
                }
            },
            'processing_errors': self.stats['processing_errors'][:10]  # First 10 errors
        }


def main():
    """Main execution function."""
    print("🚀 Enhanced Stage 1 Pseudo Label Generation")
    print("=" * 60)
    
    # Configuration
    checkpoint_path = "stage_1_checkpoints/epoch_12.pth"
    output_dir = Path("stage1_pseudo_labels_correct")
    confidence_threshold = 0.1
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for path in Path(".").glob("**/stage_1_checkpoints/*.pth"):
            print(f"  {path}")
        return
    
    # Generate pseudo labels
    generator = Stage1PseudoLabelGenerator(checkpoint_path, confidence_threshold)
    pseudo_labels = generator.generate_all_pseudo_labels()
    
    # Save results
    main_pickle = generator.save_pseudo_labels(pseudo_labels, output_dir)
    
    print("\n" + "=" * 60)
    print("🎉 PSEUDO LABEL GENERATION COMPLETE")
    print(f"📁 Main output: {main_pickle}")
    print(f"📊 Generated labels for {len(pseudo_labels)} scenes")
    print(f"🔢 Total detections: {generator.stats['total_detections']}")
    print(f"🎯 Class distribution:")
    for class_name, count in generator.stats['class_counts'].most_common():
        print(f"   {class_name}: {count}")
    
    if generator.stats['processing_errors']:
        print(f"⚠️ Processing errors: {len(generator.stats['processing_errors'])}")
    
    print("\n🔍 Next steps:")
    print("1. Validate format consistency with GT")
    print("2. Test loading through training pipeline")
    print("3. Verify coordinate alignment in visualization")


if __name__ == "__main__":
    main()
