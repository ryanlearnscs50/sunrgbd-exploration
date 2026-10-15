#!/usr/bin/env python3
"""
Multi-stage pseudo-label generation script for incremental learning.

This script generates pseudo-labels for any stage using the previous stage's model.
It automatically determines the correct target classes and file paths.
"""
import os
import sys
import torch
import pickle
import numpy as np
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mmcv import Config
from mmdet3d.apis import inference_detector
from mmdet3d.models.builder import build_detector
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type
from mmcv.parallel import collate, scatter
from copy import deepcopy


def get_stage_config(stage_id, config_file=None):
    """Get stage configuration including previous stage classes.
    
    Args:
        stage_id (int): Current stage ID  
        config_file (str): Path to config file. If None, uses default 7x5 config.
    """
    if config_file is None:
        config_file = str(
            project_root
            / 'configs/incremental/scannet/tr3d_dynamic_head_s5_pure_finetuning.py'
        )
    
    # Load stage definitions from config file
    stage_definitions = None
    if config_file and os.path.exists(config_file):
        try:
            # Load config and extract stage definitions
            cfg = Config.fromfile(config_file)
            if hasattr(cfg, 'stage_definitions'):
                stage_definitions = cfg.stage_definitions
            else:
                raise AttributeError("Config file does not contain 'stage_definitions'")
        except Exception as e:
            print(f"⚠️ Failed to load config {config_file}: {e}")
            print("Using fallback stage definitions (frequency-based split)")

    if stage_definitions is None:
        stage_definitions = [
            {
                'stage_id': 1,
                'stage_name': 'Stage 1 - Most Frequent Classes',
                'class_indices': [0, 2, 4, 5, 6, 19, 34]
            },
            {
                'stage_id': 2,
                'stage_name': 'Stage 2 - High Frequency Classes',
                'class_indices': [1, 3, 7, 10, 11, 14, 28]
            },
            {
                'stage_id': 3,
                'stage_name': 'Stage 3 - Medium Frequency Classes',
                'class_indices': [8, 9, 13, 15, 16, 17, 18]
            },
            {
                'stage_id': 4,
                'stage_name': 'Stage 4 - Lower Frequency Classes',
                'class_indices': [12, 20, 21, 22, 23, 26, 27]
            },
            {
                'stage_id': 5,
                'stage_name': 'Stage 5 - Least Frequent Classes',
                'class_indices': [24, 25, 29, 30, 31, 32, 33]
            }
        ]
    
    if stage_id < 2 or stage_id > 5:
        raise ValueError(f"Invalid stage_id {stage_id}. Must be 2-5 for pseudo-labeling.")
    
    # Get all previous stage classes (cumulative)
    previous_classes = []
    for stage in stage_definitions[:stage_id-1]:  # All stages before current
        previous_classes.extend(stage['class_indices'])
    
    current_stage = stage_definitions[stage_id-1]
    
    return {
        'stage_id': stage_id,
        'stage_name': current_stage['stage_name'],
        'previous_classes': sorted(previous_classes),
        'num_previous_classes': len(previous_classes)
    }


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    print(f"🔧 Loading model from: {checkpoint_path}")
    
    # Load config and build model
    base_cfg_path = project_root / 'configs/tr3d/tr3d_scannet-3d-35class.py'
    cfg = Config.fromfile(str(base_cfg_path))
    
    # Build model
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Add cfg attribute to model (required for inference)
    model.cfg = cfg
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = model.cuda()
    model.eval()
    
    print("✅ Model loaded successfully")
    return model


def inference_detector_with_alignment(model, pcd_file, axis_align_matrix):
    """Run inference with proper axis alignment matrix for consistent coordinates.
    
    This ensures pseudo-labels are generated in the same coordinate system as ground truth.
    """
    cfg = model.cfg
    device = next(model.parameters()).device

    # Build the data pipeline
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)

    # Create data dict with ACTUAL axis_align_matrix (not identity!)
    data = dict(
        pts_filename=pcd_file,
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        # CRITICAL: Use actual axis_align_matrix instead of identity
        ann_info=dict(axis_align_matrix=axis_align_matrix),
        sweeps=[],
        timestamp=[0],
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[]
    )
    
    # Process through pipeline (including GlobalAlignment with real matrix)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device.index])[0]
    else:
        data['img_metas'] = data['img_metas'][0].data
        data['points'] = data['points'][0].data
    
    # Forward through model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    
    return result


def filter_predictions_by_classes(predictions, target_classes, confidence_threshold=0.3):
    """Filter model predictions to only include target classes above confidence threshold."""
    if not predictions or 'scores_3d' not in predictions:
        return None
    
    # Extract predictions
    scores = predictions['scores_3d'].cpu().numpy()
    labels = predictions['labels_3d'].cpu().numpy()
    
    # Handle different bounding box formats
    boxes_3d = predictions['boxes_3d']
    if hasattr(boxes_3d, 'tensor'):
        boxes = boxes_3d.tensor.cpu().numpy()
    elif hasattr(boxes_3d, 'cpu'):
        boxes = boxes_3d.cpu().numpy()
    else:
        boxes = np.array(boxes_3d)
    
    # Filter by target classes and confidence
    class_mask = np.isin(labels, target_classes)
    conf_mask = scores >= confidence_threshold
    final_mask = class_mask & conf_mask
    
    if not final_mask.any():
        return None
    
    return {
        'boxes': boxes[final_mask],
        'labels': labels[final_mask],
        'scores': scores[final_mask],
        'count': final_mask.sum()
    }


def generate_pseudo_labels(model, target_classes, confidence_threshold=0.3):
    """Generate pseudo labels for all training scenes using ALIGNED coordinates."""
    print(f"🏷️ Generating pseudo labels in ALIGNED coordinates...")
    print(f"   Target classes: {target_classes} ({len(target_classes)} classes)")
    print(f"   Confidence threshold: {confidence_threshold}")
    print(f"   🔧 Using axis_align_matrix for coordinate consistency with ground truth")
    
    # Load training scenes
    data_file = 'data/scannet/scannet_infos_train_40class_corrected.pkl'
    with open(data_file, 'rb') as f:
        data_infos = pickle.load(f)
    
    print(f"   Total scenes to process: {len(data_infos)}")
    
    pseudo_labels = {}
    processed = 0
    alignment_errors = 0
    
    with torch.no_grad():
        for i, data_info in enumerate(data_infos):
            if i % 200 == 0:
                print(f"   Progress: {i}/{len(data_infos)}")
            
            scene_id = data_info['point_cloud']['lidar_idx']
            points_file = f'data/scannet/points/{scene_id}.bin'
            
            if not os.path.exists(points_file):
                continue
            
            # Get axis_align_matrix for this scene
            axis_align_matrix = None
            if 'annos' in data_info and 'axis_align_matrix' in data_info['annos']:
                axis_align_matrix = np.array(data_info['annos']['axis_align_matrix'])
            else:
                alignment_errors += 1
                # Fall back to identity matrix if no alignment matrix available
                axis_align_matrix = np.eye(4)
            
            try:
                # Run inference with proper axis alignment
                results = inference_detector_with_alignment(model, points_file, axis_align_matrix)
                
                # Extract predictions from aligned inference
                if isinstance(results, list) and len(results) > 0:
                    predictions = results[0]  # Direct access to first result
                    
                    # Filter predictions
                    filtered = filter_predictions_by_classes(
                        predictions, target_classes, confidence_threshold
                    )
                    
                    if filtered is not None:
                        pseudo_labels[scene_id] = filtered
                        processed += 1
                            
            except Exception as e:
                print(f"     Failed for {scene_id}: {e}")
                continue
    
    if alignment_errors > 0:
        print(f"   ⚠️ Warning: {alignment_errors} scenes missing axis_align_matrix")
    
    print(f"   ✅ Generated pseudo labels for {processed}/{len(data_infos)} scenes")
    print(f"   📐 All pseudo labels are now in ALIGNED coordinates (same as ground truth)")
    return pseudo_labels


def save_pseudo_labels(pseudo_labels, output_dir, stage_id):
    """Save pseudo labels in both raw and ScanNet formats."""
    print(f"💾 Saving pseudo labels to: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw pseudo labels
    raw_file = os.path.join(output_dir, f'stage_{stage_id}_pseudo_labels.pkl')
    with open(raw_file, 'wb') as f:
        pickle.dump(pseudo_labels, f)
    
    # Create ScanNet-style annotation format
    pseudo_infos = []
    for scene_id, data in pseudo_labels.items():
        info = {
            'scene_id': scene_id,
            'point_cloud': {'lidar_idx': scene_id},
            'annos': {
                'gt_num': data['count'],
                'name': [f'pseudo_class_{label}' for label in data['labels']],
                'class': data['labels'].astype(np.int64),
                'gt_boxes_upright_depth': data['boxes'].astype(np.float32)
            },
            'pseudo_source': f'stage_{stage_id-1}_model',
            'confidence_scores': data['scores']
        }
        pseudo_infos.append(info)
    
    # Save ScanNet-style format
    scannet_file = os.path.join(output_dir, 'pseudo_labels_scannet_format.pkl')
    with open(scannet_file, 'wb') as f:
        pickle.dump(pseudo_infos, f)
    
    # Print statistics
    total_objects = sum(data['count'] for data in pseudo_labels.values())
    avg_conf = np.mean([np.mean(data['scores']) for data in pseudo_labels.values()])
    
    print(f"📊 Statistics:")
    print(f"   Scenes with pseudo labels: {len(pseudo_labels)}")
    print(f"   Total pseudo objects: {total_objects}")
    print(f"   Average confidence: {avg_conf:.3f}")
    
    # Class distribution
    all_labels = np.concatenate([data['labels'] for data in pseudo_labels.values()])
    unique_labels, counts = np.unique(all_labels, return_counts=True)
    print(f"   Class distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"     Class {label}: {count} objects")
    
    print(f"💾 Files saved:")
    print(f"   Raw format: {raw_file}")
    print(f"   ScanNet format: {scannet_file}")
    
    return raw_file, scannet_file


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo labels for incremental learning')
    parser.add_argument('--stage', type=int, required=True, choices=[2, 3, 4, 5], 
                       help='Stage number (2-5) to generate pseudo labels for')
    parser.add_argument('--base-dir', default='.',
                       help='Base experiment directory')
    parser.add_argument('--confidence-threshold', type=float, default=0.3, 
                       help='Confidence threshold for pseudo labels')
    parser.add_argument('--config-file', type=str, 
                       default=str(project_root / 'configs/incremental/scannet/tr3d_dynamic_head_s5_pure_finetuning.py'),
                       help='Path to incremental config file containing stage definitions')
    
    args = parser.parse_args()
    
    # Get stage configuration from config file
    stage_config = get_stage_config(args.stage, args.config_file)
    
    print("🔬 Multi-Stage Pseudo-Label Generation")
    print("=" * 50)
    print(f"📋 Configuration:")
    print(f"   Stage: {stage_config['stage_id']} ({stage_config['stage_name']})")
    print(f"   Previous classes: {stage_config['previous_classes']}")
    print(f"   Number of target classes: {stage_config['num_previous_classes']}")
    print(f"   Confidence threshold: {args.confidence_threshold}")
    print(f"   Base directory: {args.base_dir}")
    
    # Paths
    previous_stage = args.stage - 1
    checkpoint_path = os.path.join(args.base_dir, f'stage_{previous_stage}', 'latest.pth')
    output_dir = os.path.join(args.base_dir, f'stage_{args.stage}', 'pseudo_labels')
    
    print(f"📁 Model checkpoint: {checkpoint_path}")
    print(f"📁 Output directory: {output_dir}")
    
    # Validate paths
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return 1
    
    try:
        # Load model
        model = load_model(checkpoint_path)
        
        # Generate pseudo labels
        pseudo_labels = generate_pseudo_labels(
            model, stage_config['previous_classes'], args.confidence_threshold
        )
        
        if pseudo_labels:
            # Save pseudo labels
            raw_file, scannet_file = save_pseudo_labels(pseudo_labels, output_dir, args.stage)
            print(f"✅ Successfully generated pseudo labels for Stage {args.stage}")
            return 0
        else:
            print("❌ No pseudo labels generated")
            return 1
            
    except Exception as e:
        print(f"❌ Pseudo-label generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
