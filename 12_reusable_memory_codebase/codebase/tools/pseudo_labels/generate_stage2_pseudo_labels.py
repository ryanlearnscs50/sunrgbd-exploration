#!/usr/bin/env python3
"""
Generate Stage 2 Pseudo Labels from Stage 1 Checkpoint

This script generates pseudo labels for Stage 2 training using the Stage 1 checkpoint.
The pseudo labels are stored with NYU40 IDs for compatibility across incremental stages.

Usage:
    python generate_stage2_pseudo_labels.py

The script will:
1. Load Stage 1 checkpoint (stage_1_checkpoints/epoch_12.pth)
2. Generate predictions for Stage 2 training scenes
3. Filter to Stage 1 classes (0-6) with confidence threshold
4. Convert to NYU40 IDs for storage
5. Save to stage2_pseudo_labels/stage2_pseudo_labels.pkl
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mmdet3d.apis import init_model, inference_detector
from mmdet3d.datasets.incremental_mappings import create_mapping_from_config
from mmdet3d.datasets import build_dataset
from mmcv import Config


def generate_stage2_pseudo_labels():
    """Generate Stage 2 pseudo labels from Stage 1 checkpoint."""
    
    print("🏷️  Generating Stage 2 Pseudo Labels from Stage 1 Checkpoint")
    print("=" * 60)
    
    # Configuration
    stage1_checkpoint = project_root / "stage_1_checkpoints/epoch_12.pth"
    stage2_config_path = project_root / "configs/incremental/scannet/tr3d_dynamic_head_s5_pure_finetuning.py"
    output_dir = project_root / "stage2_pseudo_labels"
    output_file = output_dir / "stage2_pseudo_labels.pkl"
    
    # Validation
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_checkpoint}")
    
    if not stage2_config_path.exists():
        raise FileNotFoundError(f"Stage 2 config not found: {stage2_config_path}")
    
    print(f"📦 Stage 1 checkpoint: {stage1_checkpoint}")
    print(f"⚙️  Stage 2 config: {stage2_config_path}")
    print(f"💾 Output file: {output_file}")
    
    # Load config to get Stage 2 dataset configuration
    cfg = Config.fromfile(str(stage2_config_path))
    
    # Extract incremental learning configuration
    stage_definitions = cfg.get('stage_definitions')
    if not stage_definitions:
        raise ValueError("No stage_definitions found in config")
    
    # Get Stage 1 and Stage 2 definitions
    stage1_def = stage_definitions[0]  # Stage 1
    stage2_def = stage_definitions[1]  # Stage 2
    
    stage1_classes = stage1_def['class_indices']  # [0, 1, 2, 3, 4, 5, 6] - NEW classes for Stage 1
    
    # Get cumulative classes seen up to Stage 2
    from mmdet3d.datasets.incremental_mappings import get_all_seen_classes_up_to_stage
    stage2_cumulative = get_all_seen_classes_up_to_stage(stage_definitions, 2)  # [0, 1, ..., 13] - ALL classes by Stage 2
    
    print(f"🎯 Stage 1 classes: {stage1_classes} ({len(stage1_classes)} classes)")
    print(f"🎯 Stage 2 cumulative: {stage2_cumulative} ({len(stage2_cumulative)} classes)")
    
    # Create mappings for conversion - call directly with stage_definitions
    mappings = create_mapping_from_config(stage_definitions)
    model_to_nyu40 = mappings['model_idx_to_nyu40']
    nyu40_to_model = mappings['nyu40_to_model_idx']
    
    print(f"📍 Loaded mappings: {len(model_to_nyu40)} model->NYU40, {len(nyu40_to_model)} NYU40->model")
    
    # Build Stage 2 dataset to get the scenes we need pseudo labels for
    # Temporarily override stage to Stage 2 to get Stage 2 training scenes
    stage2_cfg = cfg.copy()
    stage2_cfg.data.train.stage_definition = stage2_def
    stage2_cfg.data.train.all_stage_definitions = stage_definitions
    
    # Build dataset
    stage2_dataset = build_dataset(stage2_cfg.data.train)
    
    print(f"📊 Stage 2 training dataset: {len(stage2_dataset)} scenes")
    
    # Load Stage 1 model
    print(f"🚀 Loading Stage 1 model...")
    
    # Create a temporary config for Stage 1 model loading
    stage1_cfg = cfg.copy()
    stage1_cfg.model.bbox_head.num_classes = len(stage1_classes)  # 7 classes for Stage 1
    
    # Initialize model
    model = init_model(stage1_cfg, str(stage1_checkpoint), device='cuda:0')
    model.eval()
    
    print(f"✅ Stage 1 model loaded successfully")
    
    # Generate pseudo labels
    pseudo_labels = {}
    confidence_threshold = 0.45  # Use reasonable threshold
    processed_scenes = 0
    total_detections = 0
    
    print(f"🔍 Generating pseudo labels with confidence >= {confidence_threshold}")
    print("   Processing scenes...")
    
    # Get innermost dataset to access data_infos
    dataset = stage2_dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    
    for i, data_info in enumerate(dataset.data_infos):
        if i % 100 == 0 and i > 0:
            print(f"   Progress: {i}/{len(dataset.data_infos)} scenes ({processed_scenes} with labels)")
        
        scene_id = data_info['point_cloud']['lidar_idx']
        
        # Get point cloud file path
        points_file = os.path.join(dataset.data_root, 'points', f'{scene_id}.bin')
        
        if not os.path.exists(points_file):
            continue
        
        # Run inference
        try:
            with torch.no_grad():
                results = inference_detector(model, points_file)
                
                # Extract predictions
                if isinstance(results, tuple) and len(results) > 0:
                    predictions_list = results[0]
                    if isinstance(predictions_list, list) and len(predictions_list) > 0:
                        predictions = predictions_list[0]
                        
                        # Filter predictions by Stage 1 classes and confidence
                        filtered = filter_stage1_predictions(
                            predictions, 
                            stage1_classes, 
                            confidence_threshold,
                            model_to_nyu40
                        )
                        
                        if filtered is not None and len(filtered['labels']) > 0:
                            pseudo_labels[scene_id] = filtered
                            processed_scenes += 1
                            total_detections += len(filtered['labels'])
                            
        except Exception as e:
            print(f"   ⚠️  Inference failed for scene {scene_id}: {e}")
            continue
    
    print(f"✅ Generated pseudo labels for {processed_scenes}/{len(dataset.data_infos)} scenes")
    print(f"📈 Total detections: {total_detections}")
    
    # Save pseudo labels
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'wb') as f:
        pickle.dump(pseudo_labels, f)
    
    print(f"💾 Saved pseudo labels to: {output_file}")
    
    # Statistics
    if pseudo_labels:
        scene_counts = [len(labels['labels']) for labels in pseudo_labels.values()]
        avg_detections = np.mean(scene_counts)
        print(f"📊 Statistics:")
        print(f"   Scenes with labels: {len(pseudo_labels)}")
        print(f"   Average detections per scene: {avg_detections:.1f}")
        print(f"   Min/Max detections per scene: {min(scene_counts)}/{max(scene_counts)}")
        
        # Count by NYU40 ID
        all_labels = np.concatenate([labels['labels'] for labels in pseudo_labels.values()])
        unique, counts = np.unique(all_labels, return_counts=True)
        print(f"   Detections by NYU40 ID: {dict(zip(unique, counts))}")
    
    print("🎉 Stage 2 pseudo label generation completed!")
    return output_file


def filter_stage1_predictions(predictions, stage1_classes, confidence_threshold, model_to_nyu40):
    """Filter predictions to Stage 1 classes and convert to NYU40 IDs."""
    
    if not predictions or 'scores_3d' not in predictions:
        return None
    
    # Get predictions
    scores = predictions['scores_3d'].cpu().numpy()
    labels = predictions['labels_3d'].cpu().numpy()  # Model class indices
    
    # Handle bounding boxes
    boxes_3d = predictions['boxes_3d']
    if hasattr(boxes_3d, 'tensor'):
        boxes = boxes_3d.tensor.cpu().numpy()
    elif hasattr(boxes_3d, 'cpu'):
        boxes = boxes_3d.cpu().numpy()
    else:
        boxes = np.array(boxes_3d)
    
    # Filter by Stage 1 classes and confidence
    class_mask = np.isin(labels, stage1_classes)
    conf_mask = scores >= confidence_threshold
    final_mask = class_mask & conf_mask
    
    if not final_mask.any():
        return None
    
    # Apply filters
    filtered_boxes = boxes[final_mask]
    filtered_labels = labels[final_mask]
    filtered_scores = scores[final_mask]
    
    # Convert model indices to NYU40 IDs
    nyu40_labels = np.array([model_to_nyu40[idx] for idx in filtered_labels])
    
    return {
        'boxes': filtered_boxes.astype(np.float32),
        'labels': nyu40_labels.astype(np.int64),  # NYU40 IDs
        'scores': filtered_scores.astype(np.float32),
        'num_detections': len(nyu40_labels),
        'metadata': {
            'stage_generated': 1,
            'model_classes': stage1_classes,
            'confidence_threshold': confidence_threshold,
            'timestamp': str(np.datetime64('now'))
        }
    }


if __name__ == '__main__':
    try:
        output_file = generate_stage2_pseudo_labels()
        print(f"\n✅ Success! Pseudo labels ready for Stage 3 training.")
        print(f"📁 File: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
