#!/usr/bin/env python3
"""Generate pseudo-labels using stage 1 model for stage 2 training scenes."""
import argparse
import os
import sys
from pathlib import Path

import torch
import pickle
import numpy as np

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mmcv import Config
from mmdet3d.apis import inference_detector
from mmdet3d.models.builder import build_detector


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Generate ScanNet pseudo labels for Stage 2 scenes",
    )
    parser.add_argument(
        "--base-cfg",
        default=str(project_root / "configs/tr3d/tr3d_scannet-3d-35class.py"),
        help="Base config used to build the detector for inference",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint path to load (typically the previous-stage model)",
    )
    parser.add_argument(
        "--train-data-file",
        default="data/scannet/scannet_infos_train_40class_corrected.pkl",
        help="Training infos .pkl used to enumerate scenes",
    )
    parser.add_argument(
        "--output-dir",
        default="pseudo_labels/stage_2",
        help="Output directory for generated pseudo labels",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for keeping pseudo labels",
    )
    parser.add_argument(
        "--target-model-indices",
        type=int,
        nargs="+",
        default=[0, 2, 4, 5, 6, 19, 34],
        help="Model label indices to keep (label-space depends on checkpoint/model)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of scenes (debug)",
    )
    args = parser.parse_args()

    base_cfg_path = _resolve_path(args.base_cfg)
    checkpoint_path = _resolve_path(args.checkpoint)
    train_data_file = _resolve_path(args.train_data_file)
    output_dir = _resolve_path(args.output_dir)
    
    print("🏷️ Generating Pseudo-Labels for Stage 2")
    print(f"   Base cfg: {base_cfg_path}")
    print(f"   Model: {checkpoint_path}")
    print(f"   Output: {output_dir}")
    
    # Load config and build model using the same approach as working training
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
    
    # Load training scenes
    with open(train_data_file, 'rb') as f:
        data_infos = pickle.load(f)
    
    print(f"📁 Loaded {len(data_infos)} training scenes")
    
    # Stage 1 classes for pseudo-labeling (these are 35-class model indices)
    stage1_model_indices = args.target_model_indices
    confidence_threshold = args.confidence_threshold
    
    # Import 35-class mapping to convert model indices to NYU40 IDs
    sys.path.append(str(project_root / 'configs/_base_/class_mappings'))
    try:
        from scannet_35class_mapping import MODEL_IDX_TO_NYU40_35CLASS
    except ImportError:
        print("❌ Cannot import 35-class mappings")
        return False
    
    pseudo_labels = {}
    processed = 0
    
    print(f"🎯 Processing scenes (target model indices: {stage1_model_indices})...")
    
    with torch.no_grad():
        for i, data_info in enumerate(data_infos):
            if args.limit is not None and i >= args.limit:
                break
            if i % 200 == 0:
                print(f"   Progress: {i}/{len(data_infos)}")
            
            scene_id = data_info['point_cloud']['lidar_idx']
            points_file = project_root / 'data/scannet/points' / f'{scene_id}.bin'
            
            if not points_file.exists():
                continue
            
            try:
                # Run inference
                results = inference_detector(model, str(points_file))
                
                # Extract predictions
                if isinstance(results, tuple) and len(results) > 0:
                    predictions_list = results[0]
                    if isinstance(predictions_list, list) and len(predictions_list) > 0:
                        predictions = predictions_list[0]
                        
                        if 'scores_3d' in predictions:
                            scores = predictions['scores_3d'].cpu().numpy()
                            labels_model = predictions['labels_3d'].cpu().numpy()  # Model indices
                            boxes_3d = predictions['boxes_3d']
                            
                            if hasattr(boxes_3d, 'tensor'):
                                boxes = boxes_3d.tensor.cpu().numpy()
                            else:
                                boxes = boxes_3d.cpu().numpy()
                            
                            # Filter for stage 1 classes with confidence
                            class_mask = np.isin(labels_model, stage1_model_indices)
                            conf_mask = scores >= confidence_threshold
                            final_mask = class_mask & conf_mask
                            
                            if final_mask.any():
                                # CRITICAL: Convert model indices to NYU40 IDs before storage
                                filtered_model_labels = labels_model[final_mask]
                                filtered_nyu40_labels = np.array([
                                    MODEL_IDX_TO_NYU40_35CLASS[model_idx] 
                                    for model_idx in filtered_model_labels
                                ])
                                
                                pseudo_labels[scene_id] = {
                                    'boxes': boxes[final_mask],
                                    'labels': filtered_nyu40_labels,  # Store as NYU40 IDs
                                    'scores': scores[final_mask],
                                    'count': final_mask.sum()
                                }
                                processed += 1
                                
            except Exception as e:
                print(f"     Failed for {scene_id}: {e}")
                continue
    
    print(f"✅ Generated pseudo labels for {processed} scenes")
    
    if pseudo_labels:
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw pseudo labels
        output_file = output_dir / 'stage_2_pseudo_labels.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump(pseudo_labels, f)
        
        # Create ScanNet format
        pseudo_infos = []
        for scene_id, data in pseudo_labels.items():
            info = {
                'scene_id': scene_id,
                'point_cloud': {'lidar_idx': scene_id},
                'annos': {
                    'gt_num': data['count'],
                    'name': [f'pseudo_class_{label}' for label in data['labels']],
                    'class': data['labels'].astype(np.int64),  # Now NYU40 IDs
                    'gt_boxes_upright_depth': data['boxes'].astype(np.float32)
                },
                'pseudo_source': 'stage_1_model',
                'confidence_scores': data['scores']
            }
            pseudo_infos.append(info)
        
        scannet_file = output_dir / 'pseudo_labels_scannet_format.pkl'
        with open(scannet_file, 'wb') as f:
            pickle.dump(pseudo_infos, f)
        
        print(f"💾 Saved to: {output_file}")
        print(f"💾 ScanNet format: {scannet_file}")
        
        # Statistics
        total_objects = sum(data['count'] for data in pseudo_labels.values())
        avg_conf = np.mean([np.mean(data['scores']) for data in pseudo_labels.values()])
        
        print(f"📊 Statistics:")
        print(f"   Total pseudo objects: {total_objects}")
        print(f"   Average confidence: {avg_conf:.3f}")
        
        return True
    else:
        print("❌ No pseudo labels generated")
        return False

if __name__ == '__main__':
    success = main()
    print("=" * 50)
    if success:
        print("✅ Pseudo-label generation COMPLETED")
    else:
        print("❌ Pseudo-label generation FAILED")
