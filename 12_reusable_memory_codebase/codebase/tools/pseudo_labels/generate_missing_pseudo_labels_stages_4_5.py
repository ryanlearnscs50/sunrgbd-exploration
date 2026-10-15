#!/usr/bin/env python3
"""
Generate Missing Pseudo Labels for Stages 4-5

This script generates pseudo labels for stages 4-5 using completed Stage 3 checkpoints
from the memory experiments to fix the pseudo experiment failures.

Date: 2025-09-08
"""

import os
import sys
from pathlib import Path
import json
import argparse
from datetime import datetime

project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from mmdet3d.utils.pregenerate_pseudo_labels import pregenerate_pseudo_labels_for_stage


def _resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def main():
    parser = argparse.ArgumentParser(
        description="Generate missing ScanNet pseudo labels for Stages 4-5",
    )
    parser.add_argument(
        "--stage3-checkpoint",
        required=True,
        help="Stage 3 checkpoint (used to generate Stage 4 pseudo labels)",
    )
    parser.add_argument(
        "--stage4-checkpoint",
        required=True,
        help="Stage 4 checkpoint (used to generate Stage 5 pseudo labels)",
    )
    parser.add_argument(
        "--train-data-file",
        default="data/scannet/scannet_infos_train_40class_corrected.pkl",
        help="ScanNet training infos .pkl",
    )
    parser.add_argument(
        "--output-dir",
        default="pseudo_labels",
        help="Output directory for generated pseudo labels",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.05,
        help="Confidence threshold for pseudo labels (filtering can happen later)",
    )
    parser.add_argument(
        "--metadata-name",
        default="stages_4_5_metadata.json",
        help="Metadata JSON filename written under output-dir",
    )
    args = parser.parse_args()

    print("🎯 GENERATING MISSING PSEUDO LABELS FOR STAGES 4-5")
    print("=" * 60)
    
    stage_3_checkpoint = _resolve_path(args.stage3_checkpoint)
    stage_4_checkpoint = _resolve_path(args.stage4_checkpoint)
    
    if not stage_3_checkpoint.exists():
        raise FileNotFoundError(f"Stage 3 checkpoint not found: {stage_3_checkpoint}")
    
    if not stage_4_checkpoint.exists():
        raise FileNotFoundError(f"Stage 4 checkpoint not found: {stage_4_checkpoint}")
    
    print(f"✅ Using Stage 3 checkpoint: {stage_3_checkpoint}")
    print(f"✅ Using Stage 4 checkpoint: {stage_4_checkpoint}")
    
    # Data file
    train_data_file = str(_resolve_path(args.train_data_file))
    
    # Output directory for global pseudo labels
    output_dir = _resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Generate Stage 4 pseudo labels using Stage 3 checkpoint
        print(f"\n📝 Generating Stage 4 pseudo labels...")
        stage_4_output = pregenerate_pseudo_labels_for_stage(
            stage_id=3,  # Using Stage 3 model to generate for Stage 4
            checkpoint_path=str(stage_3_checkpoint),
            train_data_file=train_data_file,
            memory_bank_file=None,  # No memory bank for global generation
            confidence_threshold=args.confidence_threshold,
            output_dir=str(output_dir),
            config_suffix="global_stage4"
        )
        print(f"✅ Stage 4 pseudo labels generated: {stage_4_output}")
        
        # Generate Stage 5 pseudo labels using Stage 4 checkpoint
        print(f"\n📝 Generating Stage 5 pseudo labels...")
        stage_5_output = pregenerate_pseudo_labels_for_stage(
            stage_id=4,  # Using Stage 4 model to generate for Stage 5
            checkpoint_path=str(stage_4_checkpoint),
            train_data_file=train_data_file,
            memory_bank_file=None,  # No memory bank for global generation
            confidence_threshold=args.confidence_threshold,
            output_dir=str(output_dir),
            config_suffix="global_stage5"
        )
        print(f"✅ Stage 5 pseudo labels generated: {stage_5_output}")
        
        # Create metadata file
        metadata = {
            "generation_date": datetime.now().strftime("%Y-%m-%d"),
            "stage_3_checkpoint": str(stage_3_checkpoint),
            "stage_4_checkpoint": str(stage_4_checkpoint),
            "stage_4_labels": stage_4_output,
            "stage_5_labels": stage_5_output,
            "total_stages_generated": 2,
            "notes": "Generated to fix pseudo experiment failures at Stage 3->4 transition"
        }
        
        metadata_file = output_dir / args.metadata_name
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🎉 SUCCESS! Generated pseudo labels for stages 4-5")
        print(f"📁 Output directory: {output_dir}")
        print(f"📋 Metadata: {metadata_file}")
        print(f"\n📊 Generated files:")
        print(f"   - {Path(stage_4_output).name}")
        print(f"   - {Path(stage_5_output).name}")
        
    except Exception as e:
        print(f"❌ ERROR generating pseudo labels: {e}")
        raise


if __name__ == "__main__":
    main()
