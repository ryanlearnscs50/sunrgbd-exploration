#!/usr/bin/env python3
"""
Create Empty Pseudo Labels as Workaround

This script creates empty pseudo label files for stages 4-5 to prevent
experiments from hanging while trying to generate them on-the-fly.

This is a temporary workaround for the issue where Stage 3/4 checkpoints
are not producing any detections for pseudo label generation.

Date: 2025-09-08
"""

import pickle
from pathlib import Path
import json

project_root = Path(__file__).resolve().parents[2]

def create_empty_pseudo_labels(stage_id, output_dir):
    """Create an empty but valid pseudo label file."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create empty pseudo labels structure
    pseudo_labels = {}
    
    # Save to pickle file
    output_file = output_dir / f"stage_{stage_id}_empty_pseudo_labels.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(pseudo_labels, f)
    
    # Create metadata
    metadata = {
        "stage_id": stage_id,
        "num_scenes": 0,
        "num_detections": 0,
        "note": "Empty pseudo labels as workaround for generation issues"
    }
    
    metadata_file = output_dir / f"stage_{stage_id}_pseudo_labels_stats.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Created empty pseudo labels for Stage {stage_id}")
    print(f"   File: {output_file}")
    print(f"   Metadata: {metadata_file}")
    
    return str(output_file)


def main():
    print("🔧 CREATING EMPTY PSEUDO LABELS AS WORKAROUND")
    print("=" * 60)
    print("This will allow experiments to continue past Stage 3")
    print("even though pseudo label generation is failing.")
    print()
    
    # Create global pseudo labels directory
    global_dir = project_root / "pseudo_labels"
    
    # Create empty pseudo labels for stages 4 and 5
    for stage_id in [4, 5]:
        create_empty_pseudo_labels(stage_id, global_dir)
    
    print()
    print("📝 IMPORTANT: These are empty pseudo label files!")
    print("   They will allow experiments to continue but")
    print("   pseudo label training will have no effect.")
    print()
    print("🎯 Next steps:")
    print("   1. Copy these files to experiment directories")
    print("   2. Restart failed experiments")
    print("   3. Monitor to ensure they progress past Stage 3")


if __name__ == "__main__":
    main()
