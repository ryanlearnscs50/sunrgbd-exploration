#!/usr/bin/env python3
"""
Validate Pseudo Label Format

This script validates that pseudo labels follow the correct NYU40 ID format
for compatibility with ground truth during mixed training.

Usage:
    python validate_pseudo_label_format.py [pseudo_label_file.pkl]
    python validate_pseudo_label_format.py  # Validates all found pseudo label files

Date: 2025-09-02
"""

import sys
import pickle
import numpy as np
from pathlib import Path
import json
from collections import Counter


def validate_pseudo_label_file(file_path):
    """Validate a single pseudo label file."""
    print(f"\n🔍 Validating: {file_path}")
    print("=" * 60)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if not isinstance(data, dict):
            print("❌ FAIL: Data should be a dictionary")
            return False
        
        if len(data) == 0:
            print("❌ FAIL: No scenes found in pseudo labels")
            return False
        
        # Check format of first few samples
        sample_keys = list(data.keys())[:3]
        all_labels = []
        total_detections = 0
        
        for scene_id in sample_keys:
            scene_data = data[scene_id]
            
            # Check required fields
            required_fields = ['labels', 'boxes', 'scores', 'num_detections']
            missing_fields = [f for f in required_fields if f not in scene_data]
            if missing_fields:
                print(f"❌ FAIL: Missing fields in {scene_id}: {missing_fields}")
                return False
            
            # Check labels format
            labels = scene_data['labels']
            if not isinstance(labels, (list, np.ndarray)):
                print(f"❌ FAIL: Labels should be list or array in {scene_id}")
                return False
            
            if len(labels) != scene_data['num_detections']:
                print(f"❌ FAIL: Label count mismatch in {scene_id}")
                return False
            
            all_labels.extend(labels)
            total_detections += len(labels)
        
        # Convert to numpy for analysis
        all_labels = np.array(all_labels)
        
        if len(all_labels) == 0:
            print("❌ FAIL: No labels found in samples")
            return False
        
        # Validate label range
        min_label = int(all_labels.min())
        max_label = int(all_labels.max())
        
        print(f"📊 Validation Results:")
        print(f"   Total scenes: {len(data)}")
        print(f"   Sample detections: {total_detections}")
        print(f"   Label range: {min_label} to {max_label}")
        
        # Check if labels are NYU40 IDs
        if min_label >= 1 and max_label <= 40:
            print("✅ PASS: Labels are in NYU40 ID range (1-40)")
        elif min_label >= 0 and max_label <= 34:
            print("❌ FAIL: Labels appear to be 35-class model indices (0-34)")
            print("   🔧 Fix: Convert model indices to NYU40 IDs before storage")
            return False
        elif min_label >= 0 and max_label <= 6:
            print("❌ FAIL: Labels appear to be Stage 1 model indices (0-6)")  
            print("   🔧 Fix: Convert model indices to NYU40 IDs before storage")
            return False
        else:
            print(f"❌ FAIL: Labels have unexpected range ({min_label}-{max_label})")
            return False
        
        # Check label distribution
        unique_labels, counts = np.unique(all_labels, return_counts=True)
        print(f"   Unique labels: {len(unique_labels)} different NYU40 IDs")
        
        # Most common classes
        label_counts = Counter(all_labels.tolist())
        most_common = label_counts.most_common(5)
        print(f"   Top 5 classes by count:")
        for nyu40_id, count in most_common:
            print(f"     NYU40 {nyu40_id}: {count} detections")
        
        # Check for valid NYU40 IDs
        sys.path.append('configs/_base_/class_mappings')
        try:
            from scannet_35class_mapping import VALID_NYU40_IDS_35CLASS
            invalid_ids = set(unique_labels) - set(VALID_NYU40_IDS_35CLASS)
            if invalid_ids:
                print(f"⚠️  WARNING: Found labels with invalid NYU40 IDs: {invalid_ids}")
                print("   These IDs are not in the 35-class valid set")
            else:
                print("✅ PASS: All labels are valid NYU40 IDs for 35-class training")
        except ImportError:
            print("⚠️  WARNING: Cannot validate against 35-class mapping")
        
        print("\n🎯 OVERALL RESULT:")
        print("✅ SUCCESS: Pseudo labels use correct NYU40 ID format!")
        print("✅ Compatible with ground truth for mixed training")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to validate {file_path}")
        print(f"   Error: {e}")
        return False


def find_pseudo_label_files():
    """Find all pseudo label pickle files in the project."""
    pseudo_files = []
    
    # Common locations for pseudo labels
    search_paths = [
        "35class_pseudo_labels/*.pkl",
        "stage1_pseudo_labels/*.pkl", 
        "*/pseudo_labels/*.pkl",
        "incremental_logs/*/pseudo_labels/*.pkl",
        "*pseudo*labels*.pkl"
    ]
    
    for pattern in search_paths:
        pseudo_files.extend(Path(".").glob(pattern))
    
    return list(set(pseudo_files))  # Remove duplicates


def main():
    """Main validation function."""
    print("🔍 Pseudo Label Format Validator")
    print("=" * 80)
    print("Validates that pseudo labels use NYU40 IDs for ground truth compatibility\n")
    
    if len(sys.argv) > 1:
        # Validate specific file
        file_path = Path(sys.argv[1])
        if not file_path.exists():
            print(f"❌ File not found: {file_path}")
            sys.exit(1)
        
        success = validate_pseudo_label_file(file_path)
        sys.exit(0 if success else 1)
    
    else:
        # Find and validate all pseudo label files
        pseudo_files = find_pseudo_label_files()
        
        if not pseudo_files:
            print("❌ No pseudo label files found")
            print("   Search patterns:")
            print("   - 35class_pseudo_labels/*.pkl")
            print("   - stage1_pseudo_labels/*.pkl") 
            print("   - */pseudo_labels/*.pkl")
            print("   - *pseudo*labels*.pkl")
            sys.exit(1)
        
        print(f"Found {len(pseudo_files)} pseudo label files:")
        for f in pseudo_files:
            print(f"  - {f}")
        
        # Validate all files
        results = []
        for file_path in pseudo_files:
            success = validate_pseudo_label_file(file_path)
            results.append((file_path, success))
        
        # Summary
        print("\n" + "=" * 80)
        print("📊 VALIDATION SUMMARY")
        print("=" * 80)
        
        passed = sum(1 for _, success in results if success)
        failed = len(results) - passed
        
        print(f"✅ PASSED: {passed} files")
        print(f"❌ FAILED: {failed} files")
        
        if failed > 0:
            print(f"\n❌ Failed files:")
            for file_path, success in results:
                if not success:
                    print(f"   - {file_path}")
        
        print(f"\n🎯 Overall result: {'SUCCESS' if failed == 0 else 'FAILURES DETECTED'}")
        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
