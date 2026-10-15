#!/usr/bin/env python3
"""
ScanNet Info Files Inspector

This script provides comprehensive inspection of scannet_infos_*.pkl files,
focusing on class information and bounding boxes in scenes.

Usage:
    python tools/inspect_scannet_infos.py --info-file data/scannet/scannet_infos_train_18class.pkl
    python tools/inspect_scannet_infos.py --info-file data/scannet/scannet_infos_train_40class_corrected.pkl --scene-id scene0000_00
"""

import argparse
import pickle
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mmdet3d.datasets.scannet.label_maps import get_scannet_classes, NYU40_CLASSES


def load_info_file(info_path):
    """Load scannet info pickle file."""
    print(f"🔍 Loading info file: {info_path}")
    with open(info_path, 'rb') as f:
        data = pickle.load(f)
    print(f"✅ Loaded {len(data)} scenes")
    return data


def analyze_overall_stats(data):
    """Analyze overall statistics of the dataset."""
    print("\n" + "="*60)
    print("📊 OVERALL DATASET STATISTICS")
    print("="*60)
    
    total_scenes = len(data)
    total_objects = 0
    scenes_with_objects = 0
    class_counts = Counter()
    
    for scene_info in data:
        scene_id = scene_info['point_cloud']['lidar_idx']
        annos = scene_info['annos']
        num_objects = annos['gt_num']
        
        if num_objects > 0:
            scenes_with_objects += 1
            total_objects += num_objects
            
            # Count classes
            if 'class' in annos:
                classes = annos['class']
                class_counts.update(classes)
            elif 'gt_labels_3d' in annos:
                labels = annos['gt_labels_3d']
                class_counts.update(labels)
    
    print(f"Total scenes: {total_scenes}")
    print(f"Scenes with objects: {scenes_with_objects}")
    print(f"Scenes without objects: {total_scenes - scenes_with_objects}")
    print(f"Total objects: {total_objects}")
    print(f"Average objects per scene: {total_objects / total_scenes:.2f}")
    print(f"Average objects per scene (with objects): {total_objects / scenes_with_objects:.2f}")
    
    return class_counts


def analyze_class_distribution(class_counts, info_path):
    """Analyze class distribution in the dataset."""
    print("\n" + "="*60)
    print("🏷️  CLASS DISTRIBUTION")
    print("="*60)
    
    # Determine variant from filename
    variant = '18' if '18class' in str(info_path) else '40'
    
    try:
        class_names = get_scannet_classes(variant)
        print(f"Dataset variant: {variant}-class")
        print(f"Expected classes: {len(class_names)}")
    except:
        class_names = None
        print("Could not determine class names from variant")
    
    if not class_counts:
        print("⚠️  No class information found in annotations")
        return
    
    print(f"\nFound {len(class_counts)} unique class IDs")
    print(f"Total object instances: {sum(class_counts.values())}")
    
    # Sort classes by frequency
    sorted_classes = class_counts.most_common()
    
    print(f"\n{'Class ID':<10} {'Count':<8} {'%':<6} {'Class Name'}")
    print("-" * 50)
    
    total_instances = sum(class_counts.values())
    for class_id, count in sorted_classes:
        percentage = (count / total_instances) * 100
        
        # Try to get class name
        class_name = "Unknown"
        if class_names and 0 <= class_id < len(class_names):
            class_name = class_names[class_id]
        elif 0 <= class_id < len(NYU40_CLASSES):
            class_name = NYU40_CLASSES[class_id]
        
        print(f"{class_id:<10} {count:<8} {percentage:<5.1f}% {class_name}")


def inspect_scene_details(data, scene_id=None, max_scenes=5):
    """Inspect detailed information for specific scenes."""
    print("\n" + "="*60)
    print("🔍 SCENE DETAILS")
    print("="*60)
    
    scenes_to_inspect = []
    
    if scene_id:
        # Find specific scene
        for scene_info in data:
            if scene_info['point_cloud']['lidar_idx'] == scene_id:
                scenes_to_inspect = [scene_info]
                break
        if not scenes_to_inspect:
            print(f"❌ Scene {scene_id} not found!")
            return
    else:
        # Show scenes with most objects
        scene_objects = []
        for scene_info in data:
            num_objects = scene_info['annos']['gt_num']
            if num_objects > 0:
                scene_objects.append((num_objects, scene_info))
        
        scene_objects.sort(reverse=True)
        scenes_to_inspect = [info for _, info in scene_objects[:max_scenes]]
        print(f"Showing top {len(scenes_to_inspect)} scenes with most objects:")
    
    for scene_info in scenes_to_inspect:
        print_scene_info(scene_info)


def print_scene_info(scene_info):
    """Print detailed information for a single scene."""
    scene_id = scene_info['point_cloud']['lidar_idx']
    annos = scene_info['annos']
    
    print(f"\n📍 Scene: {scene_id}")
    print("-" * 40)
    
    # Basic info
    print(f"Point cloud file: {scene_info['pts_path']}")
    print(f"Number of objects: {annos['gt_num']}")
    
    if annos['gt_num'] == 0:
        print("  (No objects in this scene)")
        return
    
    # Bounding boxes
    if 'gt_boxes_upright_depth' in annos:
        boxes = annos['gt_boxes_upright_depth']
        print(f"Bounding boxes shape: {boxes.shape}")
        print(f"Box format: [x, y, z, dx, dy, dz, heading] (upright depth)")
        
        # Show box statistics
        centers = boxes[:, :3]
        sizes = boxes[:, 3:6]
        headings = boxes[:, 6] if boxes.shape[1] > 6 else None
        
        print(f"Center ranges: X[{centers[:, 0].min():.2f}, {centers[:, 0].max():.2f}] "
              f"Y[{centers[:, 1].min():.2f}, {centers[:, 1].max():.2f}] "
              f"Z[{centers[:, 2].min():.2f}, {centers[:, 2].max():.2f}]")
        print(f"Size ranges: W[{sizes[:, 0].min():.2f}, {sizes[:, 0].max():.2f}] "
              f"H[{sizes[:, 1].min():.2f}, {sizes[:, 1].max():.2f}] "
              f"D[{sizes[:, 2].min():.2f}, {sizes[:, 2].max():.2f}]")
        
        if headings is not None:
            print(f"Heading range: [{headings.min():.2f}, {headings.max():.2f}] radians")
    
    # Classes
    classes = None
    if 'class' in annos:
        classes = annos['class']
        class_key = 'class'
    elif 'gt_labels_3d' in annos:
        classes = annos['gt_labels_3d']
        class_key = 'gt_labels_3d'
    
    if classes is not None:
        print(f"Classes ({class_key}): {list(classes)}")
        class_counts = Counter(classes)
        for class_id, count in class_counts.most_common():
            class_name = "Unknown"
            if 0 <= class_id < len(NYU40_CLASSES):
                class_name = NYU40_CLASSES[class_id]
            print(f"  Class {class_id} ({class_name}): {count} instances")
    
    # Other annotation info
    other_keys = set(annos.keys()) - {'gt_num', 'gt_boxes_upright_depth', 'class', 'gt_labels_3d', 'axis_align_matrix'}
    if other_keys:
        print(f"Other annotation keys: {sorted(other_keys)}")


def compare_variants():
    """Compare different variants of ScanNet info files."""
    print("\n" + "="*60)
    print("🔄 COMPARING VARIANTS")
    print("="*60)
    
    info_files = {
        '18-class train': 'data/scannet/scannet_infos_train_18class.pkl',
        '18-class val': 'data/scannet/scannet_infos_val_18class.pkl',
        '40-class train': 'data/scannet/scannet_infos_train_40class_corrected.pkl',
        '40-class val': 'data/scannet/scannet_infos_val_40class_corrected.pkl',
    }
    
    results = {}
    
    for name, path in info_files.items():
        if Path(path).exists():
            try:
                data = load_info_file(path)
                class_counts = Counter()
                total_objects = 0
                
                for scene_info in data:
                    annos = scene_info['annos']
                    total_objects += annos['gt_num']
                    
                    if annos['gt_num'] > 0:
                        if 'class' in annos:
                            class_counts.update(annos['class'])
                        elif 'gt_labels_3d' in annos:
                            class_counts.update(annos['gt_labels_3d'])
                
                results[name] = {
                    'scenes': len(data),
                    'objects': total_objects,
                    'classes': len(class_counts),
                    'class_counts': class_counts
                }
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
        else:
            print(f"⚠️  File not found: {path}")
    
    # Print comparison table
    if results:
        print(f"\n{'Variant':<20} {'Scenes':<8} {'Objects':<8} {'Classes':<8} {'Avg Obj/Scene':<12}")
        print("-" * 60)
        
        for name, stats in results.items():
            avg_obj = stats['objects'] / stats['scenes'] if stats['scenes'] > 0 else 0
            print(f"{name:<20} {stats['scenes']:<8} {stats['objects']:<8} {stats['classes']:<8} {avg_obj:<12.2f}")


def main():
    parser = argparse.ArgumentParser(description='Inspect ScanNet info files')
    parser.add_argument('--info-file', type=str, help='Path to scannet_infos_*.pkl file')
    parser.add_argument('--scene-id', type=str, help='Specific scene ID to inspect (e.g., scene0000_00)')
    parser.add_argument('--max-scenes', type=int, default=5, help='Max number of scenes to show details for')
    parser.add_argument('--compare', action='store_true', help='Compare all available variants')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_variants()
        return
    
    if not args.info_file:
        print("❌ Please provide --info-file or use --compare")
        parser.print_help()
        return
    
    if not Path(args.info_file).exists():
        print(f"❌ File not found: {args.info_file}")
        return
    
    # Load and analyze the info file
    data = load_info_file(args.info_file)
    
    # Overall statistics
    class_counts = analyze_overall_stats(data)
    
    # Class distribution
    analyze_class_distribution(class_counts, args.info_file)
    
    # Scene details
    inspect_scene_details(data, args.scene_id, args.max_scenes)


if __name__ == '__main__':
    main()