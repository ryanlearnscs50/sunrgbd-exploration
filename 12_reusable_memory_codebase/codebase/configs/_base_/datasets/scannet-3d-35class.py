# Dataset configuration for ScanNet 35-class training using proper .bin files
# Uses 40-class data with 5 ignored classes: wall, floor, ceiling, otherstructure, otherprop
# Now uses centralized class mapping configuration for maintainability

import sys
# Use PYTHONPATH-based import (works when TR3D root is in PYTHONPATH)
sys.path.append('configs/_base_/class_mappings')
from scannet_35class_mapping import (
    SCANNET_35_CLASSES,
    IGNORED_CLASS_NAMES_35CLASS,
    VALID_NYU40_IDS_35CLASS,
    IGNORED_NYU40_IDS_35CLASS
)

dataset_type = 'ScanNetDataset'
data_root = 'data/scannet/'

# Use class names from external config
class_names = SCANNET_35_CLASSES

# Validation - ensure we have exactly 35 classes with 5 ignored
assert len(class_names) == 35, f"Expected 35 classes, got {len(class_names)}"
assert len(IGNORED_CLASS_NAMES_35CLASS) == 5, f"Expected 5 ignored classes, got {len(IGNORED_CLASS_NAMES_35CLASS)}"
assert set(IGNORED_CLASS_NAMES_35CLASS) == {'wall', 'floor', 'ceiling', 'otherstructure', 'otherprop'}

# Standard TR3D training pipeline
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='PointSample',
        num_points=20000),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.087266, 0.087266],
        scale_ratio_range=[.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='GlobalAlignment', rotation_axis=2),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_train_40class_corrected.pkl',  # Use corrected 40-class data
        pipeline=train_pipeline,
        filter_empty_gt=True,
        classes=class_names,
        variant='dynamic_head',  # Use dynamic head incremental learning variant
        box_type_3d='Depth'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val_40class_corrected.pkl',  # Use corrected 40-class data
        pipeline=test_pipeline,
        classes=class_names,
        variant='dynamic_head',  # Use dynamic head incremental learning variant
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_test_40class_corrected.pkl',  # Use corrected 40-class data  
        pipeline=test_pipeline,
        classes=class_names,
        variant='dynamic_head',  # Use dynamic head incremental learning variant
        test_mode=True,
        box_type_3d='Depth'))

# Evaluation settings
evaluation = dict(pipeline=test_pipeline)