"""
TR3D ScanNet 35-class STANDARD configuration

This config provides clean 35-class training following original TR3D design patterns.
Uses standard NYU40 ID ordering (cabinet, bed, chair, sofa, table...).
Completely independent of incremental learning experiments.

Expected performance: ~30-50% mAP@0.25

Usage:
    python tools/train.py configs/tr3d/tr3d_scannet-3d-35class_standard.py --work-dir ./my_work_dirs/tr3d_35class_standard --cfg-options runner.max_epochs=40 evaluation.interval=1
"""

voxel_size = .01
n_points = 100000

model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=voxel_size,
    backbone=dict(type='MinkResNet', in_channels=3, max_channels=128, depth=34, norm='batch'),
    neck=dict(
        type='TR3DNeck',
        in_channels=(64, 128, 128, 128),
        out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=6,
        n_classes=35,
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            # Standard label2level mapping for 35 classes in NYU40 ID order
            # Based on object size characteristics: 0=large objects, 1=small objects
            label2level=[
                0,  # cabinet (large)
                1,  # bed (medium-large)
                0,  # chair (small but structural)
                1,  # sofa (large)
                1,  # table (medium)
                0,  # door (large structural)
                1,  # window (large structural)
                1,  # bookshelf (large)
                0,  # picture (small)
                1,  # counter (large)
                1,  # blinds (medium)
                1,  # desk (medium)
                0,  # shelves (medium)
                0,  # curtain (medium)
                0,  # dresser (large)
                0,  # pillow (small)
                1,  # mirror (small-medium)
                0,  # floor_mat (small)
                1,  # clothes (small)
                1,  # books (small)
                0,  # refrigerator (large)
                0,  # television (medium)
                0,  # paper (small)
                1,  # towel (small)
                0,  # shower_curtain (medium)
                1,  # box (small-medium)
                1,  # whiteboard (medium)
                1,  # person (large)
                1,  # nightstand (medium)
                0,  # toilet (large)
                0,  # sink (medium-large)
                1,  # lamp (small-medium)
                1,  # bathtub (large)
                1,  # bag (small)
                0   # otherfurniture (variable)
            ]),
        bbox_loss=dict(type='AxisAlignedIoULoss', mode='diou', reduction='none'),
        train_cfg=dict(enable_class_masking=False)),  # Disable incremental learning masking
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# Standard 35-class names in NYU40 ID order (excludes wall, floor, ceiling, otherstructure, otherprop)
class_names = (
    'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
    'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves',
    'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes',
    'books', 'refrigerator', 'television', 'paper', 'towel',
    'shower_curtain', 'box', 'whiteboard', 'person', 'nightstand',
    'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherfurniture'
)

# Corresponding NYU40 IDs (1-based, excludes 1,2,22,38,40)
valid_nyu40_ids = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39)

print(f"🔧 TR3D 35-class STANDARD config:")
print(f"   Classes: {len(class_names)}")
print(f"   Valid NYU40 IDs: {valid_nyu40_ids[:5]}...{valid_nyu40_ids[-5:]}")

dataset_type = 'ScanNetDataset'
data_root = './data/scannet/'

# Standard TR3D pipeline (matches 18-class successful config)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),  # Simple form - no masks needed for standard training
    dict(type='GlobalAlignment', rotation_axis=2),
    # Original TR3D point sampling: 33% to 100% of points randomly
    dict(type='PointSample', num_points=.33),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=.5,
        flip_ratio_bev_vertical=.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.02, .02],
        scale_ratio_range=[.9, 1.1],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(type='NormalizePointsColor', color_mean=None),
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
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

# Standard dataset configuration
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=15,  # Original TR3D design
        dataset=dict(
            type=dataset_type,
            variant='35',  # Use standard 35-class variant (NYU40 ID ordering)
            data_root=data_root,
            ann_file=data_root + 'scannet_infos_train_40class_corrected.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        variant='35',  # Use standard 35-class variant (NYU40 ID ordering)
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val_40class_corrected.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        variant='35',  # Use standard 35-class variant (NYU40 ID ordering)
        data_root=data_root,
        ann_file=data_root + 'scannet_infos_val_40class_corrected.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))

evaluation = dict(interval=1, metric='mAP', classwise=True)