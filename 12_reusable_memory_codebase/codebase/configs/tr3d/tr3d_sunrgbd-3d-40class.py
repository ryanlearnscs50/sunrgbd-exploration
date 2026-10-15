"""
TR3D SUN RGB-D 40-class (raw top-40) supervised configuration.

This is the fully supervised upper bound for SUN RGB-D incremental experiments.
Class order matches `configs/_base_/class_mappings/sunrgbd_40class_mapping.py`.
"""

voxel_size = .01
n_points = 100000

import sys
sys.path.append('configs/_base_/class_mappings')
from sunrgbd_40class_mapping import SUNRGBD_40_RAW_TOP40_CLASSES  # type: ignore

class_names = tuple(SUNRGBD_40_RAW_TOP40_CLASSES)

# Large-object classes are assigned to level 1; others to level 0.
# Derived from mean train box volume with threshold ~= 0.8 (consistent with SUNRGBD-10 setup).
label2level = [
    0,  # chair
    1,  # table
    0,  # pillow
    0,  # sofa_chair
    1,  # desk
    1,  # bed
    1,  # sofa
    0,  # computer
    0,  # lamp
    0,  # box
    0,  # garbage_bin
    1,  # cabinet
    1,  # shelf
    0,  # drawer
    0,  # night_stand
    0,  # endtable
    0,  # sink
    0,  # picture
    0,  # stool
    0,  # coffee_table
    1,  # bookshelf
    0,  # painting
    0,  # keyboard
    0,  # dresser
    0,  # tv
    0,  # whiteboard
    0,  # cpu
    0,  # toilet
    0,  # paper
    0,  # ottoman
    1,  # bench
    0,  # recycle_bin
    0,  # monitor
    0,  # printer
    0,  # plant
    0,  # door
    0,  # book
    0,  # mirror
    0,  # laptop
    0,  # towel
]

model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=voxel_size,
    backbone=dict(
        type='MinkResNet',
        in_channels=3,
        depth=34,
        max_channels=128,
        norm='batch',
    ),
    neck=dict(
        type='TR3DNeck',
        in_channels=(64, 128, 128, 128),
        out_channels=128,
    ),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=8,
        n_classes=40,
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=label2level,
        ),
        bbox_loss=dict(type='RotatedIoU3DLoss', mode='diou', reduction='none'),
        train_cfg=dict(enable_class_masking=False),
    ),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01),
)

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
    ],
)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
    ),
    dict(type='LoadAnnotations3D'),
    dict(type='PointSample', num_points=n_points),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=.5,
        flip_ratio_bev_vertical=.0,
    ),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-.523599, .523599],
        scale_ratio_range=[.85, 1.15],
        translation_std=[.1, .1, .1],
        shift_height=False,
    ),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
    ),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='PointSample', num_points=n_points),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False,
            ),
            dict(type='Collect3D', keys=['points']),
        ],
    ),
]

data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            modality=dict(use_camera=False, use_lidar=True),
            data_root=data_root,
            ann_file=data_root + 'sunrgbd_infos_train_40class.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth',
        ),
    ),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_40class.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
    ),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'sunrgbd_infos_val_40class.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth',
    ),
)

evaluation = dict(interval=1, metric='mAP')
