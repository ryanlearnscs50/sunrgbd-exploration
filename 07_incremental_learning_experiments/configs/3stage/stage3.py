# AUTO-GENERATED incremental TR3D config (v4 3D-box split). DO NOT hand-edit;
# regenerate via incremental_learning/gen_configs_v4.py
# Schedule: 3stage | Stage 3/3 (scene_type: bedroom)
# Novel classes introduced this stage: ['bed', 'pillow', 'lamp', 'night_stand', 'sofa', 'dresser', 'tv', 'curtain', 'cabinet', 'door', 'mirror', 'bookshelf', 'painting']
# Cumulative classes known after this stage (30): ('chair', 'table', 'whiteboard', 'computer', 'desk', 'keyboard', 'box', 'drawer', 'shelf', 'garbage_bin', 'mouse', 'printer', 'book', 'monitor', 'paper', 'cup', 'laptop', 'bed', 'pillow', 'lamp', 'night_stand', 'sofa', 'dresser', 'tv', 'curtain', 'cabinet', 'door', 'mirror', 'bookshelf', 'painting')
voxel_size = .01
n_points = 100000

model = dict(
    type='MinkSingleStage3DDetector',
    voxel_size=voxel_size,
    backbone=dict(type='MinkResNet', in_channels=3, depth=34, max_channels=128, norm='batch'),
    neck=dict(type='TR3DNeck', in_channels=(64, 128, 128, 128), out_channels=128),
    head=dict(
        type='TR3DHead',
        in_channels=128,
        n_reg_outs=8,
        n_classes=30,                # cumulative classes through this stage
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level=[0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0]),
        bbox_loss=dict(type='RotatedIoU3DLoss', mode='diou', reduction='none')),
    train_cfg=dict(),
    test_cfg=dict(nms_pre=1000, iou_thr=.5, score_thr=.01))

optimizer = dict(type='AdamW', lr=.001, weight_decay=.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/incremental_v4/3stage/stage3'
# Stage>1: warm-start from the previous stage's model. NOTE (Step C): the classifier head
# grows by the new classes, so a plain load_from reinitialises the head on size-mismatch.
# Step C must replace this with proper classifier expansion + (distillation/pseudo-labels).
load_from = 'work_dirs/incremental_v4/3stage/stage2/latest.pth'
resume_from = None
workflow = [('train', 1)]

dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd_incremental/'
class_names = ('chair', 'table', 'whiteboard', 'computer', 'desk', 'keyboard', 'box', 'drawer', 'shelf', 'garbage_bin', 'mouse', 'printer', 'book', 'monitor', 'paper', 'cup', 'laptop', 'bed', 'pillow', 'lamp', 'night_stand', 'sofa', 'dresser', 'tv', 'curtain', 'cabinet', 'door', 'mirror', 'bookshelf', 'painting')
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True,
         load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadAnnotations3D'),
    dict(type='PointSample', num_points=n_points),
    dict(type='RandomFlip3D', sync_2d=False, flip_ratio_bev_horizontal=.5, flip_ratio_bev_vertical=.0),
    dict(type='GlobalRotScaleTrans', rot_range=[-.523599, .523599],
         scale_ratio_range=[.85, 1.15], translation_std=[.1, .1, .1], shift_height=False),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False, use_color=True,
         load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='MultiScaleFlipAug3D', img_scale=(1333, 800), pts_scale_ratio=1, flip=False,
         transforms=[
             dict(type='PointSample', num_points=n_points),
             dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
             dict(type='Collect3D', keys=['points'])
         ])
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
            ann_file=data_root + 'v4_3dbox/3stage/sunrgbd_infos_train_stage3.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'v4_3dbox/3stage/sunrgbd_infos_val_stage3.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + 'v4_3dbox/3stage/sunrgbd_infos_val_stage3.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
