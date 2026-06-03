#!/usr/bin/env python
"""Chunk 4 of Step B: generate the TR3D incremental config skeleton.

For each ordering (3stage, 6stage) and each stage t, emit a self-contained TR3D config
(no _base_, matching repo convention) whose:
  - class_names / n_classes / label2level cover the CUMULATIVE classes seen through stage t
  - train ann_file = stage t's NOVEL-only train pkl
  - val/test ann_file = stage t's CUMULATIVE val pkl
  - load_from = previous stage's checkpoint (None for stage 1)
This is SCAFFOLDING ONLY. The incremental METHOD (classifier expansion, knowledge
distillation, pseudo-labels) is Step C and is marked with explicit hooks in the harness.
By default a stage just trains on its novel GT -> that is the naive fine-tuning baseline.

label2level (per-class FPN level) is a size heuristic: horiz=max(mean dx,dy) > 1.0 -> 1 else 0,
which reproduces the canonical 10-class intent. Tunable; not the paper's hand-set values.
"""
import os, json, pickle, collections
import numpy as np

INC_DATA = '/data3/ryan/tr3d/data/sunrgbd_incremental'
SRC = '/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_train_40class.pkl'
CFG_ROOT = '/data3/ryan/tr3d/configs/tr3d_incremental'
DATA_ROOT_REL = 'data/sunrgbd_incremental/'
WORK_ROOT_REL = 'work_dirs/incremental'

def class_horiz_sizes():
    d = pickle.load(open(SRC, 'rb'))
    acc = collections.defaultdict(list)
    for e in d:
        a = e['annos']
        if a.get('gt_num', 0) > 0:
            for nm, dim in zip(a['name'], a['dimensions']):
                acc[str(nm)].append(max(float(dim[0]), float(dim[1])))
    return {c: float(np.mean(v)) for c, v in acc.items()}

TEMPLATE = '''# AUTO-GENERATED incremental TR3D config — Step B chunk 4 skeleton. DO NOT hand-edit;
# regenerate via incremental_learning/gen_incremental_configs.py
# Ordering: {ordering} | Stage {t}/{nstages} ({stage_name})
# Novel classes introduced this stage: {novel}
# Cumulative classes known after this stage ({ncls}): {classes}
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
        n_classes={ncls},                # cumulative classes through this stage
        voxel_size=voxel_size,
        assigner=dict(
            type='TR3DAssigner',
            top_pts_threshold=6,
            label2level={label2level}),
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
work_dir = '{work_dir}'
# Stage>1: warm-start from the previous stage's model. NOTE (Step C): the classifier head
# grows by the new classes, so a plain load_from reinitialises the head on size-mismatch.
# Step C must replace this with proper classifier expansion + (distillation/pseudo-labels).
load_from = {load_from}
resume_from = None
workflow = [('train', 1)]

dataset_type = 'SUNRGBDDataset'
data_root = '{data_root}'
class_names = {classes}
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
            ann_file=data_root + '{train_ann}',
            pipeline=train_pipeline,
            filter_empty_gt=False,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + '{val_ann}',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        modality=dict(use_camera=False, use_lidar=True),
        data_root=data_root,
        ann_file=data_root + '{val_ann}',
        pipeline=test_pipeline,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'))
'''

def main():
    horiz = class_horiz_sizes()
    summary = {}
    for ordering in ('3stage', '6stage'):
        rep = json.load(open(os.path.join(INC_DATA, ordering, 'carve_report.json')))
        order = rep['global_class_order']
        stages = rep['stages']
        outdir = os.path.join(CFG_ROOT, ordering)
        os.makedirs(outdir, exist_ok=True)
        nstages = len(stages)
        cum = 0
        cfg_paths = []
        for t, s in enumerate(stages, 1):
            cum += len(s['novel_classes'])
            classes = tuple(order[:cum])
            label2level = [1 if horiz[c] > 1.0 else 0 for c in classes]
            work_dir = f'{WORK_ROOT_REL}/{ordering}/stage{t}'
            if t == 1:
                load_from = 'None'
            else:
                load_from = repr(f'{WORK_ROOT_REL}/{ordering}/stage{t-1}/latest.pth')
            txt = TEMPLATE.format(
                ordering=ordering, t=t, nstages=nstages, stage_name=s['stage'] if 'stage' in s else s.get('through_stage',''),
                novel=s['novel_classes'], ncls=cum, classes=classes, label2level=label2level,
                work_dir=work_dir, load_from=load_from, data_root=DATA_ROOT_REL,
                train_ann=f'{ordering}/sunrgbd_infos_train_stage{t}.pkl',
                val_ann=f'{ordering}/sunrgbd_infos_val_stage{t}.pkl')
            p = os.path.join(outdir, f'stage{t}.py')
            open(p, 'w').write(txt)
            cfg_paths.append(p)
            print(f'[{ordering}] stage{t}: n_classes={cum} classes={classes}')
            print(f'           label2level={label2level}')
        summary[ordering] = cfg_paths
    json.dump(summary, open(os.path.join(CFG_ROOT, 'generated_configs.json'), 'w'), indent=1)
    print('wrote configs under', CFG_ROOT)

if __name__ == '__main__':
    main()
