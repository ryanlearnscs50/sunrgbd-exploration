#!/usr/bin/env python
"""v4 TR3D incremental config generator (mirrors gen_incremental_configs.py).

Reads v4_3dbox/stage_def_<sched>.json (ordered stages, classes, global_ids) and emits a
self-contained per-stage TR3D config, cumulative class_names through stage t, train
ann=novel-only pkl, val ann=cumulative pkl, load_from=prev stage. Naive fine-tuning
baseline; Step C hooks (head expansion / distillation / pseudo-labels) noted in template.

MERGE-aware: the label2level size heuristic aggregates dimensions under the merged name
(cpu->computer, sofa_chair->sofa) so it matches the carved class_names.
Writes to configs/tr3d_incremental_v4/<sched>/ and points at data/sunrgbd_incremental/v4_3dbox/.
"""
import os, json, pickle, collections
import numpy as np

V4 = '/data3/ryan/incremental_learning/v4_3dbox'
SRC = '/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_train_40class.pkl'
CFG_ROOT = '/data3/ryan/tr3d/configs/tr3d_incremental_v4'
DATA_ROOT_REL = 'data/sunrgbd_incremental/'
ANN_SUBDIR = 'v4_3dbox'
WORK_ROOT_REL = 'work_dirs/incremental_v4'


def class_horiz_sizes(merge):
    d = pickle.load(open(SRC, 'rb'))
    acc = collections.defaultdict(list)
    for e in d:
        a = e['annos']
        if a.get('gt_num', 0) > 0:
            for nm, dim in zip(a['name'], a['dimensions']):
                acc[merge.get(str(nm), str(nm))].append(max(float(dim[0]), float(dim[1])))
    return {c: float(np.mean(v)) for c, v in acc.items()}


TEMPLATE = '''# AUTO-GENERATED incremental TR3D config (v4 3D-box split). DO NOT hand-edit;
# regenerate via incremental_learning/gen_configs_v4.py
# Schedule: {ordering} | Stage {t}/{nstages} (scene_type: {stage_name})
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
    summary = {}
    for sched in ('3stage', '6stage'):
        sdef = json.load(open(os.path.join(V4, f'stage_def_{sched}.json')))
        merge = sdef['merges']
        horiz = class_horiz_sizes(merge)
        stages = sdef['stages']
        order = [c for s in stages for c in s['classes']]
        outdir = os.path.join(CFG_ROOT, sched)
        os.makedirs(outdir, exist_ok=True)
        nstages = len(stages)
        cum = 0
        cfg_paths = []
        for s in stages:
            t = s['stage']
            cum += len(s['classes'])
            classes = tuple(order[:cum])
            label2level = [1 if horiz[c] > 1.0 else 0 for c in classes]
            work_dir = f'{WORK_ROOT_REL}/{sched}/stage{t}'
            load_from = 'None' if t == 1 else repr(f'{WORK_ROOT_REL}/{sched}/stage{t-1}/latest.pth')
            txt = TEMPLATE.format(
                ordering=sched, t=t, nstages=nstages, stage_name=s['scene_type'],
                novel=s['classes'], ncls=cum, classes=classes, label2level=label2level,
                work_dir=work_dir, load_from=load_from, data_root=DATA_ROOT_REL,
                train_ann=f'{ANN_SUBDIR}/{sched}/sunrgbd_infos_train_stage{t}.pkl',
                val_ann=f'{ANN_SUBDIR}/{sched}/sunrgbd_infos_val_stage{t}.pkl')
            p = os.path.join(outdir, f'stage{t}.py')
            open(p, 'w').write(txt)
            cfg_paths.append(p)
            print(f'[{sched}] stage{t}: n_classes={cum} label2level={label2level}')
            print(f'           classes={classes}')
        summary[sched] = cfg_paths
    os.makedirs(CFG_ROOT, exist_ok=True)
    json.dump(summary, open(os.path.join(CFG_ROOT, 'generated_configs.json'), 'w'), indent=1)
    print('wrote configs under', CFG_ROOT)


if __name__ == '__main__':
    main()
