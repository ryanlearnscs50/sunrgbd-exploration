#!/usr/bin/env python
"""Step C - VARIANT 2: offline pseudo-label generation for incremental TR3D.

The naive baseline (variant 1) forgets old classes catastrophically because each
stage trains only on that stage's NOVEL-class GT - the old classes never appear
in the targets, so their head columns decay. SDCoT's offline-pseudo-label remedy:
run the previous-stage detector over the new stage's TRAIN scenes, keep its
confident detections (which can only be OLD classes - that is all the old head
knows), and merge them back in as old-class GT. The stage-t model then sees both
the novel GT *and* a synthetic supervision signal for the old classes.

This script does exactly that, OFFLINE (mentor-preferred: generated once, then a
plain train run consumes the augmented pkl - no teacher in the training loop):

  1. Build the stage_{t-1} model from its config + checkpoint.
  2. Run it (single_gpu_test, the same path tools/test.py uses) over the stage-t
     TRAIN scenes (we just point data.test.ann_file at the stage-t train pkl).
  3. For each scene keep detections with score >= --score-thr and >= --min-pts
     points inside the box. NMS is the model's own (we set test_cfg.iou_thr to
     --nms-iou = 0.25, the SDCoT default).
  4. Merge the kept boxes into a COPY of the stage-t train pkl as old-class GT
     and write <train_pkl>_pseudo.pkl. The novel-only original is NOT modified.

GLOBAL gids: the per-stage cumulative class order IS the global class order
(carve_report_v4.json name2gid), so the stage_{t-1} model's output label i is
already global gid i - no remapping needed. We assert this against name2gid.

Box convention: gt_boxes_upright_depth is interpreted by SUNRGBDDataset with
origin (0.5, 0.5, 0.5) (gravity center). DepthInstance3DBoxes stores a bottom-
centred tensor, so we rebuild the 7-vec as [gravity_center, dims, yaw].

Usage:
  python tools_incremental/make_pseudo_labels.py \
      --prev-cfg  configs/tr3d_incremental_v4/3stage/stage1.py \
      --prev-ckpt work_dirs/incremental_v4_pseudo/3stage/stage1/latest.pth \
      --train-pkl data/sunrgbd_incremental/v4_3dbox/3stage/sunrgbd_infos_train_stage2.pkl \
      --carve-report data/sunrgbd_incremental/v4_3dbox/3stage/carve_report_v4.json \
      --out-pkl    data/sunrgbd_incremental/v4_3dbox/3stage/sunrgbd_infos_train_stage2_pseudo.pkl \
      --gpu-id 0
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


def parse_args():
    p = argparse.ArgumentParser(description='Offline pseudo-label generation (incremental TR3D).')
    p.add_argument('--prev-cfg', required=True, help='stage_{t-1} config (defines the OLD class head)')
    p.add_argument('--prev-ckpt', required=True, help='stage_{t-1} trained checkpoint (.pth)')
    p.add_argument('--train-pkl', required=True, help='stage-t NOVEL-only train pkl to augment')
    p.add_argument('--carve-report', required=True, help='carve_report_v4.json (for name2gid / global order)')
    p.add_argument('--out-pkl', required=True, help='output pseudo-augmented train pkl')
    # NB: SDCoT used 0.9-0.95 for VoteNet, but TR3D's dense sigmoid-focal scores
    # are much lower-calibrated: over a thin classroom->office teacher the max
    # det score is ~0.80 (99th pctl ~0.47), so 0.9 keeps NOTHING. 0.5 captures
    # the confident tail (top ~1% of dets, e.g. 372 boxes/233 office scenes,
    # correctly chair-dominated). Recalibrated default; tune per run via harness.
    p.add_argument('--score-thr', type=float, default=0.5, help='keep dets with score >= this (TR3D-recalibrated; SDCoT used 0.9)')
    p.add_argument('--min-pts', type=int, default=5, help='drop boxes with fewer than this many interior points')
    p.add_argument('--nms-iou', type=float, default=0.25, help='model test_cfg.iou_thr for pseudo NMS (SDCoT 0.25)')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--raw-dets', default=None, help='optional cache path for raw detections (skip inference if present)')
    return p.parse_args()


def points_in_box_count(pts_xyz, center, dims, yaw):
    """# of points inside an (origin-centred, yaw-about-z) box. dims = (dx,dy,dz)."""
    p = pts_xyz - center[None, :]
    c, s = np.cos(yaw), np.sin(yaw)
    # box-local = R(yaw)^T @ (world - center)
    xl = c * p[:, 0] + s * p[:, 1]
    yl = -s * p[:, 0] + c * p[:, 1]
    zl = p[:, 2]
    hx, hy, hz = dims / 2.0
    inside = (np.abs(xl) <= hx) & (np.abs(yl) <= hy) & (np.abs(zl) <= hz)
    return int(inside.sum())


def run_inference(args):
    cfg = Config.fromfile(args.prev_cfg)
    # Point the test set at the stage-t TRAIN scenes; reuse the prev-stage model.
    cfg.data.test.ann_file = args.train_pkl
    cfg.data.test.test_mode = True
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if cfg.model.get('test_cfg') is not None:
        cfg.model.test_cfg.iou_thr = args.nms_iou
    cfg.gpu_ids = [args.gpu_id]

    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=2,
                              dist=False, shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.prev_ckpt, map_location='cpu')
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    outputs = single_gpu_test(model, loader)
    data_root = cfg.data.test.get('data_root', '')
    return outputs, dataset, data_root


def main():
    args = parse_args()

    report = json.load(open(args.carve_report))
    name2gid = report['name2gid']
    gid2name = {v: k for k, v in name2gid.items()}

    with open(args.train_pkl, 'rb') as f:
        infos = pickle.load(f)

    if args.raw_dets and os.path.exists(args.raw_dets):
        print(f'[pseudo] loading cached raw dets from {args.raw_dets}')
        with open(args.raw_dets, 'rb') as f:
            cached = pickle.load(f)
        outputs = cached['outputs']
        order_idx = cached['lidar_idx']
        data_root = cached['data_root']
    else:
        outputs, dataset, data_root = run_inference(args)
        order_idx = [d['point_cloud']['lidar_idx'] for d in dataset.data_infos]
        if args.raw_dets:
            with open(args.raw_dets, 'wb') as f:
                pickle.dump(dict(outputs=outputs, lidar_idx=order_idx, data_root=data_root), f)

    assert len(outputs) == len(infos) == len(order_idx), \
        f'length mismatch: outputs={len(outputs)} infos={len(infos)} order={len(order_idx)}'
    # Alignment guard: dataset order must match the pkl order scene-for-scene.
    for i, info in enumerate(infos):
        assert info['point_cloud']['lidar_idx'] == order_idx[i], \
            f'order mismatch at {i}: pkl {info["point_cloud"]["lidar_idx"]} vs det {order_idx[i]}'

    n_scenes_aug = 0
    n_pseudo_boxes = 0
    per_class = {}
    drop_score = drop_pts = 0

    for i, info in enumerate(infos):
        out = outputs[i]
        boxes = out['boxes_3d']
        scores = out['scores_3d'].numpy() if torch.is_tensor(out['scores_3d']) else np.asarray(out['scores_3d'])
        labels = out['labels_3d'].numpy() if torch.is_tensor(out['labels_3d']) else np.asarray(out['labels_3d'])
        if len(scores) == 0:
            continue

        # origin-(0.5,0.5,0.5) 7-vec: [gravity_center(xyz), dims(dx,dy,dz), yaw]
        gc = boxes.gravity_center.numpy()
        bt = boxes.tensor.numpy()
        dims = bt[:, 3:6]
        yaw = bt[:, 6]
        upright = np.concatenate([gc, dims, yaw[:, None]], axis=1).astype(np.float64)

        keep = scores >= args.score_thr
        drop_score += int((~keep).sum())
        if not keep.any():
            continue

        # interior-point filter (load this scene's cloud once)
        pts_path = os.path.join(data_root, info['pts_path'])
        pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 6)[:, :3].astype(np.float64)

        sel = []
        for j in np.where(keep)[0]:
            npt = points_in_box_count(pts, gc[j], dims[j], float(yaw[j]))
            if npt >= args.min_pts:
                sel.append(j)
            else:
                drop_pts += 1
        if not sel:
            continue
        sel = np.array(sel, dtype=np.int64)

        ps_upright = upright[sel]
        ps_gids = labels[sel].astype(np.int64)
        ps_names = np.array([gid2name[int(g)] for g in ps_gids], dtype='<U16')
        m = len(sel)

        # all gids must be OLD classes (the prev model can only emit those)
        assert ps_gids.max() < len(name2gid), 'pseudo gid out of global range'

        # ---- merge into this scene's annos (old pseudo + existing novel GT) ----
        a = info.get('annos', {})
        def cur(key, tail, dtype):
            v = a.get(key)
            if v is None:
                return np.zeros((0,) + tail, dtype)
            v = np.asarray(v)
            return v if v.size else np.zeros((0,) + tail, dtype)

        ex_up = cur('gt_boxes_upright_depth', (7,), np.float64)
        ex_cls = cur('class', (), np.int64)
        ex_name = cur('name', (), '<U16')
        ex_bbox = cur('bbox', (4,), np.float64)
        ex_loc = cur('location', (3,), np.float64)
        ex_dim = cur('dimensions', (3,), np.float64)
        ex_rot = cur('rotation_y', (), np.float64)

        ps_bbox = np.zeros((m, 4), np.float64)        # 2D box unused by TR3D (point-only)
        ps_loc = ps_upright[:, :3]
        ps_dim = ps_upright[:, 3:6]
        ps_rot = ps_upright[:, 6]

        new_annos = dict(a)  # shallow copy, then overwrite arrays
        new_annos['gt_boxes_upright_depth'] = np.concatenate([ex_up, ps_upright], axis=0)
        new_annos['class'] = np.concatenate([ex_cls, ps_gids], axis=0)
        new_annos['name'] = np.concatenate([ex_name, ps_names], axis=0)
        new_annos['bbox'] = np.concatenate([ex_bbox, ps_bbox], axis=0)
        new_annos['location'] = np.concatenate([ex_loc, ps_loc], axis=0)
        new_annos['dimensions'] = np.concatenate([ex_dim, ps_dim], axis=0)
        new_annos['rotation_y'] = np.concatenate([ex_rot, ps_rot], axis=0)
        new_annos['gt_num'] = int(len(new_annos['class']))
        new_annos['index'] = np.arange(new_annos['gt_num'], dtype=np.int32)
        info['annos'] = new_annos

        n_scenes_aug += 1
        n_pseudo_boxes += m
        for g in ps_gids:
            nm = gid2name[int(g)]
            per_class[nm] = per_class.get(nm, 0) + 1

    with open(args.out_pkl, 'wb') as f:
        pickle.dump(infos, f)

    print(f'[pseudo] wrote {args.out_pkl}')
    print(f'  scenes augmented : {n_scenes_aug}/{len(infos)}')
    print(f'  pseudo boxes     : {n_pseudo_boxes}  (dropped: score<{args.score_thr} {drop_score}, pts<{args.min_pts} {drop_pts})')
    print(f'  per-class pseudo : ' + ', '.join(f'{k}={v}' for k, v in sorted(per_class.items(), key=lambda kv: -kv[1])))


if __name__ == '__main__':
    main()
