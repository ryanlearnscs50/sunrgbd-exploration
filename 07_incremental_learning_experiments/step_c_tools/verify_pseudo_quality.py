#!/usr/bin/env python
"""Step A verification: pseudo-label quality / threshold analysis on the TRAIN set.

Reuses the cached teacher detections (raw_dets.pkl) written by make_pseudo_labels.py
and scores them against the TRUE old-class GT (from the 40-class source, which has the
FULL annotation for every scene -- the per-stage train pkl is novel-only and does NOT
contain old-class boxes). This lets us answer:

  * How good is the teacher on the train scenes it pseudo-labels?  -> AP@0.25 per old class.
  * What score threshold should we keep?  -> precision/recall/F1/#kept vs score_thr,
    with the SAME min-pts filter the production script applies.

Precision is the number that matters for pseudo-GT: a false pseudo box injects a wrong
old-class target. Recall matters less (missing some old objects just means weaker
anti-forgetting signal, not corrupted supervision).

Usage:
  python tools_incremental/verify_pseudo_quality.py --sched 3stage --stage 2
"""
import argparse, json, pickle, os
import numpy as np
import torch
from mmdet3d.core.bbox import DepthInstance3DBoxes

SRC = '/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_train_40class.pkl'
IOU_THR = 0.25
MIN_PTS = 5


def points_in_box_count(pts_xyz, center, dims, yaw):
    p = pts_xyz - center[None, :]
    c, s = np.cos(yaw), np.sin(yaw)
    xl = c * p[:, 0] + s * p[:, 1]
    yl = -s * p[:, 0] + c * p[:, 1]
    zl = p[:, 2]
    hx, hy, hz = dims / 2.0
    inside = (np.abs(xl) <= hx) & (np.abs(yl) <= hy) & (np.abs(zl) <= hz)
    return int(inside.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sched', required=True, choices=['3stage', '6stage'])
    ap.add_argument('--stage', type=int, required=True, help='stage t (>=2) being pseudo-labeled')
    ap.add_argument('--thrs', default='0.2,0.3,0.4,0.5,0.6,0.7')
    ap.add_argument('--data-root', default='data/sunrgbd_incremental/')
    args = ap.parse_args()

    base = f'data/sunrgbd_incremental/v4_3dbox/{args.sched}'
    report = json.load(open(f'{base}/carve_report_v4.json'))
    name2gid = report['name2gid']
    gid2name = {v: k for k, v in name2gid.items()}
    merges = report['merges']

    # old gids = union of novel gids from stages 1..t-1
    old_gids = set()
    for st in report['train_stages']:
        if st['stage'] <= args.stage - 1:
            old_gids.update(st['novel_class_gids'])
    old_gids = sorted(old_gids)
    print(f'[{args.sched} stage{args.stage}] teacher=stage{args.stage-1}, '
          f'{len(old_gids)} old classes: {[gid2name[g] for g in old_gids]}')

    rd = pickle.load(open(f'work_dirs/incremental_v4_pseudo/{args.sched}/stage{args.stage}/raw_dets.pkl', 'rb'))
    outputs = rd['outputs']
    order_idx = rd['lidar_idx']
    data_root = rd.get('data_root', args.data_root)

    # true full GT keyed by lidar_idx
    src = {d['point_cloud']['lidar_idx']: d for d in pickle.load(open(SRC, 'rb'))}

    # per-scene: dets (boxes, scores, labels, npts) and gt (boxes, labels) restricted to old classes
    per_scene = []
    train_pkl = pickle.load(open(f'{base}/sunrgbd_infos_train_stage{args.stage}.pkl', 'rb'))
    for i, lidx in enumerate(order_idx):
        out = outputs[i]
        boxes = out['boxes_3d']
        scores = out['scores_3d'].numpy() if torch.is_tensor(out['scores_3d']) else np.asarray(out['scores_3d'])
        labels = out['labels_3d'].numpy() if torch.is_tensor(out['labels_3d']) else np.asarray(out['labels_3d'])
        gc = boxes.gravity_center.numpy(); bt = boxes.tensor.numpy()
        det_up = np.concatenate([gc, bt[:, 3:6], bt[:, 6:7]], axis=1).astype(np.float64)

        # interior point counts (use the same cloud the production filter uses)
        pts_path = os.path.join(data_root, train_pkl[i]['pts_path'])
        pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 6)[:, :3].astype(np.float64)
        npts = np.array([points_in_box_count(pts, det_up[j, :3], det_up[j, 3:6], det_up[j, 6])
                         for j in range(len(scores))], dtype=np.int64)

        # true old-class GT for this scene
        a = src[lidx].get('annos', {})
        g_up, g_lab = [], []
        if a.get('gt_num', 0):
            names = list(a['name'])
            ups = np.asarray(a['gt_boxes_upright_depth'], dtype=np.float64)
            for nm, up in zip(names, ups):
                nm = merges.get(nm, nm)
                g = name2gid.get(nm)
                if g is not None and g in old_gids:
                    g_up.append(up); g_lab.append(g)
        g_up = np.asarray(g_up, dtype=np.float64).reshape(-1, 7)
        g_lab = np.asarray(g_lab, dtype=np.int64)
        per_scene.append(dict(det_up=det_up, scores=scores, labels=labels, npts=npts,
                              g_up=g_up, g_lab=g_lab))

    # precompute IoU per scene (3D IoU via DepthInstance3DBoxes.overlaps)
    for s in per_scene:
        if len(s['scores']) and len(s['g_lab']):
            pb = DepthInstance3DBoxes(torch.tensor(s['det_up'], dtype=torch.float32), origin=(0.5, 0.5, 0.5))
            gb = DepthInstance3DBoxes(torch.tensor(s['g_up'], dtype=torch.float32), origin=(0.5, 0.5, 0.5))
            s['iou'] = DepthInstance3DBoxes.overlaps(pb, gb).numpy()  # [n_det, n_gt]
        else:
            s['iou'] = np.zeros((len(s['scores']), len(s['g_lab'])))

    n_gt_total = sum(len(s['g_lab']) for s in per_scene)
    print(f'  scenes={len(per_scene)}  true old-class GT boxes={n_gt_total}')

    # ---- threshold sweep: precision/recall/F1/#kept with min_pts filter (production-faithful) ----
    print(f'\n  THRESHOLD SWEEP (IoU>={IOU_THR}, min_pts>={MIN_PTS}, class-aware match):')
    print(f'  {"thr":>5} {"#kept":>7} {"TP":>6} {"FP":>6} {"prec":>6} {"recall":>7} {"F1":>6}')
    thrs = [float(x) for x in args.thrs.split(',')]
    for thr in thrs:
        TP = FP = 0
        for s in per_scene:
            keep = (s['scores'] >= thr) & (s['npts'] >= MIN_PTS)
            idx = np.where(keep)[0]
            if len(idx) == 0:
                continue
            order = idx[np.argsort(-s['scores'][idx])]
            gt_used = np.zeros(len(s['g_lab']), dtype=bool)
            for j in order:
                lab = s['labels'][j]
                cand = [k for k in range(len(s['g_lab'])) if s['g_lab'][k] == lab and not gt_used[k]]
                if cand:
                    ious = s['iou'][j, cand]
                    best = int(np.argmax(ious))
                    if ious[best] >= IOU_THR:
                        gt_used[cand[best]] = True
                        TP += 1
                        continue
                FP += 1
        kept = TP + FP
        prec = TP / kept if kept else 0.0
        rec = TP / n_gt_total if n_gt_total else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f'  {thr:>5.2f} {kept:>7} {TP:>6} {FP:>6} {prec:>6.3f} {rec:>7.3f} {f1:>6.3f}')

    # ---- teacher detection AP@0.25 on train (all dets, no score cut) per old class ----
    print(f'\n  TEACHER AP@{IOU_THR} on TRAIN (old classes, VOC all-point AP):')
    aps = {}
    for g in old_gids:
        # gather dets of this class across scenes, with scene idx
        d_score, d_scene, d_local = [], [], []
        npos = 0
        for si, s in enumerate(per_scene):
            npos += int((s['g_lab'] == g).sum())
            m = np.where(s['labels'] == g)[0]
            for j in m:
                d_score.append(s['scores'][j]); d_scene.append(si); d_local.append(j)
        if npos == 0:
            continue
        if len(d_score) == 0:
            aps[g] = 0.0; continue
        order = np.argsort(-np.asarray(d_score))
        tp = np.zeros(len(order)); fp = np.zeros(len(order))
        gt_used = {si: np.zeros(int((per_scene[si]['g_lab'] == g).sum()), dtype=bool) for si in set(d_scene)}
        # map: for each scene, indices of gt of this class
        gt_idx_map = {si: [k for k in range(len(per_scene[si]['g_lab'])) if per_scene[si]['g_lab'][k] == g]
                      for si in set(d_scene)}
        for r, oi in enumerate(order):
            si = d_scene[oi]; j = d_local[oi]
            gids_local = gt_idx_map[si]
            if gids_local:
                ious = per_scene[si]['iou'][j, gids_local]
                best = int(np.argmax(ious))
                if ious[best] >= IOU_THR and not gt_used[si][best]:
                    gt_used[si][best] = True; tp[r] = 1
                else:
                    fp[r] = 1
            else:
                fp[r] = 1
        tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
        rec = tp_c / npos
        prec = tp_c / np.maximum(tp_c + fp_c, 1e-9)
        # all-point interpolation
        mrec = np.concatenate([[0], rec, [1]]); mpre = np.concatenate([[0], prec, [0]])
        for k in range(len(mpre) - 2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k + 1])
        ii = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[ii + 1] - mrec[ii]) * mpre[ii + 1])
        aps[g] = ap
    for g in old_gids:
        if g in aps:
            print(f'    {gid2name[g]:>14} AP={aps[g]:.3f}')
    if aps:
        print(f'    {"mean(old)":>14} AP={np.mean(list(aps.values())):.3f}  ({len(aps)} classes w/ GT)')


if __name__ == '__main__':
    main()
