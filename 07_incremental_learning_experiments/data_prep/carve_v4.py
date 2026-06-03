#!/usr/bin/env python
"""v4 carve: per-stage TRAIN (novel-only) + cumulative VAL pkls from the v4 split.

Consumes the v4 builder outputs directly:
  v4_3dbox/stage_def_<sched>.json      -> ordered stages, classes, global_ids, merges
  v4_3dbox/stage_manifest_<sched>.json -> train[stage]=lidar_idx list (strict-filtered),
                                          val[stage]=lidar_idx list (scenes of that scene_type)

Design (same contract as v2 carve_stage_pkls.py / build_val_pkls.py):
  TRAIN per stage = NOVEL-class-only GT (true class-incremental); annos filtered to the
    stage's classes; `class` -> GLOBAL id (introduction order); index re-enumerated.
  VAL per stage   = CUMULATIVE: union of val scenes of stages 1..t, GT filtered to the
    cumulative seen classes.
  MERGES (cpu->computer, sofa_chair->sofa) from stage_def are applied to source box names
    before matching/remapping, and the merged name is stored in annos['name'].

Output dir: tr3d/data/sunrgbd_incremental/v4_3dbox/<sched>/  (points/ resolves via data_root).
v2 artifacts (3stage/, 6stage/) are left untouched.

Usage: python carve_v4.py <3stage|6stage>
"""
import os, sys, json, pickle, collections
import numpy as np

V4 = '/data3/ryan/incremental_learning/v4_3dbox'
OUT_ROOT = '/data3/ryan/tr3d/data/sunrgbd_incremental/v4_3dbox'
TRAIN_SRC = '/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_train_40class.pkl'
VAL_SRC = '/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_val_40class.pkl'


def filter_annos(annos, keep_names, name2gid, merge):
    """Keep boxes whose MERGED name is in keep_names; remap class->global id;
    store merged name. No source array is mutated."""
    if annos.get('gt_num', 0) == 0 or 'name' not in annos:
        names = np.array([], dtype='<U16')
    else:
        names = annos['name']
    merged = [merge.get(str(n), str(n)) for n in names]
    mask = np.array([m in keep_names for m in merged], dtype=bool)
    out = {'gt_num': int(mask.sum())}
    if out['gt_num'] == 0:
        return out
    for k in ('bbox', 'location', 'dimensions', 'rotation_y', 'gt_boxes_upright_depth'):
        out[k] = annos[k][mask]
    kept_names = [m for m, keep in zip(merged, mask) if keep]
    out['name'] = np.array(kept_names, dtype='<U16')
    out['class'] = np.array([name2gid[m] for m in kept_names], dtype=np.int64)
    out['index'] = np.arange(out['gt_num'], dtype=np.int32)
    return out


def carve(scene_idxs, by_idx, keep, name2gid, merge):
    infos, box_by_cls, nonempty, missing = [], collections.Counter(), 0, 0
    for idx in scene_idxs:
        if idx not in by_idx:
            missing += 1; continue
        e = by_idx[idx]
        new = dict(e)
        new['annos'] = filter_annos(e['annos'], keep, name2gid, merge)
        if new['annos']['gt_num'] > 0:
            nonempty += 1
            for nm in new['annos']['name']:
                box_by_cls[str(nm)] += 1
        infos.append(new)
    return infos, box_by_cls, nonempty, missing


def main(sched):
    sdef = json.load(open(os.path.join(V4, f'stage_def_{sched}.json')))
    manifest = json.load(open(os.path.join(V4, f'stage_manifest_{sched}.json')))
    merge = sdef['merges']
    stages = sdef['stages']  # ordered
    # global order + name2gid straight from stage_def
    global_order = [c for s in stages for c in s['classes']]
    name2gid = {}
    for s in stages:
        for c, g in zip(s['classes'], s['global_ids']):
            name2gid[c] = g
    assert global_order == sorted(global_order, key=lambda c: name2gid[c])
    print(f'[{sched}] {len(global_order)} classes, order = {global_order}')

    train_src = pickle.load(open(TRAIN_SRC, 'rb'))
    val_src = pickle.load(open(VAL_SRC, 'rb'))
    tr_by_idx = {int(e['point_cloud']['lidar_idx']): e for e in train_src}
    va_by_idx = {int(e['point_cloud']['lidar_idx']): e for e in val_src}

    outdir = os.path.join(OUT_ROOT, sched)
    os.makedirs(outdir, exist_ok=True)
    report = {'sched': sched, 'global_class_order': global_order,
              'name2gid': name2gid, 'merges': merge, 'train_stages': [], 'val_stages': []}

    # ---- TRAIN: novel-only per stage ----
    for s in stages:
        t = s['stage']
        keep = set(s['classes'])
        idxs = manifest['train'][str(t)]
        infos, bbc, nonempty, missing = carve(idxs, tr_by_idx, keep, name2gid, merge)
        out_pkl = os.path.join(outdir, f'sunrgbd_infos_train_stage{t}.pkl')
        pickle.dump(infos, open(out_pkl, 'wb'))
        report['train_stages'].append({
            'stage': t, 'scene_type': s['scene_type'], 'pkl': out_pkl,
            'novel_classes': s['classes'], 'novel_class_gids': s['global_ids'],
            'scenes_assigned': len(idxs), 'scenes_written': len(infos),
            'scenes_missing_in_src': missing, 'scenes_with_boxes': nonempty,
            'scenes_empty': len(infos) - nonempty,
            'total_boxes': int(sum(bbc.values())), 'boxes_by_class': dict(bbc)})
        print(f'  TRAIN stage{t} [{s["scene_type"]}] written={len(infos)} '
              f'nonempty={nonempty} boxes={sum(bbc.values())} missing={missing}')
        for c in s['classes']:
            print(f'      {c:14s} gid={name2gid[c]:2d} boxes={bbc.get(c,0)}')

    # ---- VAL: cumulative ----
    cum_classes, cum_idxs = set(), []
    for s in stages:
        t = s['stage']
        cum_classes |= set(s['classes'])
        cum_idxs = cum_idxs + manifest['val'][str(t)]
        infos, bbc, nonempty, missing = carve(cum_idxs, va_by_idx, cum_classes, name2gid, merge)
        out_pkl = os.path.join(outdir, f'sunrgbd_infos_val_stage{t}.pkl')
        pickle.dump(infos, open(out_pkl, 'wb'))
        report['val_stages'].append({
            'stage': t, 'through_scene_type': s['scene_type'], 'pkl': out_pkl,
            'cumulative_classes': sorted(cum_classes, key=lambda c: name2gid[c]),
            'val_scenes': len(infos), 'scenes_with_boxes': nonempty,
            'scenes_empty': len(infos) - nonempty, 'scenes_missing_in_src': missing,
            'total_boxes': int(sum(bbc.values())), 'boxes_by_class': dict(bbc)})
        print(f'  VAL   stage{t} (<= stage{t}): classes={len(cum_classes)} '
              f'scenes={len(infos)} nonempty={nonempty} boxes={sum(bbc.values())}')

    json.dump(report, open(os.path.join(outdir, 'carve_report_v4.json'), 'w'), indent=1)
    print(f'[{sched}] wrote {len(stages)} train + {len(stages)} val pkls + carve_report_v4.json -> {outdir}')


if __name__ == '__main__':
    for w in ([sys.argv[1]] if len(sys.argv) > 1 else ['3stage', '6stage']):
        print(f'=== {w} ===')
        main(w)
