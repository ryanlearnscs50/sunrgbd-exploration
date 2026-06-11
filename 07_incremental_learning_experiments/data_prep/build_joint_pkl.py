#!/usr/bin/env python
"""Fully-supervised (joint) upper-bound TRAIN pkl for the v4 incremental setup.

Same train-scene pool as the incremental experiment (union of every stage's
strict-filtered train pool), but with the FULL cumulative vocabulary labelled in
every scene -- i.e. the model sees all classes jointly instead of one stage at a
time. Evaluated later on the existing final cumulative val pkl
(sunrgbd_infos_val_stage{N}.pkl), so the ONLY difference vs the incremental runs
is joint-vs-sequential label exposure on identical scenes.

Output: tr3d/data/sunrgbd_incremental/v4_3dbox/<sched>/sunrgbd_infos_train_joint.pkl

Usage: python build_joint_pkl.py <3stage|6stage>
"""
import os, sys, json, pickle, collections

# reuse the exact carve / filter logic the incremental pkls were built with
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carve_v4 import filter_annos, carve, V4, OUT_ROOT, TRAIN_SRC


def main(sched):
    sdef = json.load(open(os.path.join(V4, f'stage_def_{sched}.json')))
    manifest = json.load(open(os.path.join(V4, f'stage_manifest_{sched}.json')))
    merge = sdef['merges']
    stages = sdef['stages']
    global_order = [c for s in stages for c in s['classes']]
    name2gid = {}
    for s in stages:
        for c, g in zip(s['classes'], s['global_ids']):
            name2gid[c] = g
    keep = set(global_order)
    print(f'[{sched}] {len(global_order)} classes (joint), order = {global_order}')

    train_src = pickle.load(open(TRAIN_SRC, 'rb'))
    tr_by_idx = {int(e['point_cloud']['lidar_idx']): e for e in train_src}

    # union of every stage's train pool, de-duplicated, order-stable
    seen, union = set(), []
    per_stage = {}
    for s in stages:
        idxs = manifest['train'][str(s['stage'])]
        per_stage[s['stage']] = len(idxs)
        for i in idxs:
            if i not in seen:
                seen.add(i); union.append(i)
    print(f'  per-stage train pool sizes: {per_stage}; union (dedup) = {len(union)} scenes')

    infos, bbc, nonempty, missing = carve(union, tr_by_idx, keep, name2gid, merge)
    outdir = os.path.join(OUT_ROOT, sched)
    out_pkl = os.path.join(outdir, 'sunrgbd_infos_train_joint.pkl')
    pickle.dump(infos, open(out_pkl, 'wb'))

    report = {'sched': sched, 'mode': 'joint_fully_supervised',
              'global_class_order': global_order, 'name2gid': name2gid,
              'union_scenes': len(union), 'scenes_written': len(infos),
              'scenes_with_boxes': nonempty, 'scenes_empty': len(infos) - nonempty,
              'scenes_missing_in_src': missing,
              'total_boxes': int(sum(bbc.values())),
              'boxes_by_class': {c: bbc.get(c, 0) for c in global_order}}
    json.dump(report, open(os.path.join(outdir, 'joint_build_report.json'), 'w'), indent=1)
    print(f'  WROTE {out_pkl}')
    print(f'  scenes={len(infos)} nonempty={nonempty} empty={len(infos)-nonempty} '
          f'missing={missing} total_boxes={sum(bbc.values())}')
    for c in global_order:
        print(f'      {c:14s} gid={name2gid[c]:2d} boxes={bbc.get(c,0)}')


if __name__ == '__main__':
    for w in ([sys.argv[1]] if len(sys.argv) > 1 else ['3stage', '6stage']):
        print(f'=== {w} ===')
        main(w)
