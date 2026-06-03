#!/usr/bin/env python
"""Chunk 2/3 of Step B: carve per-stage TRAIN pkls from the 40-class source.

Source labels : /data3/sunrgbd/sunrgbd_40/sunrgbd_infos_train_40class.pkl  (decided 2026-05-31)
Scene pools   : stage_manifest.json (chunk 1), split_membership=='train'
Per-stage GT  : NOVEL-class-only (true class-incremental). annos filtered to the stage's
                box-classes; `class` remapped to a GLOBAL id (introduction order); index re-enumerated.
Exact-name match only — fine-grained variants (sofa_chair, coffee_table, dining_table, side_table)
are NOT merged into chair/sofa/table.

This script does ONE ordering at a time (arg: 3stage|6stage). Chunk 2 = 3stage.
"""
import os, sys, json, pickle, collections
import numpy as np

INC = '/data3/ryan/tr3d/data/sunrgbd_incremental'
SRC = '/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_train_40class.pkl'
MANIFEST = os.path.join(INC, 'stage_manifest', 'stage_manifest.json')

NO_BOX = {'window', 'picture', 'bag', 'light', 'counter'}

# stage class-lists copied verbatim from split_stats_v2_*.csv 'classes' column (order preserved),
# NO_BOX classes are dropped below.
STAGES = {
    '3stage': {
        'split_key': 'SceneType-3stage-bestorder',
        'stages': [
            ('Stage 1 – Classroom', ['chair','window','counter','sofa','desk','table','book','door']),
            ('Stage 2 – Office',     ['keyboard','monitor','paper','cup','cabinet','box','shelf','light']),
            ('Stage 3 – Bedroom',    ['bed','pillow','curtain','lamp','picture','mirror','bag','bottle']),
        ],
    },
    '6stage': {
        'split_key': 'SceneType-6stage-bestorder',
        'stages': [
            ('Stage 1 – Dining Area',  ['table','counter','window','chair']),
            ('Stage 2 – Classroom',    ['bottle','light','door','bag']),
            ('Stage 3 – Library',      ['pillow','shelf','desk','book']),
            ('Stage 4 – Bedroom',      ['bed','curtain','mirror','lamp']),
            ('Stage 5 – Living Room',  ['sofa','cabinet','box','picture']),
            ('Stage 6 – Office',       ['keyboard','monitor','paper','cup']),
        ],
    },
}

def filter_annos(annos, keep_names, name2gid):
    """Return a new annos dict with only boxes whose name is in keep_names.
    `class` -> global id via name2gid; index re-enumerated. None of the source arrays mutated."""
    if annos.get('gt_num', 0) == 0 or 'name' not in annos:
        names = np.array([], dtype='<U12')
    else:
        names = annos['name']
    mask = np.array([str(n) in keep_names for n in names], dtype=bool)
    out = {}
    n = int(mask.sum())
    out['gt_num'] = n
    if n == 0:
        return out  # empty scene: just gt_num=0 (matches source convention for no-box scenes)
    for k in ('name','bbox','location','dimensions','rotation_y','gt_boxes_upright_depth'):
        out[k] = annos[k][mask]
    out['class'] = np.array([name2gid[str(x)] for x in out['name']], dtype=np.int64)
    out['index'] = np.arange(n, dtype=np.int32)
    return out

def main(which):
    cfg = STAGES[which]
    manifest = json.load(open(MANIFEST))[cfg['split_key']]
    src = pickle.load(open(SRC, 'rb'))
    by_idx = {int(e['point_cloud']['lidar_idx']): e for e in src}

    # global class order = introduction order across stages, NO_BOX dropped
    global_order, per_stage_box = [], []
    for stage_name, cls in cfg['stages']:
        box_cls = [c for c in cls if c not in NO_BOX]
        per_stage_box.append((stage_name, box_cls))
        global_order.extend(box_cls)
    assert len(global_order) == len(set(global_order)), 'class appears in >1 stage!'
    name2gid = {c: i for i, c in enumerate(global_order)}
    print(f'[{which}] global class order ({len(global_order)}): {global_order}')

    outdir = os.path.join(INC, which)
    os.makedirs(outdir, exist_ok=True)

    report = {'ordering': which, 'global_class_order': global_order, 'name2gid': name2gid, 'stages': []}
    for si, (stage_name, box_cls) in enumerate(per_stage_box, 1):
        keep = set(box_cls)
        scene_recs = [r for r in manifest if r['stage'] == stage_name and r['split_membership'] == 'train']
        infos = []
        box_by_cls = collections.Counter()
        nonempty = 0
        missing = 0
        for r in scene_recs:
            idx = r['mmdet3d_idx']
            if idx not in by_idx:
                missing += 1; continue
            e = by_idx[idx]
            new = dict(e)  # shallow copy; replace annos
            new['annos'] = filter_annos(e['annos'], keep, name2gid)
            if new['annos']['gt_num'] > 0:
                nonempty += 1
                for nm in new['annos']['name']:
                    box_by_cls[str(nm)] += 1
            infos.append(new)
        out_pkl = os.path.join(outdir, f'sunrgbd_infos_train_stage{si}.pkl')
        pickle.dump(infos, open(out_pkl, 'wb'))
        gids = [name2gid[c] for c in box_cls]
        srep = {'stage': stage_name, 'stage_num': si, 'pkl': out_pkl,
                'novel_classes': box_cls, 'novel_class_gids': gids,
                'scenes_assigned': len(scene_recs), 'scenes_written': len(infos),
                'scenes_missing_in_src': missing,
                'scenes_with_boxes': nonempty, 'scenes_empty': len(infos) - nonempty,
                'total_boxes': int(sum(box_by_cls.values())),
                'boxes_by_class': dict(box_by_cls)}
        report['stages'].append(srep)
        print(f'  stage{si} {stage_name}: assigned={len(scene_recs)} written={len(infos)} '
              f'nonempty={nonempty} empty={len(infos)-nonempty} boxes={sum(box_by_cls.values())}')
        for c in box_cls:
            print(f'      {c:12s} gid={name2gid[c]:2d}  boxes={box_by_cls.get(c,0)}')

    json.dump(report, open(os.path.join(outdir, 'carve_report.json'), 'w'), indent=1)
    print(f'[{which}] wrote {len(report["stages"])} stage pkls + carve_report.json to {outdir}')

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '3stage')
