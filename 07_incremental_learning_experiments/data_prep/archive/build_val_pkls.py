#!/usr/bin/env python
"""Chunk 3: build per-stage VAL pkls (cumulative eval) for both orderings.

Class-incremental eval: at stage t the model knows classes from stages 1..t, so it is
evaluated on the union of val scenes from stages 1..t, with GT filtered to those
cumulative classes (global IDs from the train carve_report). Mirrors carve_stage_pkls.py.
"""
import os, json, pickle, collections
import numpy as np

INC = '/data3/ryan/tr3d/data/sunrgbd_incremental'
SRC = '/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_val_40class.pkl'
MANIFEST = os.path.join(INC, 'stage_manifest', 'stage_manifest.json')
SPLIT_KEY = {'3stage': 'SceneType-3stage-bestorder', '6stage': 'SceneType-6stage-bestorder'}

def filter_annos(annos, keep_names, name2gid):
    names = annos['name'] if annos.get('gt_num', 0) > 0 and 'name' in annos else np.array([], dtype='<U12')
    mask = np.array([str(n) in keep_names for n in names], dtype=bool)
    out = {'gt_num': int(mask.sum())}
    if out['gt_num'] == 0:
        return out
    for k in ('name','bbox','location','dimensions','rotation_y','gt_boxes_upright_depth'):
        out[k] = annos[k][mask]
    out['class'] = np.array([name2gid[str(x)] for x in out['name']], dtype=np.int64)
    out['index'] = np.arange(out['gt_num'], dtype=np.int32)
    return out

def main(which):
    manifest = json.load(open(MANIFEST))[SPLIT_KEY[which]]
    rep = json.load(open(os.path.join(INC, which, 'carve_report.json')))
    name2gid = rep['name2gid']
    stage_names = [s['stage'] for s in rep['stages']]
    stage_novel = {s['stage']: s['novel_classes'] for s in rep['stages']}

    src = pickle.load(open(SRC, 'rb'))
    by_idx = {int(e['point_cloud']['lidar_idx']): e for e in src}

    outdir = os.path.join(INC, which)
    report = {'ordering': which, 'mode': 'cumulative (classes seen through stage t)', 'stages': []}
    cum_classes = set()
    cum_stage_names = []
    for t, sname in enumerate(stage_names, 1):
        cum_classes |= set(stage_novel[sname])
        cum_stage_names.append(sname)
        keep = cum_classes
        val_recs = [r for r in manifest
                    if r['stage'] in cum_stage_names and r['split_membership'] == 'val']
        infos = []
        box_by_cls = collections.Counter()
        nonempty = missing = 0
        for r in val_recs:
            idx = r['mmdet3d_idx']
            if idx not in by_idx:
                missing += 1; continue
            e = by_idx[idx]
            new = dict(e)
            new['annos'] = filter_annos(e['annos'], keep, name2gid)
            if new['annos']['gt_num'] > 0:
                nonempty += 1
                for nm in new['annos']['name']:
                    box_by_cls[str(nm)] += 1
            infos.append(new)
        out_pkl = os.path.join(outdir, f'sunrgbd_infos_val_stage{t}.pkl')
        pickle.dump(infos, open(out_pkl, 'wb'))
        srep = {'stage_num': t, 'through_stage': sname, 'pkl': out_pkl,
                'cumulative_classes': sorted(keep, key=lambda c: name2gid[c]),
                'val_scenes': len(infos), 'scenes_with_boxes': nonempty,
                'scenes_empty': len(infos) - nonempty, 'scenes_missing_in_src': missing,
                'total_boxes': int(sum(box_by_cls.values())),
                'boxes_by_class': dict(box_by_cls)}
        report['stages'].append(srep)
        print(f'  val stage{t} (<= {sname}): classes={len(keep)} scenes={len(infos)} '
              f'nonempty={nonempty} boxes={sum(box_by_cls.values())}')
    json.dump(report, open(os.path.join(outdir, 'val_report.json'), 'w'), indent=1)
    print(f'[{which}] wrote {len(report["stages"])} val pkls + val_report.json')

if __name__ == '__main__':
    for w in ('3stage', '6stage'):
        print(f'=== {w} ===')
        main(w)
