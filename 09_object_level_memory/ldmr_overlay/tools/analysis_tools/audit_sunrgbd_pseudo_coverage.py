"""Offline diagnostic of retained pseudo-label precision/recall against train GT.

Builds the production dataset to apply its exact pseudo merge/filter path. Uses
old GT only for measurement; no labels or thresholds are changed for training.
Reported coverage uses greedy same-class rotated 3D IoU matching, not AP.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from mmcv import Config
from shapely.geometry import MultiPoint
from mmdet3d.core.bbox import DepthInstance3DBoxes
from mmdet3d.datasets import build_dataset


def geometry(boxes):
    if len(boxes) == 0:
        return []
    corners = DepthInstance3DBoxes(
        np.array(boxes, dtype=np.float32, copy=True), box_dim=7,
        origin=(.5, .5, .5)).corners.numpy()
    return [(MultiPoint(c[:, :2]).convex_hull, float(c[:, 2].min()),
             float(c[:, 2].max())) for c in corners]


def iou(a, b):
    z = max(0., min(a[2], b[2]) - max(a[1], b[1]))
    if z == 0 or not a[0].intersects(b[0]):
        return 0.
    intersection = a[0].intersection(b[0]).area * z
    union = a[0].area * (a[2] - a[1]) + b[0].area * (b[2] - b[1]) - intersection
    return intersection / union if union > 0 else 0.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pseudo_file', type=Path)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Analytical checks protect the standalone diagnostic's geometry.
    example = np.array([[0, 0, 0, 2, 2, 2, 0], [1, 0, 0, 2, 2, 2, 0],
                        [9, 0, 0, 2, 2, 2, .4]], dtype=np.float32)
    g = geometry(example)
    assert np.isclose(iou(g[0], g[0]), 1.)
    assert np.isclose(iou(g[0], g[1]), 1/3)
    assert iou(g[0], g[2]) == 0.
    cfg = Config.fromfile('configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_pseudo_only_matched_52211.py')
    ds_cfg = cfg.data.train.dataset.copy()
    ds_cfg.update(stage_definition=cfg.stage_definitions[args.stage - 1],
                  all_stage_definitions=cfg.stage_definitions, evaluation_mode=False,
                  use_pseudo_labels=True, pseudo_label_config=dict(
                      cfg.pseudo_label_config, pregenerated_file=str(args.pseudo_file.resolve())),
                  scene_memory_bank=None, object_memory_bank=None)
    ds = build_dataset(ds_cfg)
    original = {str(x['point_cloud']['lidar_idx']): x for x in ds.original_data_infos}
    boundary = (args.stage - 1) * 8
    counts = {cls: dict(gt=0, pseudo=0, tp25=0, tp50=0) for cls in range(boundary)}
    for info in ds.data_infos:
        key = str(info['point_cloud']['lidar_idx'])
        gt = original[key]['annos']
        gl = np.asarray(gt['class'])
        keep_gt = gl < boundary
        gb = np.asarray(gt['gt_boxes_upright_depth'])[keep_gt]
        gl = gl[keep_gt]
        pseudo = info['annos']
        pl = np.asarray(pseudo['class'])
        keep_p = pl < boundary
        pb = np.asarray(pseudo['gt_boxes_upright_depth'])[keep_p]
        pl = pl[keep_p]
        gg, pg = geometry(gb), geometry(pb)
        pairs = sorted([(iou(pg[p], gg[g]), p, g) for p in range(len(pl))
                        for g in range(len(gl)) if pl[p] == gl[g]], reverse=True)
        for cls in counts:
            counts[cls]['gt'] += int(np.sum(gl == cls))
            counts[cls]['pseudo'] += int(np.sum(pl == cls))
        for threshold, metric in ((.25, 'tp25'), (.50, 'tp50')):
            used_p, used_g = set(), set()
            for overlap, p, g in pairs:
                if overlap < threshold:
                    break
                if p not in used_p and g not in used_g:
                    counts[int(pl[p])][metric] += 1
                    used_p.add(p)
                    used_g.add(g)
    def summarize(c):
        return dict(c, precision25=c['tp25']/c['pseudo'] if c['pseudo'] else 0.,
                    recall25=c['tp25']/c['gt'] if c['gt'] else 0.,
                    precision50=c['tp50']/c['pseudo'] if c['pseudo'] else 0.,
                    recall50=c['tp50']/c['gt'] if c['gt'] else 0.)
    totals = {k: sum(c[k] for c in counts.values()) for k in ('gt', 'pseudo', 'tp25', 'tp50')}
    report = dict(stage=args.stage, pseudo_file=str(args.pseudo_file.resolve()),
                  scenes=len(ds.data_infos), injected_scenes=ds.pseudo_injected_scene_count,
                  totals=summarize(totals), per_class={k: summarize(c) for k,c in counts.items()},
                  method='Production pseudo filtering; same-class greedy rotated 3D IoU '
                         'matching on training annotations. Diagnostic only; not validation AP.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(stage=args.stage, totals=report['totals']), indent=2))


if __name__ == '__main__':
    main()
