"""CPU audit of actual placement success, hidden-object overlap and crop support.

Uses existing bank crops and production placement; full GT is used ONLY to audit
overlap after placement, never to select a training placement or train a model.
"""
import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
from mmdet3d.datasets.pipelines.exemplar_insertion import InsertExemplarObjects


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bank', type=Path)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=301)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    random.seed(args.seed)
    rng = np.random.RandomState(args.seed)
    with args.bank.open('rb') as f:
        bank = pickle.load(f)
    with open('data/sunrgbd/sunrgbd_infos_train_40class.pkl', 'rb') as f:
        infos = pickle.load(f)
    first_new = (args.stage - 1) * 8
    last_new = args.stage * 8
    candidates = []
    for info in infos:
        labels = np.asarray(info['annos'].get('class', []))
        if np.any((labels >= first_new) & (labels < last_new)):
            candidates.append(info)
    crops = [crop for cls, rows in bank['exemplars'].items()
             if int(cls) < first_new for crop in rows]
    if not crops or not candidates:
        raise ValueError('No eligible old crops or natural scenes')
    indices = rng.choice(len(candidates), min(args.samples, len(candidates)), replace=False)
    transform = InsertExemplarObjects(
        max_exemplars_per_scene=1, insertion_probability=1.,
        collision_threshold=0., placement_jitter=0., max_placement_attempts=20,
        floor_offset=0., floor_percentile=1., preserve_source_height=True)
    rows = []
    for idx in indices:
        info = candidates[int(idx)]
        labels = np.asarray(info['annos']['class'])
        boxes = np.asarray(info['annos']['gt_boxes_upright_depth'], dtype=np.float32)
        current = (labels >= first_new) & (labels < last_new)
        points = np.fromfile(Path('data/sunrgbd') / info['pts_path'], dtype=np.float32).reshape(-1, 6)
        crop = random.choice(crops)
        pasted, box = transform._find_valid_placement(
            np.asarray(crop['points']), np.asarray(crop['bbox']), boxes[current], points,
            source_floor_offset=float(crop.get('source_floor_offset', 0.)))
        row = dict(scene_id=str(info['point_cloud']['lidar_idx']),
                   class_id=int(crop['class_id']), source_scene=str(crop['scene_id']),
                   crop_points=int(len(crop['points'])), scene_points=int(len(points)),
                   old_objects=int(np.sum(labels < first_new)), accepted=pasted is not None)
        if pasted is not None:
            gravity_box = box.copy()
            gravity_box[2] += gravity_box[5] / 2
            row['overlaps_hidden_old_aabb'] = bool(transform._check_collision(
                gravity_box, boxes[labels < first_new]))
            # PointSample randomly retains 100k points; model the identical uniform
            # draw by tagging indices rather than copying the full cloud.
            total = len(points) + len(pasted)
            choice = rng.choice(total, 100000, replace=total < 100000)
            retained = pasted[choice[choice >= len(points)] - len(points), :3]
            row['retained_crop_points'] = int(len(retained))
            row['crop_voxels_1cm'] = int(len(np.unique(
                np.floor(retained / .01).astype(np.int64), axis=0)))
        rows.append(row)
    accepted = [r for r in rows if r['accepted']]
    def quantiles(key):
        vals = [r[key] for r in accepted]
        return dict(zip(['min', 'p10', 'median', 'p90', 'max'],
                        np.percentile(vals, [0, 10, 50, 90, 100]).tolist())) if vals else {}
    summary = dict(stage=args.stage, seed=args.seed, bank=str(args.bank.resolve()),
                   attempted=len(rows), accepted=len(accepted),
                   acceptance_rate=len(accepted) / len(rows),
                   hidden_old_overlap_count=sum(r['overlaps_hidden_old_aabb'] for r in accepted),
                   retained_points=quantiles('retained_crop_points'),
                   crop_voxels=quantiles('crop_voxels_1cm'),
                   accepted_below_20_points=sum(r['retained_crop_points'] < 20 for r in accepted),
                   accepted_below_6_voxels=sum(r['crop_voxels_1cm'] < 6 for r in accepted),
                   note='No-pseudo placement audit. AABB overlap is a conservative proxy; '
                        'crop voxels precede augmentation and are not assigned detector positives.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(summary=summary, samples=rows), indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
