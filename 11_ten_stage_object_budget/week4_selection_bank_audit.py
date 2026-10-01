"""CPU-only full-pool point-count selection audit, without detector training."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT / 'repo'
OUT = ROOT / 'week4_runs/selection_bank_audit'
os.chdir(REPO)
sys.path.insert(0, str(REPO))
import numpy as np
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank
from tools.train_incremental_scene import prepare_stage_config


def main():
    OUT.mkdir(exist_ok=False)
    started = time.time()
    cfg = Config.fromfile('configs/incremental/sunrgbd/week4_s10_object20_largest_pseudo.py')
    settings = dict(cfg.object_memory_config)
    settings.pop('insertion')
    bank = ObjectMemoryBank(**settings, random_seed=200)
    stage = prepare_stage_config(cfg.base_config, cfg.stage_definitions[0], 0,
                                 cfg.stage_definitions, str(OUT), incremental_cfg=cfg)
    data = stage.data.train
    while 'dataset' in data:
        data = data.dataset
    data.stage_definition = cfg.stage_definitions[0]
    data.all_stage_definitions = cfg.stage_definitions
    data.object_memory_bank = bank
    data.scene_memory_bank = None
    data.evaluation_mode = False
    dataset = build_dataset(data)
    # Ground-truth crop selection is model independent. Audit the production
    # population method for each stage against a separate exhaustive ranking.
    for definition in cfg.stage_definitions:
        dataset.stage_id = definition['stage_id']
        dataset.stage_classes = definition['class_indices']
        dataset.update_memory_bank_from_stage()
        bank.save_state(str(OUT / f'object_memory_bank_stage_{dataset.stage_id}.pkl'),
                        stage_id=dataset.stage_id)
        print('populated stage', dataset.stage_id, flush=True)
    populations = {i: [] for i in range(40)}
    for info in dataset.original_data_infos:
        sid = dataset._extract_scene_id(info)
        scene = dataset._load_scene_points(sid)
        if scene is None:
            continue
        ann = info.get('annos', {})
        for index, (cid, box) in enumerate(zip(ann.get('class', []), ann.get('gt_boxes_upright_depth', []))):
            cid = int(cid)
            if cid not in populations:
                continue
            crop = bank.extract_object_points(scene, box, crop_margin=bank.crop_margin)
            if len(crop) >= bank.min_points:
                populations[cid].append(dict(scene_id=sid, object_idx=index, point_count=len(crop)))
    rows = []
    for cid in range(40):
        expected = sorted(populations[cid], key=lambda x: -x['point_count'])[:20]
        actual = bank.exemplars[cid]
        key = lambda x: (str(x['scene_id']), x['object_idx'], x['point_count'])
        assert list(map(key, actual)) == list(map(key, expected)), cid
        rows.append(dict(class_id=cid, name=dataset.CLASSES[cid],
                         valid_candidates=len(populations[cid]), selected=len(actual),
                         points=sum(x['point_count'] for x in actual),
                         unique_scenes=len({x['scene_id'] for x in actual}),
                         minimum_points=min(x['point_count'] for x in actual)))
    report = dict(status='passed', scope='Offline GT bank construction; no detector accuracy measurement',
                  strategy='largest_point_count', seed_independent=True, classes=rows,
                  objects=bank.get_total_exemplar_count(), points=sum(r['points'] for r in rows),
                  pickle_bytes=(OUT / 'object_memory_bank_stage_10.pkl').stat().st_size,
                  seconds=time.time()-started,
                  source_sha256=hashlib.sha256((REPO / 'mmdet3d/datasets/incremental_sunrgbd.py').read_bytes()).hexdigest())
    (OUT / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'classes'}), flush=True)


if __name__ == '__main__':
    main()
