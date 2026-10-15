"""Two-stage exporter parity with the original Design-2 bank and scorer."""
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys

import pytest

from test_unidet3d_memory_export import full_infos, REPO
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.collection import fixed_pool, context_digest, training_schedule
from ldmr_unidet3d.data import stage_records
from ldmr_unidet3d.scene import replay_records, merge_replay_records
from ldmr_unidet3d.selection import reference_scene_policy, collection_payload
from ldmr_unidet3d.sequence import input_hashes, sha256
from ldmr_unidet3d.trajectories import make_snapshot, write_snapshot
from ldmr_unidet3d.learning_dynamics import reference_functions
from mmdet3d.datasets.scene_memory_bank import SceneMemoryBank


def seal_collection(path, records, protocol, stage, bank, source, memory_mode="scene", reviewing=False):
    path.mkdir()
    if memory_mode == 'object':
        from ldmr_unidet3d.collection import object_source_pool
        pool = object_source_pool(records, bank.get_all_exemplars(), protocol, stage)
    else:
        pool = merge_replay_records(stage_records(records, protocol, stage),
            replay_records(bank, {r['sample_idx']:r for r in records}, protocol, stage))
    geometry, seats = fixed_pool(pool, protocol, stage)
    steps = [0,1,2] if reviewing else [0,1]
    context = dict(protocol_id=protocol.identity, stage=stage, expected_steps=steps,
        q_metric='f1', seats=seats, geometry=geometry, sources={str(Path(__file__).resolve()):sha256(__file__)}, inputs=input_hashes([source], [pool]))
    digest = context_digest(context)
    (path/'context.json').write_text(json.dumps(context))
    artifacts = {'context.json':sha256(path/'context.json')}
    checkpoints, metrics = {}, {}
    for step in steps:
        checkpoint = path/f'checkpoint_{step}'
        checkpoint.write_text(str(step))
        checkpoints[str(checkpoint)] = sha256(checkpoint)
        measured = []
        for seat in seats:
            visible = protocol.visible_classes(stage, None if seat['save_stage']==stage else seat['save_stage'])
            classes = {}
            for cid in visible:
                count = sum(i['bbox_label_3d']==cid for i in seat['instances'])
                tp = (count if step != 1 else 0) if reviewing else (count if step else 0)
                classes[cid] = dict(gt_count=count, tp=tp, tp_count=tp, fp=0, fn=count-tp,
                    dt_count=tp, denom=count, count=count, q=2*tp/(tp+count+1e-9) if count else None)
            measured.append(dict(scene_id=seat['scene_id'], save_stage=seat['save_stage'], classes=classes))
        metrics[step] = measured
        snap = make_snapshot(measured, protocol=protocol, stage=stage, step=step,
            checkpoint_sha256=sha256(checkpoint), measurement_sha256=digest, q_metric='f1')
        name = f'snapshot_{step}.json'
        artifacts[name] = write_snapshot(path/name, snap, protocol)
    (path/'complete.json').write_text(json.dumps(dict(status='complete', artifacts=artifacts,
        measurement_sha256=digest, steps=steps, checkpoints=checkpoints)))
    return reference_functions()['scoring']['compute_learning_dynamics_design1_scores'](
        metrics, class_ids=list(range(protocol.seen_count(stage))),
        new_classes=list(protocol.visible_classes(stage)), q_metric='f1', eps=1e-9, design_version=2)


def test_two_stage_export_matches_original_selector_including_ties(tmp_path):
    import numpy as np
    source = full_infos(tmp_path)
    records = pickle.loads(source.read_bytes())['data_list']
    p = ScanNet35Protocol()
    bank = SceneMemoryBank(max_memory_scenes=2, total_training_scenes=3, random_seed=200,
        debug_mode=False, stage_scene_counts={s:len(stage_records(records,p,s)) for s in range(1,6)},
        **reference_scene_policy())
    for stage in (1,2):
        collection = tmp_path/f'collection_{stage}'
        scores = seal_collection(collection, records, p, stage, bank, source)
        payload, _ = collection_payload(collection,p,stage,records,bank)
        assert payload == dict(scores,stage_id=stage)
        from copy import deepcopy
        wrong = deepcopy(records)
        wrong[0]['axis_align_matrix'][0][3] += 1
        with pytest.raises(ValueError, match='exact selection pool'):
            collection_payload(collection,p,stage,wrong,bank)
        natural = [dict(sample_idx=r['sample_idx'], annos=dict(gt_num=len(r['instances']),
            **{'class':np.array([p.raw_ids[i['bbox_label_3d']] for i in r['instances']])},
            gt_boxes_upright_depth=np.array([i['bbox_3d'] for i in r['instances']]))) for r in records]
        bank.add_stage_scenes(stage,natural,list(range(p.seen_count(stage))),
            dict(nyu40_to_model_idx=dict(zip(p.raw_ids,range(35))),valid_nyu40_ids=list(p.raw_ids)),
            learning_dynamics_design2_payload=dict(scores,stage_id=stage))
        output = tmp_path/f'bank_{stage}'
        cmd = [sys.executable,str(REPO/'adapters/tools/export_memory.py'),'--input',str(source),
            '--output-dir',str(output),'--stage',str(stage),'--memory-mode','scene',
            '--scene-budget','2','--ld-collection',str(collection)]
        if stage>1: cmd += ['--previous-bank',str(tmp_path/'bank_1/bank.pkl')]
        result = subprocess.run(cmd,capture_output=True,text=True,env=dict(os.environ,OMP_NUM_THREADS='2'),timeout=90)
        assert result.returncode == 0, result.stdout+result.stderr
        exported = SceneMemoryBank.load_checkpoint(output/'bank.pkl')
        identity = lambda b: [(r['scene_id'],r['save_stage']) for r in b.list_memory_entries()]
        assert identity(exported) == identity(bank)
        assert (output/'selection.json').is_file()


@pytest.mark.parametrize('schedule', [[0,2], [0,0,2], [0,3,2], [1,2], [False,2]])
def test_schedule(schedule):
    if schedule == [0,2] and type(schedule[0]) is int:
        assert training_schedule(schedule,2)==schedule
    else:
        with pytest.raises(ValueError): training_schedule(schedule,2)


@pytest.mark.parametrize('reviewing', [False, True])
def test_object_design2_export_matches_original_ranker(tmp_path, reviewing):
    import numpy as np
    from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank
    from ldmr_unidet3d.selection import reference_object_policy
    from ldmr_unidet3d.objects import load_crops, selected_payloads
    source = full_infos(tmp_path)
    records = pickle.loads(source.read_bytes())['data_list']
    protocol = ScanNet35Protocol()
    config = dict(exemplars_per_class=2, max_total_exemplars=4, min_points=1,
                  random_seed=200, **reference_object_policy())
    bank = ObjectMemoryBank(**config)
    points = {r['sample_idx']: np.fromfile(r['lidar_points']['lidar_path'], np.float32).reshape(-1,6)
              for r in records}
    for stage in (1, 2):
        collection = tmp_path / f'object_collection_{stage}'
        scores = seal_collection(collection, records, protocol, stage, bank, source, 'object', reviewing=reviewing)
        payload, evidence = collection_payload(collection, protocol, stage, records, bank, 'object', reviewing=reviewing)
        if evidence.get('reviewing_weights') is not None:
            assert all(w>1 for w in evidence['reviewing_weights'].values())
            bank.apply_source_seat_replay_weights(evidence['reviewing_weights'])
        assert payload == dict(scores, stage_id=stage)
        for cid in protocol.visible_classes(stage):
            candidates = [dict(scene_id=r['sample_idx'], object_idx=idx, bbox=i['bbox_3d'],
                               class_id=cid, nyu40_id=protocol.raw_ids[cid])
                          for r in records for idx, i in enumerate(r['instances']) if i['bbox_label_3d']==cid]
            bank.add_exemplars(cid, candidates, scene_points_dict=points, stage_id=stage,
                              learning_dynamics_design2_payload=dict(scores, stage_id=stage))
        output = tmp_path / f'object_bank_{stage}'
        command = [sys.executable, str(REPO/'adapters/tools/export_memory.py'), '--input', str(source),
            '--output-dir', str(output), '--stage', str(stage), '--memory-mode', 'object',
            '--objects-per-class', '2', '--max-objects', '4', '--min-points', '1',
            '--ld-collection', str(collection)]
        if reviewing:
            command += ['--reviewing']
        if stage > 1:
            command += ['--previous-bank', str(tmp_path/'object_bank_1/bank.pkl'),
                        '--previous-crops', str(tmp_path/'object_bank_1/crops.pkl')]
        result = subprocess.run(command, capture_output=True, text=True,
                                env=dict(os.environ, OMP_NUM_THREADS='2'), timeout=90)
        assert result.returncode == 0, result.stdout + result.stderr
        exported = ObjectMemoryBank.load_state(output/'bank.pkl', expected_config=config)
        identities = lambda b: [(o['scene_id'], o['object_idx'], o['class_id'], o['stage_id'])
                               for o in b.get_all_exemplars()]
        assert identities(exported) == identities(bank)
        assert [c.get('replay_weight',1.) for c in exported.get_all_exemplars()] == [c.get('replay_weight',1.) for c in bank.get_all_exemplars()]
        crops = load_crops(output/'crops.pkl', protocol, stage)
        retained = selected_payloads(exported, crops, protocol)
        assert len(retained) == 2 * stage
        assert [c['replay_weight'] for c in retained] == [c['replay_weight'] for c in crops]
        assert (output/'selection.json').is_file()


def test_object_carrier_pool_preserves_distinct_stages_and_excludes_future_gt(tmp_path):
    from ldmr_unidet3d.collection import object_source_pool
    from copy import deepcopy
    p = ScanNet35Protocol()
    records = pickle.loads(full_infos(tmp_path).read_bytes())['data_list']
    objects = [dict(scene_id='s0', stage_id=stage, object_idx=idx,
                    class_id=records[0]['instances'][idx]['bbox_label_3d'],
                    bbox=records[0]['instances'][idx]['bbox_3d']) for stage, idx in ((1,0),(2,1))]
    geometry, seats = fixed_pool(object_source_pool(records, objects + objects, p, 3), p, 3)
    assert len(geometry) == 1
    assert [(s['scene_id'],s['save_stage']) for s in seats] == [('s0',1),('s0',2)]
    assert all(i['bbox_label_3d'] in p.visible_classes(3,s['save_stage'])
               for s in seats for i in s['instances'])
    wrong = deepcopy(objects)
    wrong[0]['bbox'][0] += 10
    with pytest.raises(ValueError, match='annotation differs'):
        object_source_pool(records, wrong, p, 3)
