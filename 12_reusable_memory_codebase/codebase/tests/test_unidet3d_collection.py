"""Collector schedules and fixed visibility prevent mismatched LD measurements."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.collection import checkpoint_schedule, fixed_pool, validate_checkpoint_epoch, context_digest

P = ScanNet35Protocol()


@pytest.mark.parametrize('change', [None, 'seed', 'sources', 'seats', 'expected_steps', 'checkpoint', 'artifact'])
def test_reuse_preserves_only_verified_same_context_observations(tmp_path, change):
    import json
    from ldmr_unidet3d.collection import reusable_prefix
    from ldmr_unidet3d.trajectories import make_snapshot, write_snapshot
    from ldmr_unidet3d.sequence import sha256
    context = dict(expected_steps=[0, 1], seed=200, sources={'code': 'a'}, seats=['fixed'])
    artifacts, checkpoints, snapshots = {}, {}, []
    rows = [dict(scene_id='memory', save_stage=1, classes={cid: dict(
        tp=0, fp=0, fn=0, denom=0, gt_count=0, tp_count=0, dt_count=0,
        count=0, q=None) for cid in range(7)})]
    for epoch in (0, 1):
        checkpoint = tmp_path / f'epoch_{epoch}.pth'
        checkpoint.write_text(str(epoch))
        checkpoints[str(checkpoint)] = sha256(checkpoint)
        snapshot = make_snapshot(rows, protocol=P, stage=2, step=epoch,
            checkpoint_sha256=sha256(checkpoint), measurement_sha256=context_digest(context))
        snapshots.append(snapshot)
        name = f'snapshot_{epoch}.json'
        artifacts[name] = write_snapshot(tmp_path / name, snapshot, P)
    (tmp_path / 'context.json').write_text(json.dumps(context))
    artifacts['context.json'] = sha256(tmp_path / 'context.json')
    (tmp_path / 'complete.json').write_text(json.dumps(dict(status='complete', steps=[0, 1],
        measurement_sha256=context_digest(context), artifacts=artifacts, checkpoints=checkpoints)))
    extended = dict(context, expected_steps=[0, 1, 2])
    checkpoints[str(tmp_path / 'epoch_2.pth')] = 'c'*64
    if change in ('seed', 'sources', 'seats', 'expected_steps'):
        extended[change] = 'changed'
    elif change == 'checkpoint':
        checkpoints[str(tmp_path / 'epoch_0.pth')] = 'd'*64
    elif change == 'artifact':
        (tmp_path / 'snapshot_0.json').write_text('{}')
    if change:
        with pytest.raises(ValueError):
            reusable_prefix(tmp_path, extended, checkpoints, P)
    else:
        reused, receipt = reusable_prefix(tmp_path, extended, checkpoints, P)
        # JSON canonicalizes class keys. Compare serialized observations.
        assert json.dumps(reused[0]['seats'], sort_keys=True) == json.dumps(snapshots[0]['seats'], sort_keys=True)
        assert str(tmp_path / 'complete.json') in receipt['hashes']


def record():
    return dict(sample_idx='scene', lidar_points=dict(lidar_path='/points'),
        super_pts_path='/superpoints', axis_align_matrix=[[1]],
        ldmr_protocol_id=P.identity, ldmr_semantic_space='nyu40',
        instances=[dict(bbox_label_3d=0, bbox_3d=[0,0,0,1,1,1]),
                   dict(bbox_label_3d=7, bbox_3d=[0,0,0,2,2,2])],
        is_replay=True, ldmr_replay_from_stage=1, ldmr_include_current=True)


def test_merged_pool_splits_visibility_and_removes_inference_annotations():
    row = record(); saved = deepcopy(row)
    geometry, seats = fixed_pool([row], P, 2)
    assert row == saved
    assert len(geometry) == 1 and 'instances' not in geometry[0]
    assert 'is_replay' not in geometry[0]
    assert [r['save_stage'] for r in seats] == [1, 2]
    assert [[i['bbox_label_3d'] for i in s['instances']] for s in seats] == [[0], [7]]


@pytest.mark.parametrize('ids', [[], ['missing'], ['scene','scene'], [True], 'scene'])
def test_pool_rejects_bad_requests(ids):
    with pytest.raises(ValueError): fixed_pool([record()], P, 2, ids)


def test_pool_rejects_duplicate_seats_and_conflicting_geometry():
    row = record()
    with pytest.raises(ValueError, match='duplicate'): fixed_pool([row, row], P, 2)
    changed = deepcopy(row); changed['super_pts_path'] = '/different'
    with pytest.raises(ValueError, match='geometry'): fixed_pool([row, changed], P, 2)


@pytest.mark.parametrize('specs', [[], ['0=/a'], ['1=/a','2=/b'],
    ['0=/a','0=/b'], ['0=/a','2=/b','1=/c'], ['00=/a','1=/b'],
    ['False=/a','1=/b'], ['0=/a','1=/a'], ['0','1=/b']])
def test_invalid_checkpoint_schedule(specs):
    with pytest.raises(ValueError): checkpoint_schedule(specs)


def test_schedule_preserves_explicit_measurement_intervals():
    assert [epoch for epoch, _ in checkpoint_schedule(['0=/a','2=/b','8=/c'])] == [0,2,8]


@pytest.mark.parametrize('meta,epoch', [({'epoch':1},0), ({},0),
    ({'epoch':True},1), ({'epoch':2},1), ({},1),
    ({'ldmr_initialized_from_stage':1, 'epoch':False},0)])
def test_wrong_checkpoint_epoch(meta, epoch):
    with pytest.raises(ValueError): validate_checkpoint_epoch(dict(meta=meta),2,epoch)


def test_valid_initialization_and_trained_checkpoint():
    validate_checkpoint_epoch(dict(meta=dict(ldmr_initialized_from_stage=1)),2,0)
    validate_checkpoint_epoch(dict(meta=dict(epoch=8)),2,8)


def test_context_binds_actual_gt_and_schedule():
    context = dict(seats=[record()], steps=[0,1], inputs={'/points':'a'*64})
    digest = context_digest(context)
    changed = deepcopy(context); changed['seats'][0]['instances'][0]['bbox_3d'][0] = 2
    assert context_digest(changed) != digest
    changed = deepcopy(context); changed['steps'] = [0,2]
    assert context_digest(changed) != digest
