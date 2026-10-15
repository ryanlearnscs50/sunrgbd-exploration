"""Incomplete measurements must not masquerade as forgetting trajectories."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.learning_dynamics import reference_functions
from ldmr_unidet3d.trajectories import (make_snapshot, validate_snapshot,
    validate_trajectory, score_design2, write_snapshot, load_snapshot)

P = ScanNet35Protocol()


def seat(scene='one', tp=1):
    return dict(scene_id=scene, save_stage=1, classes={cid: dict(
        tp=float(tp if cid == 0 else 0), fp=0, fn=float(1-tp if cid == 0 else 0),
        denom=float(cid == 0), gt_count=int(cid == 0), tp_count=tp if cid == 0 else 0,
        dt_count=tp if cid == 0 else 0, q=tp/(1+1e-9) if cid == 0 else None,
        count=int(cid == 0)) for cid in range(7)})


def snapshot(step=0, rows=None):
    return make_snapshot([seat(), seat('two')] if rows is None else rows,
        protocol=P, stage=2, step=step, checkpoint_sha256=('a' if step == 0 else 'b')*64,
        measurement_sha256='c'*64)


def test_reference_missing_seat_would_create_false_forgetting():
    before = snapshot()
    after = snapshot(4, [seat()])
    raw = reference_functions()['scoring']['compute_learning_dynamics_design1_scores'](
        {0: before['seats'], 1: after['seats']}, class_ids=list(range(7)),
        q_metric='recall', design_version=2)
    assert raw['seat_class_terms']['two'][1][0]['d'] > .99
    with pytest.raises(ValueError, match='coverage'):
        score_design2([before, after], protocol=P, expected_steps=[0, 4])


def test_complete_trajectory_is_exact_reference_scoring():
    snapshots = [snapshot(), snapshot(4, [seat(tp=0), seat('two')])]
    metrics = validate_trajectory(snapshots, protocol=P, expected_steps=[0, 4])
    expected = reference_functions()['scoring']['compute_learning_dynamics_design1_scores'](
        metrics, class_ids=list(range(7)), new_classes=list(range(7, 14)),
        q_metric='recall', eps=1e-9, design_version=2)
    assert score_design2(snapshots, protocol=P, expected_steps=[0, 4]) == expected
    assert metrics[1][0]['classes'][0]['q'] == 0  # genuine empty detection is valid


@pytest.mark.parametrize('steps', [[], [1], [0, 0], [0, 4, 2], [False], [0, 2.0]])
def test_invalid_planned_steps(steps):
    with pytest.raises(ValueError, match='steps'):
        validate_trajectory([snapshot()], protocol=P, expected_steps=steps)


@pytest.mark.parametrize('snapshots,steps', [([snapshot()], [0, 4]),
    ([snapshot(), snapshot(4)], [0]), ([snapshot(), snapshot(4)], [0, 3]),
    ([snapshot(4), snapshot()], [0, 4])])
def test_missing_extra_or_reordered_checkpoints(snapshots, steps):
    with pytest.raises(ValueError):
        validate_trajectory(snapshots, protocol=P, expected_steps=steps)


@pytest.mark.parametrize('key,value', [('measurement_sha256', 'd'*64),
    ('iou_thr', .5), ('eps', 2e-9)])
def test_mixed_context_is_rejected(key, value):
    first, second = snapshot(), snapshot(4)
    second[key] = value
    with pytest.raises(ValueError):
        validate_trajectory([first, second], protocol=P, expected_steps=[0, 4])


@pytest.mark.parametrize('change', ['duplicate_seat', 'missing_class', 'alias_class',
    'hidden_class', 'nan', 'negative', 'count', 'q', 'bool', 'fractional', 'empty',
    'future_stage', 'missing_field', 'undefined_q', 'protocol', 'version', 'digest'])
def test_corrupt_snapshot(change):
    value = snapshot()
    stat = value['seats'][0]['classes'][0]
    if change == 'duplicate_seat': value['seats'].append(deepcopy(value['seats'][0]))
    if change == 'missing_class': del value['seats'][0]['classes'][1]
    if change == 'alias_class': value['seats'][0]['classes']['0'] = deepcopy(stat)
    if change == 'hidden_class': value['seats'][0]['classes'][7] = deepcopy(stat)
    if change == 'nan': stat['tp'] = float('nan')
    if change == 'negative': stat['fn'] = -1
    if change == 'count': stat['gt_count'] = 2
    if change == 'q': stat['q'] = .2
    if change == 'bool': stat['fp'] = False
    if change == 'fractional': stat['dt_count'] = 1.5
    if change == 'empty': value['seats'] = []
    if change == 'future_stage': value['seats'][0]['save_stage'] = 3
    if change == 'missing_field': del stat['tp_count']
    if change == 'undefined_q': value['seats'][0]['classes'][1]['q'] = 0
    if change == 'protocol': value['protocol_id'] = 'wrong'
    if change == 'version': value['version'] = True
    if change == 'digest': value['checkpoint_sha256'] = 'missing'
    with pytest.raises(ValueError):
        validate_snapshot(value, P)


def test_gt_change_rejected_even_if_counts_and_quality_are_internally_valid():
    first, second = snapshot(), snapshot(4)
    second['seats'][0]['classes'][0].update(gt_count=2, count=2, denom=2., fn=1., q=1/(2+1e-9))
    with pytest.raises(ValueError, match='GT changed'):
        validate_trajectory([first, second], protocol=P, expected_steps=[0, 4])


def test_roundtrip_hashes_exclusive_write_and_no_mutation(tmp_path):
    original = snapshot()
    saved = deepcopy(original)
    path = tmp_path / 'snapshot.json'
    digest = write_snapshot(path, original, P)
    assert load_snapshot(path, P, expected_sha256=digest) == original
    assert original == saved
    with pytest.raises(FileExistsError):
        write_snapshot(path, snapshot(4), P)
    assert len(list(tmp_path.iterdir())) == 1
    with pytest.raises(ValueError, match='hash'):
        load_snapshot(path, P, expected_sha256='e'*64)
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='hash'):
        load_snapshot(path, P, expected_sha256=digest)


def test_duplicate_json_key_rejected_with_matching_artifact_hash(tmp_path):
    path = tmp_path / 'duplicate.json'
    raw = json.dumps(snapshot()).replace('"version": 1', '"version": 1, "version": 1').encode()
    path.write_bytes(raw)
    with pytest.raises(ValueError, match='duplicate'):
        load_snapshot(path, P, expected_sha256=hashlib.sha256(raw).hexdigest())


def test_roundtrip_accepts_mixed_canonical_json_and_integer_class_keys(tmp_path):
    value = snapshot()
    classes = value['seats'][0]['classes']
    classes['0'] = classes.pop(0)
    path = tmp_path / 'mixed.json'
    digest = write_snapshot(path, value, P)
    assert load_snapshot(path, P, expected_sha256=digest) == snapshot()
