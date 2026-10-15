"""Legacy stage-1 recomputation must reject ambiguous/corrupt snapshot inputs."""
import json
import pytest
from tools.train_incremental_scene import _load_ld_stage_stats_by_k


def write(tmp_path, k=0, **changes):
    payload = dict(stage_id=1, iou_mode='0.25', iou_thr=.25, eps=1e-9,
                   review_k=k, seats=[])
    payload.update(changes)
    path = tmp_path / f'memory_stats_k{k}.json'
    path.write_text(json.dumps(payload))
    return path


def load(tmp_path):
    return _load_ld_stage_stats_by_k(stage_ld_dir=tmp_path,
        filename_prefix='memory_stats', require_stage_id=1)


def test_valid_snapshots_and_legacy_optional_review_index(tmp_path):
    path = write(tmp_path)
    payload = json.loads(path.read_text()); del payload['review_k']
    path.write_text(json.dumps(payload))
    write(tmp_path, 1)
    assert load(tmp_path)['ks'] == [0, 1]


@pytest.mark.parametrize('change', [dict(eps=float('nan')), dict(eps=float('inf')),
    dict(eps=0), dict(iou_thr=float('nan')), dict(iou_thr=1),
    dict(iou_mode='nan'), dict(iou_mode='0.50'), dict(review_k=1),
    dict(review_k=False), dict(stage_id=True), dict(stage_id=1.5)])
def test_corrupt_snapshot_metadata(tmp_path, change):
    write(tmp_path, **change)
    with pytest.raises(RuntimeError):
        load(tmp_path)


def test_duplicate_numeric_index_is_not_silently_overwritten(tmp_path):
    path = write(tmp_path)
    (tmp_path / 'memory_stats_k00.json').write_bytes(path.read_bytes())
    with pytest.raises(RuntimeError, match='Duplicate'):
        load(tmp_path)


def test_missing_seat_cannot_reach_stage1_recomputed_scores(tmp_path):
    row = dict(scene_id='scene', save_stage=1, classes={'0': {'gt_count': 1}})
    write(tmp_path, seats=[row])
    write(tmp_path, 1, seats=[])
    with pytest.raises(ValueError, match='coverage'):
        load(tmp_path)


@pytest.mark.parametrize('change', ['duplicate', 'missing_class', 'gt_change', 'class_alias'])
def test_fixed_pool_coverage(change):
    from copy import deepcopy
    from mmdet3d.utils.learning_dynamics_scoring import validate_fixed_seat_coverage
    first = [dict(scene_id='scene', save_stage=1, classes={'0': {'gt_count': 1}})]
    second = deepcopy(first)
    if change == 'duplicate': second.append(deepcopy(second[0]))
    if change == 'missing_class': second[0]['classes'] = {}
    if change == 'gt_change': second[0]['classes']['0']['gt_count'] = 2
    if change == 'class_alias': second[0]['classes'][0] = {'gt_count': 1}
    with pytest.raises(ValueError):
        validate_fixed_seat_coverage({0: first, 1: second})


@pytest.mark.parametrize('location', ['metadata', 'seat', 'class', 'stat'])
def test_duplicate_json_keys_are_not_silently_overwritten(tmp_path, location):
    row = dict(scene_id='scene', save_stage=1, classes={'0': {'gt_count': 1}})
    path = write(tmp_path, seats=[row])
    raw = path.read_text()
    if location == 'metadata': raw = raw.replace('"stage_id": 1', '"stage_id": 2, "stage_id": 1')
    if location == 'seat': raw = raw.replace('"scene_id": "scene"', '"scene_id": "wrong", "scene_id": "scene"')
    if location == 'class': raw = raw.replace('"0": {"gt_count": 1}', '"0": {"gt_count": 9}, "0": {"gt_count": 1}')
    if location == 'stat': raw = raw.replace('"gt_count": 1', '"gt_count": 9, "gt_count": 1')
    path.write_text(raw)
    with pytest.raises(RuntimeError, match='[Dd]uplicate'):
        load(tmp_path)
