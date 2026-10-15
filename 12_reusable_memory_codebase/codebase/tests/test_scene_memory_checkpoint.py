import copy
import pickle

import numpy as np
import pytest

from mmdet3d.datasets.scene_memory_bank import SceneMemoryBank
from mmdet3d.datasets.memory_bank.state_io import atomic_write


def make_bank():
    return SceneMemoryBank(max_memory_scenes=4, quota_strategy='stage_ratio',
                           stage_scene_counts={1: 10, 2: 10},
                           selection_strategy='random', random_seed=42,
                           debug_mode=False, min_objects_per_scene=1)


def infos(stage):
    return [dict(sample_idx=f's{stage}_{i}', annos=dict(
        gt_num=2, **{'class': np.array([stage - 1, 9], dtype=np.int64)},
        gt_boxes_upright_depth=np.ones((2, 7), dtype=np.float32))) for i in range(8)]


def populated():
    bank = make_bank()
    bank.add_stage_scenes(1, infos(1), [0], {})
    return bank


def seats(bank):
    return [(e['scene_id'], e['save_stage']) for e in bank.list_memory_entries()]


def test_scene_checkpoint_preserves_replay_and_next_stage_selection(tmp_path):
    bank = populated()
    path = tmp_path / 'stage1.pkl'
    context = dict(classes=[[0], [1]], dataset='fixture')
    bank.save_checkpoint(path, 1, context)
    restored = SceneMemoryBank.load_checkpoint(
        path, expected_stage=1, expected_context=context,
        expected_config=make_bank()._checkpoint_config)
    assert restored.scene_snapshots == {}  # no unselected candidate cache
    before, before_ids = bank.get_replay_scenes([0], 2)
    after, after_ids = restored.get_replay_scenes([0], 2)
    assert before_ids == after_ids
    for left, right in zip(before, after):
        np.testing.assert_array_equal(left['annos']['class'], right['annos']['class'])
        assert right['annos']['class'].tolist() == [0]  # future label excluded
        np.testing.assert_array_equal(left['annos']['gt_boxes_upright_depth'],
                                      right['annos']['gt_boxes_upright_depth'])
    for candidate in (bank, restored):
        candidate.add_stage_scenes(2, infos(2), [0, 1], {})
    assert seats(bank) == seats(restored)
    assert bank.class_distribution == restored.class_distribution
    assert bank.scenes_per_stage == restored.scenes_per_stage


def test_design2_update_is_identical_after_checkpoint(tmp_path):
    config = make_bank()._checkpoint_config
    config.update(selection_strategy='learning_dynamics_design2',
                  learning_dynamics_design2=dict(
                      q_metric='f1', min_add_lower_bound=1,
                      force_accept_until_lower_bound=True,
                      supply_scaling_mode='raw', use_class_balance=True,
                      redundancy_lambda=0.3, redundancy_topk=5))
    bank = SceneMemoryBank(**config)
    bank.memory_scenes = copy.deepcopy(populated().memory_scenes)
    bank._recompute_class_distribution()
    bank._sync_active_stage_stats()
    path = tmp_path / 'ld.pkl'
    bank.save_checkpoint(path, 1)
    restored = SceneMemoryBank.load_checkpoint(path)
    terms = {sid: {1: {0: dict(g=0., r_best=1., d=1., u=1.)}}
             for sid in bank.memory_scenes}
    for info in infos(2):
        terms[info['sample_idx']] = {2: {1: dict(g=0., r_best=0., d=0., u=0.)}}
    payload = dict(stage_id=2, q_metric='f1', class_need={0: .7, 1: .3},
                   seat_class_terms=terms)
    for candidate in (bank, restored):
        candidate.add_stage_scenes(2, infos(2), [0, 1], {},
                                  learning_dynamics_design2_payload=payload)
    assert seats(bank) == seats(restored)
    assert bank.class_distribution == restored.class_distribution


@pytest.mark.parametrize('kwargs,match', [
    (dict(expected_stage=2), 'stage'),
    (dict(expected_context=dict(dataset='different')), 'protocol'),
    (dict(expected_config=dict(max_memory_scenes=99)), 'configuration'),
])
def test_scene_checkpoint_rejects_wrong_resume(tmp_path, kwargs, match):
    path = tmp_path / 'bank.pkl'
    populated().save_checkpoint(path, 1)
    with pytest.raises(ValueError, match=match):
        SceneMemoryBank.load_checkpoint(path, **kwargs)


@pytest.mark.parametrize('corruption', ['version', 'future_stage', 'budget'])
def test_scene_checkpoint_rejects_corrupted_state(tmp_path, corruption):
    path = tmp_path / 'bank.pkl'
    populated().save_checkpoint(path, 1)
    payload = pickle.loads(path.read_bytes())
    if corruption == 'version':
        payload['format_version'] = 999
    elif corruption == 'future_stage':
        scene = next(iter(payload['memory_scenes'].values()))
        scene['stages'][2] = scene['stages'].pop(1)
    else:
        payload['config']['max_memory_scenes'] = 1
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError):
        SceneMemoryBank.load_checkpoint(path)


def test_atomic_write_preserves_prior_artifact_on_serialization_failure(tmp_path):
    path = tmp_path / 'bank'
    path.write_text('prior')
    def fail(stream):
        stream.write('partial')
        raise RuntimeError('serialization failed')
    with pytest.raises(RuntimeError):
        atomic_write(path, fail)
    assert path.read_text() == 'prior'
    assert list(tmp_path.iterdir()) == [path]
