import pickle
from types import SimpleNamespace

import numpy as np
import pytest

from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank
from tools.train_incremental_scene import _validate_object_memory_resume


def objects(cid=0):
    return [dict(scene_id=f's{cid}', object_idx=i,
                 bbox=np.array([0, 0, 0, 2, 2, 2, 0], dtype=np.float32),
                 points=np.ones((20, 6), dtype=np.float32)) for i in range(2)]


def bank():
    result = ObjectMemoryBank(exemplars_per_class=2, max_total_exemplars=2)
    result.add_exemplars(0, objects(), stage_id=1)
    return result


@pytest.mark.parametrize('field,value', [
    ('coordinate_frame', 'world'), ('point_count', 999), ('class_id', 3),
    ('points', np.full((20, 6), np.nan)), ('stage_id', 3),
    ('box_size', np.zeros(3)), ('replay_weight', -1.),
])
def test_corrupted_crop_is_rejected(tmp_path, field, value):
    path = tmp_path / 'bank.pkl'
    bank().save_state(path, stage_id=1)
    payload = pickle.loads(path.read_bytes())
    payload['exemplars'][0][0][field] = value
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError):
        ObjectMemoryBank.load_state(path)


@pytest.mark.parametrize('corruption', ['version', 'capacity', 'processed', 'stage'])
def test_corrupted_envelope_is_rejected(tmp_path, corruption):
    path = tmp_path / 'bank.pkl'
    bank().save_state(path, stage_id=1)
    payload = pickle.loads(path.read_bytes())
    if corruption == 'version':
        payload['format_version'] = 999
    elif corruption == 'capacity':
        payload['config']['max_total_exemplars'] = 1
    elif corruption == 'processed':
        del payload['processed_classes']
    else:
        payload['stage_id'] = 1.5
    path.write_bytes(pickle.dumps(payload))
    with pytest.raises(ValueError):
        ObjectMemoryBank.load_state(path)


def test_v3_crop_states_remain_readable(tmp_path):
    path = tmp_path / 'bank.pkl'
    bank().save_state(path, stage_id=1)
    payload = pickle.loads(path.read_bytes())
    payload['format_version'] = 3
    del payload['processed_classes']
    payload['config'].pop('source_cache_max_bytes')
    path.write_bytes(pickle.dumps(payload))
    restored = ObjectMemoryBank.load_state(path)
    assert restored.processed_classes == {0}
    assert restored.get_total_exemplar_count() == 2


def test_processed_empty_class_survives_resume(tmp_path):
    source = bank()
    source.add_exemplars(1, [], stage_id=1)
    path = tmp_path / 'bank.pkl'
    source.save_state(path, stage_id=1)
    restored = ObjectMemoryBank.load_state(path)
    _validate_object_memory_resume(restored, 2, [dict(stage_id=1, class_indices=[0, 1])], path)
    assert restored.get_class_exemplar_count(1) == 0
    with pytest.raises(ValueError, match='coverage'):
        _validate_object_memory_resume(restored, 2, [dict(stage_id=1, class_indices=[0, 1, 2])], path)


def test_pruning_releases_points_and_reports_retained_count():
    source = bank()
    assert source.add_exemplars(1, objects(1), stage_id=2) == 1
    assert source.add_exemplars(2, objects(2), stage_id=3) == 0
    assert source.get_total_exemplar_count() == 2
    assert all('points' not in obj for xs in source.removed_exemplars.values() for obj in xs)
    stats = source.get_statistics()
    assert stats['active_point_bytes'] == 2 * 20 * 6 * 4
    assert stats['retained_removed_point_bytes'] == 0


def test_reviewing_requires_reattachment_after_restore(tmp_path):
    path = tmp_path / 'bank.pkl'
    bank().save_state(path, stage_id=1)
    restored = ObjectMemoryBank.load_state(path)
    with pytest.raises(RuntimeError, match='dataset reference'):
        restored.list_source_scene_entries()
    restored.dataset_ref = SimpleNamespace(_object_scene_info_by_id={'s0': dict(sample_idx='s0')})
    assert len(restored.list_source_scene_entries()) == 1
    restored.apply_source_seat_replay_weights({'s0_stage1': 2.0})
    assert all(x['replay_weight'] == 2.0 for x in restored.get_all_exemplars())


def test_legacy_handler_reports_actual_bank_reduction():
    from mmdet3d.datasets.object_memory_bank_handler import MemoryBankHandler
    source = bank()
    source.max_total_exemplars = 1
    handler = MemoryBankHandler(source)
    report = handler._apply_reduction_strategy(dict(
        name='legacy_suggestion', target_reduction=1))
    assert report['actual_reduction'] == 1
    assert report['final_count'] == source.get_total_exemplar_count() == 1
    assert report['strategy_name'] == 'balanced_bank_budget'


def test_resume_rejects_silently_changed_budget(tmp_path):
    path = tmp_path / 'bank.pkl'
    bank().save_state(path, stage_id=1)
    with pytest.raises(ValueError, match='configuration mismatch'):
        ObjectMemoryBank.load_state(path, expected_config=dict(
            exemplars_per_class=2, max_total_exemplars=4))


def test_resume_keeps_saved_selection_seed_with_new_training_seed(tmp_path):
    path = tmp_path / 'bank.pkl'
    bank().save_state(path, stage_id=1)
    restored = ObjectMemoryBank.load_state(path, expected_config=dict(
        exemplars_per_class=2, max_total_exemplars=2, random_seed=99,
        source_cache_max_bytes=1024))
    assert restored.random_seed == 0
    assert restored.source_cache_max_bytes == 1024
