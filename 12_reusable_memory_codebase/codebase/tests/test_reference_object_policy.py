"""Reference semantics checked against the teammate's actual bank source."""
from pathlib import Path
import random
import numpy as np
import pytest
from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank
from mmdet3d.datasets.pipelines.exemplar_insertion import InsertExemplarObjects


def reference_fixture():
    import json
    return json.loads((Path(__file__).parent / 'reference_fixtures/tr3d_obj_bank.json').read_text())


def test_replacement_matches_reference_stream_for_one_class():
    # Compare the replacement algorithm; per-class RNG differs from the
    # teammate's globally interleaved multi-class RNG, explicitly documented.
    records = [dict(class_id=0, object_idx=i) for i in range(47)]
    ours = ObjectMemoryBank(exemplars_per_class=7, selection_strategy='tr3d_obj_replace', random_seed=23)
    selected = ours._select(records, 7, 0)
    assert selected == reference_fixture()['replacement']


def test_real_reference_crop_survives_population_and_checkpoint(tmp_path):
    """The original 1cm tolerance makes a real 18-point crop eligible (25)."""
    import json
    from types import SimpleNamespace
    from mmdet3d.datasets.incremental_sunrgbd import IncrementalSUNRGBDDataset
    fixture = json.loads((Path(__file__).parent / 'reference_fixtures/tr3d_obj_real_crop.json').read_text())
    points = np.asarray(fixture['input_points'], np.float32)
    box = np.asarray(fixture['box'], np.float32)
    assert len(ObjectMemoryBank.extract_object_points(points, box)) == 18
    config = dict(exemplars_per_class=1, max_total_exemplars=1, min_points=20,
                  selection_strategy='tr3d_obj_replace', crop_tolerance=.01)
    bank = ObjectMemoryBank(**config)
    info = dict(scene_id='005114', annos={'class': np.array([0]),
                'gt_boxes_upright_depth': box[None]})
    dataset = SimpleNamespace(object_memory_bank=bank, evaluation_mode=False,
        stage_classes=[0], original_data_infos=[info], paths=None, work_dir=None,
        stage_id=1, _extract_scene_id=lambda row: row['scene_id'],
        _load_scene_points=lambda sid: points)
    IncrementalSUNRGBDDataset.update_memory_bank_from_stage(dataset)
    crop = bank.get_all_exemplars()[0]
    assert crop['point_count'] == fixture['reference_point_count'] == 25
    world = crop['points'].copy()
    c, s = np.cos(float(box[6])), np.sin(float(box[6]))
    world[:, :2] = world[:, :2] @ np.array([[c, s], [-s, c]], np.float32)
    np.testing.assert_allclose(world, fixture['reference_points'], atol=2e-6)
    path = tmp_path / 'bank.pkl'
    bank.save_state(path, stage_id=1)
    restored = ObjectMemoryBank.load_state(path, expected_config=config)
    np.testing.assert_array_equal(restored.get_all_exemplars()[0]['points'], crop['points'])
    with pytest.raises(ValueError, match='crop_tolerance'):
        ObjectMemoryBank.load_state(path, expected_config=dict(config, crop_tolerance=1e-6))


def test_class_uniform_draws_match_reference_and_allow_duplicates():
    buckets = {0: [dict(class_id=0, object_idx=0)],
               1: [dict(class_id=1, object_idx=i) for i in range(9)]}
    ours = ObjectMemoryBank()
    ours.exemplars = buckets
    insertion = InsertExemplarObjects(ours, max_exemplars_per_scene=30, sampling_policy='class_uniform')
    random.seed(42)
    actual = insertion._sample_exemplars_for_insertion([0, 1])
    assert actual == reference_fixture()['draws']
    assert len(actual) > sum(map(len, buckets.values()))


def test_reference_population_keeps_only_budgeted_crops(monkeypatch):
    import weakref
    bank = ObjectMemoryBank(exemplars_per_class=7, min_points=2,
                            selection_strategy='tr3d_obj_replace', random_seed=23)
    held = []
    peaks = []
    original = np.asarray

    class TrackedPoints(np.ndarray):
        def copy(self, *args, **kwargs):
            result = super().copy(*args, **kwargs)
            held.append(weakref.ref(result))
            peaks.append(sum(ref() is not None for ref in held))
            return result

    # Track only authoritative crop copies, not source points or box metadata.
    points = np.zeros((2, 6), dtype=np.float32).view(TrackedPoints)
    monkeypatch.setattr(np, 'asarray', lambda value, *a, **kw:
                        value if value is points else original(value, *a, **kw))
    records = [dict(scene_id='s', object_idx=i, bbox=[0, 0, 0, 2, 1, 1], points=points)
               for i in range(47)]
    bank.add_exemplars(0, records, stage_id=1)
    assert [row['object_idx'] for row in bank.get_exemplars([0])] == [
        row['object_idx'] for row in reference_fixture()['replacement']]
    assert max(peaks) <= 8  # seven retained crops plus the incoming candidate
    assert sum(ref() is not None for ref in held) == 7


@pytest.mark.parametrize('source_yaw', [0., .7, -1.2])
def test_random_yaw_preserves_box_point_geometry(source_yaw):
    insertion = InsertExemplarObjects(random_yaw=True, placement_jitter=0,
        preserve_source_height=False, floor_percentile=0)
    points = np.array([[.8, 0, -.4], [-.8, 0, .4]], dtype=np.float32)
    box = np.array([0, 0, 0, 2, 1, 1, source_yaw], dtype=np.float32)
    random.seed(1)
    placed, result = insertion._find_valid_placement(points, box, np.zeros((0, 7)),
        np.array([[-5,-5,0],[5,5,4]], dtype=np.float32))
    assert result[6] != 0 and result[2] == 0
    center = result[:3].copy(); center[2] += result[5]/2
    # Recover the original local coordinates using the returned target yaw.
    # Distances alone cannot detect an extra rotation of the pasted points.
    c, s = np.cos(result[6]), np.sin(result[6])
    recovered = (placed-center) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    np.testing.assert_allclose(recovered, points, atol=1e-6)
    assert np.all(np.abs(recovered) <= result[3:6] / 2 + 1e-6)
