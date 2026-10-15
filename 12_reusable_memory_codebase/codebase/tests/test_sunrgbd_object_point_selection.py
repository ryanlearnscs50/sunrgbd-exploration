"""Exercise the dataset population path, not just ObjectMemoryBank._select."""
from types import SimpleNamespace

import numpy as np
import pytest

from mmdet3d.datasets.incremental_sunrgbd import IncrementalSUNRGBDDataset
from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank


def populate(strategy, counts, cap=2):
    bank = ObjectMemoryBank(exemplars_per_class=cap, max_total_exemplars=cap,
                            selection_strategy=strategy, min_points=20,
                            crop_margin=0, random_seed=200)
    infos = [dict(scene_id=str(i), annos=dict(
        **{'class': np.array([0])},
        gt_boxes_upright_depth=np.array([[0, 0, 0, 2, 2, 2, 0]], dtype=np.float32)))
        for i in range(len(counts))]
    dataset = SimpleNamespace(
        object_memory_bank=bank, evaluation_mode=False, stage_classes=[0],
        original_data_infos=infos, paths=None, work_dir=None, stage_id=1,
        _extract_scene_id=lambda info: info['scene_id'],
        _load_scene_points=lambda sid: np.zeros((counts[int(sid)], 6), dtype=np.float32))
    IncrementalSUNRGBDDataset.update_memory_bank_from_stage(dataset)
    return bank.get_all_exemplars()


def test_largest_population_scans_beyond_quota_and_breaks_ties_stably():
    result = populate('largest_point_count', [22, 30, 19, 80, 80, 70])
    assert [r['scene_id'] for r in result] == ['3', '4']
    assert [r['point_count'] for r in result] == [80, 80]


@pytest.mark.parametrize('strategy', ['random', 'largest_point_count'])
def test_population_keeps_valid_crops_when_support_is_below_cap(strategy):
    result = populate(strategy, [0, 19, 21], cap=4)
    assert len(result) == 1
    assert result[0]['point_count'] == 21


def test_random_population_preserves_seeded_candidate_order():
    result = populate('random', [30] * 8)
    expected = np.random.RandomState(200).permutation(8)[:2]
    assert [r['scene_id'] for r in result] == [str(i) for i in expected]
