from types import SimpleNamespace

import numpy as np
import pytest

from mmdet3d.datasets.incremental_sunrgbd import IncrementalSUNRGBDDataset
from mmdet3d.datasets.incremental_scannet import IncrementalScanNetDataset
from mmdet3d.datasets.memory_bank.source_cache import SourcePointCache
from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank


def populate(dataset_name, strategy, cache_bytes):
    bank = ObjectMemoryBank(exemplars_per_class=2, max_total_exemplars=4,
                            min_points=5, selection_strategy=strategy,
                            random_seed=42, source_cache_max_bytes=cache_bytes)
    boxes = np.array([[0, 0, 0, 2, 2, 2, 0]] * 4, dtype=np.float32)
    labels = [0, 0, 1, 1] if dataset_name == 'sunrgbd' else [3, 3, 4, 4]
    infos = [dict(scene_id=str(i), annos=dict(
        gt_num=4, **{'class': np.array(labels)},
        gt_boxes_upright_depth=boxes.copy())) for i in range(4)]
    arrays = {str(i): np.full((10 + i, 6), i * .1, dtype=np.float32) for i in range(4)}
    loads = []
    def loader(sid):
        loads.append(sid)
        return arrays[sid].copy()
    dataset = SimpleNamespace(
        object_memory_bank=bank, memory_handler=None, evaluation_mode=False,
        stage_classes=[0, 1], stage_id=1, stage_idx=0,
        original_data_infos=infos, paths=None, work_dir=None,
        mappings=dict(nyu40_to_model_idx={3: 0, 4: 1}, valid_nyu40_ids=[3, 4]),
        _extract_scene_id=lambda info: info['scene_id'],
        _get_scene_id_from_data_info=lambda info: info['scene_id'],
        _load_scene_points=loader)
    cls = IncrementalSUNRGBDDataset if dataset_name == 'sunrgbd' else IncrementalScanNetDataset
    cls.update_memory_bank_from_stage(dataset)
    return bank, loads


@pytest.mark.parametrize('dataset_name', ['sunrgbd', 'scannet'])
@pytest.mark.parametrize('strategy', ['random', 'largest_point_count'])
def test_cache_preserves_actual_dataset_population(dataset_name, strategy):
    uncached, old_loads = populate(dataset_name, strategy, 0)
    cached, new_loads = populate(dataset_name, strategy, 4096)
    assert len(new_loads) < len(old_loads)
    assert cached.last_population_io['peak_cached_point_bytes'] <= 4096
    for left, right in zip(uncached.get_all_exemplars(), cached.get_all_exemplars()):
        assert (left['class_id'], left['scene_id'], left['object_idx']) == (
            right['class_id'], right['scene_id'], right['object_idx'])
        np.testing.assert_array_equal(left['points'], right['points'])
        assert left['source_floor_offset'] == right['source_floor_offset']
    assert uncached.get_total_exemplar_count() == cached.get_total_exemplar_count()


def test_cache_evicts_by_bytes_and_does_not_cache_oversized_arrays():
    cache = SourcePointCache(lambda key: np.ones((int(key), 6), dtype=np.float32), 480)
    cache.get('10')
    cache.get('11')
    assert list(cache.entries) == ['11']
    cache.get('100')
    assert list(cache.entries) == ['11']
    assert cache.peak_bytes <= 480
    assert cache.loads == 3
