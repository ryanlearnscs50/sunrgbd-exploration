from __future__ import annotations

import pytest

from mmdet3d.datasets.incremental_sunrgbd import IncrementalSUNRGBDDataset


def test_sunrgbd_reviewing_sampling_preserves_memory_coverage():
    ds = IncrementalSUNRGBDDataset.__new__(IncrementalSUNRGBDDataset)
    ds.data_infos = [
        {'scene_id': 'nat_a', 'annos': {'gt_num': 1}},
        {
            'scene_id': 'rep_high',
            'is_replay': True,
            'replay_unique_id': 's1_stage1',
            'annos': {'gt_num': 1},
        },
        {
            'scene_id': 'rep_low',
            'is_replay': True,
            'replay_unique_id': 's2_stage1',
            'annos': {'gt_num': 1},
        },
        {
            'scene_id': 'merged_mid',
            'is_merged': True,
            'replay_unique_ids': ['s2_stage1', 's3_stage1'],
            'annos': {'gt_num': 1},
        },
    ]
    ds.reviewing_sampling = {
        'enabled': True,
        'target_length': 600,
        'weights_by_replay_unique_id': {
            's1_stage1': 12.0,
            's2_stage1': 1.0,
            's3_stage1': 8.0,
        },
        'strict_memory_coverage': True,
        'seed': 7,
    }

    ds._apply_reviewing_sampling_if_enabled()

    sampled = ds.data_infos
    assert len(sampled) == 600

    n_high = sum(1 for x in sampled if x.get('replay_unique_id') == 's1_stage1')
    n_low = sum(1 for x in sampled if x.get('replay_unique_id') == 's2_stage1')
    n_merged = sum(1 for x in sampled if bool(x.get('is_merged', False)))
    assert n_high > n_low
    assert n_merged > n_low
    assert n_high >= 1
    assert n_low >= 1
    assert n_merged >= 1

    dbg = getattr(ds, 'reviewing_sampling_debug', {}) or {}
    assert dbg.get('memory_candidate_count') == 3
    assert dbg.get('memory_seen_count') == 3
    assert dbg.get('memory_never_seen_count') == 0
    assert float(dbg.get('memory_coverage_ratio')) == pytest.approx(1.0)


def test_sunrgbd_reviewing_sampling_strict_coverage_guard():
    ds = IncrementalSUNRGBDDataset.__new__(IncrementalSUNRGBDDataset)
    ds.data_infos = [
        {
            'scene_id': f'rep_{i}',
            'is_replay': True,
            'replay_unique_id': f's{i}_stage1',
            'annos': {'gt_num': 1},
        }
        for i in range(5)
    ]
    ds.reviewing_sampling = {
        'enabled': True,
        'target_length': 3,  # smaller than number of memory candidates
        'weights_by_replay_unique_id': {
            f's{i}_stage1': 1.0 for i in range(5)
        },
        'strict_memory_coverage': True,
        'seed': 1,
    }
    with pytest.raises(RuntimeError, match='cannot guarantee memory coverage'):
        ds._apply_reviewing_sampling_if_enabled()


def test_sunrgbd_reviewing_sampling_non_strict_expands_target_for_coverage():
    ds = IncrementalSUNRGBDDataset.__new__(IncrementalSUNRGBDDataset)
    ds.data_infos = [
        {
            'scene_id': f'rep_{i}',
            'is_replay': True,
            'replay_unique_id': f's{i}_stage1',
            'annos': {'gt_num': 1},
        }
        for i in range(5)
    ]
    ds.reviewing_sampling = {
        'enabled': True,
        'target_length': 3,  # smaller than number of memory candidates
        'weights_by_replay_unique_id': {
            f's{i}_stage1': 1.0 for i in range(5)
        },
        'strict_memory_coverage': False,
        'seed': 1,
    }
    ds._apply_reviewing_sampling_if_enabled()
    assert len(ds.data_infos) == 5
    seen_uids = {
        x.get('replay_unique_id')
        for x in ds.data_infos
        if x.get('is_replay', False)
    }
    assert seen_uids == {f's{i}_stage1' for i in range(5)}
