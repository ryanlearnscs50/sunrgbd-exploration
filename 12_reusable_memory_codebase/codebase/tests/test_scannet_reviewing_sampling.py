from __future__ import annotations

import pytest

from mmdet3d.datasets.incremental_scannet import IncrementalScanNetDataset


def test_scannet_reviewing_sampling_uses_replay_and_merged_weights():
    ds = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
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
    # Coverage-preserving behavior: each memory candidate appears at least once.
    assert n_high >= 1
    assert n_low >= 1
    assert n_merged >= 1
    dbg = getattr(ds, 'reviewing_sampling_debug', {}) or {}
    assert dbg.get('memory_candidate_count') == 3
    assert dbg.get('memory_seen_count') == 3
    assert dbg.get('memory_never_seen_count') == 0
    assert float(dbg.get('memory_coverage_ratio')) == pytest.approx(1.0)


def test_scannet_reviewing_sampling_boosts_memory_at_cost_of_natural():
    base_infos = []
    for i in range(10):
        base_infos.append({'scene_id': f'nat_{i}', 'annos': {'gt_num': 1}})
    for i in range(3):
        base_infos.append(
            {
                'scene_id': f'rep_{i}',
                'is_replay': True,
                'replay_unique_id': f's{i}_stage1',
                'annos': {'gt_num': 1},
            }
        )

    # Neutral weights: memory stays at baseline coverage-only count.
    ds_neutral = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
    ds_neutral.data_infos = list(base_infos)
    ds_neutral.reviewing_sampling = {
        'enabled': True,
        'target_length': 13,
        'weights_by_replay_unique_id': {
            's0_stage1': 1.0,
            's1_stage1': 1.0,
            's2_stage1': 1.0,
        },
        'seed': 13,
        'strict_memory_coverage': True,
    }
    ds_neutral._apply_reviewing_sampling_if_enabled()
    neutral_mem = sum(1 for x in ds_neutral.data_infos if bool(x.get('is_replay', False)))
    neutral_nat = len(ds_neutral.data_infos) - neutral_mem
    assert neutral_mem == 3
    assert neutral_nat == 10

    # Boost one memory UID: extra memory revisits should reduce natural picks.
    ds_boost = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
    ds_boost.data_infos = list(base_infos)
    ds_boost.reviewing_sampling = {
        'enabled': True,
        'target_length': 13,
        'weights_by_replay_unique_id': {
            's0_stage1': 10.0,
            's1_stage1': 1.0,
            's2_stage1': 1.0,
        },
        'seed': 13,
        'strict_memory_coverage': True,
    }
    ds_boost._apply_reviewing_sampling_if_enabled()
    boost_mem = sum(1 for x in ds_boost.data_infos if bool(x.get('is_replay', False)))
    boost_nat = len(ds_boost.data_infos) - boost_mem

    assert boost_mem > neutral_mem
    assert boost_nat < neutral_nat
    # Coverage is still preserved under boosting.
    seen_uids = {x.get('replay_unique_id') for x in ds_boost.data_infos if x.get('is_replay', False)}
    assert {'s0_stage1', 's1_stage1', 's2_stage1'}.issubset(seen_uids)


def test_scannet_reviewing_sampling_strict_coverage_guard():
    ds = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
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
        'seed': 1,
        'strict_memory_coverage': True,
    }
    with pytest.raises(RuntimeError, match='cannot guarantee memory coverage'):
        ds._apply_reviewing_sampling_if_enabled()


def test_scannet_reviewing_sampling_non_strict_expands_target_for_coverage():
    ds = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
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
        'seed': 1,
        'strict_memory_coverage': False,
    }
    ds._apply_reviewing_sampling_if_enabled()
    assert len(ds.data_infos) == 5
    seen_uids = {x.get('replay_unique_id') for x in ds.data_infos if x.get('is_replay', False)}
    assert seen_uids == {f's{i}_stage1' for i in range(5)}
    dbg = getattr(ds, 'reviewing_sampling_debug', {}) or {}
    assert dbg.get('memory_candidate_count') == 5
    assert dbg.get('memory_seen_count') == 5
    assert dbg.get('memory_never_seen_count') == 0
