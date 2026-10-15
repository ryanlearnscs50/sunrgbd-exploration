from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from mmdet3d.datasets.scene_memory_bank import SceneMemoryBank
from mmdet3d.utils.incremental_paths import IncrementalPaths


@dataclass
class _DummyDatasetRef:
    paths: IncrementalPaths


def _make_scene_info(scene_id: str, labels: List[int]) -> dict:
    labels_arr = np.asarray(labels, dtype=np.int64)
    return {
        'sample_idx': str(scene_id),
        'point_cloud': {'lidar_idx': str(scene_id)},
        'annos': {
            'gt_num': int(labels_arr.shape[0]),
            'class': labels_arr,
        },
    }


def test_stage_ratio_random_writes_actions_and_passes_quota_check(tmp_path):
    paths = IncrementalPaths(tmp_path)
    dataset_ref = _DummyDatasetRef(paths=paths)

    mb = SceneMemoryBank(
        max_memory_scenes=6,
        quota_strategy='stage_ratio',
        stage_scene_counts={1: 10, 2: 10},
        selection_strategy='random',
        debug_mode=False,
        min_objects_per_scene=1,
    )
    mappings = {}

    stage1_infos = [_make_scene_info(f"s1_{i}", [0, 0]) for i in range(8)]
    stage1_replay_priority = {f"s1_{i}": {1: float(i + 1)} for i in range(8)}
    mb.add_stage_scenes(
        stage_id=1,
        scene_infos=stage1_infos,
        seen_classes=[0],
        mappings=mappings,
        dataset_ref=dataset_ref,
        learning_dynamics_replay_priority_by_seat=stage1_replay_priority,
    )

    stage2_infos = [_make_scene_info(f"s2_{i}", [0, 1]) for i in range(8)]
    mb.add_stage_scenes(
        stage_id=2,
        scene_infos=stage2_infos,
        seen_classes=[0, 1],
        mappings=mappings,
        dataset_ref=dataset_ref,
    )

    comp_path = paths.memory_bank_actions_dir() / 'memory_bank_composition_stage_2.json'
    assert comp_path.exists()
    report = json.loads(comp_path.read_text())
    assert report['quota_check_passed'] is True
    assert report['quotas'] == {'1': 3, '2': 3}
    assert report['actual_counts'] == {'1': 3, '2': 3}

    ids_path = paths.memory_bank_actions_dir() / 'memory_bank_selected_scenes_stage_2.txt'
    assert ids_path.exists()
    seat_ids = [ln for ln in ids_path.read_text().splitlines() if ln.strip()]
    assert len(seat_ids) == 6
    assert all('_stage' in ln for ln in seat_ids)


def test_stage_ratio_random_raises_on_quota_violation(tmp_path):
    paths = IncrementalPaths(tmp_path)
    dataset_ref = _DummyDatasetRef(paths=paths)

    mb = SceneMemoryBank(
        max_memory_scenes=6,
        quota_strategy='stage_ratio',
        stage_scene_counts={1: 10, 2: 10},
        selection_strategy='random',
        debug_mode=False,
        min_objects_per_scene=1,
    )
    mappings = {}

    stage1_infos = [_make_scene_info(f"s1_{i}", [0, 0]) for i in range(8)]
    stage1_replay_priority = {f"s1_{i}": {1: float(i + 1)} for i in range(8)}
    mb.add_stage_scenes(
        stage_id=1,
        scene_infos=stage1_infos,
        seen_classes=[0],
        mappings=mappings,
        dataset_ref=dataset_ref,
        learning_dynamics_replay_priority_by_seat=stage1_replay_priority,
    )

    # Only 1 stage-2 scene => cannot satisfy the stage-ratio quota of 3 seats.
    stage2_infos = [_make_scene_info("s2_only", [0, 1])]
    with pytest.raises(RuntimeError) as exc_info:
        mb.add_stage_scenes(
            stage_id=2,
            scene_infos=stage2_infos,
            seen_classes=[0, 1],
            mappings=mappings,
            dataset_ref=dataset_ref,
        )
    assert 'quotas violated' in str(exc_info.value)

    comp_path = paths.memory_bank_actions_dir() / 'memory_bank_composition_stage_2.json'
    assert comp_path.exists()
    report = json.loads(comp_path.read_text())
    assert report['quota_check_passed'] is False


def test_stage_ratio_learning_dynamics_prunes_and_adds_by_scores(tmp_path):
    paths = IncrementalPaths(tmp_path)
    dataset_ref = _DummyDatasetRef(paths=paths)

    mb = SceneMemoryBank(
        max_memory_scenes=6,
        quota_strategy='stage_ratio',
        stage_scene_counts={1: 10, 2: 10},
        selection_strategy='learning_dynamics',
        debug_mode=False,
        min_objects_per_scene=1,
    )
    mappings = {}

    stage1_infos = [_make_scene_info(f"s1_{i}", [0, 0]) for i in range(8)]
    stage1_replay_priority = {f"s1_{i}": {1: float(i + 1)} for i in range(8)}
    mb.add_stage_scenes(
        stage_id=1,
        scene_infos=stage1_infos,
        seen_classes=[0],
        mappings=mappings,
        dataset_ref=dataset_ref,
        learning_dynamics_replay_priority_by_seat=stage1_replay_priority,
    )

    stage1_seats = sorted(str(sid) for sid in mb.memory_scenes.keys())
    assert len(stage1_seats) == 6
    forgetness_by_seat = {
        sid: {1: float(i)} for i, sid in enumerate(stage1_seats)
    }

    stage2_infos = [_make_scene_info(f"s2_{i}", [0, 1]) for i in range(8)]
    replay_priority_by_seat = {
        f"s2_{i}": {2: float(i)} for i in range(8)
    }
    expected_added = [f"s2_{i}" for i in (7, 6, 5)]

    mb.add_stage_scenes(
        stage_id=2,
        scene_infos=stage2_infos,
        seen_classes=[0, 1],
        mappings=mappings,
        dataset_ref=dataset_ref,
        learning_dynamics_forgetness_by_seat=forgetness_by_seat,
        learning_dynamics_replay_priority_by_seat=replay_priority_by_seat,
    )

    comp_path = paths.memory_bank_actions_dir() / 'memory_bank_composition_stage_2.json'
    report = json.loads(comp_path.read_text())
    assert report['quota_check_passed'] is True
    assert report['selection_strategy'] == 'learning_dynamics'

    ld = report.get('learning_dynamics')
    assert isinstance(ld, dict)
    assert ld.get('quota_pruned_count') == 3

    added_entries = ld.get('added_entries') or []
    added_scene_ids = [e.get('scene_id') for e in added_entries]
    assert added_scene_ids == expected_added


def test_stage_ratio_learning_dynamics_design1_reports_shortage_explicitly(tmp_path):
    paths = IncrementalPaths(tmp_path)
    dataset_ref = _DummyDatasetRef(paths=paths)

    mb = SceneMemoryBank(
        max_memory_scenes=6,
        quota_strategy='stage_ratio',
        stage_scene_counts={1: 10, 2: 10},  # stage2 quota upper bound = 3
        selection_strategy='learning_dynamics_design1',
        learning_dynamics_design1=dict(
            q_metric='f1',
            min_add_lower_bound=1,
            use_compatibility_kernel=False,
            force_accept_until_lower_bound=True,
        ),
        enforce_unique_scene_ids=False,
        debug_mode=False,
        min_objects_per_scene=1,
    )
    mappings = {}

    # Prefill stage-1 seats directly (equivalent to a previous stage update).
    for i in range(6):
        sid = f"s1_{i}"
        info = _make_scene_info(sid, [0, 0])
        snap = {
            'save_stage': 1,
            'present_classes': [0],
            'object_counts': {0: 2},
            'data_info': info,
        }
        mb._add_scene_to_memory(sid, 1, snap, importance=1.0)

    stage2_infos = [_make_scene_info('s2_only', [0, 1])]
    seat_terms = {
        f"s1_{i}": {
            1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}
        }
        for i in range(6)
    }
    seat_terms['s2_only'] = {
        2: {
            0: {'g': 0.0, 'r_best': 0.2, 'd': 0.0, 'u': 0.1, 'r_start': 0.2, 'r_end': 0.2},
            1: {'g': 0.0, 'r_best': 0.0, 'd': 0.0, 'u': 0.0, 'r_start': 0.0, 'r_end': 0.0},
        }
    }
    design1_payload = {
        'stage_id': 2,
        'q_metric': 'f1',
        'class_need': {0: 0.7, 1: 0.3},
        'seat_class_terms': seat_terms,
    }

    mb.add_stage_scenes(
        stage_id=2,
        scene_infos=stage2_infos,
        seen_classes=[0, 1],
        mappings=mappings,
        dataset_ref=dataset_ref,
        learning_dynamics_design1_payload=design1_payload,
    )

    comp_path = paths.memory_bank_actions_dir() / 'memory_bank_composition_stage_2.json'
    assert comp_path.exists()
    report = json.loads(comp_path.read_text())
    assert report['selection_strategy'] == 'learning_dynamics_design1'
    # Stage-2 quota is 3 but only one candidate exists, so shortfall is explicit.
    assert report['quota_check_passed'] is False
    d1 = report.get('learning_dynamics_design1', {})
    assert int(d1.get('current_stage_target', -1)) == 3
    assert int(d1.get('required_add_t', -1)) == 3
    assert int(d1.get('shortfall_t', -1)) == 2
    assert int(d1.get('current_stage_added', -1)) == 1


def test_stage_ratio_learning_dynamics_design2_writes_design2_report_key(tmp_path):
    paths = IncrementalPaths(tmp_path)
    dataset_ref = _DummyDatasetRef(paths=paths)

    mb = SceneMemoryBank(
        max_memory_scenes=6,
        quota_strategy='stage_ratio',
        stage_scene_counts={1: 10, 2: 10},  # stage2 quota upper bound = 3
        selection_strategy='learning_dynamics_design2',
        learning_dynamics_design2=dict(
            q_metric='f1',
            min_add_lower_bound=1,
            force_accept_until_lower_bound=True,
            use_class_balance=True,
            supply_scaling_mode='cap_log1p',
            supply_cap=20,
            w_max=10.0,
            redundancy_lambda=0.3,
            redundancy_topk=5,
            min_class_quota=5,
        ),
        enforce_unique_scene_ids=False,
        debug_mode=False,
        min_objects_per_scene=1,
    )
    mappings = {}

    # Prefill stage-1 seats directly.
    for i in range(6):
        sid = f"s1_{i}"
        info = _make_scene_info(sid, [0, 0])
        snap = {
            'save_stage': 1,
            'present_classes': [0],
            'object_counts': {0: 2},
            'data_info': info,
        }
        mb._add_scene_to_memory(sid, 1, snap, importance=1.0)

    stage2_infos = [_make_scene_info('s2_only', [0, 1])]
    seat_terms = {
        f"s1_{i}": {
            1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}
        }
        for i in range(6)
    }
    seat_terms['s2_only'] = {
        2: {
            0: {'g': 0.0, 'r_best': 0.2, 'd': 0.0, 'u': 0.1, 'r_start': 0.2, 'r_end': 0.2},
            1: {'g': 0.0, 'r_best': 0.0, 'd': 0.0, 'u': 0.0, 'r_start': 0.0, 'r_end': 0.0},
        }
    }
    design2_payload = {
        'stage_id': 2,
        'q_metric': 'f1',
        'class_need': {0: 0.7, 1: 0.3},
        'seat_class_terms': seat_terms,
    }

    mb.add_stage_scenes(
        stage_id=2,
        scene_infos=stage2_infos,
        seen_classes=[0, 1],
        mappings=mappings,
        dataset_ref=dataset_ref,
        learning_dynamics_design2_payload=design2_payload,
    )

    comp_path = paths.memory_bank_actions_dir() / 'memory_bank_composition_stage_2.json'
    assert comp_path.exists()
    report = json.loads(comp_path.read_text())
    assert report['selection_strategy'] == 'learning_dynamics_design2'
    assert 'learning_dynamics_design2' in report
    assert 'learning_dynamics_design1' not in report
