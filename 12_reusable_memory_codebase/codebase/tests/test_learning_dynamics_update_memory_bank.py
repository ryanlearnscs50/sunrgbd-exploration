import numpy as np

from mmdet3d.datasets.scene_memory_bank import SceneMemoryBank
from mmdet3d.utils.learning_dynamics_scoring import (
    compute_learning_dynamics_scores,
    compute_reviewing_entry_weights_ld_drop,
)


def _make_snapshot(save_stage, present_classes, object_counts=None):
    return {
        'save_stage': int(save_stage),
        'present_classes': [int(x) for x in present_classes],
        'object_counts': object_counts or {},
        'data_info': {'annos': {'gt_num': 1}},
    }


def _make_candidate(scene_id, stage_id, present_classes, object_counts=None):
    snap = _make_snapshot(stage_id, present_classes, object_counts=object_counts)
    return {
        'scene_id': str(scene_id),
        'snapshot': snap,
        'present_classes': snap['present_classes'],
        'stage_id': int(stage_id),
    }


def test_learning_dynamics_update_evicts_lowest_forgetness_and_adds_top_priority():
    mb = SceneMemoryBank(
        memory_budget_ratio=1.0,
        max_memory_scenes=4,
        total_training_scenes=4,
        quota_strategy='stage_ratio',
        selection_strategy='learning_dynamics',
        stage_scene_counts=[3, 1],  # stage2 target = 1 (with budget=4)
        learning_dynamics_update={},
        random_seed=0,
        debug_mode=False,
    )

    # Fill memory with 4 stage1 seats.
    mb._add_scene_to_memory('scene_a', 1, _make_snapshot(1, [0]), importance=1.0)
    mb._add_scene_to_memory('scene_b', 1, _make_snapshot(1, [0]), importance=1.0)
    mb._add_scene_to_memory('scene_c', 1, _make_snapshot(1, [0]), importance=1.0)
    mb._add_scene_to_memory('scene_d', 1, _make_snapshot(1, [0]), importance=1.0)
    assert mb._count_scene_stage_pairs() == 4

    candidates = [
        _make_candidate('scene_new_hi', 2, [1]),
        _make_candidate('scene_new_lo', 2, [1]),
    ]

    # Evict the least-forgotten old seat (lowest score).
    forgetness_by_seat = {
        'scene_a': {1: 0.5},
        'scene_b': {1: 0.2},
        'scene_c': {1: 0.1},
        'scene_d': {1: 0.0},  # should be evicted
    }

    # Admit the highest-priority new seat.
    replay_priority_by_seat = {
        'scene_new_hi': {2: 0.9},
        'scene_new_lo': {2: 0.1},
    }

    _, quotas, actual, report = mb._apply_stage_ratio_update_learning_dynamics(
        stage_id=2,
        candidate_scenes=candidates,
        forgetness_by_seat=forgetness_by_seat,
        replay_priority_by_seat=replay_priority_by_seat,
    )

    assert quotas == {1: 3, 2: 1}
    assert report['quota_pruned_count'] == 1
    pruned = report['quota_pruned_entries']
    assert len(pruned) == 1
    assert pruned[0]['scene_id'] == 'scene_d'
    assert pruned[0]['save_stage'] == 1
    assert pruned[0]['reason'] == 'quota_prune_old_stage'

    added = report['added_entries']
    assert len(added) == 1
    assert added[0]['scene_id'] == 'scene_new_hi'
    assert added[0]['save_stage'] == 2

    assert actual[2] == 1
    assert mb._count_scene_stage_pairs() == 4

    # Old seats in the eviction pool get their scores persisted.
    assert mb.memory_scenes['scene_a']['stages'][1]['learning_dynamics_forgetness'] == 0.5
    assert mb.memory_scenes['scene_b']['stages'][1]['learning_dynamics_forgetness'] == 0.2
    assert mb.memory_scenes['scene_c']['stages'][1]['learning_dynamics_forgetness'] == 0.1
    assert 'scene_d' not in mb.memory_scenes or 1 not in mb.memory_scenes.get('scene_d', {}).get('stages', {})

    # Added seat gets replay priority persisted.
    assert mb.memory_scenes['scene_new_hi']['stages'][2]['learning_dynamics_replay_priority'] == 0.9


def test_learning_dynamics_update_requires_scores_for_all_old_seats():
    mb = SceneMemoryBank(
        memory_budget_ratio=1.0,
        max_memory_scenes=2,
        total_training_scenes=2,
        quota_strategy='stage_ratio',
        selection_strategy='learning_dynamics',
        stage_scene_counts=[1, 1],
        learning_dynamics_update={},
        random_seed=0,
        debug_mode=False,
    )
    mb._add_scene_to_memory('scene_a', 1, _make_snapshot(1, [0]), importance=1.0)
    mb._add_scene_to_memory('scene_b', 1, _make_snapshot(1, [0]), importance=1.0)

    candidates = [_make_candidate('scene_new', 2, [1])]
    replay_priority = {'scene_new': {2: 1.0}}

    try:
        mb._apply_stage_ratio_update_learning_dynamics(
            stage_id=2,
            candidate_scenes=candidates,
            forgetness_by_seat={'scene_a': {1: 1.0}},  # missing scene_b
            replay_priority_by_seat=replay_priority,
        )
        assert False, "Expected missing score error"
    except RuntimeError as e:
        assert 'missing' in str(e).lower()


def test_learning_dynamics_update_skips_duplicate_scene_ids_and_fills_with_next_candidate():
    mb = SceneMemoryBank(
        memory_budget_ratio=1.0,
        max_memory_scenes=4,
        total_training_scenes=4,
        quota_strategy='stage_ratio',
        selection_strategy='learning_dynamics',
        stage_scene_counts=[2, 2],  # stage2 target = 2 (with budget=4)
        learning_dynamics_update={},
        random_seed=0,
        debug_mode=False,
    )

    # Fill memory with 4 stage1 seats.
    mb._add_scene_to_memory('scene_a', 1, _make_snapshot(1, [0]), importance=1.0)
    mb._add_scene_to_memory('scene_b', 1, _make_snapshot(1, [0]), importance=1.0)
    mb._add_scene_to_memory('scene_c', 1, _make_snapshot(1, [0]), importance=1.0)
    mb._add_scene_to_memory('scene_d', 1, _make_snapshot(1, [0]), importance=1.0)
    assert mb._count_scene_stage_pairs() == 4

    # Candidate list includes:
    # - Duplicate new scene id (scene_new_hi appears twice)
    # - An existing memory scene id (scene_a) which must be skipped
    # - A second unique new scene (scene_new_lo)
    candidates = [
        _make_candidate('scene_new_hi', 2, [1]),
        _make_candidate('scene_new_hi', 2, [1]),
        _make_candidate('scene_a', 2, [1]),
        _make_candidate('scene_new_lo', 2, [1]),
    ]

    forgetness_by_seat = {
        'scene_a': {1: 0.9},
        'scene_b': {1: 0.8},
        'scene_c': {1: 0.1},
        'scene_d': {1: 0.0},
    }

    replay_priority_by_seat = {
        'scene_new_hi': {2: 0.9},
        'scene_new_lo': {2: 0.1},
        'scene_a': {2: 1.0},  # should be ignored since scene_a is already in memory
    }

    _, quotas, actual, report = mb._apply_stage_ratio_update_learning_dynamics(
        stage_id=2,
        candidate_scenes=candidates,
        forgetness_by_seat=forgetness_by_seat,
        replay_priority_by_seat=replay_priority_by_seat,
    )

    assert quotas == {1: 2, 2: 2}
    assert actual[1] == 2
    assert actual[2] == 2
    assert mb._count_scene_stage_pairs() == 4
    assert mb._count_scene_stage_pairs() == len(mb.memory_scenes)

    added_ids = {e['scene_id'] for e in report['added_entries']}
    assert 'scene_new_hi' in added_ids
    assert 'scene_new_lo' in added_ids
    assert 'scene_a' not in added_ids


def test_ld_scoring_reviewing_weights_and_memory_update_are_consistent():
    """Single-design consistency: seat stats -> LD forgetness -> review weights -> bank action."""

    def _tp_fn_for_q(q: float):
        q = float(q)
        assert 0.0 < q <= 1.0
        # With fp=0, gt_count=1: q = 2tp / (2tp + fn) and fn = 1 - tp.
        tp = q / (2.0 - q)
        fn = 1.0 - tp
        return float(tp), float(fn)

    mb = SceneMemoryBank(
        memory_budget_ratio=1.0,
        max_memory_scenes=4,
        total_training_scenes=4,
        quota_strategy='stage_ratio',
        selection_strategy='learning_dynamics',
        stage_scene_counts=[3, 1],
        learning_dynamics_update={},
        random_seed=0,
        debug_mode=False,
    )

    # Fill memory with 4 stage1 seats.
    for sid in ['scene_a', 'scene_b', 'scene_c', 'scene_d']:
        mb._add_scene_to_memory(sid, 1, _make_snapshot(1, [0]), importance=1.0)
    assert mb._count_scene_stage_pairs() == 4

    # Build 2-point seat stats (k=0->1) for old class 0.
    q1_by_scene = {
        'scene_a': 0.50,  # drop 0.50 (highest)
        'scene_b': 0.80,  # drop 0.20
        'scene_c': 0.90,  # drop 0.10
        'scene_d': 1.00,  # drop 0.00 (lowest)
    }
    k0 = []
    k1 = []
    for sid, q1 in q1_by_scene.items():
        k0.append(
            {'scene_id': sid, 'save_stage': 1, 'classes': {0: {'tp': 1.0, 'fp': 0.0, 'fn': 0.0, 'gt_count': 1}}}
        )
        tp1, fn1 = _tp_fn_for_q(q1)
        k1.append(
            {'scene_id': sid, 'save_stage': 1, 'classes': {0: {'tp': tp1, 'fp': 0.0, 'fn': fn1, 'gt_count': 1}}}
        )
    scores = compute_learning_dynamics_scores(
        {0: k0, 1: k1},
        old_classes=[0],
        new_classes=[],
        iou_mode='0.25',
        slope_k_start=0,
        slope_k_end=1,
        object_count_cap=20,
        eps=1e-9,
        return_trajectories=False,
    )
    forgetness_by_seat = scores['forgetness_by_seat']

    # Reviewing weights from the same two snapshots must induce the same ordering.
    weights = compute_reviewing_entry_weights_ld_drop(
        k0,
        k1,
        old_classes=[0],
        iou_mode='0.25',
        object_count_cap=20,
        eps=1e-9,
        eta=5.0,
        normalize_by_gt_weight=True,
        w_entry_max=None,
    )['weights_by_uid']
    assert float(weights['scene_a_stage1']) > float(weights['scene_b_stage1']) > float(weights['scene_c_stage1']) > float(weights['scene_d_stage1'])

    candidates = [
        _make_candidate('scene_new_hi', 2, [1]),
        _make_candidate('scene_new_lo', 2, [1]),
    ]
    replay_priority_by_seat = {
        'scene_new_hi': {2: 0.9},
        'scene_new_lo': {2: 0.1},
    }

    _, _, _, report = mb._apply_stage_ratio_update_learning_dynamics(
        stage_id=2,
        candidate_scenes=candidates,
        forgetness_by_seat=forgetness_by_seat,
        replay_priority_by_seat=replay_priority_by_seat,
    )

    pruned = report['quota_pruned_entries']
    assert len(pruned) == 1
    assert pruned[0]['scene_id'] == 'scene_d'


def test_learning_dynamics_design1_ratio_target_adds_to_stage_quota():
    mb = SceneMemoryBank(
        memory_budget_ratio=1.0,
        max_memory_scenes=4,
        total_training_scenes=4,
        quota_strategy='stage_ratio',
        selection_strategy='learning_dynamics_design1',
        stage_scene_counts=[3, 1],  # stage2 upper bound = 1
        learning_dynamics_design1=dict(
            q_metric='f1',
            min_add_lower_bound=1,
            use_compatibility_kernel=False,
            force_accept_until_lower_bound=True,
        ),
        enforce_unique_scene_ids=False,
        random_seed=0,
        debug_mode=False,
    )

    for sid in ['scene_a', 'scene_b', 'scene_c', 'scene_d']:
        mb._add_scene_to_memory(
            sid,
            1,
            _make_snapshot(1, [0], object_counts={0: 1}),
            importance=1.0,
        )
    assert mb._count_scene_stage_pairs() == 4

    candidates = [
        _make_candidate('scene_new', 2, [0], object_counts={0: 1}),
    ]
    payload = {
        'stage_id': 2,
        'q_metric': 'f1',
        'class_need': {0: 1.0},
        'seat_class_terms': {
            'scene_a': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}},
            'scene_b': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}},
            'scene_c': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}},
            'scene_d': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}},
            'scene_new': {2: {0: {'g': 0.0, 'r_best': 0.0, 'd': 0.0, 'u': 0.0, 'r_start': 0.0, 'r_end': 0.0}}},
        },
    }

    _, quotas, actual, report = mb._apply_stage_ratio_update_learning_dynamics_design1(
        stage_id=2,
        candidate_scenes=candidates,
        learning_dynamics_design1_payload=payload,
    )
    assert quotas == {1: 3, 2: 1}
    assert actual[2] == 1
    assert report['current_stage_target'] == 1
    assert report['required_add_t'] == 1
    assert report['current_stage_added'] == 1
    assert report['shortfall_t'] == 0
    assert report['exact_target_match'] is True
    assert 'scene_new' in mb.memory_scenes
    assert 2 in mb.memory_scenes['scene_new']['stages']


def test_learning_dynamics_design1_ratio_target_does_not_use_delta_rejection():
    mb = SceneMemoryBank(
        memory_budget_ratio=1.0,
        max_memory_scenes=4,
        total_training_scenes=4,
        quota_strategy='stage_ratio',
        selection_strategy='learning_dynamics_design1',
        stage_scene_counts=[3, 1],  # stage2 upper bound = 1
        learning_dynamics_design1=dict(
            q_metric='f1',
            min_add_lower_bound=0,
            use_compatibility_kernel=False,
            force_accept_until_lower_bound=False,
        ),
        enforce_unique_scene_ids=False,
        random_seed=0,
        debug_mode=False,
    )

    for sid in ['scene_a', 'scene_b', 'scene_c', 'scene_d']:
        mb._add_scene_to_memory(
            sid,
            1,
            _make_snapshot(1, [0], object_counts={0: 1}),
            importance=1.0,
        )
    assert mb._count_scene_stage_pairs() == 4

    candidates = [
        _make_candidate('scene_new', 2, [0], object_counts={0: 1}),
    ]
    payload = {
        'stage_id': 2,
        'q_metric': 'f1',
        'class_need': {0: 1.0},
        'seat_class_terms': {
            'scene_a': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}},
            'scene_b': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}},
            'scene_c': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}},
            'scene_d': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0, 'r_start': 1.0, 'r_end': 0.0}}},
            'scene_new': {2: {0: {'g': 0.0, 'r_best': 0.0, 'd': 0.0, 'u': 0.0, 'r_start': 0.0, 'r_end': 0.0}}},
        },
    }

    _, quotas, actual, report = mb._apply_stage_ratio_update_learning_dynamics_design1(
        stage_id=2,
        candidate_scenes=candidates,
        learning_dynamics_design1_payload=payload,
    )
    assert quotas == {1: 3, 2: 1}
    assert actual.get(2, 0) == 1
    assert report['required_add_t'] == 1
    assert report['shortfall_t'] == 0
    assert report['current_stage_added'] == 1
    assert report['exact_target_match'] is True
    assert 'scene_new' in mb.memory_scenes


def test_learning_dynamics_design1_recall_q_metric_is_supported_in_update():
    mb = SceneMemoryBank(
        memory_budget_ratio=1.0,
        max_memory_scenes=4,
        total_training_scenes=4,
        quota_strategy='stage_ratio',
        selection_strategy='learning_dynamics_design1',
        stage_scene_counts=[3, 1],  # stage2 upper bound = 1
        learning_dynamics_design1=dict(
            q_metric='recall',
            min_add_lower_bound=1,
            use_compatibility_kernel=False,
            force_accept_until_lower_bound=True,
        ),
        enforce_unique_scene_ids=False,
        random_seed=0,
        debug_mode=False,
    )

    for sid in ['scene_a', 'scene_b', 'scene_c', 'scene_d']:
        mb._add_scene_to_memory(
            sid,
            1,
            _make_snapshot(1, [0], object_counts={0: 1}),
            importance=1.0,
        )

    candidates = [_make_candidate('scene_new', 2, [0], object_counts={0: 1})]
    payload = {
        'stage_id': 2,
        'q_metric': 'recall',
        'class_need': {0: 1.0},
        'seat_class_terms': {
            'scene_a': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0}}},
            'scene_b': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0}}},
            'scene_c': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0}}},
            'scene_d': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0}}},
            'scene_new': {2: {0: {'g': 0.0, 'r_best': 0.0, 'd': 0.0, 'u': 0.0}}},
        },
    }

    _, quotas, actual, report = mb._apply_stage_ratio_update_learning_dynamics_design1(
        stage_id=2,
        candidate_scenes=candidates,
        learning_dynamics_design1_payload=payload,
    )
    assert mb.learning_dynamics_design1_q_metric == 'recall'
    assert report['q_metric'] == 'recall'
    assert report['supply_scaling_mode'] == 'raw'
    assert report['supply_cap'] is None
    assert quotas == {1: 3, 2: 1}
    assert actual[2] == 1
    assert report['required_add_t'] == 1
    assert report['shortfall_t'] == 0
    assert report['current_stage_added'] == 1
    assert report['exact_target_match'] is True
    assert 'scene_new' in mb.memory_scenes
    assert 2 in mb.memory_scenes['scene_new']['stages']


def test_learning_dynamics_design2_strategy_uses_explicit_design2_block():
    mb = SceneMemoryBank(
        memory_budget_ratio=1.0,
        max_memory_scenes=4,
        total_training_scenes=4,
        quota_strategy='stage_ratio',
        selection_strategy='learning_dynamics_design2',
        stage_scene_counts=[3, 1],  # stage2 upper bound = 1
        learning_dynamics_design2=dict(
            q_metric='f1',
            min_add_lower_bound=1,
            force_accept_until_lower_bound=True,
            supply_scaling_mode='cap_log1p',
            supply_cap=20,
            use_class_balance=True,
            w_max=10.0,
            redundancy_lambda=0.3,
            redundancy_topk=5,
            min_class_quota=5,
        ),
        enforce_unique_scene_ids=False,
        random_seed=0,
        debug_mode=False,
    )

    for sid in ['scene_a', 'scene_b', 'scene_c', 'scene_d']:
        mb._add_scene_to_memory(
            sid,
            1,
            _make_snapshot(1, [0], object_counts={0: 1}),
            importance=1.0,
        )

    candidates = [_make_candidate('scene_new', 2, [0], object_counts={0: 1})]
    payload = {
        'stage_id': 2,
        'q_metric': 'f1',
        'class_need': {0: 1.0},
        'seat_class_terms': {
            'scene_a': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0}}},
            'scene_b': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0}}},
            'scene_c': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0}}},
            'scene_d': {1: {0: {'g': 0.0, 'r_best': 1.0, 'd': 1.0, 'u': 1.0}}},
            'scene_new': {2: {0: {'g': 0.0, 'r_best': 0.0, 'd': 0.0, 'u': 0.0}}},
        },
    }

    _, quotas, actual, report = mb._apply_stage_ratio_update_learning_dynamics_design1(
        stage_id=2,
        candidate_scenes=candidates,
        learning_dynamics_design1_payload=payload,
    )
    assert quotas == {1: 3, 2: 1}
    assert actual[2] == 1
    assert report['policy'] == 'learning_dynamics_design2'
    assert int(report['design_version']) == 2
    assert np.isclose(float(report['design2_redundancy_lambda']), 0.3)
    assert int(report['design2_redundancy_topk']) == 5
    assert int(report['design2_min_class_quota']) == 5
