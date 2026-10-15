from __future__ import annotations

import pytest

from mmdet3d.utils.ld_strategy_config import (
    LD_DESIGN1_STRATEGY,
    LD_DESIGN2_STRATEGY,
    validate_scene_memory_ld_strategy_config,
)


def test_design1_requires_design1_block():
    with pytest.raises(ValueError) as exc_info:
        validate_scene_memory_ld_strategy_config(
            dict(selection_strategy=LD_DESIGN1_STRATEGY)
        )
    assert "requires" in str(exc_info.value)
    assert "learning_dynamics_design1 block" in str(exc_info.value)


def test_design2_requires_design2_block():
    with pytest.raises(ValueError) as exc_info:
        validate_scene_memory_ld_strategy_config(
            dict(selection_strategy=LD_DESIGN2_STRATEGY)
        )
    assert "requires" in str(exc_info.value)
    assert "learning_dynamics_design2 block" in str(exc_info.value)


def test_design1_requires_explicit_q_metric():
    with pytest.raises(ValueError) as exc_info:
        validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=LD_DESIGN1_STRATEGY,
                learning_dynamics_design1=dict(
                    use_compatibility_kernel=True,
                    compatibility_weight=1.0,
                ),
            )
        )
    msg = str(exc_info.value)
    assert "q_metric" in msg
    assert "explicitly set" in msg


def test_design2_requires_explicit_q_metric():
    with pytest.raises(ValueError) as exc_info:
        validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=LD_DESIGN2_STRATEGY,
                learning_dynamics_design2=dict(
                    w_max=10.0,
                    redundancy_lambda=0.3,
                    redundancy_topk=5,
                    min_class_quota=5,
                ),
            )
        )
    msg = str(exc_info.value)
    assert "q_metric" in msg
    assert "explicitly set" in msg


def test_design1_rejects_design2_block():
    with pytest.raises(ValueError) as exc_info:
        validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=LD_DESIGN1_STRATEGY,
                learning_dynamics_design1=dict(q_metric='f1'),
                learning_dynamics_design2=dict(q_metric='f1', w_max=10.0),
            )
        )
    assert "forbids" in str(exc_info.value)
    assert "learning_dynamics_design2 block" in str(exc_info.value)


def test_design2_rejects_design1_block():
    with pytest.raises(ValueError) as exc_info:
        validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=LD_DESIGN2_STRATEGY,
                learning_dynamics_design1=dict(q_metric='f1'),
                learning_dynamics_design2=dict(q_metric='f1', w_max=10.0),
            )
        )
    assert "forbids" in str(exc_info.value)
    assert "learning_dynamics_design1 block" in str(exc_info.value)


def test_design1_rejects_design2_only_keys():
    with pytest.raises(ValueError) as exc_info:
        validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=LD_DESIGN1_STRATEGY,
                learning_dynamics_design1=dict(
                    q_metric='f1',
                    w_max=10.0,
                ),
            )
        )
    assert "Design-2-only" in str(exc_info.value)
    assert "w_max" in str(exc_info.value)


def test_design2_rejects_design1_only_keys():
    with pytest.raises(ValueError) as exc_info:
        validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=LD_DESIGN2_STRATEGY,
                learning_dynamics_design2=dict(
                    q_metric='f1',
                    compatibility_weight=1.0,
                ),
            )
        )
    assert "Design-1-only" in str(exc_info.value)
    assert "compatibility_weight" in str(exc_info.value)


def test_design_blocks_reject_unknown_keys():
    with pytest.raises(ValueError) as exc_info1:
        validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=LD_DESIGN1_STRATEGY,
                learning_dynamics_design1=dict(
                    q_metric='f1',
                    unexpected_knob=123,
                ),
            )
        )
    assert "unknown/forbidden keys" in str(exc_info1.value)
    assert "unexpected_knob" in str(exc_info1.value)

    with pytest.raises(ValueError) as exc_info2:
        validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=LD_DESIGN2_STRATEGY,
                learning_dynamics_design2=dict(
                    q_metric='f1',
                    unexpected_knob=456,
                ),
            )
        )
    assert "unknown/forbidden keys" in str(exc_info2.value)
    assert "unexpected_knob" in str(exc_info2.value)


def test_legacy_mixed_config_emits_explicit_migration_error():
    with pytest.raises(ValueError) as exc_info:
        validate_scene_memory_ld_strategy_config(
            dict(
                selection_strategy=LD_DESIGN1_STRATEGY,
                learning_dynamics_design1=dict(
                    q_metric='f1',
                    design_version=2,
                ),
            )
        )
    msg = str(exc_info.value)
    assert "Legacy mixed LD config is no longer supported" in msg
    assert "selection_strategy='learning_dynamics_design2'" in msg


def test_shared_overlap_keys_allowed_for_both_designs():
    d1 = validate_scene_memory_ld_strategy_config(
        dict(
            selection_strategy=LD_DESIGN1_STRATEGY,
            learning_dynamics_design1=dict(
                q_metric='f1',
                stage1_scores_mode='precomputed',
                stage1_scores_file='foo.json',
                stage1_stats_dir='bar',
                min_add_lower_bound=1,
                force_accept_until_lower_bound=True,
                supply_scaling_mode='raw',
                use_class_balance=True,
                use_compatibility_kernel=True,
                compatibility_weight=1.0,
            ),
        )
    )
    assert d1['active_ld_block_key'] == 'learning_dynamics_design1'
    assert d1['selection_strategy'] == LD_DESIGN1_STRATEGY

    d2 = validate_scene_memory_ld_strategy_config(
        dict(
            selection_strategy=LD_DESIGN2_STRATEGY,
            learning_dynamics_design2=dict(
                q_metric='f1',
                stage1_scores_mode='precomputed',
                stage1_scores_file='foo.json',
                stage1_stats_dir='bar',
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
        )
    )
    assert d2['active_ld_block_key'] == 'learning_dynamics_design2'
    assert d2['selection_strategy'] == LD_DESIGN2_STRATEGY
