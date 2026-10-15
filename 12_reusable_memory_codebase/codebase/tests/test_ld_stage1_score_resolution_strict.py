from __future__ import annotations

from pathlib import Path

import pytest

from tools.train_incremental_scene import _resolve_stage1_ld_scores_path_for_checkpoint


def _make_stage1_layout(tmp_path: Path):
    run_dir = tmp_path / 'run_a'
    ckpt = run_dir / 'checkpoints' / 'stage_1' / 'epoch_1.pth'
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text('ckpt')
    stage1_ld_dir = run_dir / 'learning_dynamics' / 'stage_1'
    stage1_ld_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt, stage1_ld_dir


def test_explicit_stage1_score_path_missing_fails_without_fallback(tmp_path):
    run_dir, ckpt, stage1_ld_dir = _make_stage1_layout(tmp_path)
    canonical = stage1_ld_dir / 'learning_dynamics_design1_scores.json'
    canonical.write_text('{}')

    explicit_missing = tmp_path / 'missing_explicit.json'
    cfg = dict(
        learning_dynamics_design1=dict(
            stage1_scores_file=str(explicit_missing),
        )
    )

    with pytest.raises(RuntimeError) as exc_info:
        _resolve_stage1_ld_scores_path_for_checkpoint(
            checkpoint_path=str(ckpt),
            scene_memory_config=cfg,
            config_block_key='learning_dynamics_design1',
            score_filename='learning_dynamics_design1_scores.json',
            strategy_name='learning_dynamics_design1',
        )
    msg = str(exc_info.value)
    assert str(explicit_missing) in msg
    assert str(canonical) not in msg


def test_explicit_stage1_score_path_exists_passes(tmp_path):
    _, ckpt, _ = _make_stage1_layout(tmp_path)
    explicit = tmp_path / 'explicit_scores.json'
    explicit.write_text('{}')

    cfg = dict(
        learning_dynamics_design2=dict(
            stage1_scores_file=str(explicit),
        )
    )

    resolved = _resolve_stage1_ld_scores_path_for_checkpoint(
        checkpoint_path=str(ckpt),
        scene_memory_config=cfg,
        config_block_key='learning_dynamics_design2',
        score_filename='learning_dynamics_design2_scores.json',
        strategy_name='learning_dynamics_design2',
    )
    assert resolved == explicit.resolve()


def test_default_canonical_stage1_score_path_is_used_when_explicit_absent(tmp_path):
    _, ckpt, stage1_ld_dir = _make_stage1_layout(tmp_path)
    canonical = stage1_ld_dir / 'learning_dynamics_design1_scores.json'
    canonical.write_text('{}')

    cfg = dict(learning_dynamics_design1=dict())
    resolved = _resolve_stage1_ld_scores_path_for_checkpoint(
        checkpoint_path=str(ckpt),
        scene_memory_config=cfg,
        config_block_key='learning_dynamics_design1',
        score_filename='learning_dynamics_design1_scores.json',
        strategy_name='learning_dynamics_design1',
    )
    assert resolved == canonical.resolve()


def test_design1_strategy_does_not_accept_design2_filename(tmp_path):
    _, ckpt, stage1_ld_dir = _make_stage1_layout(tmp_path)
    (stage1_ld_dir / 'learning_dynamics_design2_scores.json').write_text('{}')

    cfg = dict(learning_dynamics_design1=dict())
    with pytest.raises(RuntimeError) as exc_info:
        _resolve_stage1_ld_scores_path_for_checkpoint(
            checkpoint_path=str(ckpt),
            scene_memory_config=cfg,
            config_block_key='learning_dynamics_design1',
            score_filename='learning_dynamics_design1_scores.json',
            strategy_name='learning_dynamics_design1',
        )
    assert 'learning_dynamics_design1_scores.json' in str(exc_info.value)


def test_design2_strategy_does_not_accept_design1_filename(tmp_path):
    _, ckpt, stage1_ld_dir = _make_stage1_layout(tmp_path)
    (stage1_ld_dir / 'learning_dynamics_design1_scores.json').write_text('{}')

    cfg = dict(learning_dynamics_design2=dict())
    with pytest.raises(RuntimeError) as exc_info:
        _resolve_stage1_ld_scores_path_for_checkpoint(
            checkpoint_path=str(ckpt),
            scene_memory_config=cfg,
            config_block_key='learning_dynamics_design2',
            score_filename='learning_dynamics_design2_scores.json',
            strategy_name='learning_dynamics_design2',
        )
    assert 'learning_dynamics_design2_scores.json' in str(exc_info.value)
