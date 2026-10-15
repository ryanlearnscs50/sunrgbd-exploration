"""`tools/eval_incremental.py` must derive the stage from the checkpoint's head
width and the config's stage definitions, not from a hard-coded 7-classes-per-stage
assumption. The latter silently mis-evaluates every protocol except ScanNet 5-stage.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from mmcv import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / 'configs' / 'incremental'

# (config, per-stage cumulative class counts)
PROTOCOLS = {
    'sunrgbd/tr3d_dynamic_head_20x10x10_pseudo_memory_ld_design2_reviewing_521.py':
        [20, 30, 40],
    'sunrgbd/tr3d_dynamic_head_8x5_pseudo_memory_ld_design2_reviewing_52211.py':
        [8, 16, 24, 32, 40],
    'sunrgbd/tr3d_dynamic_head_4x10_pseudo_memory_ld_design2_reviewing_6111111111.py':
        [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
    'scannet/tr3d_dynamic_head_s3_15_10_10_pseudo_memory_ld_design2_reviewing.py':
        [15, 25, 35],
    'scannet/tr3d_dynamic_head_s5_pseudo_memory_ld_design2_reviewing.py':
        [7, 14, 21, 28, 35],
    'scannet/tr3d_dynamic_head_s10_4444433333_pseudo_memory_ld_design2_reviewing.py':
        [4, 8, 12, 16, 20, 23, 26, 29, 32, 35],
}


@pytest.fixture(scope='module')
def eval_incremental():
    """Import the CLI module without executing it."""
    sys.path.insert(0, str(REPO_ROOT / 'configs' / '_base_' / 'class_mappings'))
    spec = importlib.util.spec_from_file_location(
        'eval_incremental', REPO_ROOT / 'tools' / 'eval_incremental.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_checkpoint(path, n_classes, stage_id=None):
    """A checkpoint is identified by the width of head.cls_conv.kernel."""
    meta = {'CLASSES': tuple(f'c{i}' for i in range(40))}
    if stage_id is not None:
        meta['stage_id'] = stage_id
    torch.save({'meta': meta,
                'state_dict': {'head.cls_conv.kernel': torch.zeros(128, n_classes)}},
               path)
    return path


@pytest.mark.parametrize('config_rel,expected_counts', PROTOCOLS.items(),
                         ids=lambda v: v if isinstance(v, str) else '')
def test_cumulative_class_counts_match_protocol(eval_incremental, config_rel, expected_counts):
    cfg = Config.fromfile(str(CONFIGS / config_rel))
    counts = eval_incremental.cumulative_class_counts(cfg.stage_definitions)
    assert counts == expected_counts


@pytest.mark.parametrize('config_rel,expected_counts', PROTOCOLS.items(),
                         ids=lambda v: v if isinstance(v, str) else '')
def test_every_stage_is_detected(eval_incremental, tmp_path, config_rel, expected_counts):
    cfg = Config.fromfile(str(CONFIGS / config_rel))
    stage_definitions = cfg.stage_definitions

    for stage_id, n_classes in enumerate(expected_counts, start=1):
        ckpt = _fake_checkpoint(tmp_path / f'stage_{stage_id:02d}.pth', n_classes)
        got_stage, got_n = eval_incremental.detect_checkpoint_stage(ckpt, stage_definitions)
        assert (got_stage, got_n) == (stage_id, n_classes)


def test_recorded_stage_id_never_overrides_head_width(eval_incremental, tmp_path):
    """A stage-10 SUN RGB-D checkpoint must not be read as stage 5 / 35 classes."""
    cfg = Config.fromfile(str(
        CONFIGS / 'sunrgbd/tr3d_dynamic_head_4x10_pseudo_memory_ld_design2_reviewing_6111111111.py'))
    ckpt = _fake_checkpoint(tmp_path / 'stage_10.pth', n_classes=40, stage_id=10)
    assert eval_incremental.detect_checkpoint_stage(ckpt, cfg.stage_definitions) == (10, 40)


def test_mismatched_protocol_raises(eval_incremental, tmp_path):
    """A 25-class ScanNet checkpoint has no counterpart in the SUN RGB-D 4x10 protocol."""
    cfg = Config.fromfile(str(
        CONFIGS / 'sunrgbd/tr3d_dynamic_head_4x10_pseudo_memory_ld_design2_reviewing_6111111111.py'))
    ckpt = _fake_checkpoint(tmp_path / 'foreign.pth', n_classes=25)
    with pytest.raises(ValueError, match='Could not determine the incremental stage'):
        eval_incremental.detect_checkpoint_stage(ckpt, cfg.stage_definitions)


def test_sunrgbd_val_dataset_is_not_rewritten_to_scannet(eval_incremental):
    """setup_incremental_dataset must not turn a SUN RGB-D val set into a ScanNet one."""
    cfg = Config.fromfile(str(
        CONFIGS / 'sunrgbd/tr3d_dynamic_head_4x10_pseudo_memory_ld_design2_reviewing_6111111111.py'))
    assert cfg.data.val.type == 'SUNRGBDDataset'

    info = eval_incremental.detect_incremental_config(cfg)
    info['n_classes'] = 12                       # pretend a stage-3 checkpoint
    cfg = eval_incremental.setup_incremental_dataset(cfg, info)

    assert cfg.data.val.type == 'SUNRGBDDataset'
    assert not hasattr(cfg.data.val, 'variant')
    assert cfg.model.head.n_classes == 12


def test_scannet_val_dataset_keeps_dynamic_head_variant(eval_incremental):
    cfg = Config.fromfile(str(
        CONFIGS / 'scannet/tr3d_dynamic_head_s5_pseudo_memory_ld_design2_reviewing.py'))
    info = eval_incremental.detect_incremental_config(cfg)
    info['n_classes'] = 21                       # pretend a stage-3 checkpoint
    cfg = eval_incremental.setup_incremental_dataset(cfg, info)

    assert cfg.data.val.type == 'ScanNetDataset'
    assert cfg.data.val.variant == 'dynamic_head'
    assert cfg.model.head.n_classes == 21


@pytest.mark.parametrize('config_rel', list(PROTOCOLS))
def test_training_split_unwraps_repeat_and_preserves_family(eval_incremental, config_rel):
    cfg = Config.fromfile(str(CONFIGS / config_rel))
    cfg = eval_incremental.setup_incremental_dataset(cfg, eval_incremental.detect_incremental_config(cfg))
    original_val = cfg.data.val.ann_file
    source = cfg.data.train
    while source.get('type') == 'RepeatDataset':
        source = source.dataset
    selected = eval_incremental.select_evaluation_dataset_config(cfg, 'train')
    assert selected.ann_file == source.ann_file
    assert selected.data_root == source.data_root
    assert selected.type == cfg.data.val.type
    assert selected.pipeline == cfg.data.val.pipeline
    assert cfg.data.val.ann_file == original_val
    if selected.type == 'SUNRGBDDataset':
        assert 'variant' not in selected
