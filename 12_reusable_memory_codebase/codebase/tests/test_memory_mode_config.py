"""Public mode selection must preserve legacy configs and fail before training."""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

from mmcv import Config
import pytest

from tools.incremental_scene.memory_config import normalize_memory_mode


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = 'configs/memory/sunrgbd_s5_random.py'


@pytest.mark.parametrize('config,expected', [
    ({}, 'scene'),
    ({'scene_memory_config': None}, 'none'),
    ({'use_scene_memory': False, 'object_memory_config': {}}, 'object'),
    ({'use_scene_memory': False, 'use_object_memory': False,
      'object_memory_config': {}}, 'none'),
])
def test_legacy_mode_defaults(config, expected):
    assert normalize_memory_mode(config) == expected
    assert normalize_memory_mode(config) == expected  # normalization is idempotent


def test_ambiguous_legacy_modes_fail():
    with pytest.raises(ValueError, match='cannot be enabled together'):
        normalize_memory_mode({'object_memory_config': {}})


@pytest.mark.parametrize('mode', ['scene', 'object', 'none'])
def test_explicit_mode_preserves_both_bank_configs(mode):
    cfg = dict(memory_mode=mode, use_scene_memory=True, use_object_memory=True,
               scene_memory_config=dict(max_memory_scenes=10),
               object_memory_config=dict(exemplars_per_class=5))
    banks = copy.deepcopy(cfg)
    assert normalize_memory_mode(cfg) == mode
    assert cfg['use_scene_memory'] == (mode == 'scene')
    assert cfg['use_object_memory'] == (mode == 'object')
    for key in ('scene_memory_config', 'object_memory_config'):
        assert cfg[key] == banks[key]
    assert normalize_memory_mode(cfg, 'object') == 'object'


@pytest.mark.parametrize('mode', ['scene', 'object'])
@pytest.mark.parametrize('bank_config', [None, [], 'random'])
def test_explicit_switch_requires_bank_mapping(mode, bank_config):
    with pytest.raises(ValueError, match='requires a'):
        normalize_memory_mode({f'{mode}_memory_config': bank_config}, mode)


def test_invalid_mode_is_rejected():
    with pytest.raises(ValueError, match='memory_mode must'):
        normalize_memory_mode({'memory_mode': 'both'})


@pytest.mark.parametrize('dataset_hint', [
    dict(stage_setting='scannet35_s5_freqorder'),
    dict(data=dict(train=dict(dataset=dict(type='IncrementalScanNetDataset')))),
])
def test_scannet_object_design2_fails_before_training(dataset_hint):
    cfg = dict(dataset_hint, memory_mode='object', object_memory_config=dict(
        selection_strategy='learning_dynamics_design2'))
    with pytest.raises(ValueError, match='ScanNet object Design-2'):
        normalize_memory_mode(cfg)


@pytest.mark.parametrize('dataset,mode,legacy', [
    ('sunrgbd', 'scene', 'tr3d_dynamic_head_8x5_scene_memory_random_ratio_52211.py'),
    ('sunrgbd', 'object', 'tr3d_dynamic_head_8x5_object_memory_random_52211.py'),
    ('scannet', 'scene', 'tr3d_dynamic_head_s5_scene_memory_random.py'),
    ('scannet', 'object', 'tr3d_dynamic_head_s5_object_memory_random.py'),
])
def test_example_preserves_existing_random_baseline_settings(monkeypatch, dataset, mode, legacy):
    monkeypatch.chdir(ROOT)
    actual = Config.fromfile(f'configs/memory/{dataset}_s5_random.py')
    expected = Config.fromfile(f'configs/incremental/{dataset}/' + legacy)
    normalize_memory_mode(actual, mode)
    normalize_memory_mode(expected)
    for key in ('stage_definitions', 'data', 'model', 'optimizer',
                'use_scene_memory', 'use_object_memory', 'use_pseudo_labels',
                f'{mode}_memory_config'):
        # ConfigDict compares nested configuration values rather than identities.
        assert actual[key] == expected[key], key
    assert not actual.reviewing.enabled


@pytest.mark.parametrize('entrypoint,mode,example', [
    ('train_incremental.py', 'scene', EXAMPLE),
    ('train_incremental.py', 'object', EXAMPLE),
    ('train_incremental.py', 'none', EXAMPLE),
    ('train_incremental_scene.py', 'object', EXAMPLE),
    ('train_incremental.py', 'scene', 'configs/memory/scannet_s5_random.py'),
    ('train_incremental.py', 'object', 'configs/memory/scannet_s5_random.py'),
])
def test_public_cli_preflight_does_not_create_work_dir(tmp_path, entrypoint, mode, example):
    work_dir = tmp_path / 'must_not_be_created'
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools' / entrypoint), example,
         '--memory-mode', mode, '--validate-memory-config',
         '--work-dir', str(work_dir)], cwd=ROOT, text=True, capture_output=True,
        env=dict(os.environ, OMP_NUM_THREADS='2', CUDA_VISIBLE_DEVICES=''), timeout=90)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report['memory_mode'] == mode
    assert report['use_scene_memory'] == (mode == 'scene')
    assert report['use_object_memory'] == (mode == 'object')
    assert not work_dir.exists()


def test_cli_override_wins_over_cfg_options_and_rejects_invalid_bank():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools/train_incremental.py'), EXAMPLE,
         '--memory-mode', 'object', '--validate-memory-config',
         '--cfg-options', 'memory_mode=scene',
         'object_memory_config.exemplars_per_class=0'], cwd=ROOT,
        text=True, capture_output=True, timeout=90)
    assert result.returncode != 0
    assert 'Object-memory budgets must be positive' in result.stderr
