"""Exercise scheduler selection through the actual stage-config handoff."""
import copy
from types import SimpleNamespace
from mmcv import Config
from mmcv.runner.hooks.lr_updater import CosineAnnealingLrUpdaterHook
import pytest
from tools.train_incremental_scene import prepare_stage_config
PREFIX = 'configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_'

@pytest.mark.parametrize('policy', ['pseudo_only_cosine', 'object_memory_pseudo_dose25_cosine'])
def test_cosine_reaches_each_stage_and_decays(tmp_path, policy):
    cfg = Config.fromfile(PREFIX + policy + '_52211.py')
    before = copy.deepcopy(cfg.base_config.lr_config)
    for idx in range(1, 5):
        stage = prepare_stage_config(cfg.base_config, cfg.stage_definitions[idx], idx,
                                     cfg.stage_definitions, str(tmp_path), incremental_cfg=cfg)
        options = dict(stage.lr_config)
        assert options.pop('policy') == 'CosineAnnealing'
        assert 'step' not in options
        hook = CosineAnnealingLrUpdaterHook(**options)
        n = stage.runner.max_epochs * 100
        rates = [hook.get_lr(SimpleNamespace(iter=i, max_iters=n), stage.optimizer.lr)
                 for i in range(n)]
        assert rates[0] == .001
        assert .0001 < rates[-1] < .000101
        assert all(a > b for a, b in zip(rates, rates[1:]))
        stage.lr_config.min_lr_ratio = .9
        assert cfg.lr_config.min_lr_ratio == .1
    assert cfg.base_config.lr_config == before

def test_existing_policy_and_missing_override_keep_base_schedule(tmp_path):
    cfg = Config.fromfile(PREFIX + 'pseudo_only_matched_52211.py')
    for incremental in (cfg, None):
        stage = prepare_stage_config(cfg.base_config, cfg.stage_definitions[1], 1,
                                     cfg.stage_definitions, str(tmp_path), incremental_cfg=incremental)
        assert stage.lr_config == cfg.base_config.lr_config
