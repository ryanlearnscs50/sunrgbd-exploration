from __future__ import annotations

from mmcv import Config


def test_scannet_config_matrix_parses_required_files():
    cfg_paths = [
        'configs/incremental/scannet/tr3d_dynamic_head_s5_pure_finetuning.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s5_pseudo_only.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s5_scene_memory_random.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s5_pseudo_memory_random.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s5_scene_memory_ld_design2_reviewing.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s5_pseudo_memory_ld_design2_reviewing.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s3_15_10_10_pure_finetuning.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s10_4444433333_pure_finetuning.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s3_15_10_10_pseudo_memory_ld_design2_reviewing.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s10_4444433333_pseudo_memory_ld_design2_reviewing.py',
    ]

    for path in cfg_paths:
        cfg = Config.fromfile(path)
        stage_defs = list(cfg.stage_definitions)
        assert stage_defs, path
        stage_ids = [int(sd['stage_id']) for sd in stage_defs]
        assert stage_ids == list(range(1, len(stage_defs) + 1)), path
        if 'ld_design2' in path:
            scene_memory_cfg = dict(cfg.get('scene_memory_config', {}) or {})
            assert scene_memory_cfg, path
            assert scene_memory_cfg.get('selection_strategy') == 'learning_dynamics_design2', path
            assert isinstance(scene_memory_cfg.get('learning_dynamics_design2'), dict), path
            assert 'learning_dynamics_design1' not in scene_memory_cfg, path
