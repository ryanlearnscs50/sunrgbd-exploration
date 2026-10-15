"""
SUNRGBD Design-1 tuning preset: log1p supply scaling + compatibility_weight=1.

Rationale (synthetic diagnostics): log1p compression makes unary and
compatibility terms naturally similar in scale (ratio ~1), so weight=1 is a
reasonable baseline.
"""

from mmcv import Config

base_config = Config.fromfile(
    'configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_scene_memory_ld_design1_ratio_52211.py'
)

d1 = dict(base_config.scene_memory_config.get('learning_dynamics_design1', {}) or {})
d1.update(dict(
    q_metric='f1',
    supply_scaling_mode='log1p',
    compatibility_weight=1.0,
    min_add_lower_bound=1,
))
base_config.scene_memory_config['learning_dynamics_design1'] = d1

reviewing_cfg = dict(base_config.get('reviewing', {}) or {})
reviewing_cfg['enabled'] = False
base_config['reviewing'] = reviewing_cfg

locals().update(base_config)
