"""
SUNRGBD Design-1 tuning preset: raw supply scaling + compatibility_weight=10.

Rationale (synthetic diagnostics): with raw object-count supply, median unary
magnitude is ~10x median compatibility-kernel magnitude, so weight~10 gives
comparable influence in swap deltas.
"""

from mmcv import Config

base_config = Config.fromfile(
    'configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_scene_memory_ld_design1_ratio_52211.py'
)

d1 = dict(base_config.scene_memory_config.get('learning_dynamics_design1', {}) or {})
d1.update(dict(
    q_metric='f1',
    supply_scaling_mode='raw',
    compatibility_weight=10.0,
    min_add_lower_bound=1,
))
base_config.scene_memory_config['learning_dynamics_design1'] = d1

reviewing_cfg = dict(base_config.get('reviewing', {}) or {})
reviewing_cfg['enabled'] = False
base_config['reviewing'] = reviewing_cfg

locals().update(base_config)
