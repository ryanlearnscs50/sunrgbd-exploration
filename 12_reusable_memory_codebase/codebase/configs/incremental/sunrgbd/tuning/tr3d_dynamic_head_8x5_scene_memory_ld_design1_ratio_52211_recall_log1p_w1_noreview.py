"""
SUNRGBD Design-1 recall preset: q_metric='recall', log1p supply scaling,
compatibility_weight=1, reviewing disabled.

NOTE:
- This preset keeps reviewing.disabled for controlled tuning, not because of
  a compatibility limitation.
- Design-1 reviewing with `reviewing.weight_policy.type='ld_drop'` supports
  both q_metric='f1' and q_metric='recall'.
"""

from mmcv import Config

base_config = Config.fromfile(
    'configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_scene_memory_ld_design1_ratio_52211.py'
)

d1 = dict(base_config.scene_memory_config.get('learning_dynamics_design1', {}) or {})
d1.update(dict(
    q_metric='recall',
    supply_scaling_mode='log1p',
    compatibility_weight=1.0,
    min_add_lower_bound=1,
))
base_config.scene_memory_config['learning_dynamics_design1'] = d1

reviewing_cfg = dict(base_config.get('reviewing', {}) or {})
reviewing_cfg['enabled'] = False
base_config['reviewing'] = reviewing_cfg

locals().update(base_config)
