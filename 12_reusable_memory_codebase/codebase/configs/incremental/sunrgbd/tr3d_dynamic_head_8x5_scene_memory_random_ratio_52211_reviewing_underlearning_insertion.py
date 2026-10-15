"""
SUN RGB-D Incremental Learning (8×5) - Reviewing + Under-learning Insertion

This config layers under-learning insertion on top of the existing reviewing
config (which also enables forgetness-based eviction).

Behavior:
- Reviewing (train(memory_bank_subset)) computes old-class AP drops → forgetness eviction
- Train(natural) evaluation computes new-class AP@0.25 → under-learning insertion

This is intended for the "both on" ablation.
"""

from mmcv import Config

base_config = Config.fromfile(
    'configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_scene_memory_random_ratio_52211_reviewing.py'
)

base_config.scene_memory_config.setdefault(
    'underlearning_insertion',
    dict(
        enabled=True,
        score_mode='object_count_sum',
        ap_iou_thr=0.25,
        # Optional speed knob (deterministic subsample); set to None for full pool.
        eval_max_scenes=None,
        eval_seed_offset=11000,
    ),
)

locals().update(base_config)

