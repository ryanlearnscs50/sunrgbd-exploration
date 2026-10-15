"""
SUN RGB-D Incremental Learning (8×5) - FAST DEBUG + Under-learning Insertion

This debug config enables under-learning insertion while keeping the fast
1-1-1-1-1 schedule from the existing SUNRGBD debug config.
"""

from mmcv import Config

base_config = Config.fromfile(
    'configs/incremental/sunrgbd/debug/tr3d_dynamic_head_8x5_scene_memory_random_ratio_52211_debug.py'
)

base_config.scene_memory_config.setdefault(
    'underlearning_insertion',
    dict(
        enabled=True,
        score_mode='object_count_sum',
        ap_iou_thr=0.25,
        # Keep evaluation fast for debug.
        eval_max_scenes=64,
        eval_seed_offset=11000,
    ),
)

locals().update(base_config)

