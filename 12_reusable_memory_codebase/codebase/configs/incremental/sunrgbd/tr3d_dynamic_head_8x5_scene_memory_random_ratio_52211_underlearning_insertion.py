"""
SUN RGB-D Incremental Learning (8×5) - Scene Memory Bank

Variant: Under-learning insertion (TRAIN-only, new-class AP@0.25)

This config enables the under-learning based insertion policy for the SUNRGBD
scene memory bank:
- Compute new-class AP on the *train* natural pool at the end of each stage
- Convert to under-learning weights u(c)=1-AP(c)
- Insert current-stage scenes by object_count_sum over new classes (top-K)

Eviction behavior remains the baseline stage-ratio quota update (random).
"""

from mmcv import Config

base_config = Config.fromfile(
    'configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_scene_memory_random_ratio_52211.py'
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

