"""
SUN RGB-D Incremental Learning (8×5) - Reviewing (GT-only on Memory Bank)

This config enables the SUNRGBD reviewing mechanism implemented in
`tools/train_incremental_scene.py`:
- Evaluates AP/AR at IoU=0.25 and IoU=0.50 on memory bank seats at fixed fractions of stage progress
- Computes per-class drops vs the last evaluation (clamped at 0)
- Derives class/seat weights (drop-dominant) and oversamples memory seats by
  materializing duplicates into the dataset for the next segment
- Keeps inner dataset length fixed (memory ↑ ⇒ natural ↓)

Pseudo labels remain disabled for SUNRGBD.
"""

from mmcv import Config

base_config = Config.fromfile(
    'configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_scene_memory_random_ratio_52211.py'
)

# Enable forgetness-based memory bank eviction (GLOBAL across stages).
# - Forgetness: old-class AP drop between stage-start and stage-end reviewing eval.
# - Eviction: remove least-forgotten entries to make room for new scenes.
# - Insertion: remains random and uses stage_scene_counts-derived quotas.
# NOTE: Eviction does NOT maintain per-stage quotas (future work may make this finer-grained).
base_config.scene_memory_config.setdefault(
    'forgetness_eviction',
    dict(
        enabled=True,
        score_mode='presence_sum',
        protect_new_stage=True,
    ),
)

# Enable reviewing (explicit; no silent techniques).
reviewing = dict(
    enabled=True,
    # Default: 4 within-stage updates (recommended).
    review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5],
    # Always report both IoUs during reviewing evaluation.
    eval_iou_thrs=[0.25, 0.5],
    # Choose which IoU (0.25 or 0.50) is used to derive drop/class/entry weights.
    weight_iou_thr=0.25,
    compare_to='last',
    drop_clamp_min=0.0,
    weight_policy=dict(
        type='drop_dominant_sum',
        alpha_drop=1.0,
        beta_ap=0.1,
        gamma=5.0,
        w_max=10.0,
        eta=5.0,
    ),
    sampling=dict(
        mode='coverage_preserving',
        weight_space='combined',
        memory_share_max=0.9,
        strict_memory_coverage=True,
        seed_offset=9000,
    ),
    resume_optimizer=True,
)

# Explicitly keep legacy pseudo-consistency segmentation disabled.
reviewing_legacy_pseudo_consistency = dict(enabled=False)

locals().update(base_config)
