"""
SUN RGB-D Incremental Learning (4x10) - Scene Memory Bank (LD Design-2 + Stage-Ratio)

Key points:
- Stage setting: sunrgbd40_s10_freqorder_split (10 stages x 4 classes)
- Dynamic head expansion: 4 -> 8 -> 12 -> ... -> 40
- Per-stage epochs: 6-1-1-1-1-1-1-1-1-1
- Scene memory replay enabled
- selection_strategy='learning_dynamics_design2'

LD Design-2 changes vs Design-1:
- supply_scaling_mode='cap_log1p' with supply_cap=20 (compressed supply)
- Stronger class balance: 1/(1+count) capped at w_max=10.0
- Redundancy penalty instead of compatibility reward (lambda=0.3)
- Minimum per-class scene quota N=5
- Per-scene aggregation (Bug C.1 fix)
- Single-checkpoint fallback (Bug C.4 fix)

Ablation controls:
- redundancy_lambda=0   -> pure unary (no redundancy penalty)
- min_class_quota=0      -> disable quota boost
- w_max=9999             -> effectively uncapped balance
- supply_scaling_mode='raw', supply_cap removed -> revert to Design-1 supply
"""

from mmcv import Config

# Base supervised config
base_config = Config.fromfile('configs/tr3d/tr3d_sunrgbd-3d-40class.py')

# Incremental SUNRGBD dataset wrapper
base_config.data.train.dataset.type = 'IncrementalSUNRGBDDataset'
base_config.data.train.dataset.filter_empty_gt = True

# Keep RepeatDataset repetition explicit
base_config.data.train.times = 15

# Enable incremental masking hooks
base_config.model.head.train_cfg.enable_class_masking = True

# Dynamic head expansion enabled
use_dynamic_head = True

# 4x10 stage setting and stage definitions
stage_setting = 'sunrgbd40_s10_freqorder_split'
incremental = dict(stage_setting=stage_setting)

import sys
sys.path.append('configs/_base_/class_mappings')
from sunrgbd_40class_mapping import get_stage_definitions  # type: ignore

epoch_map = {
    1: 6,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
}

stage_definitions = []
for stage_def in get_stage_definitions(stage_setting=stage_setting):
    enhanced = stage_def.copy()
    enhanced['epochs'] = int(epoch_map[int(stage_def['stage_id'])])
    enhanced['lr'] = 0.001
    stage_definitions.append(enhanced)

# Enable scene memory replay
use_scene_memory = True

SCORING = dict(
    LD_IOU_MODE='0.50',
)

scene_memory_config = dict(
    memory_budget_ratio=0.1,
    total_training_scenes=5285,

    selection_strategy='learning_dynamics_design2',
    quota_strategy='stage_ratio',
    enforce_unique_scene_ids=False,

    min_objects_per_scene=1,
    debug_mode=True,

    use_drop_weights=False,
    use_current_stage_weights=False,
    enforce_current_quota=False,
    min_current_stage_quota=0,
    stage_ratio_counts=[],

    learning_dynamics_update=dict(
        eps=1e-9,
        object_count_cap=20,
        report_topk=30,
    ),

    learning_dynamics_design2=dict(
        q_metric='recall',         # {'f1', 'recall'} — matches Design-1 experiments

        min_add_lower_bound=1,
        use_class_balance=True,

        # Step 1A: compressed supply scaling.
        # cap_log1p = log(1 + min(count, cap)) prevents high-count scenes
        # from dominating unary scores.  cap=20 chosen to match
        # object_count_cap (counts above 20 are already extreme outliers
        # in SUNRGBD).
        supply_scaling_mode='cap_log1p',
        supply_cap=20,

        # Step 1B: stronger class balance with cap.
        # w_max=10: a zero-count class gets at most 10x the weight of a
        # well-represented class.  Prevents single-class explosions while
        # meaningfully boosting rare classes.
        w_max=10.0,

        # Step 1C: minimum per-class scene quota.
        # If any class has fewer than N=5 scenes in the bank, candidates
        # containing that class receive a priority boost.
        min_class_quota=5,

        # Step 2: redundancy penalty.
        # redundancy_lambda=0.3: moderate diversity pressure without
        # overwhelming the unary quality signal.  lambda=0 gives pure
        # unary behaviour (useful ablation).
        redundancy_lambda=0.3,
        redundancy_topk=5,         # top-k bank neighbours for penalty

        force_accept_until_lower_bound=True,
    ),
)

scene_dedup_strategy = 'merge_labels'

# Pseudo labels disabled
use_pseudo_labels = False
pseudo_label_config = None

# Reviewing disabled (required for recall-mode runs currently)
reviewing = dict(
    enabled=False,
    review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5],
)
reviewing_legacy_pseudo_consistency = dict(enabled=False)

evaluation = dict(interval=1, metric='mAP')

locals().update(base_config)
