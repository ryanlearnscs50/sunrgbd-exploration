"""
SUN RGB-D Incremental Learning (8x5) - Scene Memory Bank (LD Design-1 + Stage-Ratio) (5-2-2-1-1)

This config enables the new memory-bank strategy:
  selection_strategy='learning_dynamics_design1'

Key points:
- Dynamic head expansion: 8 -> 16 -> 24 -> 32 -> 40
- Stage-ratio quotas are used as current-stage upper bounds
- Lower-bound admission is configurable via learning_dynamics_design1.min_add_lower_bound
- Multi-snapshot scenes are allowed (enforce_unique_scene_ids=False)
- Pseudo labels are disabled in this baseline
"""

from mmcv import Config

# Base supervised (upper bound) config
base_config = Config.fromfile('configs/tr3d/tr3d_sunrgbd-3d-40class.py')

# Use incremental dataset wrapper for stage-wise GT filtering
base_config.data.train.dataset.type = 'IncrementalSUNRGBDDataset'
base_config.data.train.dataset.filter_empty_gt = True

# RepeatDataset repetition kept explicit for reproducibility
base_config.data.train.times = 15

# Incremental masking hooks
base_config.model.head.train_cfg.enable_class_masking = True

# Dynamic head expansion enabled
use_dynamic_head = True

# Stage definitions (8x5)
import sys
sys.path.append('configs/_base_/class_mappings')
from sunrgbd_40class_mapping import get_stage_definitions  # type: ignore

epoch_map = {1: 5, 2: 2, 3: 2, 4: 1, 5: 1}

stage_definitions = []
for stage_def in get_stage_definitions():
    enhanced = stage_def.copy()
    enhanced['epochs'] = int(epoch_map[int(stage_def['stage_id'])])
    enhanced['lr'] = 0.001
    stage_definitions.append(enhanced)

# Scene memory replay enabled
use_scene_memory = True

SCORING = dict(
    # Learning-dynamics IoU threshold for per-k TP/FP/FN matching.
    LD_IOU_MODE='0.50',
)

scene_memory_config = dict(
    # Budget
    memory_budget_ratio=0.1,
    total_training_scenes=5285,

    # New Design-1 strategy
    selection_strategy='learning_dynamics_design1',
    quota_strategy='stage_ratio',
    enforce_unique_scene_ids=False,

    # Quality thresholds
    min_objects_per_scene=1,

    # Debug/logging
    debug_mode=True,

    # Keep non-LD weighting knobs disabled for SUNRGBD
    use_drop_weights=False,
    use_current_stage_weights=False,
    enforce_current_quota=False,
    min_current_stage_quota=0,
    stage_ratio_counts=[],  # prevent ScanNet defaults from being injected

    # Shared LD scoring controls
    learning_dynamics_update=dict(
        eps=1e-9,
        object_count_cap=20,
        report_topk=30,
    ),

    # Design-1 controls
    learning_dynamics_design1=dict(
        q_metric='recall',  # {'f1', 'recall'}; canonical default = recall
        # Stage-1 skip prework modes:
        # - 'precomputed' (default): load stage1_scores_file or checkpoint-linked default.
        # - 'recompute_from_stats': rebuild scores from memory_stats_k*/natural_stats_k*.
        # stage1_scores_mode='precomputed',
        # Optional explicit file override for --start-stage 2 strict prework:
        # stage1_scores_file='incremental_logs/SUN_RGBD/<run>/learning_dynamics/stage_1/learning_dynamics_design1_scores.json',
        min_add_lower_bound=1,
        use_compatibility_kernel=True,
        compatibility_weight=1.0,
        # Supply scaling for unary term:
        # - 'raw' (default): uses raw object counts
        # - 'cap': min(count, supply_cap)
        # - 'log1p': log(1 + count)
        # - 'cap_log1p': log(1 + min(count, supply_cap))
        supply_scaling_mode='raw',
        # supply_cap is required for 'cap' and 'cap_log1p'
        # supply_cap=20,
        force_accept_until_lower_bound=True,
    ),
)

# Dedup strategy between replay and natural occurrences
scene_dedup_strategy = 'merge_labels'

# Pseudo labels disabled in this baseline
use_pseudo_labels = False
pseudo_label_config = None

# Reviewing can remain disabled; trainer will still run segmented LD collection.
# NOTE:
# - reviewing.enabled=False is a baseline default, not a recall limitation.
# - For Design-1 + reviewing.weight_policy.type='ld_drop', both
#   q_metric='f1' and q_metric='recall' are supported.
reviewing = dict(
    enabled=False,
    review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5],
)

evaluation = dict(interval=1, metric='mAP')

locals().update(base_config)
