"""
SUN RGB-D Incremental Learning (4x10) - Scene Memory Bank (LD Design-1 + Stage-Ratio)

Key points:
- Stage setting: sunrgbd40_s10_freqorder_split (10 stages x 4 classes)
- Dynamic head expansion: 4 -> 8 -> 12 -> ... -> 40
- Per-stage epochs: 6-1-1-1-1-1-1-1-1-1
- Scene memory replay enabled
- selection_strategy='learning_dynamics_design1'
- Reviewing disabled by default
- Pseudo labels disabled in this baseline
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

    selection_strategy='learning_dynamics_design1',
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
        compatibility_weight=10.0,
        supply_scaling_mode='raw',  # {'raw', 'cap', 'log1p', 'cap_log1p'}
        # supply_cap=20,  # required for 'cap' and 'cap_log1p'
        force_accept_until_lower_bound=True,
    ),
)

scene_dedup_strategy = 'merge_labels'

# Pseudo labels disabled
use_pseudo_labels = False
pseudo_label_config = None

# Reviewing disabled in this baseline; both q_metric='recall' (canonical default)
# and q_metric='f1' are supported when reviewing is enabled.
reviewing = dict(
    enabled=False,
    review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5],
)
reviewing_legacy_pseudo_consistency = dict(enabled=False)

evaluation = dict(interval=1, metric='mAP')

locals().update(base_config)
