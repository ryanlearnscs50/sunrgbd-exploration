"""
SUN RGB-D Incremental Learning (4x10) - Scene Memory Bank (Random + Stage-Ratio Quotas) (6-1-1-1-1-1-1-1-1-1)

This config is a SUNRGBD memory-only baseline for the 10-stage (4 classes/stage) setting:
- Stage setting: sunrgbd40_s10_freqorder_split (10 stages x 4 classes)
- Dynamic head expansion: 4 -> 8 -> 12 -> ... -> 40
- Per-stage epochs: 6-1-1-1-1-1-1-1-1-1
- Scene memory replay enabled
- Memory selection: random during the run (seeded)
- Memory budget: 10% of SUNRGBD train split (5285 -> 528)
- Stage-ratio quotas: allocate memory entries proportional to stage train-scene counts
- Dedup between replay + natural scenes: merge_labels
- Pseudo labels: DISABLED (enable via --cfg-options)
- Reviewing: DISABLED (enable via --cfg-options)
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

# Enable scene memory replay (random selection + stage-ratio quotas)
use_scene_memory = True

scene_memory_config = dict(
    memory_budget_ratio=0.1,
    total_training_scenes=5285,

    selection_strategy='random',
    quota_strategy='stage_ratio',

    min_objects_per_scene=1,
    debug_mode=True,

    use_drop_weights=False,
    use_current_stage_weights=False,
    enforce_current_quota=False,
    min_current_stage_quota=0,
    stage_ratio_counts=[],
)

scene_dedup_strategy = 'merge_labels'

# Pseudo labels disabled by default (enable via --cfg-options)
use_pseudo_labels = False
pseudo_label_config = None

# Reviewing disabled by default (enable via --cfg-options)
reviewing = dict(
    enabled=False,
    review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5],
)
reviewing_legacy_pseudo_consistency = dict(enabled=False)

evaluation = dict(interval=1, metric='mAP')

locals().update(base_config)
