"""
SUN RGB-D Incremental Learning (8×5) - Pseudo Labels + Scene Memory Bank (Random + Stage-Ratio Quotas) (5-2-2-1-1)

Strong SUNRGBD baseline:
- Dynamic head expansion: 8 → 16 → 24 → 32 → 40
- Pseudo labels: ENABLED (pre-generated once per stage using stage t-1 checkpoint)
- Scene memory replay: ENABLED (random selection with stage-ratio quotas)
- Per-stage epochs: 5-2-2-1-1 (Stage 1..5)

Pseudo label conventions:
- Saved boxes match SUNRGBD GT arrays (`gt_boxes_upright_depth`):
  gravity-centred (origin=(0.5,0.5,0.5)) with yaw (7D).
- During training, pseudo labels are injected pre-pipeline with unified policy:
  natural scenes by default, or natural+replay if
  `pseudo_label_config.apply_to_memory_scenes=True`.
"""

from mmcv import Config

# Base supervised (upper bound) config
base_config = Config.fromfile('configs/tr3d/tr3d_sunrgbd-3d-40class.py')

# Use incremental dataset wrapper for stage-wise GT filtering
base_config.data.train.dataset.type = 'IncrementalSUNRGBDDataset'

# Training: drop empty scenes after stage filtering
base_config.data.train.dataset.filter_empty_gt = True

# Make RepeatDataset repetition explicit (important for interpreting epochs)
base_config.data.train.times = 15

# Enable incremental loss masking hooks (keeps behavior explicit)
base_config.model.head.train_cfg.enable_class_masking = True

# Dynamic head expansion enabled
use_dynamic_head = True

# Stage definitions (8×5)
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

# Enable scene memory bank (random selection + stage-ratio quotas)
use_scene_memory = True

scene_memory_config = dict(
    # Budget (scenes)
    memory_budget_ratio=0.1,
    total_training_scenes=5285,

    # Selection: random during the run (seeded)
    selection_strategy='random',
    quota_strategy='stage_ratio',

    # Quality thresholds
    min_objects_per_scene=1,

    # Debug/logging
    debug_mode=True,

    # Explicitly disable ScanNet-oriented weighting knobs
    use_drop_weights=False,
    use_current_stage_weights=False,
    enforce_current_quota=False,
    min_current_stage_quota=0,
    stage_ratio_counts=[],  # prevent ScanNet defaults from being injected
)

# Dedup strategy between replay and natural occurrences
scene_dedup_strategy = 'merge_labels'

# Enable pseudo labels (pre-generated once at stage start).
use_pseudo_labels = True
pseudo_label_config = dict(
    use_pregenerated=True,
    apply_to_memory_scenes=False,
    confidence_threshold=0.50,
    nms_threshold=0.30,
    max_pseudo_per_scene=100,
    pseudo_vs_gt_iou_thr=0.25,
    pseudo_nms_iou_thr=0.30,
)

# Evaluation settings
evaluation = dict(interval=1, metric='mAP')

# Export the modified base config
locals().update(base_config)
