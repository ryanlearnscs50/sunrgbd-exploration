"""
SUN RGB-D Incremental Learning (8×5) - Scene Memory Bank (Random + Stage-Ratio Quotas) (5-2-2-1-1)

This config is a SUNRGBD memory-only baseline:
- Dynamic head expansion: 8 → 16 → 24 → 32 → 40
- Scene memory replay enabled
- Memory selection happens during the run (no precomputed score files)
- Deterministic random selection via seeding (see `tools/train_incremental_scene.py`)
- Memory budget: 10% of SUNRGBD train split (5285 → 528)
- Stage-ratio quotas: allocate memory entries proportional to stage train-scene counts `N_s`
- Dedup between replay + natural scenes: `merge_labels`
- Pseudo labels: DISABLED

IMPORTANT ABOUT "EPOCHS" IN THIS REPO:
- Training uses a RepeatDataset wrapper (`data.train.times`).
- One "epoch" means iterating over the repeated dataset (times × inner_dataset_length).
- This config sets `data.train.times=15` explicitly for reproducibility and to match
  the common ScanNet incremental convention used in this repo.

When running with `--start-stage 2`, Stage 1 training is skipped, so the "5"
epochs in 5-2-2-1-1 are not executed (effective trained epochs: 0-2-2-1-1).
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

# Explicitly disable pseudo labels (baseline scope)
use_pseudo_labels = False
pseudo_label_config = None

# Evaluation settings
evaluation = dict(interval=1, metric='mAP')

# Export the modified base config
locals().update(base_config)

