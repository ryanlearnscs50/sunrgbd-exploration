"""
SUN RGB-D Incremental Learning (4×10) - Pure Finetuning + Dynamic Head (6-1-1-1-1-1-1-1-1-1)

Stage split and global class order are defined by:
  `configs/_base_/class_mappings/sunrgbd_40class_mapping.py`

This config introduces a NEW 10-stage setting by splitting the existing 8×5
frequency-ordered schedule into two sub-stages each (contiguous split), while
preserving the exact same global class order.

Key properties:
- Stage setting: `sunrgbd40_s10_freqorder_split`
- 10 stages × 4 classes/stage
- Dynamic head expansion: 4 → 8 → 12 → ... → 40
- No pseudo labels
- No memory replay
- Per-stage epochs: 6-1-1-1-1-1-1-1-1-1 (Stage 1..10)

IMPORTANT ABOUT "EPOCHS" IN THIS REPO:
- Training uses a RepeatDataset wrapper (`data.train.times`).
- One "epoch" means iterating over the repeated dataset (times × inner_dataset_length).
- This config sets `data.train.times=15` explicitly for reproducibility and to match
  the common ScanNet incremental convention used in this repo.
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

# Stage setting (4×10, frequency-ordered; contiguous split)
stage_setting = 'sunrgbd40_s10_freqorder_split'
incremental = dict(stage_setting=stage_setting)

# Stage definitions (4×10)
import sys
sys.path.append('configs/_base_/class_mappings')
from sunrgbd_40class_mapping import get_stage_definitions  # type: ignore

epoch_map = {1: 6, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1}

stage_definitions = []
for stage_def in get_stage_definitions(stage_setting=stage_setting):
    enhanced = stage_def.copy()
    enhanced['epochs'] = int(epoch_map[int(stage_def['stage_id'])])
    enhanced['lr'] = 0.001
    stage_definitions.append(enhanced)

# No memory bank for this baseline
scene_memory_config = None
use_scene_memory = False
scene_dedup_strategy = 'merge_labels'  # irrelevant when memory disabled

# Explicitly disable pseudo labels (baseline scope)
use_pseudo_labels = False
pseudo_label_config = None

# Evaluation settings
evaluation = dict(interval=1, metric='mAP')

# Export the modified base config
locals().update(base_config)

