"""
SUN RGB-D Incremental Learning (8 classes × 5 stages) - Pure Finetuning + Dynamic Head

Stage split and class order are defined by:
  `configs/_base_/class_mappings/sunrgbd_40class_mapping.py`

This config is the baseline for SUN RGB-D incremental experiments:
- No pseudo labels
- No memory replay
- Dynamic head expansion: 8 → 16 → 24 → 32 → 40
"""

from mmcv import Config

# Base (upper bound) supervised config
base_config = Config.fromfile('configs/tr3d/tr3d_sunrgbd-3d-40class.py')

# Use incremental dataset wrapper for stage-wise GT filtering
base_config.data.train.dataset.type = 'IncrementalSUNRGBDDataset'

# Training: drop empty scenes after stage filtering
base_config.data.train.dataset.filter_empty_gt = True

# Enable incremental loss masking hooks (dynamic head still works without it,
# but keeping it on makes the behavior explicit and consistent).
base_config.model.head.train_cfg.enable_class_masking = True

# Dynamic head expansion enabled
use_dynamic_head = True

# Stage definitions (8×5)
import sys
sys.path.append('configs/_base_/class_mappings')
from sunrgbd_40class_mapping import get_stage_definitions  # type: ignore

stage_definitions = []
for stage_def in get_stage_definitions():
    enhanced = stage_def.copy()
    enhanced['epochs'] = 12
    enhanced['lr'] = 0.001
    stage_definitions.append(enhanced)

# Disable advanced incremental components for the baseline
scene_memory_config = None
use_scene_memory = False

# Evaluation settings
evaluation = dict(interval=1, metric='mAP')

# Export the modified base config
locals().update(base_config)

