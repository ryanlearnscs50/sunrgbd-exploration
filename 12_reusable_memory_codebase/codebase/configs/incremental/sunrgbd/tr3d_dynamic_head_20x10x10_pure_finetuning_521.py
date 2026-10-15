"""
SUN RGB-D Incremental Learning (20/10/10) - Pure Finetuning + Dynamic Head (5-2-2)

Stage setting:
- `sunrgbd40_s3_20_10_10_freqorder`
- Stage sizes: 20 -> 10 -> 10
- Dynamic head: 20 -> 30 -> 40
"""

from mmcv import Config

# Base supervised (upper bound) config
base_config = Config.fromfile('configs/tr3d/tr3d_sunrgbd-3d-40class.py')

# Use incremental dataset wrapper for stage-wise GT filtering
base_config.data.train.dataset.type = 'IncrementalSUNRGBDDataset'
base_config.data.train.dataset.filter_empty_gt = True

# Keep RepeatDataset repetition explicit
base_config.data.train.times = 15

# Enable incremental masking hooks
base_config.model.head.train_cfg.enable_class_masking = True

# Dynamic head expansion enabled
use_dynamic_head = True

# 3-stage setting and stage definitions
stage_setting = 'sunrgbd40_s3_20_10_10_freqorder'
incremental = dict(stage_setting=stage_setting)

import sys
sys.path.append('configs/_base_/class_mappings')
from sunrgbd_40class_mapping import get_stage_definitions  # type: ignore

epoch_map = {1: 5, 2: 2, 3: 2}

stage_definitions = []
for stage_def in get_stage_definitions(stage_setting=stage_setting):
    enhanced = stage_def.copy()
    enhanced['epochs'] = int(epoch_map[int(stage_def['stage_id'])])
    enhanced['lr'] = 0.001
    stage_definitions.append(enhanced)

# No memory bank for this baseline
scene_memory_config = None
use_scene_memory = False
scene_dedup_strategy = 'merge_labels'

# No pseudo labels
use_pseudo_labels = False
pseudo_label_config = None

evaluation = dict(interval=1, metric='mAP')

locals().update(base_config)
