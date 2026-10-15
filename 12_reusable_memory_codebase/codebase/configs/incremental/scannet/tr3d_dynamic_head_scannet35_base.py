"""
ScanNet 35-class incremental base (dynamic head).

Default stage setting: 5-stage frequency split (7-7-7-7-7).
Derived configs override stage_setting and feature toggles.
"""

from mmcv import Config

base_config = Config.fromfile('configs/tr3d/tr3d_scannet-3d-35class.py')

# Incremental dataset wrapper.
base_config.data.train.dataset.type = 'IncrementalScanNetDataset'
base_config.data.train.dataset.use_sequential_gci = True
base_config.data.val.variant = 'dynamic_head'

# RepeatDataset remains explicit for reproducibility.
base_config.data.train.times = 15

# Incremental loss masking hooks.
base_config.model.head.train_cfg.enable_class_masking = True

use_dynamic_head = True
class_ordering = 'frequency'
stage_setting = 'scannet35_s5_freqorder'

import sys
sys.path.append('configs/_base_/class_mappings')
from scannet_dynamic_head_mappings import get_stage_definitions  # type: ignore


def _build_stage_definitions(stage_setting_key, epoch_map=None, lr=0.001):
    defs = get_stage_definitions(
        strategy=class_ordering,
        stage_setting=stage_setting_key,
    )
    out = []
    for sd in defs:
        e = sd.copy()
        sid = int(e['stage_id'])
        e['epochs'] = int(epoch_map.get(sid, 12)) if isinstance(epoch_map, dict) else 12
        e['lr'] = float(lr)
        out.append(e)
    return out


stage_definitions = _build_stage_definitions(stage_setting)

# Feature toggles (overridden by derived configs).
use_scene_memory = False
scene_memory_config = dict()
scene_dedup_strategy = 'merge_labels'

use_pseudo_labels = False
pseudo_label_config = dict()

# Reviewing defaults (used by LD+reviewing derived configs).
reviewing = dict(
    enabled=False,
    review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5],
)

# Shared scoring knobs for LD flows.
SCORING = dict(
    LD_IOU_MODE='0.25',
)

evaluation = dict(interval=1, metric='mAP')

locals().update(base_config)
