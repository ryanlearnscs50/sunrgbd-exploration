"""ScanNet 35-class incremental (S10 4-4-4-4-4-3-3-3-3-3): pure finetuning baseline."""

_base_ = './tr3d_dynamic_head_scannet35_base.py'

stage_setting = 'scannet35_s10_freqorder_4444433333'
import sys
sys.path.append('configs/_base_/class_mappings')
from scannet_dynamic_head_mappings import get_stage_definitions  # type: ignore

stage_definitions = []
for _sd in get_stage_definitions(strategy='frequency', stage_setting=stage_setting):
    _e = _sd.copy()
    _e['epochs'] = 12
    _e['lr'] = 0.001
    stage_definitions.append(_e)

use_scene_memory = False
scene_memory_config = None
use_pseudo_labels = False
pseudo_label_config = None

reviewing = dict(
    enabled=False,
    review_fractions=[1 / 10, 2 / 10, 3 / 10, 4 / 10, 5 / 10, 6 / 10, 7 / 10, 8 / 10, 9 / 10],
)
