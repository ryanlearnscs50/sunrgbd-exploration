"""ScanNet 35-class incremental (S3 15-10-10): pure finetuning baseline."""

_base_ = './tr3d_dynamic_head_scannet35_base.py'

stage_setting = 'scannet35_s3_freqorder_15_10_10'
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

reviewing = dict(enabled=False, review_fractions=[1 / 3, 2 / 3])
