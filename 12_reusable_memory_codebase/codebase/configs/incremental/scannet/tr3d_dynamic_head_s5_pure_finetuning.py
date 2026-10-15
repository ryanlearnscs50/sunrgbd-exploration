"""ScanNet 35-class incremental (S5 7-7-7-7-7): pure finetuning baseline."""

_base_ = './tr3d_dynamic_head_scannet35_base.py'

stage_setting = 'scannet35_s5_freqorder'

use_scene_memory = False
scene_memory_config = None
use_pseudo_labels = False
pseudo_label_config = None

reviewing = dict(enabled=False, review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5])
