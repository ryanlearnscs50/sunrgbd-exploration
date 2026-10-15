"""ScanNet 35-class incremental (S3 15-10-10): pseudo + LD Design2 memory + reviewing.

Final-stage mAP@0.25 = 37.81 (seed 201).
Review granularity: 2 review checkpoints -> I = 3 sub-stages.
"""

_base_ = './tr3d_dynamic_head_s5_pseudo_memory_ld_design2_reviewing.py'

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

reviewing = dict(
    enabled=True,
    review_fractions=[1 / 3, 2 / 3],
    eval_iou_thrs=[0.25, 0.50],
    weight_iou_thr=0.25,
    weight_policy=dict(
        type='ld_drop',
        eta=5.0,
        normalize_by_gt_weight=True,
        object_count_cap=20,
        w_entry_max=10.0,
    ),
)
