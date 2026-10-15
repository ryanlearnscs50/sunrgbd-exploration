"""SUNRGBD smoke config: 8x5 pseudo+random-memory with 1-epoch stages."""

_base_ = '../tr3d_dynamic_head_8x5_pseudo_memory_random_ratio_52211.py'

pseudo_label_config = dict(
    _delete_=True,
    use_pregenerated=True,
    apply_to_memory_scenes=False,
    confidence_threshold=0.50,
    nms_threshold=0.30,
    max_pseudo_per_scene=100,
    pseudo_vs_gt_iou_thr=0.25,
    pseudo_nms_iou_thr=0.30,
)

import sys
sys.path.append('configs/_base_/class_mappings')
from sunrgbd_40class_mapping import get_stage_definitions  # type: ignore

stage_definitions = []
for _sd in get_stage_definitions():
    _e = dict(_sd)
    _e['epochs'] = 1
    _e['lr'] = float(_e.get('lr', 0.001))
    stage_definitions.append(_e)

data = dict(
    workers_per_gpu=2,
    train=dict(
        times=1,
    ),
)

evaluation = dict(interval=1, metric='mAP')
