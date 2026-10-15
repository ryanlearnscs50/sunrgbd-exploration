"""ScanNet 35-class incremental (S5): pseudo + LD Design2 memory + reviewing.

Final-stage mAP@0.25 = 27.46 (seed 201).
Review granularity: 4 review checkpoints -> I = 5 sub-stages.
"""

_base_ = './tr3d_dynamic_head_s5_scene_memory_ld_design2_reviewing.py'

use_pseudo_labels = True
pseudo_label_config = dict(
    _delete_=True,
    use_pregenerated=True,
    apply_to_memory_scenes=True,
    confidence_threshold=0.45,
    # Thresholds are looked up per stage id; entries beyond the protocol's
    # stage count are simply unused, so one table serves the 3-, 5- and
    # 10-stage configs that inherit from this file.
    stage_thresholds={
        2: 0.45, 3: 0.40, 4: 0.35, 5: 0.30,
        6: 0.30, 7: 0.30, 8: 0.30, 9: 0.30, 10: 0.30,
    },
    nms_threshold=0.30,
    max_pseudo_per_scene=100,
    pseudo_vs_gt_iou_thr=0.25,
    pseudo_nms_iou_thr=0.30,
)
