"""ScanNet 35-class incremental (S5): pre-generated pseudo labels only."""

_base_ = './tr3d_dynamic_head_scannet35_base.py'

stage_setting = 'scannet35_s5_freqorder'

use_scene_memory = False
scene_memory_config = None

use_pseudo_labels = True
pseudo_label_config = dict(
    use_pregenerated=True,
    apply_to_memory_scenes=False,
    confidence_threshold=0.45,
    stage_thresholds={2: 0.45, 3: 0.40, 4: 0.35, 5: 0.30},
    nms_threshold=0.30,
    max_pseudo_per_scene=100,
    pseudo_vs_gt_iou_thr=0.25,
    pseudo_nms_iou_thr=0.30,
)

reviewing = dict(enabled=False, review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5])
