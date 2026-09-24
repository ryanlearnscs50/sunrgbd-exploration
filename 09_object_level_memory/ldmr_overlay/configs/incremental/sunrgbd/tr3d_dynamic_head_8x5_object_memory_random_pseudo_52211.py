"""Week 3: isolate pseudo supervision on the existing random-object baseline."""

_base_ = './tr3d_dynamic_head_8x5_object_memory_random_52211.py'

use_pseudo_labels = True
pseudo_label_config = dict(
    _delete_=True,
    use_pregenerated=True,
    apply_to_memory_scenes=False,
    confidence_threshold=0.50,
    nms_threshold=0.30,
    pseudo_nms_iou_thr=0.30,
    max_pseudo_per_scene=100,
    pseudo_vs_gt_iou_thr=0.25,
)
