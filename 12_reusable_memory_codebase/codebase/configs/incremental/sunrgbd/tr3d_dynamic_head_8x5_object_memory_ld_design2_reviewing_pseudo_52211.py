"""Complete SUN RGB-D S5 object-memory extension.

Combines Design-2 object selection, source-seat ``ld_drop`` reviewing, and
old-class pseudo labels on the current natural scenes.  Pasted crops already
carry their ground-truth object boxes, so pseudo labels are not generated for
the synthetic objects themselves.
"""

_base_ = './tr3d_dynamic_head_8x5_object_memory_ld_design2_reviewing_52211.py'

use_pseudo_labels = True
pseudo_label_config = dict(
    _delete_=True,
    use_pregenerated=True,
    # Object replay does not append memory scenes; this remains enabled for
    # compatibility with the unified natural/replay pseudo-label policy.
    apply_to_memory_scenes=True,
    confidence_threshold=0.50,
    nms_threshold=0.30,
    max_pseudo_per_scene=100,
    pseudo_vs_gt_iou_thr=0.25,
    pseudo_nms_iou_thr=0.30,
)

reviewing_legacy_pseudo_consistency = dict(enabled=False)
