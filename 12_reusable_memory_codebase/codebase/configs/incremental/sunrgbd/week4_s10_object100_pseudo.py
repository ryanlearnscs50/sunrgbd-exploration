"""Week 4 S10 object-budget baseline, full-training seeds.

TR3D_OBJ nominal budget/dose with local corrected box geometry and uniform
bank selection. This extends the reference S5 object condition to S10; it is
not an exact reference-code reproduction. Local insertion has no extra crop
yaw and samples objects without replacement rather than classes with replacement.
"""

_base_ = './tr3d_dynamic_head_4x10_scene_memory_random_ratio_6111111111.py'

use_scene_memory = False
scene_memory_config = None
use_object_memory = True
object_memory_config = dict(
    exemplars_per_class=100,
    max_total_exemplars=4000,
    selection_strategy='random',
    min_points=20,
    crop_margin=0.0,
    floor_percentile=0.0,
    insertion=dict(
        max_exemplars_per_scene=3,
        insertion_probability=0.7,
        collision_threshold=0.3,
        placement_jitter=1.0,
        max_placement_attempts=10,
        floor_offset=0.0,
        floor_percentile=0.0,
        preserve_source_height=False,
    ),
)
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
reviewing = dict(enabled=False)
# Reference rule: step at epochs-1. Stage 1 has 6 epochs -> decay for epoch 6.
# Stages 2-10 have one epoch: step [5] and reference step [1] both leave all
# training iterations at the initial LR. No cosine or Week 3 dose tuning here.
lr_config = dict(policy='step', warmup=None, step=[5])
