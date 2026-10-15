"""TR3D_OBJ-style object+pseudo method (scannet); no object Design-2/reviewing.

Reference replacement intentionally favors late candidates, as the teammate's
implementation does. ScanNet transfers the policy to axis-aligned geometry and
keeps the original LDMR ScanNet 12x15 exposure; SUN retains 5/2/2/1/1 x15.
"""
_base_ = '../incremental/scannet/tr3d_dynamic_head_s5_object_memory_random.py'
memory_mode = 'object'
use_scene_memory = False
use_object_memory = True
reviewing = dict(enabled=False)
stage_lr_policy = 'tr3d_obj'
object_memory_config = dict(
    _delete_=True,
    exemplars_per_class=100,
    max_total_exemplars=3500,
    selection_strategy='tr3d_obj_replace',
    min_points=20,
    crop_margin=0.0,
    crop_tolerance=0.01,
    floor_percentile=0.0,
    insertion=dict(
        max_exemplars_per_scene=3, insertion_probability=0.7,
        sampling_policy='class_uniform', collision_threshold=0.3,
        placement_jitter=1.0, max_placement_attempts=10,
        floor_percentile=0.0, preserve_source_height=False, random_yaw=False,
    ),
)
use_pseudo_labels = True
pseudo_label_config = dict(
    _delete_=True, use_pregenerated=True, apply_to_memory_scenes=False,
    confidence_threshold=0.5, nms_threshold=0.3, max_pseudo_per_scene=100,
    pseudo_vs_gt_iou_thr=0.25, pseudo_nms_iou_thr=0.3,
)
