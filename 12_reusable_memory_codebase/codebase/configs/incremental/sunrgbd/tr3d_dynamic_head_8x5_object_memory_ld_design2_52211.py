"""SUN RGB-D S5 object replay with LDMR Design-2 object selection.

This isolates selection quality: object crops are replayed with the validated
one-object dose, while reviewing and pseudo labels remain disabled. A full
stage-1 run supplies genuine source-scene/class learning dynamics.
"""

_base_ = './tr3d_dynamic_head_8x5_object_memory_random_52211.py'

SCORING = dict(LD_IOU_MODE='0.50')

object_memory_config = dict(
    _delete_=True,
    exemplars_per_class=20,
    max_total_exemplars=800,
    selection_strategy='learning_dynamics_design2',
    min_points=5,
    crop_margin=0.0,
    floor_percentile=1.0,
    learning_dynamics_update=dict(
        eps=1e-9,
        object_count_cap=20,
        report_topk=30,
    ),
    learning_dynamics_design2=dict(
        q_metric='recall',
        min_add_lower_bound=1,
        use_class_balance=True,
        supply_scaling_mode='cap_log1p',
        supply_cap=20,
        w_max=10.0,
        min_class_quota=5,
        redundancy_lambda=0.5,
        redundancy_topk=5,
        force_accept_until_lower_bound=True,
    ),
    insertion=dict(
        max_exemplars_per_scene=1,
        insertion_probability=1.0,
        collision_threshold=0.0,
        placement_jitter=0.0,
        max_placement_attempts=20,
        floor_offset=0.0,
        floor_percentile=1.0,
        preserve_source_height=True,
    ),
)

use_scene_memory = False
scene_memory_config = None
use_object_memory = True
use_pseudo_labels = False
pseudo_label_config = None
reviewing = dict(
    enabled=False,
    review_fractions=[0.2, 0.4, 0.6, 0.8],
    eval_iou_thrs=[0.25, 0.50],
    weight_iou_thr=0.50,
)
