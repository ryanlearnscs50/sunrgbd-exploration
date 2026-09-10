"""SUN RGB-D S5 object Design-2 with source-seat intra-stage reviewing.

Old object crops inherit LDMR ``ld_drop`` sampling weights from evaluations
of their original carrier-scene seats. Pseudo labels remain disabled so this
isolates reviewing from the later old-object pseudo-label ablation.
"""

_base_ = './tr3d_dynamic_head_8x5_object_memory_ld_design2_52211.py'

reviewing = dict(
    _delete_=True,
    enabled=True,
    review_fractions=[0.2, 0.4, 0.6, 0.8],
    eval_iou_thrs=[0.25, 0.50],
    weight_iou_thr=0.50,
    compare_to='last',
    drop_clamp_min=0.0,
    resume_optimizer=True,
    weight_policy=dict(
        type='ld_drop',
        eta=3,
        normalize_by_gt_weight=True,
    ),
    sampling=dict(
        mode='coverage_preserving',
        weight_space='object_source_seat',
        memory_share_max=0.9,
        seed_offset=9000,
        strict_memory_coverage=True,
    ),
)
