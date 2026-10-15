"""ScanNet 35-class incremental (S5): LD Design2 memory management + reviewing."""

_base_ = './tr3d_dynamic_head_scannet35_base.py'

stage_setting = 'scannet35_s5_freqorder'

use_scene_memory = True
scene_memory_config = dict(
    memory_budget_ratio=0.1,
    total_training_scenes=1201,
    selection_strategy='learning_dynamics_design2',
    quota_strategy='stage_ratio',
    enforce_unique_scene_ids=True,
    min_objects_per_scene=1,
    debug_mode=True,
    use_drop_weights=False,
    use_current_stage_weights=False,
    enforce_current_quota=False,
    min_current_stage_quota=0,
    stage_ratio_counts=[],
    learning_dynamics_update=dict(
        eps=1e-9,
        object_count_cap=20,
        report_topk=30,
    ),
    learning_dynamics_design2=dict(
        q_metric='f1',
        min_add_lower_bound=1,
        use_class_balance=True,
        supply_scaling_mode='cap_log1p',
        supply_cap=20,
        force_accept_until_lower_bound=True,
        w_max=10.0,
        redundancy_lambda=0.3,
        redundancy_topk=5,
        min_class_quota=5,
    ),
)
scene_dedup_strategy = 'merge_labels'

use_pseudo_labels = False
pseudo_label_config = None

reviewing = dict(
    enabled=True,
    review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5],
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

SCORING = dict(LD_IOU_MODE='0.25')
