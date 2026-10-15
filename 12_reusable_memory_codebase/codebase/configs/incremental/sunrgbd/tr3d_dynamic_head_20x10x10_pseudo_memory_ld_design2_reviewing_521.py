"""
SUN RGB-D Incremental Learning (20/10/10) - LDMR (pseudo + LD Design-2 memory + reviewing)

Final-stage mAP@0.25 = 29.12 (seed 200).

Stage setting:
- `sunrgbd40_s3_20_10_10_freqorder`
- Stage sizes: 20 -> 10 -> 10
- Dynamic head: 20 -> 30 -> 40
- Per-stage epochs: 5-2-2

LDMR components:
- Human-like intra-stage review: 6 review checkpoints -> I = 7 sub-stages
- Scene-aware cross-stage memory evolution: budget 10% of the training set,
  learnability/diversity trade-off lambda = 0.5
- Pseudo labels for old classes, regenerated per stage from the previous model
"""

from mmcv import Config

# Base supervised (upper bound) config
base_config = Config.fromfile('configs/tr3d/tr3d_sunrgbd-3d-40class.py')

# Use incremental dataset wrapper for stage-wise GT filtering
base_config.data.train.dataset.type = 'IncrementalSUNRGBDDataset'
base_config.data.train.dataset.filter_empty_gt = True

# Keep RepeatDataset repetition explicit
base_config.data.train.times = 15

# Enable incremental masking hooks
base_config.model.head.train_cfg.enable_class_masking = True

# Dynamic head expansion enabled
use_dynamic_head = True

# 3-stage setting and stage definitions
stage_setting = 'sunrgbd40_s3_20_10_10_freqorder'
incremental = dict(stage_setting=stage_setting)

import sys
sys.path.append('configs/_base_/class_mappings')
from sunrgbd_40class_mapping import get_stage_definitions  # type: ignore

epoch_map = {1: 5, 2: 2, 3: 2}

stage_definitions = []
for stage_def in get_stage_definitions(stage_setting=stage_setting):
    enhanced = stage_def.copy()
    enhanced['epochs'] = int(epoch_map[int(stage_def['stage_id'])])
    enhanced['lr'] = 0.001
    stage_definitions.append(enhanced)

# Learning-dynamics recall is measured at IoU 0.50 (Sec. 5.1)
SCORING = dict(LD_IOU_MODE='0.50')

# ---------------------------------------------------------------------------
# Scene-aware cross-stage memory evolution (Sec. 4.3)
# ---------------------------------------------------------------------------
use_scene_memory = True
scene_dedup_strategy = 'merge_labels'

scene_memory_config = dict(
    memory_budget_ratio=0.1,
    total_training_scenes=5285,

    selection_strategy='learning_dynamics_design2',
    quota_strategy='stage_ratio',
    enforce_unique_scene_ids=True,

    min_objects_per_scene=1,

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
        q_metric='recall',                # per-class recall, Eq. 3
        min_add_lower_bound=1,
        use_class_balance=True,
        supply_scaling_mode='cap_log1p',  # log(1 + min(n, cap)) count weight, Eq. 9
        supply_cap=20,
        w_max=10.0,
        min_class_quota=5,
        redundancy_lambda=0.5,            # lambda in Eq. 14
        redundancy_topk=5,
        force_accept_until_lower_bound=True,
    ),
)

# ---------------------------------------------------------------------------
# Pseudo labels for old classes
# ---------------------------------------------------------------------------
use_pseudo_labels = True
pseudo_label_config = dict(
    use_pregenerated=True,
    apply_to_memory_scenes=True,
    confidence_threshold=0.50,
    nms_threshold=0.30,
    max_pseudo_per_scene=100,
    pseudo_vs_gt_iou_thr=0.25,
    pseudo_nms_iou_thr=0.30,
)

# ---------------------------------------------------------------------------
# Human-like intra-stage review (Sec. 4.2)
# ---------------------------------------------------------------------------
reviewing = dict(
    enabled=True,
    # 6 review checkpoints -> I = 7 sub-stages. These are the literal values
    # used for the published run (i/7 rounded to 6 decimal places, i = 1..6).
    review_fractions=[0.142857, 0.285714, 0.428571, 0.571429, 0.714286, 0.857143],
    eval_iou_thrs=[0.25, 0.50],
    weight_iou_thr=0.50,
    compare_to='last',
    drop_clamp_min=0.0,
    resume_optimizer=True,
    weight_policy=dict(
        type='ld_drop',
        eta=3,                            # review emphasis eta, Eq. 5
        normalize_by_gt_weight=True,
    ),
    sampling=dict(
        mode='coverage_preserving',
        weight_space='combined',
        memory_share_max=0.9,
        seed_offset=9000,
    ),
)
reviewing_legacy_pseudo_consistency = dict(enabled=False)

evaluation = dict(interval=1, metric='mAP')

locals().update(base_config)
