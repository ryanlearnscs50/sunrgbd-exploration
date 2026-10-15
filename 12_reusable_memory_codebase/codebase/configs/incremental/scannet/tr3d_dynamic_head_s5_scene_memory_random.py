"""ScanNet 35-class incremental (S5): random scene memory replay."""

_base_ = './tr3d_dynamic_head_scannet35_base.py'

stage_setting = 'scannet35_s5_freqorder'

use_scene_memory = True
scene_memory_config = dict(
    memory_budget_ratio=0.1,
    total_training_scenes=1201,
    selection_strategy='random',
    quota_strategy='stage_ratio',
    min_objects_per_scene=1,
    debug_mode=True,
    # Keep random baseline explicit.
    use_drop_weights=False,
    use_current_stage_weights=False,
    enforce_current_quota=False,
    min_current_stage_quota=0,
    stage_ratio_counts=[],
)
scene_dedup_strategy = 'merge_labels'

use_pseudo_labels = False
pseudo_label_config = None

reviewing = dict(enabled=False, review_fractions=[1 / 5, 2 / 5, 3 / 5, 4 / 5])
