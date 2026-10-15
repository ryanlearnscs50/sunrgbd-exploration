"""Switchable ScanNet-35 S5 random-memory baseline.

No pseudo labels or reviewing. Scene and object budgets have different units.
"""

_base_ = '../incremental/scannet/tr3d_dynamic_head_s5_scene_memory_random.py'

memory_mode = 'scene'
object_memory_config = dict(
    exemplars_per_class=20,
    max_total_exemplars=700,
    selection_strategy='random',
    min_points=5,
    crop_margin=0.0,
    floor_percentile=1.0,
    insertion=dict(
        max_exemplars_per_scene=3,
        insertion_probability=0.7,
        collision_threshold=0.0,
        placement_jitter=0.0,
        max_placement_attempts=20,
        floor_offset=0.0,
        floor_percentile=1.0,
        preserve_source_height=True,
    ),
)
