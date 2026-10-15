"""Switchable SUN RGB-D S5 random-memory baseline (no pseudo labels/reviewing).

Run from the repository root. --memory-mode selects scene/object/none.
Budgets use different units and are not a matched-storage comparison.
"""

_base_ = '../incremental/sunrgbd/tr3d_dynamic_head_8x5_scene_memory_random_ratio_52211.py'

memory_mode = 'scene'

# Same object settings as the existing S5 random-object baseline.
object_memory_config = dict(
    exemplars_per_class=20,
    max_total_exemplars=800,
    selection_strategy='random',
    min_points=5,
    crop_margin=0.0,
    floor_percentile=1.0,
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
reviewing = dict(enabled=False)
