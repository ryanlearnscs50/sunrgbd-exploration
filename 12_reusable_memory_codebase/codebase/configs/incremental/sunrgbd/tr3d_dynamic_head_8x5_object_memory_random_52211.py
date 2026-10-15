"""SUN RGB-D 40-class S5 with object-level exemplar replay.

This follows the same 8x5 class/epoch protocol as the random scene-memory
baseline, but stores individual object crops and pastes old-class exemplars into
new-stage natural scenes.
"""

_base_ = './tr3d_dynamic_head_8x5_scene_memory_random_ratio_52211.py'

use_scene_memory = False
scene_memory_config = None

use_object_memory = True
object_memory_config = dict(
    exemplars_per_class=20,
    max_total_exemplars=800,  # 20 objects x 40 classes
    selection_strategy='random',
    min_points=5,
    crop_margin=0.0,
    floor_percentile=1.0,
    insertion=dict(
        # Same-seed SUN RGB-D ablation: one attempted paste per sample retained
        # old classes as well as the heavier 3-at-0.7 policy while improving
        # novel-class plasticity. See OBJECT_MEMORY_FINDINGS.md.
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

use_pseudo_labels = False
pseudo_label_config = None
reviewing = dict(enabled=False)
