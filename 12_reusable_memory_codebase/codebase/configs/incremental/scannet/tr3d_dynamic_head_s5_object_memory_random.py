"""ScanNet 35-class incremental (S5): object-level exemplar replay.

This is the object-memory counterpart to the random scene-memory baseline. It
stores cropped objects after each stage and inserts old-class exemplars into
natural scenes before geometric augmentation.
"""

_base_ = './tr3d_dynamic_head_scannet35_base.py'

stage_setting = 'scannet35_s5_freqorder'

use_scene_memory = False
scene_memory_config = None

use_object_memory = True
object_memory_config = dict(
    exemplars_per_class=20,
    max_total_exemplars=700,  # 20 objects x 35 classes
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

use_pseudo_labels = False
pseudo_label_config = None
reviewing = dict(enabled=False)
