"""
SUN RGB-D Incremental Learning (8×5) - FAST DEBUG CONFIG

Purpose: quick validation of stage transitions (e.g., dynamic head expansion)
without the long RepeatDataset schedule used for full experiments.

Key overrides vs `configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_scene_memory_random_ratio_52211.py`:
- `data.train.times = 1` (RepeatDataset repetition)
- Per-stage epochs: 1-1-1-1-1
"""

from mmcv import Config


# Base incremental config (memory-only baseline, seeded random selection)
base_config = Config.fromfile(
    'configs/incremental/sunrgbd/tr3d_dynamic_head_8x5_scene_memory_random_ratio_52211.py'
)

# Make RepeatDataset repetition explicit and small for fast debugging
base_config.data.train.times = 1

# Override per-stage epochs to 1 each (fast)
epoch_map = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

stage_definitions = []
for stage_def in base_config.stage_definitions:
    enhanced = stage_def.copy()
    enhanced['epochs'] = int(epoch_map[int(stage_def['stage_id'])])
    stage_definitions.append(enhanced)

# Persist overrides back into the Config object
base_config.stage_definitions = stage_definitions

# Export config (keep all other settings unchanged)
locals().update(base_config)
