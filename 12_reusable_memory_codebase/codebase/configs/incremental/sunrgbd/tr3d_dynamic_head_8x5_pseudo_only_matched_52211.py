"""Week 3: pseudo-only control matched to random object replay plus pseudo."""

_base_ = './tr3d_dynamic_head_8x5_object_memory_random_pseudo_52211.py'

use_object_memory = False
object_memory_config = None
