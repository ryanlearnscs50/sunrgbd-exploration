"""Week 3 replay-dose ablation: attempt one object on 25% of scenes."""

_base_ = './tr3d_dynamic_head_8x5_object_memory_random_pseudo_52211.py'

object_memory_config = dict(insertion=dict(insertion_probability=0.25))
