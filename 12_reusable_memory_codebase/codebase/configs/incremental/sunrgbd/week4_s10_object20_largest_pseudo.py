"""Week 4 criterion comparison: same B=20, select crops with most points."""
_base_ = './week4_s10_object20_pseudo.py'
object_memory_config = dict(selection_strategy='largest_point_count')
