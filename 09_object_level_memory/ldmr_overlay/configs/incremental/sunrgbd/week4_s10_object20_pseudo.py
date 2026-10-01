"""Week 4 phase 3b: B=20, with all non-capacity baseline settings fixed."""
_base_ = './week4_s10_object100_pseudo.py'
object_memory_config = dict(exemplars_per_class=20, max_total_exemplars=800)
