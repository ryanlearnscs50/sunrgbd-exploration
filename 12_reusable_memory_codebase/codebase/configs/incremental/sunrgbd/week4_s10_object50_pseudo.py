"""Week 4 phase 3a: halve stored-object capacity; freeze the B=100 protocol."""

_base_ = './week4_s10_object100_pseudo.py'

object_memory_config = dict(exemplars_per_class=50, max_total_exemplars=2000)
