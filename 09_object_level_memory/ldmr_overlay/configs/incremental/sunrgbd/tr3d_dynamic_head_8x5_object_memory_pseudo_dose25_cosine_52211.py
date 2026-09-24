"""Week 3: per-iteration LR decay within each incremental stage."""

_base_ = './tr3d_dynamic_head_8x5_object_memory_pseudo_dose25_52211.py'

# Reset each stage: 0.001 toward 0.0001 over that stage's actual iterations.
lr_config = dict(_delete_=True, policy='CosineAnnealing', by_epoch=False,
                 min_lr_ratio=0.1, warmup=None)
