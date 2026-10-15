# optimizer
# This schedule is mainly used by models on indoor dataset,
# e.g., VoteNet on SUNRGBD and ScanNet
# FIXED: Reduced lr and weight_decay to prevent NaN loss in 35-class training
lr = 0.002  # max learning rate - reduced from 0.008 to prevent gradient explosion
optimizer = dict(type='AdamW', lr=lr, weight_decay=0.001)  # reduced from 0.01 to 0.001
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[24, 32])
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=36)
