#!/usr/bin/env python
"""Step C — classifier-head expansion for incremental TR3D.

The TR3D head's classifier (`head.cls_conv`) has one output channel per class, so
when the cumulative class count grows from stage t-1 to stage t a plain
`load_from` would size-mismatch this layer and silently REINITIALISE it, throwing
away everything the old head learned. This script grows the classifier instead:

  * `head.cls_conv.kernel`  (in_channels, C_old)  ->  (in_channels, C_new)
  * `head.cls_conv.bias`    (1, C_old)            ->  (1, C_new)

Old-class columns are copied verbatim; the C_new - C_old new columns are
initialised exactly as TR3DHead.init_weights would (kernel ~ N(0, .01),
bias = bias_init_with_prob(.01)). The classifier is a per-class SIGMOID focal
head, so appending columns leaves the old classes' logits mathematically
untouched. `head.bbox_conv` is class-agnostic (n_reg_outs=8) and is left as-is;
backbone/neck weights pass through unchanged.

The result is written as a fresh checkpoint that the stage-t config can
`load_from` (warm start) with no size mismatch. This is the weight-transfer that
makes the *naive fine-tuning* baseline a genuine warm start rather than a head
reset; the pseudo-labeling variant reuses the exact same expansion.

Usage:
  python tools_incremental/expand_head.py \
      --prev work_dirs/incremental_v4/3stage/stage1/latest.pth \
      --cfg  configs/tr3d_incremental_v4/3stage/stage2.py \
      --out  work_dirs/incremental_v4/3stage/stage2/init_expanded.pth
"""
import argparse
import os

import torch
from mmcv import Config
from mmcv.cnn import bias_init_with_prob

CLS_KERNEL = 'head.cls_conv.kernel'
CLS_BIAS = 'head.cls_conv.bias'


def parse_args():
    p = argparse.ArgumentParser(description='Expand TR3D classifier head for the next stage.')
    p.add_argument('--prev', required=True, help='previous-stage checkpoint (.pth) to grow')
    p.add_argument('--cfg', required=True, help='next-stage config (defines new n_classes/class_names)')
    p.add_argument('--out', required=True, help='output checkpoint path')
    p.add_argument('--seed', type=int, default=0, help='seed for new-column init (reproducibility)')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = Config.fromfile(args.cfg)
    new_n = int(cfg.model.head.n_classes)
    new_names = tuple(cfg.class_names)
    assert len(new_names) == new_n, f'cfg class_names ({len(new_names)}) != n_classes ({new_n})'

    ck = torch.load(args.prev, map_location='cpu')
    sd = ck['state_dict']
    assert CLS_KERNEL in sd and CLS_BIAS in sd, f'no cls_conv in {args.prev}'

    old_kernel = sd[CLS_KERNEL]          # (in_channels, C_old)
    old_bias = sd[CLS_BIAS]              # (1, C_old)
    in_ch, old_n = old_kernel.shape
    assert old_bias.shape == (1, old_n), f'unexpected bias shape {tuple(old_bias.shape)}'
    assert new_n >= old_n, f'new n_classes ({new_n}) < old ({old_n}); not an incremental step'

    # Sanity: the old model's classes must be the prefix of the new cumulative order,
    # otherwise copying columns 0..old_n-1 would map weights onto the wrong classes.
    old_names = ck.get('meta', {}).get('CLASSES') if isinstance(ck.get('meta'), dict) else None
    if old_names is not None:
        old_names = tuple(old_names)
        assert new_names[:len(old_names)] == old_names, (
            'class-order mismatch: previous CLASSES is not a prefix of the new config '
            f'class_names.\n prev={old_names}\n new ={new_names[:len(old_names)]}')
        assert len(old_names) == old_n, f'prev meta CLASSES ({len(old_names)}) != checkpoint cls width ({old_n})'

    n_new = new_n - old_n

    # Grow kernel: copy old columns, init the new ones like TR3DHead.init_weights (N(0, .01)).
    new_kernel = old_kernel.new_empty((in_ch, new_n))
    new_kernel[:, :old_n] = old_kernel
    if n_new > 0:
        torch.nn.init.normal_(new_kernel[:, old_n:], std=.01)

    # Grow bias: copy old, set new ones to the focal-loss prior bias.
    prior_bias = bias_init_with_prob(.01)
    new_bias = old_bias.new_empty((1, new_n))
    new_bias[:, :old_n] = old_bias
    if n_new > 0:
        new_bias[:, old_n:] = prior_bias

    sd[CLS_KERNEL] = new_kernel
    sd[CLS_BIAS] = new_bias

    # Drop optimizer state (load_from ignores it; keeps the init checkpoint small/clean)
    # and stamp the new class list into meta so the next expansion can prefix-check it.
    out_ck = {'state_dict': sd}
    meta = dict(ck.get('meta', {})) if isinstance(ck.get('meta'), dict) else {}
    meta['CLASSES'] = new_names
    meta['expand_head'] = dict(prev=os.path.abspath(args.prev), old_n=old_n, new_n=new_n,
                               prior_bias=float(prior_bias))
    out_ck['meta'] = meta

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out_ck, args.out)

    print(f'[expand_head] {args.prev}')
    print(f'  cls_conv.kernel {tuple(old_kernel.shape)} -> {tuple(new_kernel.shape)}')
    print(f'  cls_conv.bias   {tuple(old_bias.shape)} -> {tuple(new_bias.shape)}')
    print(f'  copied {old_n} old class columns, init {n_new} new (bias={prior_bias:.5f})')
    print(f'  wrote {args.out}')


if __name__ == '__main__':
    main()
