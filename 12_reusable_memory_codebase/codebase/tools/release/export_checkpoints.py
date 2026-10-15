#!/usr/bin/env python
"""Export per-stage LDMR checkpoints from a training run into a release folder.

An incremental run writes ``<run-dir>/checkpoints/stage_N/epoch_M.pth``. Each of
those files carries the model weights *and* the optimizer state; the optimizer is
roughly two thirds of the file and is never needed to evaluate a stage or to seed
the next one (``--checkpoint-path`` only reads ``state_dict``). This script copies
the final checkpoint of every stage, drops the optimizer by default, and records
where each file came from in ``meta['ldmr']``.

Example:

    python tools/release/export_checkpoints.py \
        --run-dir  /path/to/incremental_logs/SUN_RGBD/<run> \
        --protocol sunrgbd_10stage \
        --out-dir  release_checkpoints

Writes ``release_checkpoints/sunrgbd_10stage/stage_01.pth ... stage_10.pth`` plus
a ``manifest.json`` describing every exported file.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch

EVAL_LINE = re.compile(r'stage=(\d+)\s+mAP25=([0-9.]+)')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--run-dir', required=True, type=Path,
                   help='Training run directory containing checkpoints/stage_*/')
    p.add_argument('--protocol', required=True,
                   help="Release name for this run, e.g. 'scannet_5stage'")
    p.add_argument('--out-dir', required=True, type=Path,
                   help='Destination root; a <protocol>/ subfolder is created')
    p.add_argument('--keep-optimizer', action='store_true',
                   help='Keep optimizer state (~3x larger files)')
    p.add_argument('--extra-eval-log', type=Path, nargs='*', default=[],
                   help='Additional eval_summary logs to read scores from, for stages '
                        'the run itself did not evaluate (e.g. a base stage trained by '
                        'a separate prep run). The run\'s own logs take precedence.')
    p.add_argument('--dry-run', action='store_true',
                   help='Report what would be exported without writing anything')
    return p.parse_args()


def stage_mAP(run_dir, extra_logs=()):
    """Read final mAP@0.25 per stage from the run's eval summary, if present.

    ``extra_logs`` are parsed first so that any stage the run evaluated itself
    overrides a value inherited from another run's log.
    """
    own = sorted(run_dir.glob('eval_summary_*.log'))
    if len(own) > 1:
        print(f'  warning: {len(own)} eval_summary logs in {run_dir.name}; later '
              f'timestamps win per stage. Logs: {[p.name for p in own]}')
    out = {}
    for log in list(extra_logs) + own:
        if not log.is_file():
            raise SystemExit(f'eval log not found: {log}')
        for line in log.read_text(errors='replace').splitlines():
            m = EVAL_LINE.search(line)
            if m:
                out[int(m.group(1))] = float(m.group(2))
    return out


def final_checkpoint(stage_dir):
    """The checkpoint to publish: resolve latest.pth, else the highest epoch_N."""
    latest = stage_dir / 'latest.pth'
    if latest.exists():
        return latest.resolve()
    epochs = sorted(stage_dir.glob('epoch_*.pth'),
                    key=lambda p: int(re.search(r'epoch_(\d+)', p.name).group(1)))
    return epochs[-1] if epochs else None


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(chunk), b''):
            h.update(block)
    return h.hexdigest()


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    ckpt_root = run_dir / 'checkpoints'
    if not ckpt_root.is_dir():
        raise SystemExit(f'No checkpoints/ directory under {run_dir}')

    stage_dirs = sorted(ckpt_root.glob('stage_*'),
                        key=lambda p: int(p.name.split('_')[1]))
    if not stage_dirs:
        raise SystemExit(f'No stage_*/ directories under {ckpt_root}')

    scores = stage_mAP(run_dir, args.extra_eval_log)
    out_dir = args.out_dir / args.protocol
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    entries, total = [], 0
    for stage_dir in stage_dirs:
        stage = int(stage_dir.name.split('_')[1])
        src = final_checkpoint(stage_dir)
        if src is None:
            print(f'  stage {stage:>2}: no checkpoint found, skipping')
            continue

        dst = out_dir / f'stage_{stage:02d}.pth'
        if args.dry_run:
            print(f'  stage {stage:>2}: {src.name:<14} -> {dst.name}'
                  f'   mAP@0.25={scores.get(stage, float("nan")):.4f}')
            continue

        ckpt = torch.load(src, map_location='cpu')
        released = {'meta': dict(ckpt.get('meta', {})),
                    'state_dict': ckpt['state_dict']}
        if args.keep_optimizer and 'optimizer' in ckpt:
            released['optimizer'] = ckpt['optimizer']
        released['meta']['ldmr'] = {
            'protocol': args.protocol,
            'stage': stage,
            'source_run': run_dir.name,
            'source_file': src.name,
            'mAP@0.25': scores.get(stage),
            'optimizer_stripped': not args.keep_optimizer,
        }
        torch.save(released, dst)

        size = dst.stat().st_size
        total += size
        entries.append({
            'stage': stage,
            'file': dst.name,
            'bytes': size,
            'sha256': sha256(dst),
            'mAP@0.25': scores.get(stage),
        })
        print(f'  stage {stage:>2}: {src.name:<14} -> {dst.name}  '
              f'{size / 1e6:6.1f} MB  mAP@0.25={scores.get(stage)}')

    if args.dry_run:
        return

    manifest = {
        'protocol': args.protocol,
        'source_run': run_dir.name,
        # Recorded as <run>/<log> rather than a full path: the manifest is
        # published, and absolute paths would leak the training host's layout.
        'extra_eval_logs': [f'{p.parent.name}/{p.name}' for p in args.extra_eval_log],
        'num_stages': len(entries),
        'optimizer_stripped': not args.keep_optimizer,
        'total_bytes': total,
        'final_stage_mAP@0.25': entries[-1]['mAP@0.25'] if entries else None,
        'stages': entries,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'\n{len(entries)} checkpoints, {total / 1e9:.2f} GB -> {out_dir}')
    if entries and entries[-1]['mAP@0.25'] is not None:
        print(f"final-stage mAP@0.25 = {entries[-1]['mAP@0.25']}")


if __name__ == '__main__':
    main()
