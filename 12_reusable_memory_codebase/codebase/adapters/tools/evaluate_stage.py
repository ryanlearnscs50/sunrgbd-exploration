#!/usr/bin/env python3
"""Evaluate a protocol-tagged UniDet3D checkpoint using its resolved stage config.

Run in the isolated host environment. No bank or training data is required;
the validation manifest and original point/mask/superpoint files are required.
This is an evaluation entry point, not an interrupted-training resume command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--upstream-root', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--val-info', type=Path, help='optional replacement stage validation manifest')
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--workers', type=int, default=0)
    args = parser.parse_args()
    if args.work_dir.exists():
        parser.error('work directory exists; refusing to overwrite evaluation evidence')
    if args.workers < 0:
        parser.error('workers must be nonnegative')
    sys.path.insert(0, str(args.upstream_root.resolve()))
    import torch
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner, set_random_seed
    from mmdet3d.registry import MODELS
    from ldmr_unidet3d import host, ScanNet35Protocol
    from ldmr_unidet3d.data import validate_stage_manifest
    from ldmr_unidet3d.checkpoints import validate_checkpoint_state

    cfg = Config.fromfile(str(args.config))
    context = cfg.ldmr_protocol
    protocol = ScanNet35Protocol(context.stage_setting)
    stage = context.stage
    protocol.seen_count(stage)
    if context.identity != protocol.identity:
        raise ValueError('resolved config protocol mismatch')
    if list(cfg.model.decoder.datasets_classes) != [protocol.decoder_classes(stage)]:
        raise ValueError('resolved config classifier classes differ from protocol')
    datasets = cfg.test_dataloader.dataset.datasets
    if len(datasets) != 1 or datasets[0].type != 'LDMRScanNetDataset':
        raise ValueError('expected single ScanNet-35 host evaluation dataset')
    info = (args.val_info or Path(datasets[0].ann_file)).resolve()
    manifest = validate_stage_manifest(info, protocol, stage, context.memory_mode, evaluation=True)
    datasets[0].ann_file = str(info)
    cfg.test_dataloader.update(batch_size=1, num_workers=args.workers,
                               persistent_workers=args.workers > 0)
    cfg.train_dataloader = cfg.train_cfg = cfg.optim_wrapper = cfg.param_scheduler = None
    cfg.val_dataloader = cfg.val_cfg = cfg.val_evaluator = None
    cfg.custom_hooks = []
    cfg.load_from = str(args.checkpoint.resolve())
    cfg.resume = False
    cfg.work_dir = str(args.work_dir.resolve())
    cfg.randomness = dict(seed=args.seed)
    init_default_scope('mmdet3d')
    set_random_seed(args.seed)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model = MODELS.build(cfg.model)
    validate_checkpoint_state(checkpoint, model.state_dict(), protocol, stage)
    del model, checkpoint
    args.work_dir.mkdir(parents=True, exist_ok=False)
    cfg.dump(str(args.work_dir / 'resolved_evaluation.py'))
    metrics = Runner.from_cfg(cfg).test()
    from ldmr_unidet3d.metrics import json_metrics
    values, undefined = json_metrics(metrics)
    # Empty-class AP can be undefined in small subsets: record it as null,
    # explicitly naming those metrics instead of writing nonstandard JSON NaN.
    report = dict(protocol_id=protocol.identity, stage=stage, seed=args.seed,
        evaluated_classes=list(protocol.names[:protocol.seen_count(stage)]),
        scenes=len(manifest['data_list']),
        metrics=values,
        undefined_metrics=undefined, checkpoint_sha256=sha256(args.checkpoint),
        config_sha256=sha256(args.config), annotation_sha256=sha256(info))
    with tempfile.NamedTemporaryFile(mode='w', dir=args.work_dir, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    try:
        os.link(temporary, args.work_dir / 'metrics.json')
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
