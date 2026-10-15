#!/usr/bin/env python3
"""Run or preflight one LDMR ScanNet-35 stage in a separate UniDet3D runtime.

Metadata must already contain current-class natural scenes plus selected replay
seats for scene mode. Optional frozen teacher artifacts require explicit box
targets. Optional LD boundaries capture running weights and collect a sealed
trajectory after training. Optional scene reviewing updates sampling online.
"""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'adapters'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream-root', required=True, type=Path)
    parser.add_argument('--train-info', required=True, type=Path)
    parser.add_argument('--val-info', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--stage-setting', default='scannet35_s5_freqorder')
    parser.add_argument('--memory-mode', choices=['none', 'scene', 'object'], required=True)
    parser.add_argument('--crop-path', type=Path)
    parser.add_argument('--object-policy', choices=['ldmr_weighted', 'tr3d_obj'], default='ldmr_weighted')
    parser.add_argument('--target-mode', choices=['masks', 'boxes'], default='masks')
    parser.add_argument('--pseudo-labels', type=Path,
                        help='protocol-tagged teacher JSON; requires box targets and previous checkpoint')
    parser.add_argument('--previous-checkpoint', type=Path)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--validation-interval', type=int, default=1)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--ld-epochs', type=int, nargs='+', help='explicit measurement epochs including 0 and final epoch')
    parser.add_argument('--object-source-info', type=Path, help='full training metadata for object Design-2 measurement')
    parser.add_argument('--reviewing', action='store_true', help='ScanNet S5 online scene/object review policy')
    args = parser.parse_args()
    if args.object_policy == "tr3d_obj" and (args.memory_mode != "object" or args.reviewing):
        parser.error("TR3D_OBJ policy requires object mode without Design-2 reviewing")
    if args.reviewing:
        from ldmr_unidet3d.reviewing import review_boundaries
        if args.memory_mode not in ('scene', 'object'):
            parser.error('online reviewing requires scene or object memory')
        boundaries = review_boundaries(args.epochs, args.stage_setting)
        if args.ld_epochs is None:
            args.ld_epochs = boundaries
        elif boundaries != args.ld_epochs:
            parser.error('--ld-epochs must equal the original reviewing boundaries')
    if args.ld_epochs is not None:
        from ldmr_unidet3d.collection import training_schedule
        training_schedule(args.ld_epochs, args.epochs)
    if bool(args.object_source_info) != (args.memory_mode == 'object' and args.ld_epochs is not None):
        parser.error('object LD training requires --object-source-info only with object --ld-epochs')
    if args.pseudo_labels and (args.target_mode != 'boxes' or args.stage < 2 or not args.previous_checkpoint):
        parser.error('pseudo labels require --target-mode boxes, stage 2+ and --previous-checkpoint')
    if args.stage > 1 and not args.previous_checkpoint and not args.validate_only:
        parser.error('stage 2+ training requires the preceding protocol-tagged checkpoint')
    if args.stage == 1 and args.previous_checkpoint:
        parser.error('stage 1 has no preceding checkpoint')
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error('epochs and batch size must be positive; workers nonnegative')
    if not args.validate_only and args.work_dir.exists():
        parser.error('work directory exists; refusing to overwrite or guess resume state')
    from ldmr_unidet3d import ScanNet35Protocol
    from ldmr_unidet3d.data import validate_stage_manifest, validate_disjoint_splits
    protocol = ScanNet35Protocol(args.stage_setting)
    train = validate_stage_manifest(args.train_info, protocol, args.stage, args.memory_mode, evaluation=False)
    val = validate_stage_manifest(args.val_info, protocol, args.stage, args.memory_mode, evaluation=True)
    validate_disjoint_splits(train['data_list'], val['data_list'])
    if args.object_source_info:
        import pickle
        from ldmr_unidet3d.data import stage_records
        full = pickle.loads(args.object_source_info.read_bytes())['data_list']
        if stage_records(full, protocol, args.stage) != train['data_list']:
            raise ValueError('object source metadata differs from natural training pool')
        validate_disjoint_splits(full, val['data_list'])
    sys.path.insert(0, str(args.upstream_root.resolve()))
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner, set_random_seed
    from mmengine.registry import init_default_scope
    from mmdet3d.registry import DATASETS, MODELS
    from ldmr_unidet3d import host  # explicit host registration
    from ldmr_unidet3d.configuration import stage_config
    from ldmr_unidet3d.checkpoints import expand_checkpoint
    from ldmr_unidet3d.sequence import sha256
    init_default_scope('mmdet3d')
    base = Config.fromfile(str(args.upstream_root / 'configs/unidet3d_1xb8_scannet.py'))
    cfg = Config(stage_config(base.to_dict(), protocol, args.stage,
        train_info=args.train_info.resolve(), val_info=args.val_info.resolve(),
        work_dir=args.work_dir.resolve(), memory_mode=args.memory_mode,
        epochs=args.epochs, crop_path=args.crop_path, target_mode=args.target_mode,
        validation_interval=args.validation_interval,
        object_policy=args.object_policy,
        pseudo_path=args.pseudo_labels,
        teacher_sha256=sha256(args.previous_checkpoint) if args.pseudo_labels else None))
    if args.ld_epochs is not None:
        cfg.custom_hooks.append(dict(type='LDMRTrajectoryCheckpointHook', epochs=args.ld_epochs,
            stage=args.stage, total_epochs=args.epochs))
        cfg.ldmr_protocol.ld_epochs = args.ld_epochs
        if args.memory_mode == 'object':
            cfg.ldmr_protocol.ld_crop_path = str(args.crop_path.resolve()) if args.crop_path else None
    if args.reviewing:
        from ldmr_unidet3d import reviewing_host
        cfg.custom_imports.imports.append('ldmr_unidet3d.reviewing_host')
        cfg.custom_hooks.append(dict(type='LDMROnlineReviewHook', stage=args.stage,
            total_epochs=args.epochs, upstream_root=str(args.upstream_root.resolve()),
            train_info=str(args.train_info.resolve()), seed=args.seed,
            stage_setting=args.stage_setting, memory_mode=args.memory_mode,
            object_source_info=str(args.object_source_info.resolve()) if args.object_source_info else None))
        if args.memory_mode == 'scene':
            cfg.train_dataloader.sampler = dict(type='LDMRReviewSampler', shuffle=True)
        cfg.train_dataloader.drop_last = False
        cfg.ldmr_protocol.reviewing = True
    cfg.randomness = dict(seed=args.seed)
    for split in ('train', 'val', 'test'):
        cfg[split + '_dataloader'].update(num_workers=args.workers,
            persistent_workers=args.workers > 0, batch_size=args.batch_size if split == 'train' else 1)
    set_random_seed(args.seed)
    if args.validate_only:
        for split in ('train', 'val'):
            dataset = DATASETS.build(cfg[split + '_dataloader'].dataset)
            sample = dataset[0]
            print(f'{split}: {len(dataset)} records; first points={tuple(sample["inputs"]["points"].shape)}')
        model = MODELS.build(cfg.model)
        print(f'model built: {protocol.seen_count(args.stage)} seen classes; no training or AP evaluation')
        return
    args.work_dir.mkdir(parents=True, exist_ok=False)
    if args.previous_checkpoint:
        model = MODELS.build(cfg.model)
        model.init_weights()
        previous = torch.load(args.previous_checkpoint, map_location='cpu')
        expanded = expand_checkpoint(previous, model.state_dict(), protocol, args.stage)
        expanded_path = args.work_dir / f'initialized_stage_{args.stage}.pth'
        torch.save(expanded, expanded_path)
        cfg.load_from = str(expanded_path.resolve())
        del model, previous, expanded
    cfg.dump(str(args.work_dir / 'resolved_stage.py'))
    runner = Runner.from_cfg(cfg)
    runner.train()
    if args.ld_epochs is not None:
        import subprocess
        # Release detector/optimizer GPU memory before isolated inference.
        del runner
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        command = [sys.executable, str(Path(__file__).with_name('collect_ld.py')),
            '--config', str(args.work_dir / 'resolved_stage.py'),
            '--train-info', str(args.train_info), '--upstream-root', str(args.upstream_root),
            '--output', str(args.work_dir / 'ld_collection'), '--seed', str(args.seed),
            '--q-metric', 'f1']
        if args.object_source_info:
            command += ['--object-source-info', str(args.object_source_info)]
        for epoch in args.ld_epochs:
            command += ['--checkpoint', f'{epoch}={args.work_dir / "ld_checkpoints" / f"epoch_{epoch}.pth"}']
        if args.reviewing and args.stage > 1 and len(boundaries) > 2:
            command += ['--reuse-collection', str(args.work_dir / 'reviewing' / f'review_{len(boundaries) - 2}')]
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
