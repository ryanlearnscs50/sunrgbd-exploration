#!/usr/bin/env python3
"""Run ScanNet-35 stage training and bank export using two isolated runtimes.

Launch this supervisor detached for training. It runs children sequentially,
records each exact command and exit status, and stops on the first failure.
Every run needs a new work directory. This is not mid-training recovery.
Use --continue-from to inherit verified completed stages from a stopped run.
"""
import argparse
import json
import os
from pathlib import Path
import pickle
import sys
from datetime import datetime, timezone

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host-python', required=True, type=Path)
    parser.add_argument('--ldmr-python', required=True, type=Path)
    parser.add_argument('--upstream-root', required=True, type=Path)
    parser.add_argument('--train-info', required=True, type=Path)
    parser.add_argument('--val-info', required=True, type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--continue-from', type=Path,
                        help='stopped sequence whose completed stages should be reused read-only')
    parser.add_argument('--memory-mode', choices=['none', 'scene', 'object'], required=True)
    parser.add_argument('--target-mode', choices=['masks', 'boxes'], default='masks')
    parser.add_argument('--recipe', choices=['custom', 'ldmr', 'tr3d_obj'], default='custom',
                        help='reference method; --epochs are nominal epochs with 15 repeats for named recipes')
    parser.add_argument('--object-policy', choices=['ldmr_weighted', 'tr3d_obj'], default='ldmr_weighted')
    parser.add_argument('--object-selector', choices=['random', 'largest_point_count', 'tr3d_obj_replace'], default='random')
    parser.add_argument('--min-points', type=int, default=5)
    parser.add_argument('--pseudo', action='store_true',
                        help='generate previous-stage teacher predictions; requires box targets')
    parser.add_argument('--stage-setting', default='scannet35_s5_freqorder')
    parser.add_argument('--last-stage', type=int)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--validation-interval', type=int, default=None,
                        help='host passes between validation/checkpoints; default 15 for named recipes, 1 for custom')
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='default 8 for reference recipes, 2 for custom runs')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--scene-budget', type=int, default=120)
    parser.add_argument('--objects-per-class', type=int, default=20)
    parser.add_argument('--max-objects', type=int, default=1000)
    parser.add_argument('--ld-epochs', type=int, nargs='+', help='ScanNet Design-2 measurement boundaries, including 0 and final epoch')
    parser.add_argument("--reviewing", action="store_true", help="ScanNet S5 online scene/object reviewing")
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = 8 if args.recipe != "custom" else 2
    from ldmr_unidet3d.recipes import apply_reference_recipe
    try:
        apply_reference_recipe(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.object_policy == 'tr3d_obj' and (args.memory_mode != 'object' or args.reviewing or args.ld_epochs):
        parser.error('TR3D_OBJ replay requires object mode without Design-2/reviewing')
    if args.reviewing:
        if args.memory_mode not in ('scene', 'object'):
            parser.error('online reviewing requires scene or object memory')
        from ldmr_unidet3d.reviewing import review_boundaries
        boundaries = review_boundaries(args.epochs, args.stage_setting)
        if args.ld_epochs is None:
            args.ld_epochs = boundaries
        elif boundaries != args.ld_epochs:
            parser.error("LD schedule must equal the original reviewing boundaries")
    if args.ld_epochs is not None:
        from ldmr_unidet3d.collection import training_schedule
        training_schedule(args.ld_epochs, args.epochs)
        if args.memory_mode not in ('scene', 'object') or args.stage_setting != 'scannet35_s5_freqorder':
            parser.error('Design-2 sequence currently requires scene/object memory and original S5 policy')
    if args.pseudo and args.target_mode != 'boxes':
        parser.error('--pseudo requires --target-mode boxes')
    from ldmr_unidet3d import ScanNet35Protocol
    from ldmr_unidet3d.data import stage_records, verify_point_files, validate_disjoint_splits
    from ldmr_unidet3d.sequence import (sha256, verify_hashes, input_hashes,
        continuation, stage_directory, seal_stage)
    from ldmr_unidet3d.processes import run_managed
    protocol = ScanNet35Protocol(args.stage_setting)
    if args.pseudo:
        from ldmr_unidet3d.pseudo import scannet_policy
        scannet_policy(2, args.stage_setting)
    last_stage = len(protocol.stages) if args.last_stage is None else args.last_stage
    protocol.seen_count(last_stage)
    if min(args.epochs, args.batch_size, args.scene_budget, args.objects_per_class, args.max_objects) < 1 or args.workers < 0:
        parser.error('epochs, batch size and budgets must be positive; workers nonnegative')
    if args.validation_interval < 1 or args.epochs % args.validation_interval:
        parser.error('validation interval must be positive and divide stage epochs')
    for name in ('host_python', 'ldmr_python', 'upstream_root', 'train_info', 'val_info', 'work_dir'):
        # Keep venv interpreter symlinks: resolving python itself selects the
        # system interpreter and loses pyvenv.cfg.
        setattr(args, name, getattr(args, name).absolute())
    if args.work_dir.exists():
        parser.error('work directory exists; refusing to overwrite or guess resume state')
    if args.continue_from:
        args.continue_from = args.continue_from.absolute()
    datasets = []
    for path in (args.train_info, args.val_info):
        with path.open('rb') as stream:
            records = pickle.load(stream)['data_list']
        stage_records(records, protocol, last_stage, evaluation=True)
        for record in records:
            verify_point_files(record)
        datasets.append(records)
    validate_disjoint_splits(*datasets)
    for stage in range(1, last_stage + 1):
        if not stage_records(datasets[0], protocol, stage):
            raise ValueError(f'no natural training scenes at stage {stage}')
        if not stage_records(datasets[1], protocol, stage, evaluation=True):
            raise ValueError(f'no validation scenes at stage {stage}')
    repo = TOOLS.parents[1]
    sources = (list(TOOLS.parent.rglob('*.py'))
               + list((args.upstream_root / 'unidet3d').glob('*.py'))
               + [args.upstream_root / 'configs/unidet3d_1xb8_scannet.py',
                  repo / 'configs/_base_/class_mappings/scannet_dynamic_head_mappings.py']
               + list((repo / 'mmdet3d/datasets/memory_bank').glob('*.py'))
               + [repo / 'mmdet3d/datasets/scene_memory_bank.py',
                  repo / 'mmdet3d/datasets/object_memory_bank.py',
                  repo / 'mmdet3d/datasets/pseudo_label_utils.py',
                  repo / 'mmdet3d/core/evaluation/scene_class_stats.py',
                  repo / 'mmdet3d/utils/learning_dynamics_scoring.py',
                  repo / 'mmdet3d/utils/reviewing_sampling.py',
                  repo / 'tools/incremental_scene/reviewing_ld.py',
                  repo / 'configs/memory/scannet_s5_tr3d_obj.py',
                  repo / 'configs/incremental/scannet/tr3d_dynamic_head_s5_scene_memory_ld_design2_reviewing.py',
                  repo / 'configs/incremental/scannet/tr3d_dynamic_head_s5_pseudo_memory_ld_design2_reviewing.py'])
    hashes = {str(path): sha256(path) for path in sorted(sources)}
    inputs = input_hashes((args.train_info, args.val_info), datasets)
    state = dict(status='running', pid=os.getpid(), protocol_id=protocol.identity,
                 memory_mode=args.memory_mode, commands=[], completed_stages=[],
                 inherited_stages={}, artifact_hashes={})
    if args.continue_from:
        state.update(continuation(args.continue_from, vars(args), hashes, inputs, protocol, last_stage))
    args.work_dir.mkdir(parents=True, exist_ok=False)
    (args.work_dir / 'source_hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
    (args.work_dir / 'input_hashes.json').write_text(json.dumps(inputs, indent=2) + '\n')
    (args.work_dir / 'arguments.json').write_text(json.dumps(vars(args), default=str, indent=2) + '\n')

    def save():
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        temporary = args.work_dir / '.status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        os.replace(temporary, args.work_dir / 'status.json')

    def run(python, tool, options, label):
        verify_hashes(hashes, 'source')
        command = list(map(str, [python, TOOLS / tool, *options]))
        entry = dict(label=label, argv=command, log=str(args.work_dir / (label + '.log')))
        state['commands'].append(entry)
        save()
        # Do not inherit a caller's legacy/host PYTHONPATH across environments.
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        env.setdefault('OMP_NUM_THREADS', '2')
        def started(pid):
            entry.update(pid=pid, process_group=pid)
            save()
        with open(entry['log'], 'wb') as log:
            try:
                entry['returncode'] = run_managed(command, stdout=log, env=env,
                                                   on_started=started)
            except BaseException as exc:
                entry['interrupted'] = f'{type(exc).__name__}: {exc}'
                save()
                raise
        save()
        if entry['returncode']:
            raise RuntimeError(f'{label} failed ({entry["returncode"]}); see {entry["log"]}')
        verify_hashes(hashes, 'source')

    previous_checkpoint = None
    first_stage = len(state['completed_stages']) + 1
    if first_stage > 1:
        previous_checkpoint = (stage_directory(args.work_dir, first_stage - 1, state)
                               / 'training' / f'epoch_{args.epochs}.pth')
    try:
        save()
        for stage in range(first_stage, last_stage + 1):
            current = args.work_dir / f'stage_{stage}'
            current.mkdir()
            previous_bank = stage_directory(args.work_dir, stage - 1, state) / 'bank'
            common = ['--stage', stage, '--stage-setting', args.stage_setting,
                      '--memory-mode', args.memory_mode]
            for split, source in (('train', args.train_info), ('val', args.val_info)):
                options = ['--input', source, '--output', current / (split + '.pkl'),
                           '--split', split, *common]
                if stage > 1 and split == 'train' and args.memory_mode == 'scene':
                    options += ['--replay-records', previous_bank / 'replay_records.pkl']
                run(args.host_python, 'prepare_stage_infos.py', options, f's{stage}_{split}_infos')
            if args.pseudo and stage > 1:
                teacher_config = stage_directory(args.work_dir, stage - 1, state) / 'training/resolved_stage.py'
                run(args.host_python, 'generate_pseudo.py', [
                    '--teacher-config', teacher_config, '--checkpoint', previous_checkpoint,
                    '--train-info', current / 'train.pkl', '--upstream-root', args.upstream_root,
                    '--stage', stage, '--stage-setting', args.stage_setting,
                    '--seed', args.seed, '--policy', 'tr3d_obj' if args.object_policy == 'tr3d_obj' else 'ldmr',
                    '--output', current / 'pseudo.json'], f's{stage}_pseudo')
            options = ['--upstream-root', args.upstream_root,
                       '--train-info', current / 'train.pkl', '--val-info', current / 'val.pkl',
                       '--work-dir', current / 'training', '--epochs', args.epochs,
                       '--validation-interval', args.validation_interval,
                       '--seed', args.seed, '--batch-size', args.batch_size,
                       '--workers', args.workers, '--target-mode', args.target_mode,
                       '--object-policy', args.object_policy, *common]
            if args.pseudo and stage > 1:
                options += ['--pseudo-labels', current / 'pseudo.json']
            if previous_checkpoint is not None:
                options += ['--previous-checkpoint', previous_checkpoint]
            if stage > 1 and args.memory_mode == 'object':
                options += ['--crop-path', previous_bank / 'crops.pkl']
            if args.ld_epochs is not None:
                options += ['--ld-epochs', *args.ld_epochs]
                if args.memory_mode == 'object':
                    options += ['--object-source-info', args.train_info]
            if args.reviewing:
                options += ['--reviewing']
            run(args.host_python, 'train_stage.py', options, f's{stage}_training')
            previous_checkpoint = current / 'training' / f'epoch_{args.epochs}.pth'
            if not previous_checkpoint.is_file():
                raise RuntimeError(f'training did not produce {previous_checkpoint}')
            if args.memory_mode != 'none':
                options = ['--input', args.train_info, '--output-dir', current / 'bank',
                           '--seed', args.seed, '--scene-budget', args.scene_budget,
                           '--objects-per-class', args.objects_per_class,
                           '--max-objects', args.max_objects, '--min-points', args.min_points,
                           '--object-selector', args.object_selector,
                           '--object-policy', args.object_policy, *common]
                if stage > 1:
                    options += ['--previous-bank', previous_bank / 'bank.pkl']
                    if args.memory_mode == 'object':
                        options += ['--previous-crops', previous_bank / 'crops.pkl']
                if args.ld_epochs is not None:
                    options += ['--ld-collection', current / 'training' / 'ld_collection']
                    if args.reviewing and args.memory_mode == 'object':
                        options += ['--reviewing']
                run(args.ldmr_python, 'export_memory.py', options, f's{stage}_bank')
            state['artifact_hashes'][str(stage)] = seal_stage(
                current, stage, args.epochs, args.memory_mode, len(protocol.stages), args.pseudo, args.ld_epochs, args.reviewing)
            state['completed_stages'].append(stage)
            save()
        state['status'] = 'complete'
    except BaseException as exc:
        state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    from ldmr_unidet3d.processes import termination_handler
    with termination_handler():
        main()
