#!/usr/bin/env python3
"""
Incremental Learning Model Evaluation Script for TR3D

This script is specifically designed to evaluate incremental learning models that use:
- Dynamic head expansion (7→14→21→28→35 classes)
- Sequential GCI (Global Class Index) mapping (0-34)
- Scene-based or object-based memory replay

Key differences from standard test.py:
1. Automatically detects incremental learning configurations
2. Uses proper class mappings (GCI to class names)
3. Handles dynamic head models correctly
4. Supports evaluation of any stage checkpoint

Usage:
    python tools/test_incremental.py config checkpoint --eval mAP
    
Example:
    python tools/test_incremental.py \
        configs/incremental/scannet/tr3d_dynamic_head_s5_pure_finetuning.py \
        incremental_logs/.../stage_5/latest.pth \
        --eval mAP
"""

import argparse
import os
import warnings
import sys
from os import path as osp

# Ensure repo root is on PYTHONPATH when running as a script, e.g.
# `python tools/eval_incremental.py ...` (otherwise `import mmdet3d` fails).
REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet.datasets import replace_ImageToTensor

# Import incremental learning utilities
sys.path.append(osp.join(osp.dirname(__file__), '..', 'configs', '_base_', 'class_mappings'))
try:
    from scannet_dynamic_head_mapping import (
        SCANNET_DYNAMIC_HEAD_CLASSES,
        DYNAMIC_HEAD_GCI_TO_NAME,
        DYNAMIC_HEAD_STAGE_DEFINITIONS
    )
except ImportError as e:
    print(f"Warning: Could not import dynamic head mapping: {e}")
    SCANNET_DYNAMIC_HEAD_CLASSES = None
    DYNAMIC_HEAD_GCI_TO_NAME = None
    DYNAMIC_HEAD_STAGE_DEFINITIONS = None


def cumulative_class_counts(stage_definitions):
    """Number of classes seen after each stage, e.g. [7, 14, 21, 28, 35]."""
    counts, total = [], 0
    for stage_def in stage_definitions or []:
        total += len(stage_def.get('class_indices', stage_def.get('class_names', [])))
        counts.append(total)
    return counts


def detect_checkpoint_stage(checkpoint_path, stage_definitions=None):
    """Detect which incremental stage a checkpoint came from.

    The width of the classification head is authoritative: a stage-k checkpoint
    has one channel per class seen up to and including stage k. We read it and
    then look the stage up in the config's stage definitions, so this works for
    any protocol (SUN RGB-D 4/8/20 classes per stage, ScanNet 3/4/7/15, ...).

    Args:
        checkpoint_path: path to a per-stage .pth
        stage_definitions: cfg.stage_definitions, used to map classes -> stage id

    Returns:
        tuple: (stage_id, n_classes)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    meta = checkpoint.get('meta', {}) if isinstance(checkpoint.get('meta'), dict) else {}
    counts = cumulative_class_counts(stage_definitions)

    n_classes = None
    cls_conv = checkpoint.get('state_dict', {}).get('head.cls_conv.kernel')
    if cls_conv is not None and getattr(cls_conv, 'ndim', 0) >= 2:
        n_classes = int(cls_conv.shape[1])  # [feature_dim, n_classes]

    # Stage id: prefer the class count, fall back to what the run recorded.
    stage_id = None
    if n_classes is not None and n_classes in counts:
        stage_id = counts.index(n_classes) + 1
    else:
        recorded = meta.get('stage_id') or meta.get('ldmr', {}).get('stage')
        if isinstance(recorded, int) and 1 <= recorded <= len(counts):
            stage_id = recorded
            if n_classes is None:
                n_classes = counts[stage_id - 1]

    if stage_id is None or n_classes is None:
        raise ValueError(
            f'Could not determine the incremental stage of {checkpoint_path}. '
            f'Head width: {n_classes}; classes after each stage of this config: {counts}. '
            'Check that the checkpoint and the config describe the same protocol.')

    if n_classes != counts[stage_id - 1]:
        raise ValueError(
            f'{checkpoint_path} has a {n_classes}-class head, but stage {stage_id} of this '
            f'config expects {counts[stage_id - 1]} classes (per-stage totals: {counts}). '
            'The checkpoint and the config are for different protocols.')

    return stage_id, n_classes

if mmdet.__version__ > '2.23.0':
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate incremental learning 3D detector')
    parser.add_argument('config', help='incremental test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--metrics-out', help='new JSON path for metrics, class scope and input hashes')
    parser.add_argument('--eval-class-scope', choices=['seen', 'all'], default='seen',
                        help='Evaluate seen classes (default), or explicitly include future classes')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--dataset-split',
        type=str,
        choices=['train', 'val'],
        default='val',
        help='Dataset split to evaluate on (train or val)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    return parser.parse_args()


def detect_incremental_config(cfg):
    """Detect if config is for incremental learning and return configuration info."""
    incremental_info = {
        'is_incremental': False,
        'use_dynamic_head': False,
        'use_sequential_gci': False,
        'stage_definitions': None,
        'n_classes': None
    }
    
    # Check for dynamic head markers
    if hasattr(cfg, 'use_dynamic_head') and cfg.use_dynamic_head:
        incremental_info['is_incremental'] = True
        incremental_info['use_dynamic_head'] = True
        incremental_info['use_sequential_gci'] = getattr(cfg, 'use_sequential_gci', True)
        incremental_info['stage_definitions'] = getattr(cfg, 'stage_definitions', None)
    
    # Check for incremental dataset types
    train_type = cfg.data.get('train', {}).get('type', '')
    if 'Incremental' in train_type:
        incremental_info['is_incremental'] = True
        
    # Determine number of classes from model config
    if hasattr(cfg, 'model') and hasattr(cfg.model, 'head'):
        incremental_info['n_classes'] = cfg.model.head.get('n_classes', 35)
    
    return incremental_info


def setup_incremental_dataset(cfg, incremental_info):
    """Setup dataset configuration for incremental learning evaluation."""
    
    # For dynamic head models, use dynamic_head variant
    if incremental_info['use_dynamic_head']:
        print("📊 Setting up dynamic head evaluation dataset")

        # Use validation dataset (same as training evaluation)
        if hasattr(cfg.data, 'val'):
            # Evaluate on the plain (non-incremental) counterpart of the val set.
            # The config already names the right dataset; only strip the
            # Incremental* wrapper, never rewrite one dataset family into another.
            val_type = cfg.data.val.type
            if val_type.startswith('Incremental'):
                val_type = val_type[len('Incremental'):]
                cfg.data.val.type = val_type

            cfg.data.val.test_mode = True

            # The dynamic_head variant / 35-class palette is a ScanNet notion.
            if val_type == 'ScanNetDataset':
                cfg.data.val.variant = 'dynamic_head'
                try:
                    from scannet_dynamic_head_mapping import SCANNET_DYNAMIC_HEAD_CLASSES as DH_CLASSES
                    cfg.data.val.classes = list(DH_CLASSES)
                except Exception:
                    # Fall back to the dataset's own dynamic_head default.
                    if hasattr(cfg.data.val, 'classes'):
                        delattr(cfg.data.val, 'classes')

            # Remove incremental-specific parameters that could cause issues
            incremental_params = ['stage_definition', 'mappings', 'memory_bank',
                                'evaluation_mode', 'all_stage_definitions',
                                'use_sequential_gci']
            for param in incremental_params:
                if hasattr(cfg.data.val, param):
                    delattr(cfg.data.val, param)

        # Ensure model uses correct class count for the checkpoint's stage
        n_classes = incremental_info['n_classes']
        if n_classes and hasattr(cfg.model, 'head'):
            cfg.model.head.n_classes = n_classes
            print(f"📊 Model configured for {n_classes} classes")

    return cfg


def select_evaluation_dataset_config(cfg, split):
    """Use validation transforms with the requested raw annotation split.

    Training configs may wrap the dataset in RepeatDataset. Evaluation should
    use its annotations once and preserve the validation dataset family.
    """
    import copy
    selected = copy.deepcopy(cfg.data.val)
    selected.test_mode = True
    if split == 'train':
        source = cfg.data.train
        while source.get('type') == 'RepeatDataset':
            source = source['dataset']
        if 'ann_file' not in source or 'data_root' not in source:
            raise ValueError('training-split evaluation requires one dataset with ann_file and data_root')
        selected.ann_file = source.ann_file
        selected.data_root = source.data_root
    elif split != 'val':
        raise ValueError('evaluation split must be train or val')
    return selected


def load_incremental_checkpoint(model, checkpoint_path):
    """Load checkpoint with proper handling for incremental learning models."""
    
    # Convert to absolute path to avoid path resolution issues
    checkpoint_path = osp.abspath(checkpoint_path)
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Validate checkpoint file exists
    if not osp.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found: {checkpoint_path}")
        
        # Check if it's a broken symbolic link
        if osp.islink(checkpoint_path):
            link_target = os.readlink(checkpoint_path)
            print(f"🔗 This is a symbolic link pointing to: {link_target}")
            print(f"❌ The target file does not exist")
            
            # Try to find alternatives in the same directory
            checkpoint_dir = osp.dirname(checkpoint_path)
            if osp.exists(checkpoint_dir):
                pth_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth') and not osp.islink(osp.join(checkpoint_dir, f))]
                if pth_files:
                    print(f"🔍 Found alternative checkpoint files in {checkpoint_dir}:")
                    for pth_file in sorted(pth_files):
                        full_path = osp.join(checkpoint_dir, pth_file)
                        if osp.exists(full_path):
                            print(f"  - {pth_file}")
                    print(f"\n💡 Try using one of these files instead:")
                    print(f"   CUDA_VISIBLE_DEVICES=1 python tools/eval_incremental.py \\")
                    print(f"       [config] {osp.join(checkpoint_dir, pth_files[-1])} --eval mAP")
                else:
                    print(f"❌ No alternative .pth files found in {checkpoint_dir}")
        else:
            print(f"❌ The file path does not exist")
        
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Set CUDA context before loading checkpoint for MinkowskiEngine compatibility
    if torch.cuda.is_available():
        # Use CUDA_VISIBLE_DEVICES to determine target device
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
        device_id = int(cuda_devices.split(',')[0]) if cuda_devices else 0
        if device_id < torch.cuda.device_count():
            torch.cuda.set_device(device_id)
            print(f"🎯 Set CUDA device context to {device_id} for MinkowskiEngine")
            map_location = f'cuda:{device_id}'
        else:
            print(f"⚠️ Requested CUDA device {device_id} not available, using CPU")
            map_location = 'cpu'
    else:
        map_location = 'cpu'
    
    checkpoint = load_checkpoint(model, checkpoint_path, map_location=map_location)
    
    # Extract stage information if available
    stage_info = None
    if 'meta' in checkpoint and isinstance(checkpoint['meta'], dict):
        stage_info = {
            'stage_id': checkpoint['meta'].get('stage_id'),
            'stage_name': checkpoint['meta'].get('stage_name'),
        }
        if stage_info['stage_id']:
            print(f"📊 Checkpoint from Stage {stage_info['stage_id']}: {stage_info['stage_name']}")
    
    return stage_info


def print_incremental_results(results, dataset, incremental_info, stage_info=None):
    """Print evaluation results with proper incremental learning formatting."""
    
    if not results:
        print("⚠️ No evaluation results to display")
        return
        
    print("\n" + "="*80)
    print("🎯 INCREMENTAL LEARNING EVALUATION RESULTS")
    print("="*80)
    
    if stage_info and stage_info['stage_id']:
        print(f"📊 Stage: {stage_info['stage_id']} ({stage_info['stage_name']})")
    
    if incremental_info['use_dynamic_head']:
        print(f"📊 Model Type: Dynamic Head Expansion")
        print(f"📊 Sequential GCI Mapping: {incremental_info['use_sequential_gci']}")
        print(f"📊 Total Classes: {incremental_info['n_classes']}")
    else:
        print(f"📊 Model Type: Fixed Head Incremental Learning")
    
    # Extract and display main mAP metrics
    if isinstance(results, dict):
        mAP_025 = results.get('mAP_0.25')
        mAP_050 = results.get('mAP_0.50') or results.get('mAP_0.5')  # Handle both formats
        
        if mAP_025 is not None or mAP_050 is not None:
            print("\n🎯 OVERALL PERFORMANCE:")
            if mAP_025 is not None:
                print(f"   mAP@0.25: {mAP_025:.4f}")
            if mAP_050 is not None:
                print(f"   mAP@0.50: {mAP_050:.4f}")
            
            # Show improvement context for Stage 1 models
            if stage_info and stage_info['stage_id'] == 1 and mAP_025 is not None:
                if mAP_025 > 0.1:
                    print(f"   ✅ Good Stage 1 performance - properly evaluated on seen classes only")
                elif mAP_025 < 0.01:
                    print(f"   ⚠️  Very low mAP - check if evaluating on correct class subset")
    
    print("="*80)


def main():
    args = parse_args()
    if args.metrics_out:
        if args.format_only or not args.eval:
            raise ValueError('--metrics-out requires --eval and cannot be used with --format-only')
        if osp.exists(args.metrics_out):
            raise FileExistsError(args.metrics_out)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # STEP 1: Detect stage from checkpoint BEFORE building model. The config's
    # stage definitions tell us how many classes each stage adds.
    stage_id, n_classes = detect_checkpoint_stage(
        args.checkpoint, cfg.get('stage_definitions'))
    print(f"📊 Evaluating Stage {stage_id} checkpoint ({n_classes} classes)")

    # Safety gate: prevent accidental runs of research configs.
    cfg_path = args.config.replace('\\', '/').lstrip('./')
    if '/configs/experimental/' in f'/{cfg_path}':
        assert cfg.get('allow_experimental', False) is True, (
            'Refusing to run an experimental config without explicit opt-in. '
            'Re-run with `--cfg-options allow_experimental=True`.'
        )
    
    cfg = compat_cfg(cfg)
    
    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    
    # Detect incremental learning configuration
    incremental_info = detect_incremental_config(cfg)
    # The checkpoint, not the config, decides how wide the head is.
    incremental_info['n_classes'] = n_classes

    if incremental_info['is_incremental']:
        print("🎯 Detected incremental learning configuration")
        cfg = setup_incremental_dataset(cfg, incremental_info)
    else:
        print("⚠️ Standard configuration detected - consider using tools/test.py for non-incremental models")
    
    # in case the validation dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.val, dict):
        cfg.data.val.test_mode = True
        samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
    elif isinstance(cfg.data.val, list):
        for ds_cfg in cfg.data.val:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.val])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.val:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use distributed testing for '
                      'multi-gpu testing!')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.out is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(osp.dirname(args.out)))
        
    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader - select dataset split based on args
    dataset_config = select_evaluation_dataset_config(cfg, args.dataset_split)
    dataset = build_dataset(dataset_config)
    print(f'📊 Evaluating {dataset_config.type} {args.dataset_split} split ({len(dataset)} samples)')
    
    # STEP 2: Set seen_classes_for_eval based on detected stage, so mAP is
    # computed only over the classes the model has seen up to this stage.
    seen_classes = list(range(n_classes))
    dataset.seen_classes_for_eval = seen_classes if args.eval_class_scope == 'seen' else None
    evaluated_count = n_classes if args.eval_class_scope == 'seen' else len(dataset.CLASSES)
    print(f"📊 Evaluation scope: {args.eval_class_scope}; {evaluated_count} classes (stage {stage_id})")
    
    workers_per_gpu = getattr(cfg.data, 'workers_per_gpu', 4)  # Default to 4 if not specified
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    
    # CRITICAL: Configure model head to match checkpoint's stage before building model
    # Stage detection gives us the correct class count for this checkpoint
    if hasattr(cfg.model, 'head'):
        cfg.model.head.n_classes = n_classes
        print(f"📊 Model head configured for Stage {stage_id}: {n_classes} classes")
    
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    # Load checkpoint with incremental learning handling
    stage_info = load_incremental_checkpoint(model, args.checkpoint)
    
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        
        kwargs = {}
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        else:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args and training-only flags
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals', 'by_epoch'
            ]:
                eval_kwargs.pop(key, None)
            
            # Set evaluation metrics properly
            if args.eval:
                if isinstance(args.eval, list):
                    eval_kwargs['metric'] = args.eval[0] if len(args.eval) > 0 else 'mAP'
                else:
                    eval_kwargs['metric'] = args.eval
            else:
                eval_kwargs['metric'] = 'mAP'  # Default to mAP
            
            # Ensure we use the standard evaluation parameters for ScanNet
            eval_kwargs.update({
                'classwise': True
            })
            eval_kwargs.update(kwargs)
            
            print(f"🔍 Evaluating with metrics: {eval_kwargs.get('metric')}")
            results = dataset.evaluate(outputs, **eval_kwargs)
            if args.metrics_out:
                from tools.evaluation_artifacts import write_evaluation_report
                write_evaluation_report(args.metrics_out, results, dataset=dataset,
                    config=cfg, checkpoint=args.checkpoint, seed=args.seed,
                    stage=stage_id, trained_classes=n_classes)
            
            # Create stage info for display
            stage_info = {
                'stage_id': stage_id,
                'stage_name': f'Stage {stage_id} - {n_classes} trained classes; {evaluated_count} evaluated classes'
            }
            
            # Print results with incremental learning formatting
            print_incremental_results(results, dataset, incremental_info, stage_info)
            
            # Print detailed results
            print("\n📊 Detailed Results:")
            print(results)


if __name__ == '__main__':
    main()
