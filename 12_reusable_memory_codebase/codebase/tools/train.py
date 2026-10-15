# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division
import argparse
import copy
import os
import sys
import time
import warnings
from os import path as osp

# Ensure repo root is on PYTHONPATH when running as a script, e.g.
# `python tools/train.py ...` (otherwise `import mmdet3d` fails).
REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

try:
    # If mmdet version > 2.20.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
except ImportError:
    from mmdet3d.utils import setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='resume from the latest checkpoint automatically')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='(Deprecated, please use --gpu-id) number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--diff-seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
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
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    parser.add_argument(
        '--debug-eval',
        action='store_true',
        help='debug mode: reduce training to 10%% of iterations for quick evaluation testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Safety gate: prevent accidental runs of research configs.
    cfg_path = args.config.replace('\\', '/').lstrip('./')
    if '/configs/experimental/' in f'/{cfg_path}':
        assert cfg.get('allow_experimental', False) is True, (
            'Refusing to run an experimental config without explicit opt-in. '
            'Re-run with `--cfg-options allow_experimental=True`.'
        )

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # Generate timestamp for unique folder names
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    
    # work_dir is determined in this priority: CLI > segment in file > filename
    # ALWAYS append timestamp to prevent experiment overwriting
    if args.work_dir is not None:
        # Always append timestamp to user-specified work_dir to prevent experiment overwriting
        base_work_dir = args.work_dir.rstrip('/')  # Remove trailing slash if any
        cfg.work_dir = f"{base_work_dir}_{timestamp}"
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        config_name = osp.splitext(osp.basename(args.config))[0]
        cfg.work_dir = osp.join('./work_dirs', f'{config_name}_{timestamp}')
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from

    if args.auto_resume:
        cfg.auto_resume = args.auto_resume
        warnings.warn('`--auto-resume` is only supported when mmdet'
                      'version >= 2.20.0 for 3D detection model or'
                      'mmsegmentation verision >= 0.21.0 for 3D'
                      'segmentation model')

    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn('`--gpus` is deprecated because we only support '
                      'single GPU mode in non-distributed training. '
                      'Use `gpus=1` now.')
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed training. Use the first GPU '
                      'in `gpu_ids` now.')
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    if args.autoscale_lr:
        # apply the linear scaling rule (https://arxiv.org/abs/1706.02677)
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # debug_eval mode: reduce training to 10% of full training
    if args.debug_eval:
        print("\n" + "="*60)
        print("🐛 DEBUG EVAL MODE ACTIVATED")
        print("   - Training epochs reduced to 10% of normal")
        print("   - Evaluation interval reduced for quicker feedback")
        print("   - Use this to quickly test evaluation output")
        print("="*60 + "\n")
        
        # 1. Reduce training epochs to 10% (main time saver)
        original_epochs = cfg.runner.max_epochs
        debug_epochs = max(2, int(original_epochs * 0.1))  # At least 2 epochs
        cfg.runner.max_epochs = debug_epochs
        print(f"🔧 Max epochs: {original_epochs} → {debug_epochs}")
        
        # 2. Adjust learning rate schedule for shorter training
        if hasattr(cfg, 'lr_config') and cfg.lr_config.get('step'):
            original_steps = cfg.lr_config['step']
            # Scale LR steps proportionally to new epoch count
            debug_steps = [max(1, int(step * debug_epochs / original_epochs)) for step in original_steps]
            # Remove duplicates and ensure steps are valid
            debug_steps = sorted(list(set([s for s in debug_steps if s < debug_epochs])))
            if debug_steps:
                cfg.lr_config['step'] = debug_steps
                print(f"🔧 LR steps: {original_steps} → {debug_steps}")
            else:
                # If no valid steps, use linear decay
                cfg.lr_config = dict(policy='linear', warmup=None)
                print(f"🔧 LR schedule: step → linear (no valid steps for {debug_epochs} epochs)")
        
        # 3. Reduce evaluation interval for quicker feedback
        if hasattr(cfg, 'evaluation'):
            original_interval = cfg.evaluation.get('interval', 1)
            debug_interval = max(1, min(original_interval, debug_epochs // 2))  
            cfg.evaluation['interval'] = debug_interval
            print(f"🔧 Evaluation interval: {original_interval} → {debug_interval}")
        
        # 4. Reduce checkpoint interval  
        if hasattr(cfg, 'checkpoint_config'):
            original_ckpt_interval = cfg.checkpoint_config.get('interval', 1)
            debug_ckpt_interval = max(1, min(original_ckpt_interval, debug_epochs // 2))
            cfg.checkpoint_config['interval'] = debug_ckpt_interval
            print(f"🔧 Checkpoint interval: {original_ckpt_interval} → {debug_ckpt_interval}")
        
        # 5. Optional: Slightly reduce batch size to speed up iterations
        original_samples = cfg.data.samples_per_gpu
        debug_samples = max(1, int(original_samples * 0.7))  # 30% reduction
        cfg.data.samples_per_gpu = debug_samples
        print(f"🔧 Samples per GPU: {original_samples} → {debug_samples}")
        
        # 6. Ensure validation is enabled for debug mode
        if args.no_validate:
            print("⚠️  WARNING: --no-validate ignored in debug mode, enabling validation")
            args.no_validate = False

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    # specify logger name, if we still use 'mmdet', the output info will be
    # filtered and won't be saved in the log_file
    # TODO: ugly workaround to judge whether we are training det or seg model
    if cfg.model.type in ['EncoderDecoder3D']:
        logger_name = 'mmseg'
    else:
        logger_name = 'mmdet'
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=logger_name)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    
    # Log debug eval mode status
    if args.debug_eval:
        logger.info("🐛 DEBUG EVAL MODE: Training reduced to 10% for quick evaluation testing")
        logger.info(f"   Max epochs: {cfg.runner.max_epochs}")
        logger.info(f"   Samples per GPU: {cfg.data.samples_per_gpu}")
        if hasattr(cfg, 'evaluation'):
            logger.info(f"   Evaluation interval: {cfg.evaluation.get('interval', 'N/A')}")
        if hasattr(cfg, 'lr_config'):
            if cfg.lr_config.get('step'):
                logger.info(f"   LR steps: {cfg.lr_config['step']}")
            else:
                logger.info(f"   LR policy: {cfg.lr_config.get('policy', 'N/A')}")
    
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model.init_weights()

    logger.info(f'Model:\n{model}')
    datasets = [build_dataset(cfg.data.train)]
    
    # Log actual dataset size in debug mode
    if args.debug_eval:
        logger.info(f"🐛 DEBUG EVAL: Training dataset size: {len(datasets[0])} samples")
    
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        # in case we use a dataset wrapper
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        # set test_mode=False here in deep copied config
        # which do not affect AP/AR calculation later
        # refer to https://mmdetection3d.readthedocs.io/en/latest/tutorials/customize_runtime.html#customize-workflow  # noqa
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE  # for segmentors
            if hasattr(datasets[0], 'PALETTE') else None)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
