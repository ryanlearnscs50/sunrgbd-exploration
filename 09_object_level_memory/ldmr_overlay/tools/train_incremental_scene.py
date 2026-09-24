#!/usr/bin/env python3
"""
Scene-Based Incremental Learning Training Script for TR3D

This version uses scene-based memory replay instead of object-based exemplars.
Complete scenes from previous stages are replayed with filtered labels.

Key improvements:
1. Explicit stage mapping in config files
2. Dynamic head expansion (7→14→21→28→35) with proper masking
3. Loss masking for unseen classes
4. Clear validation and error handling

Usage:
    python tools/train_incremental_explicit.py \
        configs/incremental/tr3d_scannet_35class_3stages_explicit.py \
        --work-dir ./my_work_dirs/incremental_explicit
"""

import argparse
import copy
import os
import json
import pickle
import re
import shlex
import shutil
import socket
import sys
import time
from os import path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Ensure repo root is on PYTHONPATH when running as a script, e.g.
# `python tools/train_incremental_scene.py ...` (otherwise `import mmdet3d` fails).
REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist

from mmdet3d.apis import init_random_seed, train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger
from mmdet.apis import set_random_seed
from mmdet3d.datasets.incremental_mappings import (
    create_mapping_from_config,
    get_seen_classes_mask,
    validate_incremental_mappings
)
from mmdet3d.datasets.pseudo_label_utils import (
    evaluate_pseudo_label_file_hits,
    log_pseudo_hit_metrics,
)
from mmdet3d.utils.incremental_paths import IncrementalPaths
from mmdet3d.utils.forgetting_metrics import (
    calculate_forgetting_metrics_from_stage_metrics_json,
)
from mmdet3d.utils.pseudo_consistency import (
    generate_pseudo_set_for_indices,
    compute_consistency_drop,
    save_jsonl,
)
from mmdet3d.utils.ld_strategy_config import (
    LD_DESIGN1_STRATEGY,
    LD_DESIGN2_STRATEGY,
    get_ld_design_stage1_filenames,
    validate_scene_memory_ld_strategy_config,
)
from tools.incremental_scene.cfg_utils import (
    cfg_get as _cfg_get_impl,
    cfg_has_key as _cfg_has_key_impl,
    unwrap_train_dataset_cfg as _unwrap_train_dataset_cfg_impl,
    validate_unified_replay_pseudo_cfg_or_raise as _validate_unified_replay_pseudo_cfg_or_raise_impl,
)
from tools.incremental_scene.orchestrator import (
    resolve_segment_times as _resolve_segment_times_impl,
    segmented_mode_label as _segmented_mode_label_impl,
    should_use_sunrgbd_segmented_path as _should_use_sunrgbd_segmented_path_impl,
)
from tools.incremental_scene.pseudo_pipeline import (
    build_pseudo_config_suffix as _build_pseudo_config_suffix_impl,
    resolve_stage_pseudo_file as _resolve_stage_pseudo_file_impl,
    validate_pseudo_labels_nonfatal as _validate_pseudo_labels_nonfatal_impl,
)
from tools.incremental_scene.reviewing_ld import (
    build_review_weight_policy as _build_review_weight_policy_impl,
    build_reviewing_eval_payload as _build_reviewing_eval_payload_impl,
    resolve_effective_ld_reviewing_params as _resolve_effective_ld_reviewing_params_impl,
)
from tools.incremental_scene.runtime_snapshot import (
    sanitize_for_cfg_snapshot as _sanitize_for_cfg_snapshot_impl,
    try_get_git_info as _try_get_git_info_impl,
    write_resolved_config_snapshot as _write_resolved_config_snapshot_impl,
)
from tools.incremental_scene.stage_settings import (
    fingerprint_stage_definitions as _fingerprint_stage_definitions_impl,
    infer_sunrgbd_stage_setting as _infer_sunrgbd_stage_setting_impl,
    log_stage_groups as _log_stage_groups_impl,
    resolve_stage_setting as _resolve_stage_setting_impl,
    validate_stage_setting_or_raise as _validate_stage_setting_or_raise_impl,
)
from tools.validate_scannet_alignment_contract import (
    validate_scannet_alignment_contract,
)


def get_innermost_dataset(dataset):
    """Get the innermost dataset from potentially wrapped dataset.

    This handles RepeatDataset and other wrappers to access the underlying
    dataset that has domain-specific methods like update_scene_memory_bank_from_stage.

    Args:
        dataset: Dataset that may be wrapped

    Returns:
        The innermost dataset with domain methods
    """
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    return dataset


def _strip_scene_memory_cfg_meta(value: Any) -> Any:
    """Remove config-system-only keys before SceneMemoryBank constructor call."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if str(k) == '_delete_':
                continue
            out[k] = _strip_scene_memory_cfg_meta(v)
        return out
    if isinstance(value, list):
        return [_strip_scene_memory_cfg_meta(v) for v in value]
    return value


def _attach_object_memory_insertion(dataset_cfg, object_memory_bank,
                                    insertion_cfg=None) -> None:
    """Attach the shared bank to one pre-augmentation pipeline transform."""
    if object_memory_bank is None:
        return
    pipeline = getattr(dataset_cfg, 'pipeline', None)
    if pipeline is None:
        raise ValueError('Object replay requires a training dataset pipeline.')
    insertion = dict(insertion_cfg or {})
    insertion['type'] = 'InsertExemplarObjects'
    insertion['memory_bank'] = object_memory_bank
    for transform in pipeline:
        transform_type = transform.get('type') if hasattr(transform, 'get') else None
        if transform_type == 'InsertExemplarObjects':
            transform.update(insertion)
            return
    annotation_index = next(
        (i for i, transform in enumerate(pipeline)
         if getattr(transform, 'get', lambda *_: None)('type') == 'LoadAnnotations3D'),
        None)
    if annotation_index is None:
        raise ValueError(
            'Object replay requires LoadAnnotations3D before insertion.')
    # ScanNet points become box-aligned in GlobalAlignment. Inserting before it
    # would mix raw points with aligned boxes and transform pasted crops twice.
    alignment_indices = [
        i for i, transform in enumerate(pipeline)
        if getattr(transform, 'get', lambda *_: None)('type') == 'GlobalAlignment'
    ]
    insertion_after = max([annotation_index] + alignment_indices)
    pipeline.insert(insertion_after + 1, insertion)


def _detach_object_memory_insertion(dataset_cfg) -> None:
    """Make a dataset config natural-only for LD source-scene evaluation."""
    pipeline = getattr(dataset_cfg, 'pipeline', None)
    if pipeline is not None:
        dataset_cfg.pipeline = [
            transform for transform in pipeline
            if getattr(transform, 'get', lambda *_: None)(
                'type') != 'InsertExemplarObjects'
        ]
    dataset_cfg.object_memory_bank = None


def _validate_object_memory_resume(object_memory_bank, start_stage,
                                   stage_definitions, state_path) -> None:
    """Require a bank produced by the stage immediately before a resume."""
    start_stage = int(start_stage)
    if start_stage <= 1:
        raise ValueError(
            '--object-memory-path is only valid with --start-stage > 1.')
    expected_memory_stage = start_stage - 1
    if object_memory_bank.loaded_stage_id != expected_memory_stage:
        raise ValueError(
            'Object-memory state/checkpoint stage mismatch: expected a '
            f'stage-{expected_memory_stage} bank for --start-stage '
            f'{start_stage}, but {state_path} records stage '
            f'{object_memory_bank.loaded_stage_id}.')
    expected_classes = sorted({
        int(class_id)
        for stage in stage_definitions
        if int(stage['stage_id']) <= expected_memory_stage
        for class_id in stage.get('class_indices', [])
    })
    stored_classes = object_memory_bank.get_stored_classes()
    if stored_classes != expected_classes:
        raise ValueError(
            'Object-memory class coverage mismatch: expected classes '
            f'{expected_classes}, found {stored_classes}.')


def _cfg_get(cfg_node, key: str, default=None):
    return _cfg_get_impl(cfg_node, key, default)


def _cfg_has_key(cfg_node, key: str) -> bool:
    return _cfg_has_key_impl(cfg_node, key)


def _unwrap_train_dataset_cfg(train_cfg):
    return _unwrap_train_dataset_cfg_impl(train_cfg)


def _validate_unified_replay_pseudo_cfg_or_raise(incremental_cfg) -> None:
    _validate_unified_replay_pseudo_cfg_or_raise_impl(incremental_cfg)


def _log_prediction_head_summary(*, logger, model, prefix: str) -> None:
    """Log only the prediction head shapes (avoid full model dumps)."""
    if logger is None or model is None:
        return

    try:
        target_model = model.module if hasattr(model, 'module') else model
        head = getattr(target_model, 'head', None) or getattr(target_model, 'bbox_head', None)
        if head is None:
            logger.info(f"{prefix}: (no head found)")
            return

        parts = [f"type={head.__class__.__name__}"]

        n_classes = getattr(head, 'n_classes', None)
        if n_classes is not None:
            parts.append(f"n_classes={int(n_classes)}")

        if hasattr(head, 'cls_conv') and getattr(head.cls_conv, 'kernel', None) is not None:
            parts.append(f"cls_kernel={tuple(head.cls_conv.kernel.shape)}")
            if getattr(head.cls_conv, 'bias', None) is not None:
                parts.append(f"cls_bias={tuple(head.cls_conv.bias.shape)}")

        if hasattr(head, 'bbox_conv') and getattr(head.bbox_conv, 'kernel', None) is not None:
            parts.append(f"bbox_kernel={tuple(head.bbox_conv.kernel.shape)}")
            if getattr(head.bbox_conv, 'bias', None) is not None:
                parts.append(f"bbox_bias={tuple(head.bbox_conv.bias.shape)}")

        logger.info(f"{prefix}: " + ", ".join(parts))
    except Exception as e:
        logger.warning(f"{prefix}: (failed to summarize head: {e})")


def _get_log_verbosity(cfg) -> str:
    """Get log verbosity level from config.

    Supported values:
      - 'normal' (default): concise, no long lists
      - 'debug': include verbose lists and detailed sanity metrics
    """
    try:
        value = cfg.get('log_verbosity', 'normal')
    except Exception:
        value = getattr(cfg, 'log_verbosity', 'normal')
    return str(value).strip().lower()


def _get_artifact_profile_requested(cfg) -> str:
    """Read artifact logging profile from config.

    Supported values:
      - 'auto' (default): resolve to 'ld_path_only' for LD memory strategy,
        otherwise 'full'
      - 'full': keep all historical artifacts
      - 'ld_path_only': keep only LD score->action path artifacts
    """
    raw = None
    try:
        logging_cfg = cfg.get('logging', None)
    except Exception:
        logging_cfg = getattr(cfg, 'logging', None)

    if logging_cfg is not None:
        try:
            raw = logging_cfg.get('artifact_profile', None)
        except Exception:
            raw = getattr(logging_cfg, 'artifact_profile', None)

    if raw is None:
        try:
            raw = cfg.get('artifact_profile', None)
        except Exception:
            raw = getattr(cfg, 'artifact_profile', None)

    if raw is None:
        raw = 'auto'

    profile = str(raw).strip().lower()
    allowed = {'auto', 'full', 'ld_path_only'}
    if profile not in allowed:
        raise ValueError(
            "Invalid logging.artifact_profile. "
            f"Expected one of {sorted(allowed)}, got '{profile}'."
        )
    return profile


def _resolve_artifact_profile(*,
                              requested_profile: str,
                              learning_dynamics_strategy: bool) -> str:
    """Resolve profile, honoring auto mode."""
    requested_profile = str(requested_profile).strip().lower()
    if requested_profile == 'auto':
        return 'ld_path_only' if bool(learning_dynamics_strategy) else 'full'
    return requested_profile


def _is_ld_selection_strategy(selection_strategy: Any) -> bool:
    strategy = str(selection_strategy).strip().lower()
    return strategy in (
        'learning_dynamics',
        LD_DESIGN1_STRATEGY,
        LD_DESIGN2_STRATEGY,
    )


def _is_ld_design_selection_strategy(selection_strategy: Any) -> bool:
    strategy = str(selection_strategy).strip().lower()
    return strategy in (LD_DESIGN1_STRATEGY, LD_DESIGN2_STRATEGY)


def _fingerprint_stage_definitions(stage_definitions):
    return _fingerprint_stage_definitions_impl(stage_definitions)


def _infer_sunrgbd_stage_setting(stage_definitions: List[Dict[str, Any]],
                                 num_classes: int) -> Optional[str]:
    return _infer_sunrgbd_stage_setting_impl(stage_definitions, num_classes)


def _resolve_stage_setting(incremental_cfg,
                           stage_definitions: List[Dict[str, Any]],
                           num_classes: int) -> Tuple[Optional[str], str]:
    return _resolve_stage_setting_impl(incremental_cfg, stage_definitions, num_classes)


def _validate_stage_setting_or_raise(*,
                                     stage_setting: Optional[str],
                                     stage_definitions: List[Dict[str, Any]]) -> None:
    _validate_stage_setting_or_raise_impl(
        stage_setting=stage_setting,
        stage_definitions=stage_definitions,
        repo_root=REPO_ROOT,
    )


def _log_stage_groups(*,
                      logger,
                      stage_setting: Optional[str],
                      stage_setting_source: str,
                      stage_definitions: List[Dict[str, Any]]) -> None:
    _log_stage_groups_impl(
        logger=logger,
        stage_setting=stage_setting,
        stage_setting_source=stage_setting_source,
        stage_definitions=stage_definitions,
    )


def _load_or_compute_stage_scene_counts(
        *,
        base_cfg,
        incremental_cfg,
        stage_definitions,
        mappings,
        work_dir,
        logger,
        stats_path,
        is_main_process: bool,
        use_cache: bool = True):
    """Compute and cache per-stage train scene counts N_s (unique scenes, no RepeatDataset multiplier).

    This is used for stage-ratio quota allocation in incremental memory-bank runs.
    The cache is keyed by (ann_file, filter_empty_gt, stage_definitions_fingerprint).
    """
    # Resolve ann_file + filter_empty_gt from inner dataset config
    try:
        train_cfg = base_cfg.data.train
        inner_cfg = train_cfg.dataset  # RepeatDataset wraps inner dataset
        ann_file = str(getattr(inner_cfg, 'ann_file', ''))
        filter_empty_gt = bool(getattr(inner_cfg, 'filter_empty_gt', False))
    except Exception:
        ann_file = ''
        filter_empty_gt = False

    stage_fp = _fingerprint_stage_definitions(stage_definitions)

    # Total train scenes from info pkl (for logging + budget derivation)
    total_train_scenes = None
    if ann_file and osp.exists(ann_file):
        try:
            with open(ann_file, 'rb') as f:
                infos = pickle.load(f)
            total_train_scenes = int(len(infos))
        except Exception:
            total_train_scenes = None

    # Load cache if valid
    if use_cache and osp.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                cached = json.load(f)
            if (cached.get('ann_file') == ann_file and
                    bool(cached.get('filter_empty_gt', False)) == bool(filter_empty_gt) and
                    cached.get('stage_definitions_fingerprint') == stage_fp):
                stage_counts = cached.get('per_stage_train_scenes', None)
                if isinstance(stage_counts, list) and len(stage_counts) == len(stage_definitions):
                    if logger is not None and is_main_process:
                        logger.info(f"Loaded cached stage scene counts (N_s) from: {stats_path}")
                        logger.info(f"    ann_file: {ann_file}")
                        logger.info(f"    filter_empty_gt: {filter_empty_gt}")
                        logger.info(f"    per_stage_train_scenes: {stage_counts}")
                    return {
                        'ann_file': ann_file,
                        'filter_empty_gt': filter_empty_gt,
                        'stage_definitions_fingerprint': stage_fp,
                        'total_train_scenes': cached.get('total_train_scenes', total_train_scenes),
                        'per_stage_train_scenes': stage_counts,
                        'quota_rounding': cached.get('quota_rounding', 'largest_remainder'),
                    }
        except Exception as e:
            if logger is not None and is_main_process:
                logger.warning(f"Failed to load stage stats cache {stats_path}: {e}")

    if not is_main_process:
        # Non-main ranks: wait for rank0 to create the file (best-effort).
        return None

    # Compute N_s by building each stage's train dataset and counting non-empty scenes
    # after stage filtering (training mode) with filter_empty_gt semantics.
    if logger is not None:
        logger.info("Computing per-stage train scene counts (N_s)...")
        logger.info("    Note: counts are on the innermost dataset (unique scenes), not RepeatDataset length.")

    per_stage_counts = []
    for stage_idx, stage_def in enumerate(stage_definitions):
        train_dataset_cfg = copy.deepcopy(base_cfg.data.train)
        # Ensure we build the innermost incremental dataset with no replay/pseudo.
        try:
            train_dataset_cfg.dataset.stage_definition = stage_def
            train_dataset_cfg.dataset.mappings = mappings
            train_dataset_cfg.dataset.scene_memory_bank = None
            train_dataset_cfg.dataset.scene_dedup_strategy = 'keep_both'
            train_dataset_cfg.dataset.evaluation_mode = False
            train_dataset_cfg.dataset.all_stage_definitions = stage_definitions
            train_dataset_cfg.dataset.experiment_dir = work_dir
            # Explicitly disable pseudo labels for stats computation
            train_dataset_cfg.dataset.use_pseudo_labels = False
            train_dataset_cfg.dataset.pseudo_label_config = None
            train_dataset_cfg.dataset.pseudo_label_dir = None
        except Exception:
            pass

        ds = build_dataset(train_dataset_cfg)
        inner = get_innermost_dataset(ds)

        # N_s is defined as number of scenes that remain non-empty after stage
        # filtering (with filter_empty_gt=True semantics).
        n_s = None
        try:
            infos = getattr(inner, 'data_infos', None)
            if isinstance(infos, list):
                n_s = int(sum(
                    int((info.get('annos', {}) or {}).get('gt_num', 0) or 0) > 0
                    for info in infos
                    if isinstance(info, dict)
                ))
        except Exception:
            n_s = None

        if n_s is None:
            n_s = int(len(inner))

        per_stage_counts.append(int(n_s))
        if logger is not None:
            logger.info(f"    Stage {int(stage_def['stage_id'])}: N_s={n_s}")

    try:
        dataset_type = str(getattr(base_cfg.data.train.dataset, 'type', ''))
    except Exception:
        dataset_type = ''
    payload = {
        'dataset': str(dataset_type or 'unknown'),
        'ann_file': ann_file,
        'filter_empty_gt': bool(filter_empty_gt),
        'stage_definitions_fingerprint': stage_fp,
        'total_train_scenes': int(total_train_scenes) if total_train_scenes is not None else None,
        'per_stage_train_scenes': per_stage_counts,
        'quota_rounding': 'largest_remainder',
        'created_time': time.time(),
    }

    try:
        os.makedirs(osp.dirname(stats_path), exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(payload, f, indent=2)
        if logger is not None:
            logger.info(f"Saved stage scene counts cache to: {stats_path}")
    except Exception as e:
        if logger is not None:
            logger.warning(f"Failed to write stage stats cache {stats_path}: {e}")

    return payload


def _create_pseudo_labels_metadata(metadata_file, stage_id, pseudo_label_file,
                                  source_global, global_source, config_suffix,
                                  confidence_threshold, generation_checkpoint, logger):
    """Create or update metadata file for pseudo labels tracking.

    Args:
        metadata_file (Path): Path to metadata JSON file
        stage_id (int): Current stage ID
        pseudo_label_file (Path): Path to pseudo label file
        source_global (bool): Whether copied from global source
        global_source (str): Global source path if copied
        config_suffix (str): Configuration suffix
        confidence_threshold (float): Confidence threshold used
        generation_checkpoint (str): Checkpoint used for generation
        logger: Logger instance
    """
    import json
    from datetime import datetime

    # Load existing metadata or create new
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {
            "created": datetime.now().isoformat(),
            "stages": {}
        }

    # Add stage information
    stage_key = f"stage_{stage_id}"
    metadata["stages"][stage_key] = {
        "pseudo_label_file": str(pseudo_label_file.name),  # Just filename, not full path
        "config_suffix": config_suffix,
        "confidence_threshold": confidence_threshold,
        "copied_from_global": source_global,
        "global_source": global_source,
        "generation_checkpoint": generation_checkpoint,
        "created": datetime.now().isoformat()
    }

    metadata["last_updated"] = datetime.now().isoformat()

    # Save metadata
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    if logger:
        logger.info(f"Updated pseudo labels metadata: {metadata_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train incremental 3D detector with explicit mappings')
    parser.add_argument('config', help='incremental training config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # Removed old debug flags - now use --cfg-options runner.max_epochs=X data.train.times=Y for fine control
    parser.add_argument(
        '--load-checkpoint',
        type=str,
        help='[DEPRECATED] Use --checkpoint-path instead. Load checkpoint from previous stage.')
    parser.add_argument(
        '--start-stage',
        type=int,
        default=1,
        help='Stage to start training from (1-N). If > 1, requires --checkpoint-path.')
    parser.add_argument(
        '--end-stage',
        type=int,
        default=None,
        help=(
            'Stage to stop training after (1-N). Useful for discovery experiments. '
            'If omitted, defaults to the last configured stage.'
        ))
    parser.add_argument(
        '--checkpoint-path',
        type=str,
        help='Path to checkpoint from previous stage when using --start-stage > 1.')
    parser.add_argument(
        '--object-memory-path',
        type=str,
        help=(
            'Saved object-memory .pkl from the stage immediately before '
            '--start-stage. Required for object-memory resume at stage 3+; '
            'optional at stage 2.'))
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config settings')
    # Distributed launcher (optional)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher for distributed training')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--validate-scannet-alignment',
        dest='validate_scannet_alignment',
        action='store_true',
        help=(
            'Run ScanNet alignment contract validation at startup. '
            'Default is auto (enabled for ScanNet-35 runs).'
        ),
    )
    parser.add_argument(
        '--no-validate-scannet-alignment',
        dest='validate_scannet_alignment',
        action='store_false',
        help='Disable ScanNet alignment contract validation at startup.',
    )
    parser.set_defaults(validate_scannet_alignment=None)
    args = parser.parse_args()

    # Ensure LOCAL_RANK is set for torch.distributed
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    # Old debug mode validation removed - now use --cfg-options for fine control

    # Basic validation; full bounds are validated after loading config.
    if args.start_stage < 1:
        parser.error("--start-stage must be >= 1")
    if args.end_stage is not None and args.end_stage < 1:
        parser.error("--end-stage must be >= 1 when provided")
    if args.end_stage is not None and args.start_stage > args.end_stage:
        parser.error(
            f"--start-stage ({args.start_stage}) cannot be greater than "
            f"--end-stage ({args.end_stage})"
        )

    # Validate checkpoint requirements
    if args.start_stage > 1:
        if not args.checkpoint_path:
            parser.error(f"--checkpoint-path is required when starting from stage {args.start_stage}")
        if not osp.exists(args.checkpoint_path):
            parser.error(f"Checkpoint not found: {args.checkpoint_path}")
    elif args.checkpoint_path:
        parser.error("--checkpoint-path should only be used with --start-stage > 1")

    return args


def prepare_stage_config(base_cfg, stage_definition, stage_idx, stage_definitions, work_dir,
                        incremental_cfg=None, gpu_ids=None):
    """Prepare configuration for a specific stage using explicit definitions.

    Debug modes for different testing purposes:
    - debug_1epoch: Ultra-fast (1 epoch/stage) for rapid memory bank/code testing
    - debug_2epochs: Fast (2 epochs/stage) for quick incremental learning validation
    - test_mode: [DEPRECATED] Same as debug_2epochs, kept for backward compatibility

    Args:
        base_cfg: Base configuration to modify
        stage_definition: Stage-specific settings (epochs, classes, etc.)
        stage_idx: Current stage index (0-based)
        stage_definitions: All stage definitions for context
        work_dir: Working directory for outputs
        debug_1epoch: Enable 1-epoch ultra-fast debug mode
        debug_2epochs: Enable 2-epoch fast debug mode
        test_mode: Legacy debug mode (deprecated, use debug_2epochs instead)
        incremental_cfg: Incremental config with evaluation settings (optional)
    """
    stage_cfg = copy.deepcopy(base_cfg)

    # The incremental config owns stage-time scheduler overrides. Otherwise a
    # top-level schedule is silently replaced by the detector base schedule.
    if incremental_cfg is not None and 'lr_config' in incremental_cfg:
        stage_cfg.lr_config = copy.deepcopy(incremental_cfg.lr_config)

    # Copy evaluation config from incremental_cfg if available
    # This ensures --cfg-options evaluation.interval works properly
    if incremental_cfg and hasattr(incremental_cfg, 'evaluation'):
        stage_cfg.evaluation = incremental_cfg.evaluation

    # Copy selected top-level config sections that affect stage-time behavior.
    # This supports `--cfg-options` for these sections without requiring
    # stage-specific config edits.
    if incremental_cfg is not None:
        try:
            if 'SCORING' in incremental_cfg:
                stage_cfg.SCORING = copy.deepcopy(incremental_cfg.get('SCORING'))
        except Exception:
            pass
        try:
            if 'MEMORY' in incremental_cfg:
                stage_cfg.MEMORY = copy.deepcopy(incremental_cfg.get('MEMORY'))
        except Exception:
            pass

    # Copy data config to preserve cfg-options overrides (data.train.times, etc.)
    if incremental_cfg and hasattr(incremental_cfg, 'data'):
        if hasattr(incremental_cfg.data, 'train') and hasattr(incremental_cfg.data.train, 'times'):
            stage_cfg.data.train.times = incremental_cfg.data.train.times
            print(f"    Applied cfg-options override: data.train.times={stage_cfg.data.train.times}")

        # Copy pseudo label configuration from incremental_cfg
        if hasattr(incremental_cfg.data, 'train'):
            if hasattr(incremental_cfg.data.train, 'use_pseudo_labels'):
                stage_cfg.data.train.use_pseudo_labels = incremental_cfg.data.train.use_pseudo_labels
                print(
                    f"    Applied pseudo label config: "
                    f"use_pseudo_labels={stage_cfg.data.train.use_pseudo_labels}"
                )
            if hasattr(incremental_cfg.data.train, 'pseudo_label_config'):
                stage_cfg.data.train.pseudo_label_config = copy.deepcopy(
                    incremental_cfg.data.train.pseudo_label_config)

                # Apply stage-specific confidence threshold if available
                stage_id = stage_definition.get('stage_id', 1)
                if hasattr(incremental_cfg.data.train.pseudo_label_config, 'stage_thresholds'):
                    stage_thresholds = incremental_cfg.data.train.pseudo_label_config.stage_thresholds
                    if stage_id in stage_thresholds:
                        stage_cfg.data.train.pseudo_label_config.confidence_threshold = stage_thresholds[stage_id]
                        print(
                            f"    Applied stage-specific threshold: "
                            f"stage_id={stage_id}, confidence_threshold={stage_thresholds[stage_id]}"
                        )
                    else:
                        print(
                            f"    No stage-specific threshold found for "
                            f"stage_id={stage_id}; using default"
                        )

                # Only print confidence_threshold if it exists (may not exist if pseudo labels disabled)
                if hasattr(stage_cfg.data.train.pseudo_label_config, 'confidence_threshold'):
                    print(
                        f"    Applied pseudo label config: "
                        f"confidence_threshold={stage_cfg.data.train.pseudo_label_config.confidence_threshold}"
                    )
            if hasattr(incremental_cfg.data.train, 'scene_dedup_strategy'):
                stage_cfg.data.train.scene_dedup_strategy = incremental_cfg.data.train.scene_dedup_strategy
                print(
                    f"    Applied scene dedup strategy: "
                    f"{stage_cfg.data.train.scene_dedup_strategy}"
                )

    # Check if using dynamic head expansion
    use_dynamic_head = getattr(base_cfg, 'use_dynamic_head', False)
    if not use_dynamic_head:
        # Standard (non-dynamic) mode: use the full class space.
        all_indices = []
        for stage in stage_definitions:
            all_indices.extend([int(x) for x in stage.get('class_indices', [])])
        if all_indices:
            stage_cfg.model.head.n_classes = int(max(all_indices) + 1)
    # Note: For dynamic head mode, n_classes will be set in main training loop

    # Pass stage information for dataset filtering and loss masking
    stage_cfg.stage_definition = stage_definition
    stage_cfg.stage_idx = stage_idx
    stage_cfg.current_stage_classes = stage_definition['class_indices']

    # CRITICAL FIX: Create TWO separate masks for training vs evaluation
    # Training mask: ALL seen classes up to current stage (for loss masking with replay)
    all_stages_up_to_current = stage_definitions[:stage_idx+1]  # Include current stage
    stage_cfg.training_classes_mask = get_seen_classes_mask(
        all_stages_up_to_current, stage_definition['stage_id'])
    if torch.cuda.is_available():
        stage_cfg.training_classes_mask = stage_cfg.training_classes_mask.cuda()

    # Evaluation mask: ALL seen classes up to current stage (cumulative) - same as training now
    stage_cfg.evaluation_classes_mask = get_seen_classes_mask(
        all_stages_up_to_current, stage_definition['stage_id'])
    if torch.cuda.is_available():
        stage_cfg.evaluation_classes_mask = stage_cfg.evaluation_classes_mask.cuda()

    # Store both masks for reference
    stage_cfg.cumulative_seen_classes = []
    for s in all_stages_up_to_current:
        stage_cfg.cumulative_seen_classes.extend(s['class_indices'])
    stage_cfg.cumulative_seen_classes = sorted(list(set(stage_cfg.cumulative_seen_classes)))

    # Default to training mask for loss computation
    stage_cfg.seen_classes_mask = stage_cfg.training_classes_mask

    # Store experiment root and stage info for path management
    stage_cfg.experiment_dir = work_dir
    stage_cfg.stage_id = stage_definition['stage_id']

    # Set work_dir to checkpoint directory for mmdet3d API compatibility
    # This ensures checkpoint saving works while maintaining unified structure
    stage_work_dir = osp.join(work_dir, 'checkpoints', f"stage_{stage_definition['stage_id']}")
    stage_cfg.work_dir = stage_work_dir
    mmcv.mkdir_or_exist(stage_work_dir)

    # Ensure gpu_ids and seed are set
    if gpu_ids is not None:
        stage_cfg.gpu_ids = list(gpu_ids)
    elif not hasattr(stage_cfg, 'gpu_ids'):
        # Fallback to single-GPU if not provided
        stage_cfg.gpu_ids = [0]
    if not hasattr(stage_cfg, 'seed'):
        stage_cfg.seed = 0

    # CRITICAL FIX: Stage definition epochs ALWAYS take priority
    stage_epochs = stage_definition.get('epochs')
    if stage_epochs is None:
        raise ValueError(
            f"Stage {stage_definition.get('stage_id', '?')} missing 'epochs' field. "
            f"All stage definitions must specify epochs."
        )

    # Set epochs from stage definition (highest priority)
    stage_cfg.runner.max_epochs = stage_epochs
    print(
        f"    Stage {stage_definition['stage_id']}: epochs={stage_epochs} "
        f"(from stage definition)"
    )

    # Check for CLI override and warn if different
    if hasattr(incremental_cfg, 'runner') and hasattr(incremental_cfg.runner, 'max_epochs'):
        cli_epochs = incremental_cfg.runner.max_epochs
        if cli_epochs != stage_epochs:
            print(
                f"    NOTE: --cfg-options runner.max_epochs={cli_epochs} ignored; "
                f"using stage epochs={stage_epochs}"
            )

    # Ensure sensible evaluation scheduling for short stages
    if not hasattr(stage_cfg, 'evaluation') or stage_cfg.evaluation is None:
        stage_cfg.evaluation = dict(interval=1, metric='mAP')
    else:
        try:
            orig_interval = stage_cfg.evaluation.get('interval', 1)
        except AttributeError:
            # In case evaluation is a ConfigDict without .get behavior
            orig_interval = 1
        # If the stage runs fewer epochs than the eval interval, force per-epoch eval
        if stage_epochs < max(1, int(orig_interval)):
            print(
                f"    Forcing evaluation.interval=1 "
                f"(stage_epochs={stage_epochs} < interval={orig_interval})"
            )
            stage_cfg.evaluation.interval = 1

    # Set learning rate for stage
    stage_cfg.optimizer.lr = stage_definition.get('lr', stage_cfg.optimizer.lr)

    return stage_cfg


def _get_debug_mode_info(args, stage_definitions):
    """Return debug mode information for runtime diagnostics.

    Returns comprehensive information about what debug mode is active,
    estimated timing, and what each mode is suitable for testing.
    """
    total_stages = len(stage_definitions)

    # Calculate based on actual configuration (no more fixed debug modes)
    total_epochs = sum(stage['epochs'] for stage in stage_definitions)
    mode_name = "Configurable Training Mode"
    epoch_info = f"Configurable epochs per stage ({', '.join([str(stage['epochs']) for stage in stage_definitions])})"
    estimated_time = "Depends on --cfg-options settings"
    purpose = "Flexible training with full configuration control"
    suitable_for = [
        "Use --cfg-options runner.max_epochs=1 data.train.times=1 for ultra-fast debug",
        "Use --cfg-options runner.max_epochs=2 data.train.times=3 for quick validation",
        "Use --cfg-options data.train.times=15 for production training",
        "Any combination for your specific needs"
    ]

    console_messages = [
        f"{mode_name}",
        f"Training: {epoch_info} ({total_stages} stages = {total_epochs} total epochs)",
        f"Estimated time: {estimated_time}",
        f"Purpose: {purpose}",
        "",
        "Suitable for:"
    ]
    console_messages.extend([f"  • {item}" for item in suitable_for])

    log_messages = [
        f"{mode_name}: {epoch_info}",
        f"Total epochs across all stages: {total_epochs}",
        f"Estimated completion time: {estimated_time}",
        f"Primary purpose: {purpose}"
    ]

    return {
        'console_messages': console_messages,
        'log_messages': log_messages,
        'mode_name': mode_name,
        'total_epochs': total_epochs,
        'estimated_time': estimated_time
    }


def calculate_forgetting_metrics(
        previous_stage_results,
        current_stage_results,
        previous_stage_classes,
        mappings,
        logger,
        *,
        previous_stage_id: Optional[int] = None,
        current_stage_id: Optional[int] = None,
        verbose: bool = False):
    """Calculate forgetting metrics for previous stage classes.

    Args:
        previous_stage_results: Dict with per-class AP results from previous stage
        current_stage_results: Dict with per-class AP results from current stage
        previous_stage_classes: List of class indices from previous stage
        mappings: Class mappings dictionary
        logger: Logger instance

    Returns:
        Dict with forgetting metrics
    """
    import re

    forgetting_metrics = {}
    per_class_forgetting = {}

    def _infer_stage_prefix(results) -> str:
        if not isinstance(results, dict):
            return ''
        for k in results.keys():
            m = re.match(r'^(stage_\\d+_)', str(k))
            if m:
                return m.group(1)
        return ''

    prev_prefix = _infer_stage_prefix(previous_stage_results)
    curr_prefix = _infer_stage_prefix(current_stage_results)

    def _get_metric(results, key: str, default: float = 0.0) -> float:
        if not isinstance(results, dict):
            return float(default)
        val = results.get(key, None)
        if val is None:
            return float(default)
        try:
            return float(val)
        except Exception:
            return float(default)

    # Calculate per-class forgetting
    for class_idx in previous_stage_classes:
        class_name = mappings['model_idx_to_name'].get(class_idx, f"class_{class_idx}")

        # Get AP scores for this class (support both plain and stage-prefixed keys).
        prev_ap_25 = _get_metric(
            previous_stage_results,
            f'{class_name}_AP_0.25',
            _get_metric(previous_stage_results, f'{prev_prefix}{class_name}_AP_0.25', 0.0),
        )
        curr_ap_25 = _get_metric(
            current_stage_results,
            f'{class_name}_AP_0.25',
            _get_metric(current_stage_results, f'{curr_prefix}{class_name}_AP_0.25', 0.0),
        )

        prev_ap_50 = _get_metric(
            previous_stage_results,
            f'{class_name}_AP_0.50',
            _get_metric(previous_stage_results, f'{prev_prefix}{class_name}_AP_0.50', 0.0),
        )
        curr_ap_50 = _get_metric(
            current_stage_results,
            f'{class_name}_AP_0.50',
            _get_metric(current_stage_results, f'{curr_prefix}{class_name}_AP_0.50', 0.0),
        )

        # Conventions:
        # - `forgetting_*` is a *signed delta*: curr - prev (negative => forgetting).
        # - `forgetness_*` is a *positive drop*: max(prev - curr, 0).
        forgetting_25 = curr_ap_25 - prev_ap_25
        forgetting_50 = curr_ap_50 - prev_ap_50
        forgetness_25 = max(prev_ap_25 - curr_ap_25, 0.0)
        forgetness_50 = max(prev_ap_50 - curr_ap_50, 0.0)

        per_class_forgetting[class_idx] = {
            'name': class_name,
            'prev_AP_0.25': prev_ap_25,
            'curr_AP_0.25': curr_ap_25,
            'forgetting_0.25': forgetting_25,
            'forgetness_0.25': forgetness_25,
            'prev_AP_0.50': prev_ap_50,
            'curr_AP_0.50': curr_ap_50,
            'forgetting_0.50': forgetting_50,
            'forgetness_0.50': forgetness_50,
        }

    # Calculate average forgetting
    if per_class_forgetting:
        avg_forgetting_25 = sum(v['forgetting_0.25'] for v in per_class_forgetting.values()) / len(per_class_forgetting)
        avg_forgetting_50 = sum(v['forgetting_0.50'] for v in per_class_forgetting.values()) / len(per_class_forgetting)
        avg_forgetness_25 = sum(v['forgetness_0.25'] for v in per_class_forgetting.values()) / len(per_class_forgetting)
        avg_forgetness_50 = sum(v['forgetness_0.50'] for v in per_class_forgetting.values()) / len(per_class_forgetting)

        # Find classes with worst forgetting
        worst_forgetting_25 = min(per_class_forgetting.values(), key=lambda x: x['forgetting_0.25'])
        best_retention_25 = max(per_class_forgetting.values(), key=lambda x: x['forgetting_0.25'])

        forgetting_metrics = {
            'previous_stage_id': int(previous_stage_id) if previous_stage_id is not None else None,
            'current_stage_id': int(current_stage_id) if current_stage_id is not None else None,
            'conventions': {
                'forgetting_delta': 'forgetting = curr_AP - prev_AP (negative => forgetting, positive => improvement)',
                'forgetness_drop': 'forgetness = max(prev_AP - curr_AP, 0.0) (positive drop only)',
            },
            'per_class': per_class_forgetting,
            'average_forgetting_0.25': avg_forgetting_25,
            'average_forgetting_0.50': avg_forgetting_50,
            'average_forgetness_0.25': avg_forgetness_25,
            'average_forgetness_0.50': avg_forgetness_50,
            'worst_class': worst_forgetting_25,
            'best_class': best_retention_25,
            'num_classes_degraded': sum(1 for v in per_class_forgetting.values() if v['forgetting_0.25'] < 0),
            'num_classes_improved': sum(1 for v in per_class_forgetting.values() if v['forgetting_0.25'] > 0),
            'num_classes_stable': sum(1 for v in per_class_forgetting.values() if abs(v['forgetting_0.25']) < 0.01),
        }

        # Log forgetting analysis
        logger.info("=" * 80)
        logger.info("CATASTROPHIC FORGETTING ANALYSIS (Previous Stage Classes)")
        logger.info("=" * 80)

        logger.info("Summary Statistics:")
        if previous_stage_id is not None and current_stage_id is not None:
            logger.info(
                f"   Compared: stage {int(previous_stage_id)} -> stage {int(current_stage_id)}"
            )
        logger.info(f"   Average Forgetting (AP@0.25): {avg_forgetting_25:+.4f}")
        logger.info(f"   Average Forgetting (AP@0.50): {avg_forgetting_50:+.4f}")
        logger.info(f"   Average Forgetness (drop, AP@0.25): {avg_forgetness_25:.4f}")
        logger.info(f"   Average Forgetness (drop, AP@0.50): {avg_forgetness_50:.4f}")
        logger.info(f"   Classes with degraded performance: {forgetting_metrics['num_classes_degraded']}/{len(per_class_forgetting)}")
        logger.info(f"   Classes with improved performance: {forgetting_metrics['num_classes_improved']}/{len(per_class_forgetting)}")
        logger.info(f"   Classes with stable performance: {forgetting_metrics['num_classes_stable']}/{len(per_class_forgetting)}")

        if verbose:
            logger.info("Per-Class Performance Changes (AP@0.25):")
            for class_idx in sorted(per_class_forgetting.keys()):
                metrics = per_class_forgetting[class_idx]
                forgetting_val = metrics['forgetting_0.25']

                if forgetting_val < -0.05:
                    status = "FORGET++"
                elif forgetting_val < -0.01:
                    status = "FORGET+"
                elif forgetting_val > 0.01:
                    status = "IMPROVE"
                else:
                    status = "STABLE"

                logger.info(
                    f"   {status:8s} {metrics['name']:15s}: "
                    f"{metrics['prev_AP_0.25']:.4f} -> {metrics['curr_AP_0.25']:.4f} ({forgetting_val:+.4f})"
                )

        if worst_forgetting_25['forgetting_0.25'] < -0.01:
            logger.info(
                f"Worst Forgetting: {worst_forgetting_25['name']} "
                f"({worst_forgetting_25['forgetting_0.25']:+.4f})"
            )
        if best_retention_25['forgetting_0.25'] > 0.01:
            logger.info(
                f"Best Retention: {best_retention_25['name']} "
                f"({best_retention_25['forgetting_0.25']:+.4f})"
            )

        logger.info("=" * 80)

    return forgetting_metrics


def _strip_metric_prefix(metrics: dict, prefix: str) -> dict:
    if not isinstance(metrics, dict) or not prefix:
        return metrics
    out = {}
    for k, v in metrics.items():
        ks = str(k)
        if ks.startswith(prefix):
            out[ks[len(prefix):]] = v
        else:
            out[ks] = v
    return out


def _strip_any_stage_prefix(metrics: dict) -> Tuple[dict, str]:
    """Strip a leading `stage_{k}_` prefix from eval metric dicts (if present).

    Motivation:
      - Some evaluation backends log metrics as `stage_{stage_idx}_<metric>`.
      - Downstream metric consumers (forgetting computation, JSON artifacts) want
        stable, unprefixed keys: e.g. `mAP_0.25`, `chair_AP_0.25`,
        `cohort_stage_1_mAP_0.25`, `class_list`, etc.

    Returns:
      (stripped_metrics, detected_prefix)
    """
    if not isinstance(metrics, dict):
        return metrics, ''
    import re

    prefix = ''
    for k in metrics.keys():
        m = re.match(r'^(stage_\d+_)', str(k))
        if m:
            prefix = m.group(1)
            break
    if not prefix:
        return metrics, ''
    return _strip_metric_prefix(metrics, prefix), prefix


def _ld_score_records_to_map(records: Any, *, field_name: str) -> Dict[str, Dict[int, float]]:
    """Convert LD seat-score records to {scene_id: {save_stage: score}}."""
    if not isinstance(records, list):
        raise RuntimeError(
            f"learning_dynamics_scores.json has invalid '{field_name}' payload; expected list."
        )
    out: Dict[str, Dict[int, float]] = {}
    for rec in records:
        if not isinstance(rec, dict):
            raise RuntimeError(
                f"learning_dynamics_scores.json has non-dict record in '{field_name}'."
            )
        scene_id = rec.get('scene_id', None)
        save_stage = rec.get('save_stage', None)
        if scene_id is None or save_stage is None:
            raise RuntimeError(
                f"learning_dynamics_scores.json record in '{field_name}' is missing scene_id/save_stage."
            )
        try:
            sid = str(scene_id)
            st = int(save_stage)
            score = float(rec.get('score', 0.0))
        except Exception as e:
            raise RuntimeError(
                f"learning_dynamics_scores.json record in '{field_name}' has invalid types "
                "(scene_id/save_stage/score)."
            ) from e
        if not np.isfinite(score):
            raise RuntimeError(
                f"learning_dynamics_scores.json record in '{field_name}' has non-finite score."
            )
        out.setdefault(sid, {})[st] = float(score)
    return out


def _ld_score_map_payload_to_map(payload: Any, *,
                                 field_name: str) -> Dict[str, Dict[int, float]]:
    """Validate {scene_id: {save_stage: score}} LD score payload."""
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"learning_dynamics_scores.json has invalid '{field_name}' payload; expected dict."
        )
    out: Dict[str, Dict[int, float]] = {}
    for scene_id, by_stage in payload.items():
        if not isinstance(by_stage, dict):
            raise RuntimeError(
                f"learning_dynamics_scores.json field '{field_name}' has non-dict stage mapping."
            )
        sid = str(scene_id)
        for save_stage, score in by_stage.items():
            try:
                st = int(save_stage)
                sc = float(score)
            except Exception as e:
                raise RuntimeError(
                    f"learning_dynamics_scores.json field '{field_name}' has invalid "
                    "save_stage/score types."
                ) from e
            if not np.isfinite(sc):
                raise RuntimeError(
                    f"learning_dynamics_scores.json field '{field_name}' has non-finite score."
                )
            out.setdefault(sid, {})[st] = float(sc)
    return out


def _load_learning_dynamics_scores_for_memory_update(scores_path: Path,
                                                     *,
                                                     require_stage_id: Optional[int] = None
                                                     ) -> Dict[str, Any]:
    """Load and validate LD scores for memory-bank updates."""
    path = Path(scores_path)
    if not path.exists():
        raise RuntimeError(f"Learning-dynamics scores file does not exist: {path}")

    with open(path, 'r') as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid learning_dynamics_scores.json payload type at {path}")

    required_keys = [
        'stage_id',
        'iou_thr',
        'iou_mode',
        'eps',
        'object_count_cap',
        'slope_k_start',
        'slope_k_end',
        'old_classes',
        'new_classes',
    ]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise RuntimeError(
            "learning_dynamics_scores.json is missing required keys: "
            f"{missing}. file={path}"
        )

    try:
        stage_id = int(payload['stage_id'])
    except Exception as e:
        raise RuntimeError(
            f"learning_dynamics_scores.json has invalid stage_id at {path}: {payload.get('stage_id')!r}"
        ) from e

    if require_stage_id is not None and int(stage_id) != int(require_stage_id):
        raise RuntimeError(
            "learning_dynamics_scores.json stage mismatch: "
            f"expected stage_id={int(require_stage_id)}, got stage_id={int(stage_id)}. file={path}"
        )

    if 'learning_dynamics_forgetness_by_seat' in payload:
        forgetness_by_seat = _ld_score_map_payload_to_map(
            payload.get('learning_dynamics_forgetness_by_seat', {}),
            field_name='learning_dynamics_forgetness_by_seat',
        )
    elif 'forgetness_seat_scores' in payload:
        forgetness_by_seat = _ld_score_records_to_map(
            payload.get('forgetness_seat_scores', []),
            field_name='forgetness_seat_scores',
        )
    else:
        raise RuntimeError(
            "learning_dynamics_scores.json must include either "
            "'learning_dynamics_forgetness_by_seat' or 'forgetness_seat_scores'. "
            f"file={path}"
        )

    if 'learning_dynamics_replay_priority_by_seat' in payload:
        replay_priority_by_seat = _ld_score_map_payload_to_map(
            payload.get('learning_dynamics_replay_priority_by_seat', {}),
            field_name='learning_dynamics_replay_priority_by_seat',
        )
    elif 'replay_priority_seat_scores' in payload:
        replay_priority_by_seat = _ld_score_records_to_map(
            payload.get('replay_priority_seat_scores', []),
            field_name='replay_priority_seat_scores',
        )
    else:
        raise RuntimeError(
            "learning_dynamics_scores.json must include either "
            "'learning_dynamics_replay_priority_by_seat' or "
            "'replay_priority_seat_scores'. "
            f"file={path}"
        )

    replay_priority_policy = payload.get('replay_priority_policy', None)
    if replay_priority_policy is not None and not isinstance(replay_priority_policy, dict):
        raise RuntimeError(
            "learning_dynamics_scores.json has invalid replay_priority_policy; "
            f"expected dict, got {type(replay_priority_policy)}."
        )

    return dict(
        stage_id=int(stage_id),
        iou_thr=float(payload['iou_thr']),
        iou_mode=str(payload['iou_mode']),
        eps=float(payload['eps']),
        object_count_cap=int(payload['object_count_cap']),
        slope_k_start=int(payload['slope_k_start']),
        slope_k_end=int(payload['slope_k_end']),
        replay_priority_policy=dict(replay_priority_policy or {}),
        old_classes=[int(x) for x in (payload.get('old_classes') or [])],
        new_classes=[int(x) for x in (payload.get('new_classes') or [])],
        forgetness_by_seat=forgetness_by_seat,
        replay_priority_by_seat=replay_priority_by_seat,
        source_file=str(path),
    )


def _load_learning_dynamics_design1_scores_for_memory_update(
        scores_path: Path,
        *,
        require_stage_id: Optional[int] = None,
        strategy_name: str = LD_DESIGN1_STRATEGY,
        score_file_label: str = 'learning_dynamics_design1_scores.json',
) -> Dict[str, Any]:
    """Load and validate LD design scores for memory-bank updates."""
    path = Path(scores_path)
    if not path.exists():
        raise RuntimeError(
            f"Learning-dynamics {strategy_name} scores file does not exist: {path}"
        )

    with open(path, 'r') as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Invalid {score_file_label} payload type at {path}"
        )

    required_keys = [
        'stage_id',
        'iou_thr',
        'iou_mode',
        'eps',
        'q_metric',
        'class_need',
        'seat_class_terms',
    ]
    missing = [k for k in required_keys if k not in payload]
    if missing:
        raise RuntimeError(
            f"{score_file_label} is missing required keys: "
            f"{missing}. file={path}"
        )

    try:
        stage_id = int(payload['stage_id'])
    except Exception as e:
        raise RuntimeError(
            f"{score_file_label} has invalid stage_id at "
            f"{path}: {payload.get('stage_id')!r}"
        ) from e
    if require_stage_id is not None and int(stage_id) != int(require_stage_id):
        raise RuntimeError(
            f"{score_file_label} stage mismatch: "
            f"expected stage_id={int(require_stage_id)}, got stage_id={int(stage_id)}. "
            f"file={path}"
        )

    q_metric = str(payload.get('q_metric', '')).strip().lower()
    if q_metric not in ('f1', 'recall'):
        raise RuntimeError(
            f"{score_file_label} has invalid q_metric; "
            f"expected one of ['f1', 'recall'], got {payload.get('q_metric')!r}."
        )

    class_need_payload = payload.get('class_need', None)
    if not isinstance(class_need_payload, dict) or not class_need_payload:
        raise RuntimeError(
            f"{score_file_label} must include non-empty "
            "'class_need' dict."
        )
    class_need: Dict[int, float] = {}
    for cid, value in class_need_payload.items():
        try:
            cid_i = int(cid)
            v = float(value)
        except Exception as e:
            raise RuntimeError(
                f"{score_file_label} class_need has invalid "
                f"class_id/value: ({cid!r}, {value!r})."
            ) from e
        if not np.isfinite(v):
            raise RuntimeError(
                f"{score_file_label} class_need has non-finite "
                f"value for class {cid_i}."
            )
        class_need[cid_i] = float(v)
    if float(sum(class_need.values())) <= 0.0:
        raise RuntimeError(
            f"{score_file_label} class_need sum is non-positive."
        )

    seat_terms_payload = payload.get('seat_class_terms', None)
    if not isinstance(seat_terms_payload, dict) or not seat_terms_payload:
        raise RuntimeError(
            f"{score_file_label} must include non-empty "
            "'seat_class_terms' dict."
        )
    seat_class_terms: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {}
    numeric_term_keys = ('g', 'r_best', 'd', 'u', 'r_start', 'r_end')
    for scene_id, by_stage in seat_terms_payload.items():
        if not isinstance(by_stage, dict):
            raise RuntimeError(
                f"{score_file_label} seat_class_terms must map "
                "scene_id -> stage_map."
            )
        sid = str(scene_id)
        for save_stage, by_cls in by_stage.items():
            try:
                st = int(save_stage)
            except Exception as e:
                raise RuntimeError(
                    f"{score_file_label} has invalid save_stage "
                    f"for scene {sid}: {save_stage!r}"
                ) from e
            if not isinstance(by_cls, dict):
                raise RuntimeError(
                    f"{score_file_label} seat_class_terms must map "
                    "save_stage -> class_map."
                )
            for cid, term in by_cls.items():
                try:
                    cid_i = int(cid)
                except Exception as e:
                    raise RuntimeError(
                        f"{score_file_label} has invalid class_id "
                        f"in seat_class_terms: {cid!r}"
                    ) from e
                if not isinstance(term, dict):
                    raise RuntimeError(
                        f"{score_file_label} class term must be dict, "
                        f"got {type(term)} for scene={sid}, stage={st}, class={cid_i}."
                    )
                out_term: Dict[str, float] = {}
                for key in numeric_term_keys:
                    try:
                        val = float(term.get(key, 0.0))
                    except Exception as e:
                        raise RuntimeError(
                            f"{score_file_label} class term has invalid "
                            f"value for key='{key}' at scene={sid}, stage={st}, class={cid_i}."
                        ) from e
                    if not np.isfinite(val):
                        raise RuntimeError(
                            f"{score_file_label} class term has non-finite "
                            f"value for key='{key}' at scene={sid}, stage={st}, class={cid_i}."
                        )
                    out_term[str(key)] = float(val)
                seat_class_terms.setdefault(sid, {}).setdefault(int(st), {})[cid_i] = out_term

    if not seat_class_terms:
        raise RuntimeError(
            f"{score_file_label} has empty seat_class_terms after parsing."
        )

    def _parse_optional_class_score_map(field_name: str) -> Dict[int, float]:
        raw = payload.get(field_name, {}) or {}
        if not isinstance(raw, dict):
            raise RuntimeError(
                f"{score_file_label} has invalid '{field_name}'; expected dict."
            )
        out: Dict[int, float] = {}
        for k, v in raw.items():
            try:
                cid = int(k)
                fv = float(v)
            except Exception as e:
                raise RuntimeError(
                    f"{score_file_label} has invalid class score in "
                    f"'{field_name}': ({k!r}, {v!r})."
                ) from e
            if not np.isfinite(fv):
                raise RuntimeError(
                    f"{score_file_label} field '{field_name}' has non-finite value."
                )
            out[int(cid)] = float(fv)
        return out

    return dict(
        stage_id=int(stage_id),
        iou_thr=float(payload['iou_thr']),
        iou_mode=str(payload['iou_mode']),
        eps=float(payload['eps']),
        q_metric=str(q_metric),
        class_ids=[int(x) for x in (payload.get('class_ids') or [])],
        new_classes=[int(x) for x in (payload.get('new_classes') or [])],
        class_need=class_need,
        class_q_current=_parse_optional_class_score_map('class_q_current'),
        class_q_best=_parse_optional_class_score_map('class_q_best'),
        seat_class_terms=seat_class_terms,
        source_file=str(path),
    )


def _resolve_stage1_ld_scores_path_for_checkpoint(*,
                                                  checkpoint_path: str,
                                                  scene_memory_config: Dict[str, Any],
                                                  config_block_key: str = 'learning_dynamics_update',
                                                  score_filename: str = 'learning_dynamics_scores.json',
                                                  strategy_name: str = 'learning_dynamics') -> Path:
    """Resolve Stage-1 score file path for --start-stage 2 strict workflows."""
    assert checkpoint_path, (
        f"selection_strategy='{strategy_name}' with --start-stage 2 requires "
        "--checkpoint-path <stage1_checkpoint>."
    )

    ld_cfg = scene_memory_config.get(str(config_block_key), {}) or {}
    explicit_path = ld_cfg.get('stage1_scores_file', None)
    candidates: List[Path] = []
    if explicit_path:
        p = Path(str(explicit_path)).expanduser()
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append((Path(os.getcwd()) / p).resolve())
            candidates.append((Path(REPO_ROOT) / p).resolve())
    else:
        ckpt = Path(str(checkpoint_path)).expanduser()
        if not ckpt.is_absolute():
            ckpt = (Path(os.getcwd()) / ckpt).resolve()

        # Canonical checkpoint layout:
        #   <run_dir>/checkpoints/stage_1/epoch_*.pth
        run_dir = None
        try:
            if ckpt.parent.name.startswith('stage_') and ckpt.parent.parent.name == 'checkpoints':
                run_dir = ckpt.parent.parent.parent
        except Exception:
            run_dir = None
        if run_dir is None:
            run_dir = ckpt.parent.parent.parent

        candidates.append(
            (Path(run_dir) / 'learning_dynamics' / 'stage_1' / str(score_filename)).resolve()
        )

    # Deduplicate while preserving order.
    seen = set()
    unique_candidates = []
    for p in candidates:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        unique_candidates.append(p)

    for p in unique_candidates:
        if p.exists():
            return p

    candidate_str = "\n".join(f"  - {p}" for p in unique_candidates)
    raise RuntimeError(
        f"Could not find Stage-1 {strategy_name} scores for memory update. "
        "Checked paths:\n"
        f"{candidate_str}"
    )


def _resolve_stage1_ld_dir_for_checkpoint(*,
                                          checkpoint_path: str,
                                          scene_memory_config: Dict[str, Any],
                                          config_block_key: str = 'learning_dynamics_design1',
                                          strategy_name: str = 'learning_dynamics_design1') -> Path:
    """Resolve Stage-1 learning_dynamics/stage_1 directory for --start-stage 2."""
    assert checkpoint_path, (
        f"selection_strategy='{strategy_name}' with --start-stage 2 requires "
        "--checkpoint-path <stage1_checkpoint>."
    )

    candidates: List[Path] = []
    ld_cfg = scene_memory_config.get(str(config_block_key), {}) or {}

    def _append_path(raw_path: Any) -> None:
        if not raw_path:
            return
        p0 = Path(str(raw_path)).expanduser()
        path_candidates = [p0] if p0.is_absolute() else [
            (Path(os.getcwd()) / p0).resolve(),
            (Path(REPO_ROOT) / p0).resolve(),
        ]
        for p in path_candidates:
            if p.is_file():
                candidates.append(p.parent.resolve())
            else:
                candidates.append(p.resolve())

    explicit_stats_dir = ld_cfg.get('stage1_stats_dir', None)
    explicit_scores_file = ld_cfg.get('stage1_scores_file', None)
    _append_path(explicit_stats_dir)
    _append_path(explicit_scores_file)

    if not candidates:
        ckpt = Path(str(checkpoint_path)).expanduser()
        if not ckpt.is_absolute():
            ckpt = (Path(os.getcwd()) / ckpt).resolve()

        run_dir = None
        try:
            if ckpt.parent.name.startswith('stage_') and ckpt.parent.parent.name == 'checkpoints':
                run_dir = ckpt.parent.parent.parent
        except Exception:
            run_dir = None
        if run_dir is None:
            run_dir = ckpt.parent.parent.parent

        candidates.append((Path(run_dir) / 'learning_dynamics' / 'stage_1').resolve())

    seen = set()
    unique_candidates = []
    for p in candidates:
        s = str(p)
        if s in seen:
            continue
        seen.add(s)
        unique_candidates.append(p)

    for p in unique_candidates:
        if p.exists() and p.is_dir():
            return p

    candidate_str = "\n".join(f"  - {p}" for p in unique_candidates)
    raise RuntimeError(
        f"Could not find Stage-1 learning_dynamics directory for {strategy_name}. "
        "Checked paths:\n"
        f"{candidate_str}"
    )


def _load_ld_stage_stats_by_k(*,
                              stage_ld_dir: Path,
                              filename_prefix: str,
                              require_stage_id: Optional[int] = None
                              ) -> Dict[str, Any]:
    """Load Stage-1 LD per-k seat stats like memory_stats_k*.json."""
    base_dir = Path(stage_ld_dir).resolve()
    if not base_dir.exists() or not base_dir.is_dir():
        raise RuntimeError(f"Invalid learning_dynamics stage directory: {base_dir}")

    name_re = re.compile(rf"^{re.escape(str(filename_prefix))}_k(\d+)\.json$")
    files_by_k: Dict[int, Path] = {}
    for p in sorted(base_dir.glob(f"{filename_prefix}_k*.json")):
        m = name_re.match(p.name)
        if not m:
            continue
        files_by_k[int(m.group(1))] = p

    if not files_by_k:
        raise RuntimeError(
            f"Missing {filename_prefix}_k*.json files under {base_dir}"
        )

    ks = sorted(files_by_k.keys())
    expected = list(range(ks[0], ks[-1] + 1))
    if ks != expected or int(ks[0]) != 0:
        raise RuntimeError(
            f"Non-contiguous k indices for {filename_prefix} under {base_dir}: "
            f"found={ks}, expected={expected} with k0 required."
        )

    metrics_by_k: Dict[int, List[Dict[str, Any]]] = {}
    iou_mode_ref = None
    iou_thr_ref = None
    eps_ref = None
    stage_id_ref = None

    for k in ks:
        p = files_by_k[int(k)]
        with open(p, 'r') as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Invalid payload type in {p}; expected dict."
            )
        seats = payload.get('seats', None)
        if not isinstance(seats, list):
            raise RuntimeError(
                f"Invalid seats payload in {p}; expected list."
            )

        try:
            stage_id_i = int(payload.get('stage_id'))
            iou_mode_i = str(payload.get('iou_mode'))
            iou_thr_i = float(payload.get('iou_thr'))
            eps_i = float(payload.get('eps'))
        except Exception as e:
            raise RuntimeError(
                f"Invalid metadata in {p}; expected stage_id/iou_mode/iou_thr/eps."
            ) from e

        if require_stage_id is not None and int(stage_id_i) != int(require_stage_id):
            raise RuntimeError(
                f"{p} stage_id mismatch: expected {int(require_stage_id)}, "
                f"got {int(stage_id_i)}"
            )

        if stage_id_ref is None:
            stage_id_ref = int(stage_id_i)
            iou_mode_ref = str(iou_mode_i)
            iou_thr_ref = float(iou_thr_i)
            eps_ref = float(eps_i)
        else:
            if int(stage_id_i) != int(stage_id_ref):
                raise RuntimeError(
                    f"{filename_prefix} stage_id mismatch across k files at {p}: "
                    f"{stage_id_i} vs reference {stage_id_ref}"
                )
            if str(iou_mode_i) != str(iou_mode_ref):
                raise RuntimeError(
                    f"{filename_prefix} iou_mode mismatch across k files at {p}: "
                    f"{iou_mode_i!r} vs reference {iou_mode_ref!r}"
                )
            if abs(float(iou_thr_i) - float(iou_thr_ref)) > 1e-12:
                raise RuntimeError(
                    f"{filename_prefix} iou_thr mismatch across k files at {p}: "
                    f"{iou_thr_i} vs reference {iou_thr_ref}"
                )
            if abs(float(eps_i) - float(eps_ref)) > 1e-20:
                raise RuntimeError(
                    f"{filename_prefix} eps mismatch across k files at {p}: "
                    f"{eps_i} vs reference {eps_ref}"
                )

        metrics_by_k[int(k)] = list(seats)

    return dict(
        metrics_by_k=metrics_by_k,
        ks=[int(x) for x in ks],
        K=int(max(ks)),
        stage_id=int(stage_id_ref),
        iou_mode=str(iou_mode_ref),
        iou_thr=float(iou_thr_ref),
        eps=float(eps_ref),
        source_dir=str(base_dir),
    )


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON to avoid readers seeing partially-written files."""
    out_path = Path(path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(
        f".{out_path.name}.tmp.{os.getpid()}.{int(time.time() * 1e6)}"
    )
    try:
        with open(tmp_path, 'w') as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


def _recompute_stage1_ld_design1_scores_from_raw_stats(*,
                                                       checkpoint_path: str,
                                                       scene_memory_config: Dict[str, Any],
                                                       stage_definition: Dict[str, Any],
                                                       require_stage_id: int = 1,
                                                       config_block_key: str = 'learning_dynamics_design1',
                                                       strategy_name: str = LD_DESIGN1_STRATEGY,
                                                       design_version: int = 1,
                                                       output_dir: Optional[str] = None,
                                                       output_filename: str = (
                                                           'learning_dynamics_design1_scores_recomputed.json'
                                                       )) -> Dict[str, Any]:
    """Recompute Stage-1 LD design score payload from saved TP/FP/FN k-stats."""
    stage1_ld_dir = _resolve_stage1_ld_dir_for_checkpoint(
        checkpoint_path=str(checkpoint_path),
        scene_memory_config=scene_memory_config,
        config_block_key=str(config_block_key),
        strategy_name=str(strategy_name),
    )

    mem_pack = _load_ld_stage_stats_by_k(
        stage_ld_dir=stage1_ld_dir,
        filename_prefix='memory_stats',
        require_stage_id=int(require_stage_id),
    )
    nat_pack = _load_ld_stage_stats_by_k(
        stage_ld_dir=stage1_ld_dir,
        filename_prefix='natural_stats',
        require_stage_id=int(require_stage_id),
    )

    ks_mem = list(mem_pack.get('ks', []) or [])
    ks_nat = list(nat_pack.get('ks', []) or [])
    if ks_mem != ks_nat:
        raise RuntimeError(
            "Stage-1 raw stats k mismatch between memory and natural sets: "
            f"memory={ks_mem}, natural={ks_nat}, dir={stage1_ld_dir}"
        )
    if len(ks_mem) < 2:
        raise RuntimeError(
            "Stage-1 Design-1 recompute requires at least k0 and k1 stats files. "
            f"Found ks={ks_mem} in {stage1_ld_dir}"
        )

    if str(mem_pack.get('iou_mode')) != str(nat_pack.get('iou_mode')):
        raise RuntimeError(
            "Stage-1 raw stats iou_mode mismatch between memory and natural sets: "
            f"{mem_pack.get('iou_mode')} vs {nat_pack.get('iou_mode')}"
        )
    if abs(float(mem_pack.get('iou_thr', 0.0)) - float(nat_pack.get('iou_thr', 0.0))) > 1e-12:
        raise RuntimeError(
            "Stage-1 raw stats iou_thr mismatch between memory and natural sets: "
            f"{mem_pack.get('iou_thr')} vs {nat_pack.get('iou_thr')}"
        )

    d1_cfg = scene_memory_config.get(str(config_block_key), {}) or {}
    if 'q_metric' not in d1_cfg:
        raise RuntimeError(
            f"{config_block_key}.q_metric must be explicitly set "
            "(no implicit fallback)."
        )
    q_metric = str(d1_cfg.get('q_metric')).strip().lower()
    if q_metric not in ('f1', 'recall'):
        raise RuntimeError(
            f"{config_block_key}.q_metric must be one of ['f1', 'recall'], "
            f"got {d1_cfg.get('q_metric')!r}."
        )

    ld_update_cfg = scene_memory_config.get('learning_dynamics_update', {}) or {}
    eps = float(ld_update_cfg.get('eps', mem_pack.get('eps', 1e-9)))
    if eps <= 0.0:
        eps = 1e-9
    object_count_cap = int(ld_update_cfg.get('object_count_cap', 20))
    if object_count_cap <= 0:
        object_count_cap = 20

    class_ids = sorted({int(x) for x in (stage_definition.get('class_indices', []) or [])})
    if not class_ids:
        raise RuntimeError(
            "Stage-1 Design-1 recompute requires non-empty stage_definition.class_indices."
        )

    merged_metrics_by_k: Dict[int, List[Dict[str, Any]]] = {}
    for k in ks_mem:
        merged_metrics_by_k[int(k)] = (
            list((mem_pack.get('metrics_by_k', {}) or {}).get(int(k), []) or [])
            + list((nat_pack.get('metrics_by_k', {}) or {}).get(int(k), []) or [])
        )

    from mmdet3d.utils.learning_dynamics_scoring import (
        compute_learning_dynamics_design1_scores,
    )

    d1_scores = compute_learning_dynamics_design1_scores(
        merged_metrics_by_k,
        class_ids=list(class_ids),
        new_classes=list(class_ids),
        q_metric=str(q_metric),
        eps=float(eps),
        design_version=int(design_version),
    )
    seat_terms = d1_scores.get('seat_class_terms', {}) or {}
    if not isinstance(seat_terms, dict) or not seat_terms:
        raise RuntimeError(
            f"Stage-1 {strategy_name} recompute produced empty seat_class_terms. "
            f"stage_ld_dir={stage1_ld_dir}"
        )

    output_base_dir = Path(output_dir).resolve() if output_dir else Path(stage1_ld_dir)
    out_path = (output_base_dir / str(output_filename)).resolve()
    payload = dict(
        stage_id=int(require_stage_id),
        K=int(mem_pack.get('K')),
        K_review=max(0, int(mem_pack.get('K')) - 1),
        iou_thr=float(mem_pack.get('iou_thr')),
        iou_mode=str(mem_pack.get('iou_mode')),
        iou_thrs=[float(mem_pack.get('iou_thr'))],
        eps=float(eps),
        q_metric=str(q_metric),
        object_count_cap=int(object_count_cap),
        class_ids=[int(x) for x in class_ids],
        new_classes=[int(x) for x in class_ids],
        class_need={
            str(int(k)): float(v)
            for k, v in (d1_scores.get('class_need', {}) or {}).items()
        },
        class_q_current={
            str(int(k)): float(v)
            for k, v in (d1_scores.get('class_q_current', {}) or {}).items()
        },
        class_q_best={
            str(int(k)): float(v)
            for k, v in (d1_scores.get('class_q_best', {}) or {}).items()
        },
        seat_class_terms=seat_terms,
        source=dict(
            stage_ld_dir=str(stage1_ld_dir),
            memory_stats_dir=str(mem_pack.get('source_dir')),
            natural_stats_dir=str(nat_pack.get('source_dir')),
            ks=[int(x) for x in ks_mem],
            recomputed_at_start_stage_2=True,
        ),
    )
    _atomic_write_json(out_path, payload)

    loaded = _load_learning_dynamics_design1_scores_for_memory_update(
        out_path,
        require_stage_id=int(require_stage_id),
        strategy_name=str(strategy_name),
        score_file_label=str(output_filename),
    )
    return dict(
        scores_path=str(out_path),
        scores_payload=dict(loaded),
        k_indices=[int(x) for x in ks_mem],
    )


def _assert_stage1_design1_q_metric_matches_config(*,
                                                   payload: Dict[str, Any],
                                                   scene_memory_config: Dict[str, Any],
                                                   source_file: str,
                                                   config_block_key: str = 'learning_dynamics_design1',
                                                   strategy_name: str = LD_DESIGN1_STRATEGY) -> None:
    """Fail fast when Stage-1 design score q_metric mismatches current config."""
    d1_cfg = scene_memory_config.get(str(config_block_key), {}) or {}
    if 'q_metric' not in d1_cfg:
        raise RuntimeError(
            f"{config_block_key}.q_metric must be explicitly set "
            "(no implicit fallback)."
        )
    expected_q_metric = str(d1_cfg.get('q_metric')).strip().lower()
    if expected_q_metric not in ('f1', 'recall'):
        raise RuntimeError(
            f"{config_block_key}.q_metric must be one of ['f1', 'recall'], "
            f"got {d1_cfg.get('q_metric')!r}."
        )
    loaded_q_metric = str((payload or {}).get('q_metric', '')).strip().lower()
    if loaded_q_metric != expected_q_metric:
        raise RuntimeError(
            f"Stage-1 {strategy_name} score q_metric mismatch: "
            f"config expects '{expected_q_metric}', loaded payload has '{loaded_q_metric}'. "
            f"source={source_file}. "
            "If this is intentional after a design change, set "
            f"scene_memory_config.{config_block_key}.stage1_scores_mode='recompute_from_stats'."
        )


def _sanitize_for_cfg_snapshot(value: Any) -> Any:
    return _sanitize_for_cfg_snapshot_impl(value)


def _try_get_git_info(repo_root: str) -> Dict[str, Any]:
    return _try_get_git_info_impl(repo_root)


def _write_resolved_config_snapshot(*,
                                   cfg: Config,
                                   dest_path: str,
                                   run_meta: Dict[str, Any]) -> None:
    _write_resolved_config_snapshot_impl(cfg=cfg, dest_path=dest_path, run_meta=run_meta)


def _compute_review_segment_times(*,
                                 stage_epochs: int,
                                 repeat_times: int,
                                 review_fractions: List[float]) -> List[int]:
    return _resolve_segment_times_impl(
        stage_epochs=stage_epochs,
        repeat_times=repeat_times,
        review_fractions=review_fractions,
    )


def _filter_sunrgbd_infos_to_model_label_space(
        data_infos: List[Dict[str, Any]], num_classes: int
) -> List[Dict[str, Any]]:
    """Copy SUN RGB-D infos and remove labels the current head cannot emit.

    Object-memory reviewing evaluates the original carrier scenes.  Those
    metadata records intentionally retain all 40 annotations, including
    classes introduced in future stages, whereas an incremental head only has
    outputs for classes seen so far.  Keep all currently representable labels
    here; the downstream evaluator applies the narrower cohort filter.
    """
    num_classes = int(num_classes)
    if num_classes <= 0:
        raise ValueError(f'num_classes must be positive, got {num_classes}.')

    filtered_infos = copy.deepcopy(list(data_infos))
    for info in filtered_infos:
        if not isinstance(info, dict):
            continue
        annos = info.get('annos')
        if not isinstance(annos, dict) or 'class' not in annos:
            continue
        labels = np.asarray(annos.get('class', []), dtype=np.int64).reshape(-1)
        keep = (labels >= 0) & (labels < num_classes)
        object_count = int(labels.shape[0])
        new_annos = {}
        for key, value in annos.items():
            if key in {'gt_num', 'index'}:
                continue
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == object_count:
                new_annos[key] = value[keep]
            elif isinstance(value, list) and len(value) == object_count:
                kept_indices = np.flatnonzero(keep).tolist()
                new_annos[key] = [value[idx] for idx in kept_indices]
            else:
                new_annos[key] = value
        new_annos['gt_num'] = int(keep.sum())
        new_annos['index'] = np.arange(int(keep.sum()), dtype=np.int32)
        info['annos'] = new_annos
    return filtered_infos


def _sunrgbd_eval_memory_subset(*,
                                model,
                                stage_cfg,
                                data_infos: List[Dict[str, Any]],
                                eval_class_indices: List[int],
                                class_names: List[str],
                                iou_thrs=(0.25, 0.5),
                                stage_idx: int = 0,
                                split_name: str = 'train(memory_bank_subset)',
                                eval_purpose: Optional[str] = None,
                                review_k: Optional[int] = None,
                                logger=None,
                                # Optional raw outputs for downstream scoring.
                                raw_results_out: Optional[List[Dict[str, Any]]] = None,
                                raw_gt_annos_out: Optional[List[Dict[str, Any]]] = None,
                                raw_data_infos_out: Optional[List[Dict[str, Any]]] = None,
                                raw_box_type_3d_out: Optional[List[Any]] = None,
                                raw_box_mode_3d_out: Optional[List[Any]] = None):
    """Evaluate AP/AR at specified IoU thresholds on memory-seat subsets.

    Historical name retained for compatibility. Supports both SUNRGBD and
    ScanNet incremental datasets by selecting the corresponding in-memory
    dataset wrapper.
    """
    if not data_infos:
        return {}
    eval_class_indices = [int(x) for x in eval_class_indices]
    eval_class_indices = sorted(set(eval_class_indices))
    if not eval_class_indices:
        return {}

    # Normalize IoU threshold(s).
    if iou_thrs is None:
        iou_thrs = (0.25, 0.5)
    if isinstance(iou_thrs, (int, float)):
        iou_list = [float(iou_thrs)]
    else:
        iou_list = [float(x) for x in (list(iou_thrs) if isinstance(iou_thrs, (list, tuple)) else [iou_thrs])]
    iou_list = sorted({float(x) for x in iou_list})
    assert iou_list, 'Empty iou_thrs for reviewing evaluation.'
    for thr in iou_list:
        assert 0.0 < float(thr) < 1.0, thr

    from mmcv.parallel import MMDataParallel
    from mmdet.datasets import build_dataloader
    from mmdet3d.apis import single_gpu_test
    from mmdet3d.core.evaluation.incremental_indoor_eval import incremental_indoor_eval

    # Avoid evaluating through a DDP wrapper on rank0-only paths (can hang due
    # to buffer broadcasts). This also works for MMDataParallel wrappers.
    base_model = model.module if hasattr(model, 'module') else model

    # Determine incremental dataset family from config.
    train_dataset_type = None
    try:
        train_dataset_type = str(getattr(stage_cfg.data.train.dataset, 'type', ''))
    except Exception:
        train_dataset_type = None
    if not train_dataset_type:
        try:
            train_dataset_type = str(
                getattr(stage_cfg.data.train.dataset.dataset, 'type', '')
            )
        except Exception:
            train_dataset_type = ''

    is_scannet = (str(train_dataset_type) == 'IncrementalScanNetDataset')
    dataset_label = 'ScanNet' if is_scannet else 'SUNRGBD'

    # Derive dataset settings from stage_cfg.
    try:
        data_root = str(getattr(stage_cfg.data.train.dataset, 'data_root',
                                getattr(stage_cfg.data.val, 'data_root', '')))
    except Exception:
        data_root = ''
    if not data_root:
        try:
            data_root = str(getattr(stage_cfg.data.train.dataset.dataset, 'data_root',
                                    getattr(stage_cfg.data.val, 'data_root', '')))
        except Exception:
            data_root = ''
    try:
        box_type_3d = getattr(
            stage_cfg.data.val, 'box_type_3d',
            getattr(stage_cfg.data.train.dataset, 'box_type_3d', 'Depth'))
    except Exception:
        box_type_3d = 'Depth'
    try:
        pipeline = getattr(stage_cfg.data.val, 'pipeline', None)
    except Exception:
        pipeline = None

    if is_scannet:
        mem_dataset_cfg = dict(
            type='ScanNetMemoryDataset',
            data_infos=data_infos,
            data_root=data_root,
            pipeline=pipeline,
            classes=tuple(class_names),
            box_type_3d=box_type_3d,
            variant='dynamic_head',
            filter_empty_gt=False,
            test_mode=True,
        )
    else:
        data_infos = _filter_sunrgbd_infos_to_model_label_space(
            data_infos, len(class_names))
        try:
            modality = getattr(
                stage_cfg.data.train.dataset,
                'modality',
                dict(use_camera=False, use_lidar=True),
            )
        except Exception:
            modality = dict(use_camera=False, use_lidar=True)
        mem_dataset_cfg = dict(
            type='SUNRGBDMemoryDataset',
            data_infos=data_infos,
            data_root=data_root,
            pipeline=pipeline,
            classes=tuple(class_names),
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=False,
            test_mode=True,
        )
    mem_dataset = build_dataset(mem_dataset_cfg)
    mem_loader = build_dataloader(
        mem_dataset,
        samples_per_gpu=1,
        workers_per_gpu=stage_cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    device_id = int(torch.cuda.current_device()) if torch.cuda.is_available() else 0
    eval_model = MMDataParallel(base_model.cuda(), device_ids=[device_id])
    eval_model.eval()
    results = single_gpu_test(eval_model, mem_loader, show=False)

    if is_scannet:
        stage_mappings = None
        try:
            stage_mappings = getattr(stage_cfg, 'mappings', None)
        except Exception:
            stage_mappings = None
        if stage_mappings is None:
            try:
                stage_mappings = stage_cfg.get('mappings', None)
            except Exception:
                stage_mappings = None
        if not isinstance(stage_mappings, dict):
            raise RuntimeError(
                "ScanNet LD/eval requires stage_cfg.mappings dict for NYU40->GCI conversion."
            )
        nyu40_to_model = stage_mappings.get('nyu40_to_model_idx', {}) or {}
        if not isinstance(nyu40_to_model, dict) or not nyu40_to_model:
            raise RuntimeError(
                "ScanNet LD/eval requires non-empty mappings['nyu40_to_model_idx']."
            )
        gt_annos = []
        for info in mem_dataset.data_infos:
            ann = info.get('annos', {}) or {}
            gt_num = int(ann.get('gt_num', 0) or 0)
            if gt_num <= 0:
                gt_annos.append({'gt_num': 0})
                continue

            gt_boxes = np.asarray(
                ann.get('gt_boxes_upright_depth', np.zeros((0, 7), dtype=np.float32)),
                dtype=np.float32,
            )
            gt_labels_nyu = np.asarray(
                ann.get('class', np.zeros((0,), dtype=np.int64)),
                dtype=np.int64,
            ).reshape(-1)
            if int(gt_boxes.shape[0]) != int(gt_labels_nyu.shape[0]):
                raise RuntimeError(
                    "ScanNet LD/eval GT shape mismatch in memory subset: "
                    f"boxes={gt_boxes.shape[0]}, labels={gt_labels_nyu.shape[0]}."
                )
            mapped_labels: List[int] = []
            mapped_boxes: List[np.ndarray] = []
            unknown_ids: List[int] = []
            for idx_i, nyu_id in enumerate(gt_labels_nyu.tolist()):
                key = int(nyu_id)
                if key not in nyu40_to_model:
                    unknown_ids.append(int(key))
                    continue
                mapped_labels.append(int(nyu40_to_model[key]))
                mapped_boxes.append(gt_boxes[int(idx_i)])
            if unknown_ids:
                raise RuntimeError(
                    "ScanNet LD/eval encountered NYU40 IDs missing from "
                    f"nyu40_to_model_idx: {sorted(set(int(x) for x in unknown_ids))[:10]}"
                )
            if not mapped_labels:
                gt_annos.append({'gt_num': 0})
                continue
            gt_annos.append(
                {
                    'gt_boxes_upright_depth': np.asarray(mapped_boxes, dtype=np.float32),
                    'gt_num': int(len(mapped_labels)),
                    'class': np.asarray(mapped_labels, dtype=np.int64),
                }
            )
    else:
        gt_annos = [info.get('annos', {'gt_num': 0}) for info in mem_dataset.data_infos]
    eval_context = {
        'dataset': str(dataset_label),
        'split': str(split_name),
        'purpose': str(eval_purpose) if eval_purpose else 'subset_eval',
        'stage_id': max(0, int(stage_idx)) + 1,
    }
    if review_k is not None:
        try:
            eval_context['review_k'] = int(review_k)
        except Exception:
            pass
    if logger is not None:
        try:
            logger.info(
                f"{dataset_label} Subset Eval: split={split_name}, "
                f"purpose={eval_context['purpose']}, stage={eval_context.get('stage_id')}, "
                f"k={eval_context.get('review_k', '-')}"
            )
        except Exception:
            pass
    ret = incremental_indoor_eval(
        gt_annos,
        results,
        metric=iou_list,
        seen_classes=eval_class_indices,
        class_names=class_names,
        stage_idx=int(stage_idx),
        logger=logger,
        box_type_3d=mem_dataset.box_type_3d,
        box_mode_3d=mem_dataset.box_mode_3d,
        class_meta=None,
        eval_context=eval_context,
    )

    # Optional raw outputs (for learning-dynamics scoring / debugging).
    if raw_results_out is not None:
        raw_results_out[:] = list(results)
    if raw_gt_annos_out is not None:
        raw_gt_annos_out[:] = list(gt_annos)
    if raw_data_infos_out is not None:
        raw_data_infos_out[:] = list(mem_dataset.data_infos)
    if raw_box_type_3d_out is not None:
        raw_box_type_3d_out[:] = [mem_dataset.box_type_3d]
    if raw_box_mode_3d_out is not None:
        raw_box_mode_3d_out[:] = [mem_dataset.box_mode_3d]

    # Strip stage_{stage_idx}_ prefix for easier downstream processing.
    return _strip_metric_prefix(ret, prefix=f"stage_{int(stage_idx)}_")


def main():
    args = parse_args()

    # Load incremental learning configuration with explicit mappings
    incremental_cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        incremental_cfg.merge_from_dict(args.cfg_options)
    _validate_unified_replay_pseudo_cfg_or_raise(incremental_cfg)

    log_verbosity = _get_log_verbosity(incremental_cfg)
    log_debug = log_verbosity in ('debug', 'verbose')
    artifact_profile_requested = _get_artifact_profile_requested(incremental_cfg)
    artifact_profile_effective = 'full'
    ld_strategy_for_experiment = False
    ld_path_only_experiment = False

    # Safety gate: prevent accidental runs of research configs.
    cfg_path = args.config.replace('\\', '/').lstrip('./')
    if '/configs/experimental/' in f'/{cfg_path}':
        assert incremental_cfg.get('allow_experimental', False) is True, (
            'Refusing to run an experimental config without explicit opt-in. '
            'Re-run with `--cfg-options allow_experimental=True`.'
        )

    # Read config settings to override command line arguments (new consolidated approach)
    if hasattr(incremental_cfg, 'start_stage') and args.start_stage == 1:  # Only override if not set via command line
        args.start_stage = incremental_cfg.start_stage
        print(f"Config override: start_stage={args.start_stage}")

    if hasattr(incremental_cfg, 'end_stage') and args.end_stage is None:  # Only override if not set via command line
        args.end_stage = incremental_cfg.end_stage
        print(f"Config override: end_stage={args.end_stage}")

    if hasattr(incremental_cfg, 'stage1_checkpoint') and not args.checkpoint_path:
        args.checkpoint_path = incremental_cfg.stage1_checkpoint
        print(f"Config override: checkpoint_path={args.checkpoint_path}")

    if hasattr(incremental_cfg, 'work_dir_base') and not args.work_dir:
        args.work_dir = incremental_cfg.work_dir_base
        print(f"Config override: work_dir_base={args.work_dir}")

    if hasattr(incremental_cfg, 'seed') and args.seed == 0:
        args.seed = incremental_cfg.seed
        print(f"Config override: seed={args.seed}")

    print("Using consolidated config approach - all settings from config file")

    # Initialize distributed training if requested
    distributed = False
    world_size = 1
    try:
        base_dist_params = None
        if hasattr(incremental_cfg, 'base_config') and hasattr(incremental_cfg.base_config, 'dist_params'):
            base_dist_params = incremental_cfg.base_config.dist_params
        if args.launcher != 'none':
            distributed = True
            init_dist(args.launcher, **(base_dist_params or dict(backend='nccl')))
            _, world_size = get_dist_info()
    except Exception as e:
        print(
            f"Warning: Failed to initialize distributed env: {e}. "
            f"Falling back to single GPU."
        )
        distributed = False
        world_size = 1

    # Decide gpu_ids for this run
    if distributed:
        gpu_ids = list(range(world_size))
    else:
        try:
            visible = torch.cuda.device_count()
        except Exception:
            visible = 0
        gpu_ids = list(range(visible)) if visible > 0 else [0]

    # Extract stage definitions and validate
    stage_definitions = incremental_cfg.stage_definitions
    validate_incremental_mappings(stage_definitions, verbose=log_debug)

    # Determine max stage id from config (used for arg validation + defaults).
    try:
        stage_ids = [int(sd.get('stage_id', i + 1)) for i, sd in enumerate(stage_definitions)]
        if stage_ids != list(range(1, len(stage_definitions) + 1)):
            raise ValueError(
                "stage_definitions must use consecutive stage_id starting from 1. "
                f"Got: {stage_ids}"
            )
        max_stage_id = int(stage_ids[-1]) if stage_ids else 0
    except Exception as e:
        raise ValueError(f"Invalid stage_definitions (stage_id): {e}")

    # Default end_stage to last configured stage if not set via CLI/config.
    if args.end_stage is None:
        args.end_stage = int(max_stage_id)

    # Validate start/end stage arguments against the configured schedule.
    if args.start_stage < 1 or args.start_stage > max_stage_id:
        raise ValueError(
            f"--start-stage must be within [1, {max_stage_id}] for this config; "
            f"got {args.start_stage}"
        )
    if args.end_stage < 1 or args.end_stage > max_stage_id:
        raise ValueError(
            f"--end-stage must be within [1, {max_stage_id}] for this config; "
            f"got {args.end_stage}"
        )
    if args.start_stage > args.end_stage:
        raise ValueError(
            f"--start-stage ({args.start_stage}) cannot be greater than "
            f"--end-stage ({args.end_stage})"
        )

    # Validate checkpoint requirements again after config overrides.
    if args.start_stage > 1:
        if not args.checkpoint_path:
            raise ValueError(
                f"--checkpoint-path is required when starting from stage {args.start_stage}"
            )
        if not osp.exists(args.checkpoint_path):
            raise ValueError(f"Checkpoint not found: {args.checkpoint_path}")
    elif args.checkpoint_path:
        raise ValueError("--checkpoint-path should only be used with --start-stage > 1")

    # Show training plan (now that we know the configured stage count).
    if args.start_stage == 1 and args.end_stage == max_stage_id:
        print(f"FULL TRAINING: Stages 1-{max_stage_id} from scratch")
    elif args.start_stage == args.end_stage:
        if args.start_stage == 1:
            print(f"SINGLE STAGE: Stage {args.start_stage} from scratch")
        else:
            print(
                f"SINGLE STAGE: Stage {args.start_stage} from checkpoint "
                f"{args.checkpoint_path}"
            )
    else:
        if args.start_stage == 1:
            print(
                f"PARTIAL TRAINING: Stages {args.start_stage}-{args.end_stage} from scratch"
            )
        else:
            print(
                f"PARTIAL TRAINING: Stages {args.start_stage}-{args.end_stage} "
                f"from checkpoint {args.checkpoint_path}"
            )

    # CRITICAL VALIDATION: Ensure all stage definitions have epochs
    for i, stage_def in enumerate(stage_definitions):
        if 'epochs' not in stage_def:
            raise ValueError(
                f"Stage {i+1} (id={stage_def.get('stage_id', '?')}) missing 'epochs' field. "
                f"Check your config file: {args.config}"
            )
        stage_id = stage_def.get('stage_id', i+1)
        print(f"Stage {stage_id}: {stage_def['epochs']} epochs configured")

    # Create unified mappings from explicit config
    mappings = create_mapping_from_config(stage_definitions)
    num_classes = int(mappings.get('num_classes', 35))

    # Resolve and validate stage setting (for logs/output naming).
    stage_setting, stage_setting_source = _resolve_stage_setting(
        incremental_cfg, stage_definitions, num_classes
    )
    _validate_stage_setting_or_raise(
        stage_setting=stage_setting,
        stage_definitions=stage_definitions,
    )

    # Generate timestamp for unique folder names
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

    # Set up work directory - ALWAYS append timestamp to prevent overwriting
    if args.work_dir:
        # For debug configs, use the config name in work_dir
        base_work_dir = args.work_dir.rstrip('/')  # Remove trailing slash if any
        config_name = osp.splitext(osp.basename(args.config))[0]

        # If this is a debug config, append the config name for clarity
        if '/debug/' in args.config:
            if stage_setting:
                work_dir = f"{base_work_dir}/{config_name}_{stage_setting}_{timestamp}"
            else:
                work_dir = f"{base_work_dir}/{config_name}_{timestamp}"
        else:
            if stage_setting:
                work_dir = f"{base_work_dir}_{stage_setting}_{timestamp}"
            else:
                work_dir = f"{base_work_dir}_{timestamp}"
    else:
        # Default: use config name with timestamp in incremental_logs/scene_based directory
        config_name = osp.splitext(osp.basename(args.config))[0]

        # Check if using score-based selection and append criteria to work_dir name
        scene_memory_config = incremental_cfg.get('scene_memory_config', {})
        if (scene_memory_config and
            scene_memory_config.get('selection_strategy') == 'precomputed' and
            scene_memory_config.get('score_criteria')):
            score_criteria = scene_memory_config.get('score_criteria')
            if stage_setting:
                work_dir = osp.join(
                    './incremental_logs/scene_based',
                    f'{config_name}_{stage_setting}_{score_criteria}_{timestamp}'
                )
            else:
                work_dir = osp.join(
                    './incremental_logs/scene_based',
                    f'{config_name}_{score_criteria}_{timestamp}'
                )
        else:
            if stage_setting:
                work_dir = osp.join(
                    './incremental_logs/scene_based',
                    f'{config_name}_{stage_setting}_{timestamp}'
                )
            else:
                work_dir = osp.join('./incremental_logs/scene_based', f'{config_name}_{timestamp}')

    mmcv.mkdir_or_exist(osp.abspath(work_dir))

    # Determine rank for multi-GPU control flow
    try:
        rank, _ = get_dist_info()
    except Exception:
        rank, _ = (0, 1)
    is_main_process = (rank == 0)

    # Simple barrier helper
    def _dist_barrier():
        if 'torch' in sys.modules:
            try:
                import torch as _torch
                if _torch.distributed.is_available() and _torch.distributed.is_initialized():
                    _torch.distributed.barrier()
            except Exception:
                pass

    # Write a resolved config snapshot to work_dir for reproducibility (rank 0 only).
    # This captures `--cfg-options` (already merged into incremental_cfg) and the
    # effective CLI args in `run_meta`.
    config_dest_path = osp.join(work_dir, f'config_{timestamp}.py')
    try:
        if is_main_process:
            run_meta = {
                'timestamp': str(timestamp),
                'hostname': str(socket.gethostname()),
                'cwd': str(os.getcwd()),
                'argv': [str(x) for x in sys.argv],
                'cmd': " ".join(shlex.quote(str(x)) for x in sys.argv),
                'config_path': str(args.config),
                'work_dir': str(work_dir),
                'args': {str(k): v for k, v in vars(args).items()},
                'cfg_options': dict(args.cfg_options) if isinstance(args.cfg_options, dict) else {},
                'env': {
                    'python': str(sys.executable),
                    'python_version': str(sys.version.split()[0]),
                    'torch': str(getattr(torch, '__version__', '')),
                    'mmcv': str(getattr(mmcv, '__version__', '')),
                },
                'git': _try_get_git_info(REPO_ROOT),
            }
            _write_resolved_config_snapshot(
                cfg=incremental_cfg,
                dest_path=config_dest_path,
                run_meta=run_meta,
            )
            print(f"Resolved config saved to: {config_dest_path}")
    except Exception as e:
        if is_main_process:
            print(f"Warning: Failed to save resolved config snapshot: {e}")

    # Initialize unified path management
    paths = IncrementalPaths(work_dir)
    incremental_cfg.paths = paths
    incremental_cfg.work_dir = work_dir  # Keep for backward compatibility

    # Copy provided checkpoint to experiment folder when starting from later stage (rank 0 only)
    if is_main_process and args.checkpoint_path and args.start_stage > 1:
        checkpoint_stage = args.start_stage - 1  # The stage this checkpoint represents
        stage_checkpoint_dir = Path(work_dir) / "checkpoints" / f"stage_{checkpoint_stage}"
        stage_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_dest = stage_checkpoint_dir / "latest.pth"
        if not checkpoint_dest.exists():
            try:
                shutil.copy(args.checkpoint_path, checkpoint_dest)
                print(
                    f"Copied Stage {checkpoint_stage} checkpoint to experiment folder: "
                    f"{args.checkpoint_path} -> {checkpoint_dest}"
                )
            except Exception as e:
                print(f"Warning: Failed to copy checkpoint: {e}")
        else:
            print(
                f"Stage {checkpoint_stage} checkpoint already exists: "
                f"{checkpoint_dest}"
            )

    # Set up logging (rank 0 writes log file)
    log_file = osp.join(work_dir, f'incremental_training_explicit_{timestamp}.log')
    logger = get_root_logger(log_file=log_file if is_main_process else None, log_level='INFO', name='mmdet')

    # Log stage-setting summary + stage groups at run start.
    if is_main_process:
        _log_stage_groups(
            logger=logger,
            stage_setting=stage_setting,
            stage_setting_source=stage_setting_source,
            stage_definitions=stage_definitions,
        )

    # Display debug mode information
    debug_info = _get_debug_mode_info(args, stage_definitions)
    print("\n" + "="*60)
    print("INCREMENTAL LEARNING TRAINING")
    print("="*60)
    print(f"Work Directory: {work_dir}")
    print(f"Timestamp: {timestamp}")
    print("-"*60)
    for line in debug_info['console_messages']:
        print(line)
    print("="*60)

    if is_main_process:
        logger.info("="*60)
        logger.info("TR3D INCREMENTAL LEARNING TRAINING (EXPLICIT MAPPINGS)")
        logger.info(f"Work Directory: {work_dir}")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(
            f"Resolved config snapshot: {args.config} (+cfg_options, CLI args in run_meta) "
            f"-> {config_dest_path}"
        )
        for line in debug_info['log_messages']:
            logger.info(line)
        logger.info("="*60)

    # Initialize random seed
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}')
    set_random_seed(seed, deterministic=args.deterministic)

    # Get base configuration
    base_cfg = incremental_cfg.base_config

    # Startup guardrail: ScanNet frame-alignment contract validation.
    # Default behavior is fail-closed for ScanNet-35 runs.
    train_cfg_root = _cfg_get(_cfg_get(base_cfg, 'data', None), 'train', None)
    inner_train_ds_cfg = _unwrap_train_dataset_cfg(train_cfg_root)
    train_dataset_type = str(_cfg_get(inner_train_ds_cfg, 'type', '') or '')
    train_ann_file = str(_cfg_get(inner_train_ds_cfg, 'ann_file', '') or '')
    train_data_root = str(_cfg_get(inner_train_ds_cfg, 'data_root', '') or '')
    is_scannet35_runtime = (
        train_dataset_type == 'IncrementalScanNetDataset'
        and int(num_classes) == 35
        and '_40class' in train_ann_file
    )
    if args.validate_scannet_alignment is None:
        run_scannet_alignment_validation = bool(is_scannet35_runtime)
    else:
        run_scannet_alignment_validation = bool(args.validate_scannet_alignment)

    if run_scannet_alignment_validation:
        if not is_scannet35_runtime:
            logger.warning(
                "ScanNet alignment validation requested but current runtime does not look like "
                "ScanNet-35 incremental path. Skipping validation."
            )
        else:
            alignment_sample_scenes = int(
                _cfg_get(incremental_cfg, 'scannet_alignment_sample_scenes', 256) or 256
            )
            alignment_min_ratio = float(
                _cfg_get(incremental_cfg, 'scannet_alignment_min_aligned_ratio', 0.99) or 0.99
            )
            logger.info(
                "Running ScanNet alignment contract validation: "
                f"data_root={train_data_root}, ann_file={train_ann_file}, "
                f"sample_scenes={alignment_sample_scenes}, min_aligned_ratio={alignment_min_ratio:.3f}"
            )
            contract_report = validate_scannet_alignment_contract(
                data_root=train_data_root or 'data/scannet',
                ann_file=train_ann_file or 'scannet_infos_train_40class_corrected.pkl',
                sample_scenes=alignment_sample_scenes,
                min_aligned_center_ratio=alignment_min_ratio,
                fail_on_mismatch=False,
            )
            contract_report_path = Path(work_dir) / 'scannet_alignment_contract.json'
            contract_report_path.write_text(
                json.dumps(contract_report, indent=2, sort_keys=True),
                encoding='utf-8',
            )
            logger.info(f"Saved ScanNet alignment contract report: {contract_report_path}")
            logger.info(
                "ScanNet alignment contract summary: "
                f"ok={bool(contract_report.get('ok', False))}, "
                f"scenes_checked={contract_report.get('scenes_checked')}, "
                f"scenes_with_boxes={contract_report.get('scenes_with_boxes')}, "
                f"aligned_ratio_mean={float(contract_report.get('aligned_center_ratio_mean', 0.0)):.4f}, "
                f"raw_ratio_mean={float(contract_report.get('raw_center_ratio_mean', 0.0)):.4f}"
            )
            if not bool(contract_report.get('ok', False)):
                logger.error(
                    "ScanNet alignment contract failed: "
                    f"box_mismatch={contract_report.get('box_mismatch_scene_count', 0)}, "
                    f"low_aligned_ratio={contract_report.get('low_aligned_ratio_scene_count', 0)}, "
                    f"raw_false_pass={contract_report.get('raw_false_pass_scene_count', 0)}"
                )
                logger.error(
                    "Offending scenes (truncated): "
                    f"box_mismatch={contract_report.get('box_mismatch_scene_ids', [])[:10]}, "
                    f"low_aligned_ratio={contract_report.get('low_aligned_ratio_scene_ids', [])[:10]}, "
                    f"raw_false_pass={contract_report.get('raw_false_pass_scene_ids', [])[:10]}"
                )
                raise RuntimeError(
                    "ScanNet alignment contract validation failed. "
                    f"See {contract_report_path}"
                )
    else:
        logger.info("ScanNet alignment contract validation: disabled")

    if is_main_process:
        logger.info("Incremental Learning Setup (Explicit Mappings):")
        logger.info(f"  Total stages: {len(stage_definitions)}")
        logger.info(f"  Classes per stage: {[len(stage['class_indices']) for stage in stage_definitions]}")
        logger.info(f"  Total classes: {num_classes}")
        logger.info(f"  Stage names: {[stage['stage_name'] for stage in stage_definitions]}")

    # Memory levels are intentionally exclusive: scene replay adds whole data
    # samples, while object replay pastes self-contained crops into natural scenes.
    object_memory_bank = None
    object_memory_loaded = False
    object_memory_config = incremental_cfg.get('object_memory_config', None)
    use_object_memory = bool(
        incremental_cfg.get('use_object_memory', object_memory_config is not None))

    # Initialize scene memory bank if using scene replay
    scene_memory_bank = None
    scene_memory_config = incremental_cfg.get('scene_memory_config', {})

    # Check if scene memory is disabled (pseudo-only experiments)
    if scene_memory_config is None:
        logger.info("Scene Memory Bank: DISABLED (pseudo-only mode)")
        use_scene_memory = False
    else:
        use_scene_memory = incremental_cfg.get('use_scene_memory', True)
        if use_scene_memory and isinstance(scene_memory_config, dict):
            validate_scene_memory_ld_strategy_config(
                scene_memory_config,
                context='scene_memory_config',
            )

    if use_object_memory and use_scene_memory:
        raise ValueError(
            'Object-level and scene-level replay cannot be enabled together. '
            'Set exactly one of use_object_memory/use_scene_memory.')
    if use_object_memory:
        if not isinstance(object_memory_config, dict):
            raise ValueError(
                'use_object_memory=True requires object_memory_config.')
        validate_scene_memory_ld_strategy_config(
            object_memory_config,
            context='object_memory_config',
        )
        from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank
        object_bank_kwargs = _strip_scene_memory_cfg_meta(object_memory_config)
        object_bank_kwargs.pop('insertion', None)
        object_bank_kwargs.setdefault('random_seed', int(seed))
        object_bank_kwargs.setdefault('work_dir', str(work_dir))
        if args.object_memory_path:
            object_memory_bank = ObjectMemoryBank.load_state(
                args.object_memory_path)
            object_memory_bank.work_dir = str(work_dir)
            _validate_object_memory_resume(
                object_memory_bank, args.start_stage, stage_definitions,
                args.object_memory_path)
            stored_classes = object_memory_bank.get_stored_classes()
            object_memory_loaded = True
            logger.info(
                f'Object Memory Bank loaded from {args.object_memory_path}: '
                f'{object_memory_bank.get_total_exemplar_count()} objects, '
                f'{len(stored_classes)} classes, stage '
                f'{object_memory_bank.loaded_stage_id}')
        else:
            object_memory_bank = ObjectMemoryBank(**object_bank_kwargs)
            logger.info(
                'Object Memory Bank initialized: '
                f"{object_memory_bank.exemplars_per_class} exemplars/class, "
                f"{object_memory_bank.max_total_exemplars} objects total, "
                f"selection={object_memory_bank.selection_strategy_name}")
        if int(args.start_stage) > 2 and not object_memory_loaded:
            raise ValueError(
                'Object-memory resume at stage 3+ requires '
                '--object-memory-path pointing to the bank saved by the '
                'immediately preceding stage.')
    elif args.object_memory_path:
        raise ValueError(
            '--object-memory-path was provided, but object memory is disabled.')

    # Determine if this experiment uses LD-style memory updates.
    # This is config-level (strategy), independent from stage_id>=2 activation.
    ld_strategy_for_experiment = False
    active_memory_config = (
        object_memory_config if use_object_memory else scene_memory_config)
    if isinstance(active_memory_config, dict):
        strategy_for_experiment = str(
            active_memory_config.get('selection_strategy', '')
        ).strip().lower()
        ld_strategy_for_experiment = _is_ld_selection_strategy(strategy_for_experiment)
    artifact_profile_effective = _resolve_artifact_profile(
        requested_profile=artifact_profile_requested,
        learning_dynamics_strategy=ld_strategy_for_experiment,
    )
    if artifact_profile_effective == 'ld_path_only' and not ld_strategy_for_experiment:
        logger.warning(
            "logging.artifact_profile='ld_path_only' requested, but "
            "active memory selection_strategy is not one of "
            "['learning_dynamics', 'learning_dynamics_design1', "
            "'learning_dynamics_design2']. "
            "Falling back to artifact_profile='full'."
        )
        artifact_profile_effective = 'full'
    ld_path_only_experiment = bool(
        artifact_profile_effective == 'ld_path_only' and ld_strategy_for_experiment
    )
    logger.info(
        "Artifact logging profile: "
        f"requested={artifact_profile_requested}, "
        f"effective={artifact_profile_effective}, "
        f"ld_strategy={ld_strategy_for_experiment}"
    )

    # Log pseudo label configuration
    pseudo_label_config = incremental_cfg.get('pseudo_label_config', {})
    if pseudo_label_config and pseudo_label_config.get('enabled', True) != False:
        logger.info("PSEUDO LABEL CONFIGURATION:")
        if pseudo_label_config.get('use_pregenerated'):
            logger.info("  Mode: Pre-generated (fast)")
            stage_thresholds = pseudo_label_config.get('stage_thresholds', {})
            if stage_thresholds:
                logger.info(f"  Stage thresholds: {stage_thresholds}")
        else:
            logger.info("  Mode: On-the-fly generation")
            logger.info(f"  Confidence threshold: {pseudo_label_config.get('confidence_threshold', 0.45)}")
        logger.info(f"  NMS threshold: {pseudo_label_config.get('nms_threshold', 0.3)}")
        logger.info(f"  Max per scene: {pseudo_label_config.get('max_pseudo_per_scene', 100)}")
        logger.info(f"  Merge strategy: {pseudo_label_config.get('merge_strategy', 'append')}")
        logger.info(f"  Weight: {pseudo_label_config.get('weight_pseudo_loss', 1.0)}")
        if pseudo_label_config.get('debug_mode'):
            logger.info("  Debug mode: ENABLED")
    else:
        logger.info("Pseudo Labels: DISABLED")

    if use_scene_memory and scene_memory_config is not None:
        from mmdet3d.datasets.scene_memory_bank import SceneMemoryBank

        # Stage stats caching (for stage-ratio quota allocation).
        # Used when explicitly requested by config.
        try:
            inner_dataset_type = str(getattr(base_cfg.data.train.dataset, 'type', ''))
        except Exception:
            inner_dataset_type = ''

        selection_strategy = scene_memory_config.get('selection_strategy', None)
        quota_strategy = scene_memory_config.get('quota_strategy', None)

        if (inner_dataset_type in ('IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset') and
                quota_strategy == 'stage_ratio' and
                selection_strategy in (
                    'random',
                    'learning_dynamics',
                    LD_DESIGN1_STRATEGY,
                    LD_DESIGN2_STRATEGY,
                )):
            if stage_setting == 'sunrgbd40_s5_freqorder':
                stats_file = 'sunrgbd_top40_8x5_stage_scene_counts.json'
            elif stage_setting == 'sunrgbd40_s10_freqorder_split':
                stats_file = 'sunrgbd_top40_4x10_stage_scene_counts.json'
            elif stage_setting == 'sunrgbd40_s3_20_10_10_freqorder':
                stats_file = 'sunrgbd_top40_20_10_10_stage_scene_counts.json'
            elif stage_setting == 'scannet35_s5_freqorder':
                stats_file = 'scannet35_s5_stage_scene_counts.json'
            elif stage_setting == 'scannet35_s3_freqorder_15_10_10':
                stats_file = 'scannet35_s3_15_10_10_stage_scene_counts.json'
            elif stage_setting == 'scannet35_s10_freqorder_4444433333':
                stats_file = 'scannet35_s10_4444433333_stage_scene_counts.json'
            else:
                fp = _fingerprint_stage_definitions(stage_definitions)
                dataset_tag = (
                    'sunrgbd_top40'
                    if inner_dataset_type == 'IncrementalSUNRGBDDataset'
                    else 'scannet35'
                )
                stats_file = f"{dataset_tag}_custom_{fp[:8]}_stage_scene_counts.json"
            stats_path = osp.join(REPO_ROOT, 'analysis', stats_file)

            stage_stats = _load_or_compute_stage_scene_counts(
                base_cfg=base_cfg,
                incremental_cfg=incremental_cfg,
                stage_definitions=stage_definitions,
                mappings=mappings,
                work_dir=work_dir,
                logger=logger,
                stats_path=stats_path,
                is_main_process=is_main_process,
                use_cache=True,
            )
            _dist_barrier()
            if stage_stats is None:
                try:
                    with open(stats_path, 'r') as f:
                        stage_stats = json.load(f)
                except Exception:
                    stage_stats = None

            if isinstance(stage_stats, dict):
                per_stage = stage_stats.get('per_stage_train_scenes', None)
                if isinstance(per_stage, list) and len(per_stage) == len(stage_definitions):
                    scene_memory_config.setdefault('stage_scene_counts', per_stage)

                # Deterministic selection unless explicitly overridden
                scene_memory_config.setdefault('random_seed', int(seed))

                # Keep total_training_scenes consistent if possible
                if stage_stats.get('total_train_scenes') is not None:
                    scene_memory_config.setdefault(
                        'total_training_scenes', int(stage_stats['total_train_scenes'])
                    )
        # Inject score dir for MB weighting (used by drop/current-stage weighting).
        # NOTE: the pipeline no longer writes a `metrics/` folder under work_dir.
        try:
            scene_memory_config.setdefault(
                'metrics_dir', str(incremental_cfg.paths.memory_bank_scores_dir())
            )
        except Exception:
            pass

        # Provide stage→class mapping for current-stage weighting
        try:
            stage_class_map = {int(sd['stage_id']): [int(x) for x in sd['class_indices']] for sd in stage_definitions}
            scene_memory_config.setdefault('stage_class_map', stage_class_map)
        except Exception:
            pass

        # Avoid leaking ScanNet-specific defaults into non-NYU40 datasets (e.g. SUNRGBD).
        has_nyu40_mapping = bool(mappings.get('nyu40_to_model_idx')) and bool(mappings.get('valid_nyu40_ids'))
        if has_nyu40_mapping:
            # Default to empirically good ScanNet stage-wise ratios if not specified.
            scene_memory_config.setdefault('stage_ratio_counts', [17880, 15090, 13260, 13110, 6750])

            # Enable current-stage weighting/quota by default for ScanNet experiments.
            scene_memory_config.setdefault('use_current_stage_weights', True)
            scene_memory_config.setdefault('current_weight_strength', 0.25)
            scene_memory_config.setdefault('current_alpha', 0.0)
            scene_memory_config.setdefault('enforce_current_quota', True)
            scene_memory_config.setdefault('min_current_stage_quota', 10)
        else:
            # Non-ScanNet datasets: keep memory bank behavior explicit and simple unless enabled in config.
            scene_memory_config.setdefault('use_drop_weights', False)
            scene_memory_config.setdefault('use_current_stage_weights', False)
            scene_memory_config.setdefault('enforce_current_quota', False)
            scene_memory_config.setdefault('min_current_stage_quota', 0)
        scene_memory_config_for_bank = _strip_scene_memory_cfg_meta(scene_memory_config)
        scene_memory_bank = SceneMemoryBank(**scene_memory_config_for_bank)
        logger.info("Scene Memory Bank initialized:")

        # Display mode-specific information
        scenes_per_class = scene_memory_config.get('scenes_per_class', None)
        if scenes_per_class is not None:
            logger.info("  - Mode: Legacy (per-class limits)")
            logger.info(f"  - Scenes per class: {scenes_per_class}")
        else:
            memory_budget = scene_memory_config.get('max_memory_scenes')
            if memory_budget is None:
                budget_ratio = scene_memory_config.get('memory_budget_ratio', 0.1)
                total_scenes = scene_memory_config.get('total_training_scenes', 1201)
                memory_budget = int(total_scenes * budget_ratio)
            logger.info(f"  - Mode: Global budget ({memory_budget} scenes total)")
            logger.info(f"  - Budget ratio: {scene_memory_config.get('memory_budget_ratio', 0.1) * 100:.1f}% of {scene_memory_config.get('total_training_scenes', 1201)} scenes")

        logger.info(f"  - Selection strategy: {scene_memory_config.get('selection_strategy', 'balanced')}")
        logger.info(f"  - Dedup strategy: {scene_memory_config.get('dedup_strategy', 'keep_both')}")

    model = None
    stage_results_history = {}  # Store evaluation results for each stage
    last_pseudo_labels_enabled = None

    # Handle start-stage functionality
    if args.start_stage > 1:
        logger.info(f"RESUME MODE: Starting from stage {args.start_stage}")
        logger.info(f"Will load checkpoint: {args.checkpoint_path}")

    # Handle deprecated load_checkpoint argument
    if args.load_checkpoint:
        logger.warning("--load-checkpoint is deprecated. Please use --checkpoint-path instead.")

    # Train each stage using explicit definitions
    for stage_idx, stage_definition in enumerate(stage_definitions):
        stage_id = stage_definition['stage_id']
        stage_name = stage_definition['stage_name']
        stage_classes = stage_definition['class_indices']

        # Skip stages before start_stage
        if stage_id < args.start_stage:
            logger.info(
                f"SKIPPING STAGE {stage_id}/{len(stage_definitions)} TRAINING "
                f"(starting from stage {args.start_stage})"
            )

            # Object crops use GT and do not require model inference. When a
            # stage-1 checkpoint is supplied, build its replay bank directly
            # from the read-only source dataset before starting stage 2.
            if (object_memory_bank is not None and not object_memory_loaded
                    and args.start_stage == 2
                    and stage_id == 1):
                stage_cfg = prepare_stage_config(
                    base_cfg, stage_definition, stage_idx, stage_definitions,
                    work_dir, incremental_cfg=incremental_cfg)
                stage_cfg.mappings = mappings
                object_dataset_cfg = copy.deepcopy(stage_cfg.data.train)
                object_dataset_cfg.dataset.stage_definition = stage_definition
                object_dataset_cfg.dataset.mappings = mappings
                object_dataset_cfg.dataset.scene_memory_bank = None
                object_dataset_cfg.dataset.object_memory_bank = object_memory_bank
                object_dataset_cfg.dataset.evaluation_mode = False
                object_dataset_cfg.dataset.all_stage_definitions = stage_definitions
                object_dataset_cfg.dataset.experiment_dir = stage_cfg.experiment_dir
                _attach_object_memory_insertion(
                    object_dataset_cfg.dataset,
                    object_memory_bank,
                    (object_memory_config or {}).get('insertion', {}))
                object_dataset = build_dataset(object_dataset_cfg)
                object_inner = get_innermost_dataset(object_dataset)
                if not hasattr(object_inner, 'update_memory_bank_from_stage'):
                    raise RuntimeError(
                        f'{type(object_inner).__name__} cannot populate object memory.')
                object_ld_design2_payload = None
                object_selection_strategy = str(
                    getattr(object_memory_bank, 'selection_strategy_name', '')
                ).strip().lower()
                if object_selection_strategy == LD_DESIGN2_STRATEGY:
                    object_design_block_key = 'learning_dynamics_design2'
                    object_design_cfg = dict(
                        (object_memory_config or {}).get(
                            object_design_block_key, {}) or {})
                    object_stage1_mode = str(
                        object_design_cfg.get(
                            'stage1_scores_mode', 'precomputed')
                    ).strip().lower()
                    object_stage1_files = get_ld_design_stage1_filenames(
                        object_selection_strategy)
                    if object_stage1_mode == 'recompute_from_stats':
                        recomputed = _recompute_stage1_ld_design1_scores_from_raw_stats(
                            checkpoint_path=args.checkpoint_path,
                            scene_memory_config=object_memory_config or {},
                            stage_definition=stage_definition,
                            require_stage_id=int(stage_id),
                            config_block_key=object_design_block_key,
                            strategy_name=object_selection_strategy,
                            design_version=2,
                            output_dir=str(
                                Path(work_dir) / 'learning_dynamics' / 'stage_1'),
                            output_filename=str(
                                object_stage1_files.get('recomputed_filename')),
                        )
                        object_scores_path = Path(
                            str(recomputed.get('scores_path'))).resolve()
                        object_ld_design2_payload = dict(
                            recomputed.get('scores_payload', {}) or {})
                    elif object_stage1_mode == 'precomputed':
                        object_scores_path = _resolve_stage1_ld_scores_path_for_checkpoint(
                            checkpoint_path=args.checkpoint_path,
                            scene_memory_config=object_memory_config or {},
                            config_block_key=object_design_block_key,
                            score_filename=str(
                                object_stage1_files.get('score_filename')),
                            strategy_name=object_selection_strategy,
                        )
                        object_ld_design2_payload = (
                            _load_learning_dynamics_design1_scores_for_memory_update(
                                object_scores_path,
                                require_stage_id=int(stage_id),
                                strategy_name=object_selection_strategy,
                                score_file_label=str(
                                    object_stage1_files.get('score_filename')),
                            )
                        )
                    else:
                        raise RuntimeError(
                            f"{object_design_block_key}.stage1_scores_mode must be "
                            "one of ['precomputed', 'recompute_from_stats'], "
                            f"got {object_stage1_mode!r}.")
                    _assert_stage1_design1_q_metric_matches_config(
                        payload=object_ld_design2_payload,
                        scene_memory_config=object_memory_config or {},
                        source_file=str(object_scores_path),
                        config_block_key=object_design_block_key,
                        strategy_name=object_selection_strategy,
                    )
                    logger.info(
                        'Stage 1 skip (object learning_dynamics_design2): '
                        f'loaded scores from {object_scores_path}')
                object_inner.update_memory_bank_from_stage(
                    learning_dynamics_design2_payload=(
                        object_ld_design2_payload))
                if is_main_process:
                    state_path = osp.join(
                        str(work_dir), 'object_memory_bank',
                        'object_memory_bank_stage_1.pkl')
                    object_memory_bank.save_state(state_path, stage_id=1)
                    logger.info(
                        'Built stage-1 object memory for resume and saved it to '
                        f'{state_path}')
                _dist_barrier()
                continue

            # IMPORTANT: Scene-memory post-processing only works for --start-stage 2
            # For stage 3+ discovery experiments, memory bank requires more complex construction
            # involving multiple previous stages, which is not implemented here.
            if args.start_stage == 2 and stage_id == 1:
                # Only build memory bank if it's actually enabled
                if (use_scene_memory and scene_memory_config is not None
                        and scene_memory_bank is not None):
                    logger.info(
                        f"BUILDING Stage {stage_id} memory bank for "
                        f"Stage {args.start_stage} discovery (skip Stage 1 "
                        f"training but still compute metrics from checkpoint)…"
                    )
                else:
                    logger.info(
                        f"Skipping Stage {stage_id} memory bank "
                        f"construction (memory bank disabled)"
                    )
                    continue

                # Build Stage 1 dataset to get scene infos (needed for memory bank construction)
                stage_cfg = prepare_stage_config(
                    base_cfg, stage_definition, stage_idx, stage_definitions, work_dir,
                    incremental_cfg=incremental_cfg)
                stage_cfg.mappings = mappings

                # Build dataset for memory bank construction
                train_dataset_cfg = copy.deepcopy(stage_cfg.data.train)
                # Modify inner dataset (RepeatDataset wraps the incremental dataset)
                incremental_dataset_type = getattr(
                    train_dataset_cfg.dataset, 'type', 'IncrementalScanNetDataset')
                train_dataset_cfg.dataset.type = incremental_dataset_type
                train_dataset_cfg.dataset.stage_definition = stage_definition
                train_dataset_cfg.dataset.mappings = mappings
                train_dataset_cfg.dataset.scene_memory_bank = scene_memory_bank
                train_dataset_cfg.dataset.object_memory_bank = object_memory_bank
                _attach_object_memory_insertion(
                    train_dataset_cfg.dataset,
                    object_memory_bank,
                    (object_memory_config or {}).get('insertion', {}))
                train_dataset_cfg.dataset.scene_dedup_strategy = incremental_cfg.get('scene_dedup_strategy', 'keep_both')
                train_dataset_cfg.dataset.evaluation_mode = False
                train_dataset_cfg.dataset.all_stage_definitions = stage_definitions
                train_dataset_cfg.dataset.experiment_dir = stage_cfg.experiment_dir

                # Build the dataset
                temp_dataset = build_dataset(train_dataset_cfg)

                logger.info(
                    f"Built Stage {stage_id} dataset: {len(temp_dataset)} scenes for memory bank construction"
                )

                # Skip evaluation sanity check for now (has MinkowskiEngine context issues)
                # The important part is building the memory bank, which happens below
                logger.info(
                    f"Skipping Stage {stage_id} evaluation sanity check "
                    f"(not needed for memory bank construction)"
                )

                # Under-learning insertion (train-only): when starting from
                # Stage 2, Stage 1 training is skipped, but we still need to
                # compute Stage 1 train AP from the provided checkpoint so the
                # Stage 1 memory bank can be built deterministically.
                underlearning_enabled = bool(
                    scene_memory_bank is not None and
                    getattr(scene_memory_bank, 'underlearning_insertion_enabled', False) and
                    incremental_dataset_type in (
                        'IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset'
                    )
                )
                underlearning_class_ap_for_memory_update = None
                underlearning_new_classes_for_memory_update = None

                # BUILD MEMORY BANK: Optionally use Stage 1 checkpoint to
                # compute uncertainty/diversity metrics, so that we can skip
                # Stage 1 training but still have high-quality memory.
                score_criteria = scene_memory_config.get('score_criteria')
                selection_strategy = scene_memory_config.get(
                    'selection_strategy', 'balanced'
                )
                selection_strategy_key = str(selection_strategy).strip().lower()

                stage1_skip_ld_scores_for_memory_update = None
                stage1_skip_ld_design_payload_for_memory_update = None
                stage1_skip_ld_scores_file = None
                if (
                    incremental_dataset_type in (
                        'IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset'
                    )
                    and selection_strategy_key == 'learning_dynamics'
                ):
                    stage1_skip_ld_scores_path = _resolve_stage1_ld_scores_path_for_checkpoint(
                        checkpoint_path=args.checkpoint_path,
                        scene_memory_config=scene_memory_config,
                        config_block_key='learning_dynamics_update',
                        score_filename='learning_dynamics_scores.json',
                        strategy_name='learning_dynamics',
                    )
                    stage1_skip_ld_scores = _load_learning_dynamics_scores_for_memory_update(
                        stage1_skip_ld_scores_path,
                        require_stage_id=int(stage_id),
                    )
                    stage1_skip_ld_replay = (
                        stage1_skip_ld_scores.get('replay_priority_by_seat', None)
                    )
                    if not isinstance(stage1_skip_ld_replay, dict) or not stage1_skip_ld_replay:
                        raise RuntimeError(
                            "selection_strategy='learning_dynamics' with --start-stage 2 requires "
                            "non-empty Stage-1 replay-priority seat scores. "
                            f"Resolved file: {stage1_skip_ld_scores_path}"
                        )
                    stage1_skip_ld_scores_for_memory_update = stage1_skip_ld_scores
                    stage1_skip_ld_scores_file = str(stage1_skip_ld_scores_path)
                    logger.info(
                        "Stage 1 skip (LD): loaded Stage-1 learning-dynamics seat scores "
                        f"from {stage1_skip_ld_scores_file} "
                        f"(replay seats={len(stage1_skip_ld_replay)})"
                    )
                elif (
                    incremental_dataset_type in (
                        'IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset'
                    )
                    and _is_ld_design_selection_strategy(selection_strategy_key)
                ):
                    ld_design_meta = validate_scene_memory_ld_strategy_config(
                        scene_memory_config,
                        context='scene_memory_config',
                    )
                    design_block_key = str(ld_design_meta.get('active_ld_block_key'))
                    design_cfg = dict(ld_design_meta.get('active_ld_config', {}) or {})
                    stage1_files = get_ld_design_stage1_filenames(selection_strategy_key)
                    design_version = 2 if selection_strategy_key == LD_DESIGN2_STRATEGY else 1
                    stage1_scores_mode = str(
                        design_cfg.get('stage1_scores_mode', 'precomputed')
                    ).strip().lower()
                    if stage1_scores_mode not in ('precomputed', 'recompute_from_stats'):
                        raise RuntimeError(
                            f"{design_block_key}.stage1_scores_mode must be one of "
                            "['precomputed', 'recompute_from_stats'], "
                            f"got {design_cfg.get('stage1_scores_mode')!r}."
                        )

                    if stage1_scores_mode == 'recompute_from_stats':
                        recompute_output_dir = (
                            Path(work_dir) / "learning_dynamics" / "stage_1"
                        ).resolve()
                        stage1_skip_ld_recomputed = _recompute_stage1_ld_design1_scores_from_raw_stats(
                            checkpoint_path=args.checkpoint_path,
                            scene_memory_config=scene_memory_config,
                            stage_definition=stage_definition,
                            require_stage_id=int(stage_id),
                            config_block_key=str(design_block_key),
                            strategy_name=str(selection_strategy_key),
                            design_version=int(design_version),
                            output_dir=str(recompute_output_dir),
                            output_filename=str(stage1_files.get('recomputed_filename')),
                        )
                        stage1_skip_ld_scores_path = Path(
                            str(stage1_skip_ld_recomputed.get('scores_path'))
                        ).resolve()
                        stage1_skip_ld_design_payload = dict(
                            stage1_skip_ld_recomputed.get('scores_payload', {}) or {}
                        )
                    else:
                        stage1_skip_ld_scores_path = _resolve_stage1_ld_scores_path_for_checkpoint(
                            checkpoint_path=args.checkpoint_path,
                            scene_memory_config=scene_memory_config,
                            config_block_key=str(design_block_key),
                            score_filename=str(stage1_files.get('score_filename')),
                            strategy_name=str(selection_strategy_key),
                        )
                        stage1_skip_ld_design_payload = (
                            _load_learning_dynamics_design1_scores_for_memory_update(
                                stage1_skip_ld_scores_path,
                                require_stage_id=int(stage_id),
                                strategy_name=str(selection_strategy_key),
                                score_file_label=str(stage1_files.get('score_filename')),
                            )
                        )

                    _assert_stage1_design1_q_metric_matches_config(
                        payload=stage1_skip_ld_design_payload,
                        scene_memory_config=scene_memory_config,
                        source_file=str(stage1_skip_ld_scores_path),
                        config_block_key=str(design_block_key),
                        strategy_name=str(selection_strategy_key),
                    )
                    stage1_skip_ld_design_terms = (
                        stage1_skip_ld_design_payload.get('seat_class_terms', None)
                    )
                    if (not isinstance(stage1_skip_ld_design_terms, dict)
                            or not stage1_skip_ld_design_terms):
                        raise RuntimeError(
                            f"selection_strategy='{selection_strategy_key}' with --start-stage 2 "
                            "requires non-empty Stage-1 seat_class_terms. "
                            f"Resolved file: {stage1_skip_ld_scores_path}"
                        )
                    stage1_skip_ld_design_payload_for_memory_update = (
                        stage1_skip_ld_design_payload
                    )
                    stage1_skip_ld_scores_file = str(stage1_skip_ld_scores_path)
                    if stage1_scores_mode == 'recompute_from_stats':
                        k_idx = (
                            stage1_skip_ld_recomputed.get('k_indices', [])
                            if isinstance(stage1_skip_ld_recomputed, dict) else []
                        )
                        logger.info(
                            f"Stage 1 skip ({selection_strategy_key}): recomputed Stage-1 scores "
                            f"from raw k-stats and saved to {stage1_skip_ld_scores_file} "
                            f"(k={k_idx}, seats={len(stage1_skip_ld_design_terms)})"
                        )
                    else:
                        logger.info(
                            f"Stage 1 skip ({selection_strategy_key}): loaded Stage-1 scores "
                            f"from {stage1_skip_ld_scores_file} "
                            f"(seats={len(stage1_skip_ld_design_terms)})"
                        )

                if selection_strategy in (
                        'uncertainty_only',
                        'diversity_only',
                        'uncertainty_diversity_combined'):
                    logger.info(
                        f"Building memory bank from Stage {stage_id} using "
                        f"Stage 1 checkpoint for uncertainty/diversity scoring"
                    )
                    # Build a minimal Stage 1 model for inference
                    use_dynamic_head = getattr(
                        incremental_cfg, 'use_dynamic_head', False
                    )
                    stage1_model_cfg = copy.deepcopy(stage_cfg.model)
                    if use_dynamic_head:
                        stage1_n_classes = (
                            max(stage_cfg.cumulative_seen_classes) + 1
                            if getattr(stage_cfg, 'cumulative_seen_classes', None) else
                            len(stage_definition.get('class_indices', []))
                        )
                        stage1_model_cfg.head.n_classes = int(stage1_n_classes)
                    stage1_model = build_model(
                        stage1_model_cfg,
                        train_cfg=stage_cfg.get('train_cfg'),
                        test_cfg=stage_cfg.get('test_cfg'))
                    # Some detectors (e.g., MinkSingleStage3DDetector) call
                    # init_weights() inside __init__.
                    if not getattr(stage1_model, 'is_init', False):
                        stage1_model.init_weights()

                    # Load Stage 1 checkpoint specified via --checkpoint-path
                    if args.checkpoint_path:
                        logger.info(
                            f"Loading Stage 1 checkpoint for memory bank "
                            f"construction: {args.checkpoint_path}"
                        )
                        checkpoint = torch.load(
                            args.checkpoint_path, map_location='cpu'
                        )
                        state_dict = checkpoint.get('state_dict', {})
                        model_dict = stage1_model.state_dict()
                        filtered_dict = {}
                        for k, v in state_dict.items():
                            if k in model_dict and v.shape == model_dict[k].shape:
                                filtered_dict[k] = v
                        missing, unexpected = stage1_model.load_state_dict(
                            filtered_dict, strict=False
                        )
                        if missing:
                            logger.info(
                                f"   Missing keys when loading Stage 1 "
                                f"checkpoint (benign): {len(missing)}"
                            )
                        if unexpected:
                            logger.info(
                                f"   Unexpected keys in Stage 1 checkpoint "
                                f"(benign): {len(unexpected)}"
                            )

                    # Attach inference config so inference utilities work.
                    stage1_model.cfg = stage_cfg

                    # Move to GPU if available
                    if torch.cuda.is_available():
                        stage1_model = stage1_model.cuda()

                    innermost_dataset = get_innermost_dataset(temp_dataset)
                    # Compute under-learning weights for Stage 1 insertion if enabled.
                    if underlearning_enabled:
                        new_classes = [int(x) for x in stage_definition.get('class_indices', [])]
                        underlearning_new_classes_for_memory_update = list(new_classes)
                        ul_cfg = getattr(scene_memory_bank, 'underlearning_insertion', {}) or {}
                        ap_iou_thr = float(ul_cfg.get('ap_iou_thr', 0.25))
                        max_eval = ul_cfg.get('eval_max_scenes', None)
                        seed_offset = int(ul_cfg.get('eval_seed_offset', 11000))
                        if max_eval is not None:
                            max_eval = int(max_eval)
                            assert max_eval > 0, max_eval

                        natural_infos = list(getattr(innermost_dataset, 'data_infos', []) or [])
                        eval_infos = natural_infos
                        if max_eval is not None and len(eval_infos) > max_eval:
                            rng = np.random.RandomState(int(seed) + seed_offset + 100 * int(stage_id))
                            idx = rng.choice(len(eval_infos), size=int(max_eval), replace=False)
                            eval_infos = [eval_infos[i] for i in sorted(idx.tolist())]

                        try:
                            target_model = stage1_model.module if hasattr(stage1_model, 'module') else stage1_model
                            n_cls = int(getattr(getattr(target_model, 'head', None), 'n_classes', stage_cfg.model.head.n_classes))
                        except Exception:
                            n_cls = int(stage_cfg.model.head.n_classes)
                        class_names = [mappings['model_idx_to_name'][i] for i in range(int(n_cls))]

                        iou_key = f"{float(ap_iou_thr):.2f}"
                        score_dir = incremental_cfg.paths.memory_bank_scores_dir()
                        score_dir.mkdir(parents=True, exist_ok=True)
                        out_path = score_dir / f"underlearning_stage_{stage_id}_train_ap.json"

                        if is_main_process:
                            logger.info(
                                f"Under-learning (Stage 1 skip): evaluating train(natural) "
                                f"stage={stage_id}, new_classes={new_classes}, iou_thr={float(ap_iou_thr):.2f}, "
                                f"scenes={len(eval_infos)}/{len(natural_infos)}"
                            )
                            metrics_ul = _sunrgbd_eval_memory_subset(
                                model=stage1_model,
                                stage_cfg=stage_cfg,
                                data_infos=eval_infos,
                                eval_class_indices=new_classes,
                                class_names=class_names,
                                iou_thrs=(float(ap_iou_thr),),
                                stage_idx=max(0, int(stage_id) - 1),
                                split_name=f"train(stage_{stage_id}_natural)",
                                eval_purpose='underlearning',
                                logger=logger,
                            )

                            ap_by_class = {}
                            weights = {}
                            for cid in new_classes:
                                cid = int(cid)
                                name = mappings['model_idx_to_name'].get(cid, f"class_{cid}")
                                ap = float(metrics_ul.get(f"{name}_AP_{iou_key}", 0.0))
                                if not np.isfinite(ap):
                                    ap = 0.0
                                ap = float(max(0.0, min(1.0, ap)))
                                ap_by_class[cid] = ap
                                weights[cid] = float(max(0.0, min(1.0, 1.0 - ap)))

                            payload = {
                                'stage_id': int(stage_id),
                                'split': 'train(natural)',
                                'iou_thr': float(ap_iou_thr),
                                'new_classes': [int(x) for x in new_classes],
                                'num_scenes_total': int(len(natural_infos)),
                                'num_scenes_evaluated': int(len(eval_infos)),
                                'ap_by_class': {str(int(k)): float(v) for k, v in ap_by_class.items()},
                                'underlearning_weight_by_class': {str(int(k)): float(v) for k, v in weights.items()},
                                'score_mode': str(ul_cfg.get('score_mode', 'object_count_sum')),
                            }
                            with open(out_path, 'w') as f:
                                json.dump(payload, f, indent=2)
                            underlearning_class_ap_for_memory_update = ap_by_class
                            logger.info(
                                f"Under-learning (Stage 1 skip): saved train AP to {out_path}"
                            )

                        _dist_barrier()
                        if not is_main_process:
                            with open(out_path, 'r') as f:
                                loaded = json.load(f)
                            ap_loaded = loaded.get('ap_by_class', {}) or {}
                            ap_by_class = {}
                            for k, v in ap_loaded.items():
                                try:
                                    ap_by_class[int(k)] = float(v)
                                except Exception:
                                    continue
                            underlearning_class_ap_for_memory_update = ap_by_class

                    if hasattr(
                            innermost_dataset,
                            'update_scene_memory_bank_from_stage'):
                        innermost_dataset.update_scene_memory_bank_from_stage(
                            model=stage1_model,
                            underlearning_class_ap=underlearning_class_ap_for_memory_update,
                            underlearning_new_classes=underlearning_new_classes_for_memory_update,
                            learning_dynamics_forgetness_by_seat=(
                                (stage1_skip_ld_scores_for_memory_update or {}).get(
                                    'forgetness_by_seat', None)
                            ),
                            learning_dynamics_replay_priority_by_seat=(
                                (stage1_skip_ld_scores_for_memory_update or {}).get(
                                    'replay_priority_by_seat', None)
                            ),
                            learning_dynamics_design1_payload=(
                                stage1_skip_ld_design_payload_for_memory_update
                                if selection_strategy_key == LD_DESIGN1_STRATEGY
                                else None
                            ),
                            learning_dynamics_design2_payload=(
                                stage1_skip_ld_design_payload_for_memory_update
                                if selection_strategy_key == LD_DESIGN2_STRATEGY
                                else None
                            ),
                            dataset_ref=innermost_dataset if is_main_process else None,
                        )
                    else:
                        logger.warning(
                            f"Dataset {type(innermost_dataset)} doesn't have "
                            f"memory bank update method - skipping"
                        )
                else:
                    if score_criteria:
                        logger.info(
                            f"Building memory bank from Stage {stage_id} using pre-computed scores "
                            f"({score_criteria}.json)"
                        )
                    else:
                        logger.info(
                            f"Building memory bank from Stage {stage_id} using dynamic scoring..."
                        )

                    # Legacy behavior: build without a model, relying on
                    # precomputed scores or heuristic importance.
                    innermost_dataset = get_innermost_dataset(temp_dataset)
                    if hasattr(
                            innermost_dataset,
                            'update_scene_memory_bank_from_stage'):
                        stage1_model = None
                        # If under-learning insertion is enabled, we must compute
                        # train AP from the Stage 1 checkpoint for deterministic
                        # selection (otherwise SceneMemoryBank asserts).
                        if underlearning_enabled:
                            assert args.checkpoint_path, (
                                "Under-learning insertion requires a Stage 1 checkpoint "
                                "when running with --start-stage 2."
                            )
                            use_dynamic_head = getattr(incremental_cfg, 'use_dynamic_head', False)
                            stage1_model_cfg = copy.deepcopy(stage_cfg.model)
                            if use_dynamic_head:
                                stage1_n_classes = (
                                    max(stage_cfg.cumulative_seen_classes) + 1
                                    if getattr(stage_cfg, 'cumulative_seen_classes', None) else
                                    len(stage_definition.get('class_indices', []))
                                )
                                stage1_model_cfg.head.n_classes = int(stage1_n_classes)
                            stage1_model = build_model(
                                stage1_model_cfg,
                                train_cfg=stage_cfg.get('train_cfg'),
                                test_cfg=stage_cfg.get('test_cfg'))
                            if not getattr(stage1_model, 'is_init', False):
                                stage1_model.init_weights()

                            logger.info(
                                f"Loading Stage 1 checkpoint for under-learning insertion: {args.checkpoint_path}"
                            )
                            checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
                            state_dict = checkpoint.get('state_dict', {})
                            model_dict = stage1_model.state_dict()
                            filtered_dict = {}
                            for k, v in state_dict.items():
                                if k in model_dict and v.shape == model_dict[k].shape:
                                    filtered_dict[k] = v
                            stage1_model.load_state_dict(filtered_dict, strict=False)
                            stage1_model.cfg = stage_cfg
                            if torch.cuda.is_available():
                                stage1_model = stage1_model.cuda()

                            new_classes = [int(x) for x in stage_definition.get('class_indices', [])]
                            underlearning_new_classes_for_memory_update = list(new_classes)
                            ul_cfg = getattr(scene_memory_bank, 'underlearning_insertion', {}) or {}
                            ap_iou_thr = float(ul_cfg.get('ap_iou_thr', 0.25))
                            max_eval = ul_cfg.get('eval_max_scenes', None)
                            seed_offset = int(ul_cfg.get('eval_seed_offset', 11000))
                            if max_eval is not None:
                                max_eval = int(max_eval)
                                assert max_eval > 0, max_eval

                            natural_infos = list(getattr(innermost_dataset, 'data_infos', []) or [])
                            eval_infos = natural_infos
                            if max_eval is not None and len(eval_infos) > max_eval:
                                rng = np.random.RandomState(int(seed) + seed_offset + 100 * int(stage_id))
                                idx = rng.choice(len(eval_infos), size=int(max_eval), replace=False)
                                eval_infos = [eval_infos[i] for i in sorted(idx.tolist())]

                            try:
                                target_model = stage1_model.module if hasattr(stage1_model, 'module') else stage1_model
                                n_cls = int(getattr(getattr(target_model, 'head', None), 'n_classes', stage_cfg.model.head.n_classes))
                            except Exception:
                                n_cls = int(stage_cfg.model.head.n_classes)
                            class_names = [mappings['model_idx_to_name'][i] for i in range(int(n_cls))]

                            iou_key = f"{float(ap_iou_thr):.2f}"
                            score_dir = incremental_cfg.paths.memory_bank_scores_dir()
                            score_dir.mkdir(parents=True, exist_ok=True)
                            out_path = score_dir / f"underlearning_stage_{stage_id}_train_ap.json"

                            if is_main_process:
                                logger.info(
                                    f"Under-learning (Stage 1 skip): evaluating train(natural) "
                                    f"stage={stage_id}, new_classes={new_classes}, iou_thr={float(ap_iou_thr):.2f}, "
                                    f"scenes={len(eval_infos)}/{len(natural_infos)}"
                                )
                                metrics_ul = _sunrgbd_eval_memory_subset(
                                    model=stage1_model,
                                    stage_cfg=stage_cfg,
                                    data_infos=eval_infos,
                                    eval_class_indices=new_classes,
                                    class_names=class_names,
                                    iou_thrs=(float(ap_iou_thr),),
                                    stage_idx=max(0, int(stage_id) - 1),
                                    split_name=f"train(stage_{stage_id}_natural)",
                                    eval_purpose='underlearning',
                                    logger=logger,
                                )

                                ap_by_class = {}
                                weights = {}
                                for cid in new_classes:
                                    cid = int(cid)
                                    name = mappings['model_idx_to_name'].get(cid, f"class_{cid}")
                                    ap = float(metrics_ul.get(f"{name}_AP_{iou_key}", 0.0))
                                    if not np.isfinite(ap):
                                        ap = 0.0
                                    ap = float(max(0.0, min(1.0, ap)))
                                    ap_by_class[cid] = ap
                                    weights[cid] = float(max(0.0, min(1.0, 1.0 - ap)))

                                payload = {
                                    'stage_id': int(stage_id),
                                    'split': 'train(natural)',
                                    'iou_thr': float(ap_iou_thr),
                                    'new_classes': [int(x) for x in new_classes],
                                    'num_scenes_total': int(len(natural_infos)),
                                    'num_scenes_evaluated': int(len(eval_infos)),
                                    'ap_by_class': {str(int(k)): float(v) for k, v in ap_by_class.items()},
                                    'underlearning_weight_by_class': {str(int(k)): float(v) for k, v in weights.items()},
                                    'score_mode': str(ul_cfg.get('score_mode', 'object_count_sum')),
                                }
                                with open(out_path, 'w') as f:
                                    json.dump(payload, f, indent=2)
                                underlearning_class_ap_for_memory_update = ap_by_class
                                logger.info(
                                    f"Under-learning (Stage 1 skip): saved train AP to {out_path}"
                                )

                            _dist_barrier()
                            if not is_main_process:
                                with open(out_path, 'r') as f:
                                    loaded = json.load(f)
                                ap_loaded = loaded.get('ap_by_class', {}) or {}
                                ap_by_class = {}
                                for k, v in ap_loaded.items():
                                    try:
                                        ap_by_class[int(k)] = float(v)
                                    except Exception:
                                        continue
                                underlearning_class_ap_for_memory_update = ap_by_class

                        innermost_dataset.update_scene_memory_bank_from_stage(
                            model=stage1_model,
                            underlearning_class_ap=underlearning_class_ap_for_memory_update,
                            underlearning_new_classes=underlearning_new_classes_for_memory_update,
                            learning_dynamics_forgetness_by_seat=(
                                (stage1_skip_ld_scores_for_memory_update or {}).get(
                                    'forgetness_by_seat', None)
                            ),
                            learning_dynamics_replay_priority_by_seat=(
                                (stage1_skip_ld_scores_for_memory_update or {}).get(
                                    'replay_priority_by_seat', None)
                            ),
                            learning_dynamics_design1_payload=(
                                stage1_skip_ld_design_payload_for_memory_update
                                if selection_strategy_key == LD_DESIGN1_STRATEGY
                                else None
                            ),
                            learning_dynamics_design2_payload=(
                                stage1_skip_ld_design_payload_for_memory_update
                                if selection_strategy_key == LD_DESIGN2_STRATEGY
                                else None
                            ),
                            dataset_ref=innermost_dataset if is_main_process else None,
                        )
                    else:
                        logger.warning(
                            f"Dataset {type(innermost_dataset)} doesn't have "
                            f"memory bank update method - skipping"
                        )

                # Stage 1 VAL baseline (resume from Stage 2):
                # We need a Stage-1 val snapshot to compute `forgetting_metrics_stage_2.json`
                # (stage 1 -> stage 2). Do this once and cache it in the current work_dir.
                if ld_path_only_experiment:
                    if is_main_process:
                        logger.info(
                            "LD artifact profile ('ld_path_only'): skipping Stage 1 val baseline "
                            "for forgetting metrics."
                        )
                else:
                    stage1_metrics_file = Path(incremental_cfg.paths.stage_metrics_file(stage_id))
                    if is_main_process:
                        if stage1_metrics_file.exists():
                            logger.info(
                                f"Stage {stage_id} val baseline already exists: {stage1_metrics_file}"
                            )
                        else:
                            assert args.checkpoint_path, (
                                "Resuming with --start-stage 2 requires --checkpoint-path <stage1_ckpt> "
                                "to compute the Stage-1 val baseline for forgetting metrics."
                            )
                            logger.info(
                                f"Computing Stage {stage_id} val baseline from checkpoint "
                                f"({args.checkpoint_path}) to enable Stage 2 forgetting metrics…"
                            )
                            try:
                                from datetime import datetime
                                from mmdet3d.apis import single_gpu_test
                                from mmcv.parallel import MMDataParallel
                                from mmdet.datasets import build_dataloader

                                baseline_model = stage1_model
                                if baseline_model is None:
                                    use_dynamic_head = getattr(incremental_cfg, 'use_dynamic_head', False)
                                    stage1_model_cfg = copy.deepcopy(stage_cfg.model)
                                    if use_dynamic_head:
                                        stage1_n_classes = (
                                            max(stage_cfg.cumulative_seen_classes) + 1
                                            if getattr(stage_cfg, 'cumulative_seen_classes', None) else
                                            len(stage_definition.get('class_indices', []))
                                        )
                                        stage1_model_cfg.head.n_classes = int(stage1_n_classes)
                                    baseline_model = build_model(
                                        stage1_model_cfg,
                                        train_cfg=stage_cfg.get('train_cfg'),
                                        test_cfg=stage_cfg.get('test_cfg'))
                                    if not getattr(baseline_model, 'is_init', False):
                                        baseline_model.init_weights()
                                    baseline_model.cfg = stage_cfg

                                    ckpt = torch.load(args.checkpoint_path, map_location='cpu')
                                    state_dict = ckpt.get('state_dict', ckpt)
                                    model_dict = baseline_model.state_dict()
                                    filtered_dict = {}
                                    for k, v in state_dict.items():
                                        if k in model_dict and hasattr(v, 'shape') and v.shape == model_dict[k].shape:
                                            filtered_dict[k] = v
                                    baseline_model.load_state_dict(filtered_dict, strict=False)

                                if torch.cuda.is_available():
                                    baseline_model = baseline_model.cuda()

                                val_dataset = build_dataset(stage_cfg.data.val)
                                val_dataloader = build_dataloader(
                                    val_dataset,
                                    samples_per_gpu=1,
                                    workers_per_gpu=stage_cfg.data.workers_per_gpu,
                                    dist=False,
                                    shuffle=False,
                                )

                                eval_model = MMDataParallel(baseline_model, device_ids=[0])
                                eval_model.eval()
                                results = single_gpu_test(eval_model, val_dataloader, show=False)

                                eval_kwargs = stage_cfg.get('evaluation', {}).copy()
                                eval_kwargs.pop('interval', None)
                                eval_kwargs.pop('save_best', None)
                                eval_res = val_dataset.evaluate(results, **eval_kwargs)
                                eval_res, _ = _strip_any_stage_prefix(eval_res)

                                cls_list = [int(x) for x in stage_definition.get('class_indices', [])]
                                classes_section = []
                                for c_idx in cls_list:
                                    name = mappings['model_idx_to_name'].get(c_idx, f"class_{c_idx}")
                                    try:
                                        ap25 = float(eval_res.get(f"{name}_AP_0.25", 0.0))
                                    except Exception:
                                        ap25 = 0.0
                                    try:
                                        ap50 = float(eval_res.get(f"{name}_AP_0.50", 0.0))
                                    except Exception:
                                        ap50 = 0.0
                                    classes_section.append({
                                        'model_idx': int(c_idx),
                                        'name': str(name),
                                        'AP_0.25': float(ap25),
                                        'AP_0.50': float(ap50),
                                    })

                                payload = {
                                    'stage_id': int(stage_id),
                                    'evaluated_at_stage': int(stage_id),
                                    'timestamp': datetime.now().isoformat(),
                                    'mAP_0.25': float(eval_res.get('mAP_0.25', 0.0) or 0.0),
                                    'mAP_0.50': float(eval_res.get('mAP_0.50', 0.0) or 0.0),
                                    'classes': classes_section,
                                }
                                stage1_metrics_file.parent.mkdir(parents=True, exist_ok=True)
                                with open(stage1_metrics_file, 'w') as f:
                                    json.dump(payload, f, indent=2)
                                logger.info(
                                    f"Saved Stage {stage_id} val baseline to: {stage1_metrics_file}"
                                )
                            except Exception as e:
                                raise RuntimeError(
                                    f"Failed to compute Stage {stage_id} val baseline for forgetting metrics: {e}"
                                ) from e

                _dist_barrier()
                # Always save memory bank state
                state_path = str(incremental_cfg.paths.scene_memory_file(stage_id))
                scene_memory_bank.save_state(state_path)
                logger.info(f"Stage {stage_id} memory bank saved to {state_path}")

                if log_debug:
                    scene_memory_bank.print_summary()

            continue

        # Stop if we've reached the end_stage
        if stage_id > args.end_stage:
            logger.info(
                f"Stopping after stage {args.end_stage} (reached end_stage limit)"
            )
            break

        logger.info(f"{'='*20} STAGE {stage_id}/{len(stage_definitions)} {'='*20}")
        logger.info(f"Stage name: {stage_name}")
        logger.info(f"Stage classes: {stage_classes}")
        logger.info(f"Class names: {[mappings['model_idx_to_name'][i] for i in stage_classes]}")

        # Prepare stage configuration
        stage_cfg = prepare_stage_config(
            base_cfg, stage_definition, stage_idx, stage_definitions, work_dir,
            incremental_cfg=incremental_cfg, gpu_ids=gpu_ids)
        stage_cfg.mappings = mappings

        # CHECKPOINT LOADING for start_stage > 1
        if stage_id == args.start_stage and args.start_stage > 1:
            logger.info(f"STAGE {stage_id} RESUME MODE: Loading checkpoint and building model")
            logger.info(f"   Checkpoint: {args.checkpoint_path}")

            # For resume mode, just continue to normal training - checkpoint will be loaded there
            pass

        # Build model with dynamic head expansion
        if model is None:
            # Check if using dynamic head expansion
            use_dynamic_head = getattr(incremental_cfg, 'use_dynamic_head', False)

            if use_dynamic_head:
                # Dynamic head: build model with correct number of classes for current stage
                curr_n_classes = (
                    max(stage_cfg.cumulative_seen_classes) + 1
                    if getattr(stage_cfg, 'cumulative_seen_classes', None) else
                    len(stage_definition['class_indices'])
                )
                stage_cfg.model.head.n_classes = curr_n_classes
                logger.info(f"Dynamic head mode: Building {curr_n_classes}-class model for stage {stage_id}")
            else:
                # Standard mode: First stage - build model with stage 1 classes only
                n_stage_classes = len(stage_definition['class_indices'])
                stage_cfg.model.head.n_classes = n_stage_classes
                logger.info(f"Standard mode: Building {n_stage_classes}-class model for stage {stage_id}")

            model = build_model(
                stage_cfg.model,
                train_cfg=stage_cfg.get('train_cfg'),
                test_cfg=stage_cfg.get('test_cfg'))
            _log_prediction_head_summary(
                logger=logger,
                model=model,
                prefix=f"Stage {stage_id} head (post-build)",
            )
            # Some detectors (e.g., MinkSingleStage3DDetector) call init_weights()
            # inside __init__, so avoid double-initialization (and log spam).
            if not getattr(model, 'is_init', False):
                model.init_weights()
            # Attach a usable config reference for inference utilities without
            # hardcoding a dataset-specific base config path.
            model.cfg = stage_cfg

            # Load checkpoint if resuming from later stage
            if stage_id == args.start_stage and args.checkpoint_path:
                logger.info(f"Loading checkpoint from: {args.checkpoint_path}")
                checkpoint = torch.load(args.checkpoint_path, map_location='cpu')

                # Filter out size mismatched keys for dynamic head
                if use_dynamic_head:
                    model_dict = model.state_dict()
                    state_dict = checkpoint['state_dict']

                    # Filter out size mismatched keys
                    filtered_dict = {}
                    for k, v in state_dict.items():
                        if k in model_dict:
                            if v.shape == model_dict[k].shape:
                                filtered_dict[k] = v
                            else:
                                logger.info(f"Skipping {k} due to size mismatch: {v.shape} vs {model_dict[k].shape}")
                        else:
                            logger.info(f"Key {k} not found in current model")

                    model.load_state_dict(filtered_dict, strict=False)
                    logger.info(f"Loaded {len(filtered_dict)}/{len(state_dict)} keys from checkpoint")

                    # Manually copy head weights - FIXED: Dynamic class count detection
                    if 'head.cls_conv.kernel' in state_dict and 'head.cls_conv.bias' in state_dict:
                        old_kernel = state_dict['head.cls_conv.kernel']
                        old_bias = state_dict['head.cls_conv.bias']

                        # Detect previous stage class count from checkpoint
                        prev_classes = old_kernel.shape[1]  # Get actual class count from checkpoint
                        curr_classes = model.head.cls_conv.kernel.shape[1]  # Current model class count

                        logger.info(f"Manually copying head weights: {old_kernel.shape} -> {model.head.cls_conv.kernel.shape}")
                        logger.info(f"Copying {prev_classes} classes from checkpoint to {curr_classes}-class model")

                        # Copy ALL previous classes (not just 7)
                        model.head.cls_conv.kernel.data[:, :prev_classes] = old_kernel.to(model.head.cls_conv.kernel.device)
                        model.head.cls_conv.bias.data[0, :prev_classes] = old_bias.flatten().to(model.head.cls_conv.bias.device)

                        logger.info(f"Head weights manually copied for classes 0-{prev_classes-1}")
                        _log_prediction_head_summary(
                            logger=logger,
                            model=model,
                            prefix=f"Stage {stage_id} head (after checkpoint load)",
                        )
                else:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)  # Standard loading
                    logger.info(f"Successfully loaded checkpoint for stage {stage_id}")
        else:
            # Subsequent stages - expand model head
            # SAFETY CHECK: Handle potential DataParallel wrapping
            if hasattr(model, 'module') and not hasattr(model, 'head'):
                logger.warning("Model is wrapped with DataParallel - unwrapping for head expansion")
                model = model.module

            # Get previous and current class counts (works for any stage split)
            if hasattr(model, 'head'):
                prev_n_classes = int(getattr(model.head, 'n_classes', 0))
            elif hasattr(model, 'bbox_head'):
                prev_n_classes = int(getattr(model.bbox_head, 'n_classes', 0))
            else:
                raise RuntimeError("Model has no 'head' or 'bbox_head' attribute for expansion")

            curr_n_classes = (
                max(stage_cfg.cumulative_seen_classes) + 1
                if getattr(stage_cfg, 'cumulative_seen_classes', None) else
                len(stage_definition['class_indices'])
            )

            logger.info(f"Expanding model head from {prev_n_classes} to {curr_n_classes} classes")
            _log_prediction_head_summary(
                logger=logger,
                model=model,
                prefix=f"Stage {stage_id} head (pre-expand)",
            )

            # Expand the head
            if hasattr(model, 'head'):
                success = model.head.expand_classification_head(curr_n_classes, logger=logger)
                if not success:
                    raise RuntimeError(f"Failed to expand head for stage {stage_id}")
            else:
                raise RuntimeError("Model does not have 'head' attribute for expansion")

            # NOTE: Model weights are already preserved in memory during expand_classification_head()
            # No need to reload from checkpoint since weights are transferred automatically
            _log_prediction_head_summary(
                logger=logger,
                model=model,
                prefix=f"Stage {stage_id} head (post-expand)",
            )

        # CRITICAL FIX: Apply masks to model head for training and evaluation
        # SAFETY CHECK: Handle potential DataParallel wrapping
        target_model = model
        if hasattr(model, 'module') and not hasattr(model, 'head'):
            logger.warning("Model is wrapped with DataParallel during mask setting - using module")
            target_model = model.module

        if hasattr(target_model, 'head'):
            target_model.head.seen_classes_mask = stage_cfg.seen_classes_mask  # Legacy compatibility
            target_model.head.training_classes_mask = stage_cfg.training_classes_mask
            target_model.head.evaluation_classes_mask = stage_cfg.evaluation_classes_mask
            train_active = int(stage_cfg.training_classes_mask.sum().item())
            eval_active = int(stage_cfg.evaluation_classes_mask.sum().item())
            logger.info(
                f"Head masks set for Stage {stage_definition['stage_id']}: "
                f"train_active={train_active}, eval_active={eval_active}"
            )
            if log_debug:
                logger.info(
                    f"  train_classes={torch.where(stage_cfg.training_classes_mask)[0].tolist()}"
                )
                logger.info(
                    f"  eval_classes={torch.where(stage_cfg.evaluation_classes_mask)[0].tolist()}"
                )
        elif hasattr(target_model, 'bbox_head'):
            # Fallback for different model architectures
            target_model.bbox_head.seen_classes_mask = stage_cfg.seen_classes_mask  # Legacy compatibility
            target_model.bbox_head.training_classes_mask = stage_cfg.training_classes_mask
            target_model.bbox_head.evaluation_classes_mask = stage_cfg.evaluation_classes_mask
            train_active = int(stage_cfg.training_classes_mask.sum().item())
            eval_active = int(stage_cfg.evaluation_classes_mask.sum().item())
            logger.info(
                f"bbox_head masks set for Stage {stage_definition['stage_id']}: "
                f"train_active={train_active}, eval_active={eval_active}"
            )
        else:
            logger.warning("Could not find model head to set masks!")

        # Build datasets for current stage
        train_dataset_cfg = copy.deepcopy(stage_cfg.data.train)
        # Modify inner dataset (RepeatDataset wraps the incremental dataset)
        incremental_dataset_type = getattr(
            train_dataset_cfg.dataset, 'type', 'IncrementalScanNetDataset')
        train_dataset_cfg.dataset.type = incremental_dataset_type
        train_dataset_cfg.dataset.stage_definition = stage_definition
        train_dataset_cfg.dataset.mappings = mappings
        train_dataset_cfg.dataset.scene_memory_bank = scene_memory_bank  # Use scene memory bank
        train_dataset_cfg.dataset.object_memory_bank = object_memory_bank
        _attach_object_memory_insertion(
            train_dataset_cfg.dataset,
            object_memory_bank,
            (object_memory_config or {}).get('insertion', {}))
        train_dataset_cfg.dataset.scene_dedup_strategy = incremental_cfg.get('scene_dedup_strategy', 'keep_both')
        # Propagate GT merge IoU options if present in config
        try:
            if hasattr(incremental_cfg.data, 'train') and hasattr(incremental_cfg.data.train, 'enable_gt_merge_iou'):
                train_dataset_cfg.dataset.enable_gt_merge_iou = bool(incremental_cfg.data.train.enable_gt_merge_iou)
            elif hasattr(incremental_cfg, 'enable_gt_merge_iou'):
                train_dataset_cfg.dataset.enable_gt_merge_iou = bool(incremental_cfg.enable_gt_merge_iou)
        except Exception:
            pass
        try:
            if hasattr(incremental_cfg.data, 'train') and hasattr(incremental_cfg.data.train, 'gt_merge_iou_thr'):
                train_dataset_cfg.dataset.gt_merge_iou_thr = float(incremental_cfg.data.train.gt_merge_iou_thr)
            elif hasattr(incremental_cfg, 'gt_merge_iou_thr'):
                train_dataset_cfg.dataset.gt_merge_iou_thr = float(incremental_cfg.gt_merge_iou_thr)
        except Exception:
            pass
        # Optional: target memory ratio for replay sampling
        if hasattr(incremental_cfg.data, 'train') and hasattr(incremental_cfg.data.train, 'target_memory_ratio'):
            train_dataset_cfg.dataset.target_memory_ratio = incremental_cfg.data.train.target_memory_ratio
        elif hasattr(incremental_cfg, 'target_memory_ratio'):
            train_dataset_cfg.dataset.target_memory_ratio = incremental_cfg.target_memory_ratio
        train_dataset_cfg.dataset.evaluation_mode = False  # Training mode: only current stage classes
        train_dataset_cfg.dataset.all_stage_definitions = stage_definitions
        train_dataset_cfg.dataset.experiment_dir = stage_cfg.experiment_dir  # Pass experiment root for unified paths

        # Pseudo labeling configuration (if enabled in config)
        use_pseudo_labels = False
        pseudo_label_config = {}
        pseudo_cfg_source = None

        if hasattr(stage_cfg.data.train, 'use_pseudo_labels'):
            use_pseudo_labels = stage_cfg.data.train.use_pseudo_labels
            pseudo_label_config = copy.deepcopy(getattr(stage_cfg.data.train, 'pseudo_label_config', {}))
            pseudo_cfg_source = 'stage_cfg'
        elif hasattr(incremental_cfg, 'use_pseudo_labels'):
            # Fallback to incremental config
            use_pseudo_labels = incremental_cfg.use_pseudo_labels
            pseudo_label_config = copy.deepcopy(getattr(incremental_cfg, 'pseudo_label_config', {}))
            pseudo_cfg_source = 'incremental_cfg'

        if last_pseudo_labels_enabled is None or last_pseudo_labels_enabled != bool(use_pseudo_labels):
            logger.info(f"Pseudo labels: {'ENABLED' if use_pseudo_labels else 'DISABLED'}")
            last_pseudo_labels_enabled = bool(use_pseudo_labels)

        if log_debug and use_pseudo_labels:
            logger.info(f"Pseudo label config source: {pseudo_cfg_source}")
            if isinstance(pseudo_label_config, dict):
                thr = pseudo_label_config.get('confidence_threshold', None)
                pregenerated = pseudo_label_config.get('use_pregenerated', None)
                pregenerated_file = pseudo_label_config.get('pregenerated_file', None)
                if thr is not None:
                    logger.info(f"  confidence_threshold={thr}")
                if pregenerated is not None:
                    logger.info(f"  use_pregenerated={pregenerated}")
                if pregenerated_file:
                    logger.info(f"  pregenerated_file={pregenerated_file}")

        # Clear previous-stage pregenerated file references so each stage regenerates
        if use_pseudo_labels and stage_id > args.start_stage:
            if pseudo_label_config.get('pregenerated_file'):
                logger.info("   Clearing previous-stage pseudo label file entry for fresh generation")
                pseudo_label_config.pop('pregenerated_file', None)
            if hasattr(stage_cfg.data.train, 'dataset') and hasattr(stage_cfg.data.train.dataset, 'pseudo_label_config'):
                dataset_cfg = stage_cfg.data.train.dataset.pseudo_label_config
                if dataset_cfg.get('pregenerated_file'):
                    dataset_cfg.pop('pregenerated_file', None)

        # PRE-GENERATE PSEUDO LABELS if enabled and needed (speeds up training)
        if use_pseudo_labels and stage_id >= 2:
            # Check if we should pre-generate pseudo labels for faster experimentation
            use_pregenerated = pseudo_label_config.get('use_pregenerated', True)

            if use_pregenerated:
                generation_checkpoint_used = None
                # Build a consistent config_suffix used for file naming/metadata in all branches
                default_conf = 0.5 if incremental_dataset_type == 'IncrementalSUNRGBDDataset' else 0.45
                config_suffix = _build_pseudo_config_suffix_impl(
                    incremental_dataset_type=incremental_dataset_type,
                    use_scene_memory=bool(use_scene_memory),
                    scene_memory_config=scene_memory_config,
                    pseudo_label_config=pseudo_label_config,
                )

                # If user specified a pregenerated_file for this stage, skip internal generation
                user_pregen = pseudo_label_config.get('pregenerated_file', None)
                if user_pregen and os.path.exists(user_pregen):
                    logger.info(f"Using user-provided pre-generated pseudo labels: {user_pregen}")
                    logger.info("   Skipping internal pre-generation for this stage")
                    # Ensure downstream variables exist for unified handling
                    from pathlib import Path as _Path
                    pseudo_label_dir = _Path(work_dir) / "pseudo_labels"
                    pseudo_label_dir.mkdir(parents=True, exist_ok=True)
                    pseudo_label_file = _Path(user_pregen)
                    copied_from_global = False
                    global_source_used = None
                    generation_checkpoint = None
                    # No barrier needed; file is external and present
                else:
                    if is_main_process:
                        logger.info("Pre-generating pseudo labels for faster training (rank0 only)...")
                        logger.info("   Explicit error handling with no fallback mechanisms")

                    # Paths for current stage's pregenerated file
                    pseudo_label_file = _resolve_stage_pseudo_file_impl(
                        work_dir=str(work_dir),
                        stage_id=int(stage_id),
                        config_suffix=str(config_suffix),
                    )
                    pseudo_label_dir = pseudo_label_file.parent
                    pseudo_label_dir.mkdir(parents=True, exist_ok=True)

                    # Optionally copy an external global pregenerated file if present
                    copied_from_global = False
                    global_source_used = None
                    # NOTE: Stage N training uses Stage N-1 pseudo labels (previous stage's predictions)
                    global_sources = {}
                    if incremental_dataset_type == 'IncrementalScanNetDataset':
                        global_sources = {
                            2: "stage1_pseudo_labels_correct/stage1_pseudo_labels_corrected.pkl",  # Stage 2 uses Stage 1 pseudo labels
                            3: "stage2_pseudo_labels/stage2_pseudo_labels.pkl",  # Stage 3 uses Stage 2 pseudo labels
                        }
                    if global_sources and not pseudo_label_file.exists() and is_main_process and stage_id in global_sources:
                        global_source = Path(global_sources[stage_id])
                        if global_source.exists():
                            shutil.copy(global_source, pseudo_label_file)
                            copied_from_global = True
                            global_source_used = str(global_source)
                            logger.info(
                                f"Copied Stage {stage_id} pseudo labels from global location: "
                                f"{global_source} -> {pseudo_label_file}"
                            )
                    # Ensure other ranks see any copied file
                    _dist_barrier()

                if not pseudo_label_file.exists() and not copied_from_global:
                    # Determine which checkpoint to use for generation
                    if stage_id == args.start_stage and args.checkpoint_path:
                        generation_checkpoint = args.checkpoint_path
                        logger.info(f"   Using start-stage checkpoint: {generation_checkpoint}")
                    else:
                        prev_stage_checkpoint_dir = Path(work_dir) / "checkpoints" / f"stage_{stage_id - 1}"
                        possible_checkpoints = [
                            prev_stage_checkpoint_dir / "epoch_1.pth",
                            prev_stage_checkpoint_dir / "latest.pth"
                        ]
                        generation_checkpoint = None
                        for checkpoint_path in possible_checkpoints:
                            if checkpoint_path.exists():
                                generation_checkpoint = str(checkpoint_path)
                                logger.info(f"   Found previous stage checkpoint: {generation_checkpoint}")
                                break
                        if not generation_checkpoint:
                            logger.error(
                                f"Previous stage checkpoint not found in: {prev_stage_checkpoint_dir} "
                                f"(tried: epoch_1.pth, latest.pth)"
                            )
                            raise FileNotFoundError(
                                f"Cannot pre-generate pseudo labels without a checkpoint in "
                                f"{prev_stage_checkpoint_dir}"
                            )

                    if not Path(generation_checkpoint).exists():
                        raise FileNotFoundError(f"Checkpoint not found for pseudo label generation: {generation_checkpoint}")

                    memory_bank_file = None
                    if scene_memory_config:
                        score_file = scene_memory_config.get('score_criteria', '')
                        if score_file:
                            memory_bank_file = f"analysis/{score_file}.json"

                    ann_file = getattr(train_dataset_cfg.dataset, 'ann_file', None)
                    if not ann_file:
                        raise ValueError(
                            "Cannot pre-generate pseudo labels because `data.train.dataset.ann_file` is missing."
                        )

                    output_file = None
                    if is_main_process:
                        if incremental_dataset_type == 'IncrementalSUNRGBDDataset':
                            from mmdet3d.utils.pregenerate_pseudo_labels_sunrgbd import (
                                pregenerate_sunrgbd_pseudo_labels_for_stage,
                            )

                            stage_def_for_gen = dict(stage_definition)
                            stage_def_for_gen['filter_empty_gt'] = bool(
                                getattr(train_dataset_cfg.dataset, 'filter_empty_gt', True)
                            )

                            output_file = pregenerate_sunrgbd_pseudo_labels_for_stage(
                                stage_id=stage_id,
                                checkpoint_path=generation_checkpoint,
                                train_ann_file=str(ann_file),
                                stage_definition=stage_def_for_gen,
                                all_stage_definitions=stage_definitions,
                                confidence_threshold=pseudo_label_config.get('confidence_threshold', default_conf),
                                nms_iou_thr=pseudo_label_config.get('nms_threshold', 0.3),
                                max_pseudo_per_scene=pseudo_label_config.get('max_pseudo_per_scene', 100),
                                output_dir=str(pseudo_label_dir),
                                config_suffix=config_suffix,
                            )
                        else:
                            from mmdet3d.utils.pregenerate_pseudo_labels import pregenerate_pseudo_labels_for_stage

                            output_file = pregenerate_pseudo_labels_for_stage(
                                stage_id=stage_id,
                                checkpoint_path=generation_checkpoint,
                                train_data_file=str(ann_file),
                                stage_definitions=stage_definitions,
                                memory_bank_file=memory_bank_file,
                                confidence_threshold=pseudo_label_config.get('confidence_threshold', 0.45),
                                output_dir=str(pseudo_label_dir),
                                config_suffix=config_suffix,
                            )

                    _dist_barrier()
                    if output_file is None:
                        output_file = str(pseudo_label_file)
                    if not Path(output_file).exists():
                        raise FileNotFoundError(
                            f"Pseudo label generation finished but file is missing: {output_file}"
                        )

                    pseudo_label_config['pregenerated_file'] = str(output_file)
                    generation_checkpoint_used = generation_checkpoint
                    logger.info(f"Pre-generated pseudo labels saved to: {output_file}")

                    _validate_pseudo_labels_nonfatal_impl(
                        incremental_dataset_type=incremental_dataset_type,
                        pseudo_file=str(output_file),
                        stage_id=int(stage_id),
                        pseudo_label_config=pseudo_label_config,
                        ann_file=str(ann_file) if ann_file else None,
                        logger=logger,
                        is_main_process=bool(is_main_process),
                        log_debug=bool(log_debug),
                        source_label='generated',
                    )
                else:
                    pseudo_label_config['pregenerated_file'] = str(pseudo_label_file)
                    logger.info(f"Using existing pre-generated pseudo labels: {pseudo_label_file}")

                    _validate_pseudo_labels_nonfatal_impl(
                        incremental_dataset_type=incremental_dataset_type,
                        pseudo_file=str(pseudo_label_file),
                        stage_id=int(stage_id),
                        pseudo_label_config=pseudo_label_config,
                        ann_file=str(getattr(train_dataset_cfg.dataset, 'ann_file', None) or ''),
                        logger=logger,
                        is_main_process=bool(is_main_process),
                        log_debug=bool(log_debug),
                        source_label='existing',
                    )

                # Create or update metadata file for pseudo labels tracking
                if is_main_process and pseudo_label_file.exists():
                    metadata_file = pseudo_label_dir / "pseudo_labels_metadata.json"
                    _create_pseudo_labels_metadata(
                        metadata_file=metadata_file,
                        stage_id=stage_id,
                        pseudo_label_file=pseudo_label_file,
                        source_global=copied_from_global,
                        global_source=global_source_used,
                        config_suffix=config_suffix,
                        confidence_threshold=pseudo_label_config.get('confidence_threshold', 0.45),
                        generation_checkpoint=generation_checkpoint_used,
                        logger=logger
                    )

                    ann_file = getattr(stage_cfg.data.train.dataset, 'ann_file', None)
                    if ann_file and incremental_dataset_type == 'IncrementalScanNetDataset':
                        try:
                            metrics = evaluate_pseudo_label_file_hits(
                                pseudo_file=pseudo_label_file,
                                ann_file=Path(ann_file),
                                stage_id=stage_id,
                            )
                            log_pseudo_hit_metrics(metrics, stage_id, logger)
                        except Exception as eval_err:
                            logger.warning(f"Failed to compute pseudo label hit metrics: {eval_err}")
                    elif ann_file and incremental_dataset_type == 'IncrementalSUNRGBDDataset':
                        logger.info("SUNRGBD pseudo label hit metrics: skipped (ScanNet-only diagnostic).")

            else:
                logger.info("Using on-the-fly pseudo label generation (slower but more flexible)")

        # Apply pseudo label configuration to dataset
        if use_pseudo_labels:
            train_dataset_cfg.dataset.use_pseudo_labels = use_pseudo_labels
            train_dataset_cfg.dataset.pseudo_label_config = pseudo_label_config
            if isinstance(pseudo_label_config, dict) and pseudo_label_config.get('pregenerated_file'):
                logger.info(f"Using pre-generated pseudo labels: {pseudo_label_config['pregenerated_file']}")

        # Note: No pipeline transforms needed for scene-based approach
        # Scenes are added directly to the dataset, not inserted via transforms
        if scene_memory_bank is not None and stage_idx > 0:
            logger.info(f"Scene replay enabled for stage {stage_id} (adding scenes from memory bank)")

        # Build validation dataset for evaluation
        val_dataset_cfg = copy.deepcopy(stage_cfg.data.val)

        # Use incremental dataset for evaluation to ensure consistent class ordering.
        val_dataset_cfg.type = incremental_dataset_type

        # Configure for evaluation mode - this computes all seen classes
        val_dataset_cfg.evaluation_mode = True

        # Pass necessary parameters for incremental evaluation
        val_dataset_cfg.stage_definition = stage_definition
        val_dataset_cfg.mappings = mappings
        val_dataset_cfg.all_stage_definitions = stage_definitions
        val_dataset_cfg.work_dir = stage_cfg.work_dir
        val_dataset_cfg.experiment_dir = stage_cfg.experiment_dir

        # Disable memory banks and pseudo labels for evaluation
        val_dataset_cfg.object_memory_bank = None
        val_dataset_cfg.scene_memory_bank = None
        val_dataset_cfg.use_pseudo_labels = False
        val_dataset_cfg.pseudo_label_config = None

        # Compute cumulative seen classes for logging
        all_stages_up_to_current = stage_definitions[:stage_idx+1]
        cumulative_seen_classes = []
        for s in all_stages_up_to_current:
            cumulative_seen_classes.extend(s['class_indices'])
        cumulative_seen_classes = sorted(list(set(cumulative_seen_classes)))

        # === Training control knobs (explicit) ===
        stage_epochs = int(stage_definition.get('epochs', 1))
        original_times = int(getattr(train_dataset_cfg, 'times', 1))

        # Legacy pseudo-consistency segmentation must be explicitly enabled.
        legacy_seg_cfg = {}
        try:
            legacy_seg_cfg = incremental_cfg.get('reviewing_legacy_pseudo_consistency', {}) or {}
        except Exception:
            legacy_seg_cfg = getattr(incremental_cfg, 'reviewing_legacy_pseudo_consistency', {}) or {}
        legacy_seg_enabled = bool(getattr(legacy_seg_cfg, 'get', lambda *_: False)('enabled', False))

        times_a, times_b = original_times, 0
        do_segment = bool(legacy_seg_enabled) and (stage_id >= 2) and (stage_epochs in [1, 2])
        if do_segment and stage_epochs == 1 and original_times > 1:
            # Split RepeatDataset repetitions across 2 mini-epochs.
            import math
            times_a = int(math.ceil(original_times / 2.0))
            times_b = int(max(1, original_times - times_a))

        # Reviewing (SUNRGBD GT-only) must also be explicitly enabled.
        reviewing_cfg = {}
        try:
            reviewing_cfg = incremental_cfg.get('reviewing', {}) or {}
        except Exception:
            reviewing_cfg = getattr(incremental_cfg, 'reviewing', {}) or {}
        reviewing_enabled = bool(getattr(reviewing_cfg, 'get', lambda *_: False)('enabled', False))

        # SUNRGBD forgetness eviction (global bank eviction by old-class AP drops).
        # Only computed/available when reviewing is enabled (stage-start + stage-end eval on train(memory)).
        forgetness_class_drops_for_memory_update = None
        forgetness_eviction_enabled = bool(
            scene_memory_bank is not None and
            getattr(scene_memory_bank, 'forgetness_eviction_enabled', False)
        )
        if forgetness_eviction_enabled and incremental_dataset_type in (
                'IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset'):
            assert reviewing_enabled, (
                "SceneMemoryBank forgetness eviction is enabled, but `incremental_cfg.reviewing.enabled=False`. "
                "Forgetness eviction requires reviewing stage-start/end eval."
            )

        # Learning-dynamics scoring for memory bank updates (per-scene trajectories).
        learning_dynamics_scores_for_memory_update = None
        learning_dynamics_scores_file_for_memory_update = None
        learning_dynamics_design_payload_for_memory_update = None
        learning_dynamics_bank = (
            object_memory_bank if object_memory_bank is not None
            else scene_memory_bank)
        learning_dynamics_strategy_key = str(
            getattr(learning_dynamics_bank, 'selection_strategy', '')
        ).strip().lower()
        learning_dynamics_strategy = bool(
            learning_dynamics_bank is not None and
            learning_dynamics_strategy_key == 'learning_dynamics'
        )
        learning_dynamics_design_strategy = bool(
            learning_dynamics_bank is not None and
            _is_ld_design_selection_strategy(learning_dynamics_strategy_key)
        )
        learning_dynamics_enabled = bool(
            learning_dynamics_strategy or learning_dynamics_design_strategy
        )
        ld_path_only_logging = bool(
            artifact_profile_effective == 'ld_path_only' and learning_dynamics_enabled
        )
        if learning_dynamics_enabled:
            assert incremental_dataset_type in (
                'IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset'
            ), (
                "selection_strategy in ['learning_dynamics', "
                "'learning_dynamics_design1', 'learning_dynamics_design2'] "
                "requires IncrementalSUNRGBDDataset or IncrementalScanNetDataset."
            )
            ld_strategy_name = (
                learning_dynamics_strategy_key
                if learning_dynamics_design_strategy else
                'learning_dynamics'
            )
            if not reviewing_enabled:
                logger.info(
                    f"{ld_strategy_name} memory updates: reviewing is disabled; "
                    "will use segmented training without reviewing resampling."
                )
            if ld_path_only_logging:
                logger.info(
                    "LD artifact profile ('ld_path_only'): writing only LD score->action "
                    "path artifacts (skipping auxiliary metrics/debug dumps)."
                )

        # Store validation dataset configuration for evaluation
        stage_cfg.data.val = val_dataset_cfg

        sunrgbd_reviewing_active = bool(
            reviewing_enabled
            and incremental_dataset_type in (
                'IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset'
            )
            and int(stage_id) >= 2
        )
        sunrgbd_ld_segment_active = bool(
            learning_dynamics_enabled
            and incremental_dataset_type in (
                'IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset'
            )
        )
        sunrgbd_segmented_active = _should_use_sunrgbd_segmented_path_impl(
            incremental_dataset_type=str(incremental_dataset_type),
            stage_idx=int(stage_idx),
            sunrgbd_reviewing_active=bool(sunrgbd_reviewing_active),
            learning_dynamics_enabled=bool(sunrgbd_ld_segment_active),
            legacy_seg_enabled=bool(do_segment),
        )

        datasets = []
        if not sunrgbd_segmented_active:
            # Apply Segment-A repetitions (or full times when not segmented)
            train_dataset_cfg.times = times_a
            datasets = [build_dataset(train_dataset_cfg)]

            logger.info(
                f"Datasets: train_len={len(datasets[0])} (times={times_a}), "
                f"eval_dataset={incremental_dataset_type} (evaluation_mode=True), "
                f"seen_classes={len(cumulative_seen_classes)}"
            )
            if log_debug:
                logger.info(f"  seen_class_indices={cumulative_seen_classes}")

            if len(stage_cfg.workflow) == 2:
                val_dataset = copy.deepcopy(stage_cfg.data.val)
                val_dataset.stage_definition = stage_definition
                val_dataset.mappings = mappings
                if 'dataset' in stage_cfg.data.train:
                    val_dataset.pipeline = stage_cfg.data.train.dataset.pipeline
                else:
                    val_dataset.pipeline = stage_cfg.data.train.pipeline
                val_dataset.test_mode = False
                datasets.append(build_dataset(val_dataset))

        # Set up checkpoint configuration for stage
        if stage_cfg.checkpoint_config is not None:
            # Avoid embedding full config text with tensor() objects in checkpoint meta
            # to prevent mmcv Config.fromstring import errors on resume.
            classes_for_meta = None
            try:
                # Prefer config-provided class names (dataset defines CLASSES).
                cfg_classes = getattr(train_dataset_cfg.dataset, 'classes', None)
                if cfg_classes:
                    classes_for_meta = tuple(cfg_classes)
            except Exception:
                classes_for_meta = None
            if not classes_for_meta:
                # Fallback: use mapping-derived names.
                try:
                    classes_for_meta = tuple(
                        mappings['model_idx_to_name'].get(i, f"class_{i}")
                        for i in range(int(num_classes))
                    )
                except Exception:
                    classes_for_meta = tuple()
            stage_cfg.checkpoint_config.meta = dict(
                CLASSES=classes_for_meta,
                stage_id=stage_id,
                stage_name=stage_name,
                stage_classes=stage_classes)

        # Add classes to model for evaluation
        model.CLASSES = classes_for_meta

        if log_debug:
            logger.info(
                f"Evaluation classes (mask indices): "
                f"{torch.where(stage_cfg.evaluation_classes_mask)[0].tolist()}"
            )

        # Pseudo labels sanity check (do NOT assume `datasets[0]` exists; reviewing mode builds datasets later).
        #
        # For SUNRGBD, pseudo labels are loaded + injected inside IncrementalSUNRGBDDataset.__init__
        # via `pseudo_label_config['pregenerated_file']`. This block only validates the file and
        # logs a summary; it does not mutate the dataset.
        if bool(use_pseudo_labels) and stage_idx > 0:
            logger.info(f"Pseudo labels: validating pre-generated file for Stage {stage_id}")

            try:
                # Determine the same config suffix used for generation.
                config_suffix = ""
                if use_scene_memory and scene_memory_config:
                    memory_id = scene_memory_config.get('score_criteria', 'default')
                    config_suffix = f"pseudo_memory_{memory_id}"
                else:
                    config_suffix = "pseudo_only"

                conf_str = f"conf{int(pseudo_label_config.get('confidence_threshold', 0.45) * 100):02d}"
                config_suffix = f"{config_suffix}_{conf_str}"

                # Resolve source file: prefer user-provided, otherwise work_dir naming convention.
                user_file = (
                    pseudo_label_config.get('pregenerated_file')
                    if isinstance(pseudo_label_config, dict) else None
                )
                if user_file and os.path.exists(user_file):
                    source_file = user_file
                else:
                    pseudo_label_dir = Path(work_dir) / "pseudo_labels"
                    source_file = str(
                        pseudo_label_dir / f"stage_{stage_id}_{config_suffix}_pseudo_labels.pkl"
                    )

                if not os.path.exists(source_file):
                    raise FileNotFoundError(
                        f"Pre-generated pseudo labels not found. Tried: {user_file or 'N/A'}, {source_file}\n"
                        f"Stage {stage_id} training requires pseudo labels to be pre-generated.\n"
                        f"Ensure pre-generation completed successfully before training."
                    )

                # If we already built the dataset, prefer its in-memory counters to avoid double-loading the file.
                if datasets:
                    inner = get_innermost_dataset(datasets[0])
                    injected_scenes = getattr(inner, 'pseudo_injected_scene_count', None)
                    injected_boxes = getattr(inner, 'pseudo_injected_box_count', None)
                    if injected_scenes is not None and injected_boxes is not None:
                        logger.info(
                            f"Pseudo labels injected (dataset init): scenes={int(injected_scenes)}, "
                            f"boxes={int(injected_boxes)}"
                        )
                    else:
                        logger.info(f"Pseudo label file: {source_file}")
                else:
                    # Reviewing path: dataset is built later (possibly multiple segments). Validate file loadability.
                    logger.info(f"Pseudo label file: {source_file}")
                    with open(source_file, 'rb') as f:
                        pseudo_payload = pickle.load(f)
                    if not isinstance(pseudo_payload, dict):
                        raise ValueError(
                            f"Pseudo label file must contain a dict, got {type(pseudo_payload)}: {source_file}"
                        )
                    meta = pseudo_payload.get('__meta__', None)
                    n_keys = int(len(pseudo_payload))
                    n_scenes = int(n_keys - (1 if '__meta__' in pseudo_payload else 0))
                    if isinstance(meta, dict):
                        prev_n = meta.get('prev_head_n_classes', None)
                        ckpt = meta.get('checkpoint_used', None)
                        thr = meta.get('confidence_threshold', None)
                        logger.info(
                            f"Pseudo label payload: scenes={n_scenes}, keys={n_keys}, "
                            f"prev_head_n_classes={prev_n}, conf_thr={thr}, ckpt={ckpt}"
                        )
                    else:
                        logger.info(f"Pseudo label payload: scenes={n_scenes}, keys={n_keys}")

            except FileNotFoundError:
                raise
            except Exception as e:
                raise RuntimeError(
                    f"Failed to validate pre-generated pseudo labels for stage {stage_id}: {e}\n"
                    f"Pre-generation may have failed or files may be corrupted.\n"
                    f"Please check pseudo label generation and try again."
                ) from e

        # Sanity check: Run evaluation before training for Stage 2+ to catch issues early (rank0 only)
        if is_main_process and stage_idx > 0:  # Only for Stage 2 and later (not Stage 1)
            logger.info(f"Sanity check: pre-training evaluation for Stage {stage_id}")

            try:
                from mmdet3d.apis import single_gpu_test
                from mmcv.parallel import MMDataParallel
                from mmdet.datasets import build_dataloader

                # Build validation dataset (using build_dataset from mmdet3d.datasets already imported)
                val_dataset = build_dataset(stage_cfg.data.val)
                val_dataloader = build_dataloader(
                    val_dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=stage_cfg.data.workers_per_gpu,
                    dist=False,
                    shuffle=False
                )

                # Move model to GPU and wrap with MMDataParallel for proper device handling
                if log_debug:
                    logger.info("Sanity check: moving model to GPU (MMDataParallel)")
                model = MMDataParallel(model.cuda(), device_ids=[0])

                # Run inference on validation set
                if log_debug:
                    logger.info(
                        f"Sanity check: running inference on {len(val_dataset)} validation scenes"
                    )
                model.eval()
                results = single_gpu_test(model, val_dataloader, show=False)

                # Run evaluation
                if log_debug:
                    logger.info("Sanity check: evaluating results")
                eval_kwargs = stage_cfg.get('evaluation', {}).copy()
                eval_kwargs.pop('interval', None)
                eval_kwargs.pop('save_best', None)
                try:
                    eval_res = val_dataset.evaluate(
                        results, **eval_kwargs, eval_purpose='sanity_check'
                    )
                except TypeError:
                    # Not all dataset.evaluate implementations accept our extra kwarg.
                    eval_res = val_dataset.evaluate(results, **eval_kwargs)

                # Log sanity check results
                logger.info("Sanity check passed: evaluation pipeline working correctly")
                if log_debug:
                    logger.info("Sanity check metrics (before training):")
                    for key, value in eval_res.items():
                        if isinstance(value, float):
                            logger.info(f"  {key}: {value:.4f}")
                else:
                    key_metrics = []
                    for k in ('mAP_0.25', 'mAP_0.50', 'mAR_0.25', 'mAR_0.50'):
                        v = eval_res.get(k, None)
                        if isinstance(v, float):
                            key_metrics.append(f"{k}={v:.4f}")
                    if key_metrics:
                        logger.info("Sanity check metrics: " + ", ".join(key_metrics))

                # CRITICAL: Unwrap model after sanity check before training
                if hasattr(model, 'module'):
                    model = model.module
                    if log_debug:
                        logger.info("Sanity check: unwrapped model after evaluation")

            except Exception as e:
                logger.error(f"Sanity check failed: {str(e)}")
                logger.error("The evaluation pipeline has issues that need to be fixed")
                logger.error("Stopping to prevent wasted training time")
                raise RuntimeError(f"Pre-training evaluation sanity check failed for Stage {stage_id}") from e

        # Train with possible segmentation / SUNRGBD reviewing / learning-dynamics
        if sunrgbd_segmented_active:
            seg_mode_label = _segmented_mode_label_impl(
                sunrgbd_reviewing_active=bool(sunrgbd_reviewing_active),
                learning_dynamics_enabled=bool(learning_dynamics_enabled),
            )
            assert learning_dynamics_bank is not None, (
                "SUNRGBD segmented learning-dynamics training requires a memory bank."
            )
            assert not do_segment, (
                "Refusing to run SUNRGBD segmented training (reviewing/learning-dynamics) together with "
                "legacy pseudo-consistency segmentation. Disable `reviewing_legacy_pseudo_consistency.enabled`."
            )

            review_fractions = list(reviewing_cfg.get('review_fractions', []) or [])
            if sunrgbd_reviewing_active:
                logger.info("SUNRGBD Reviewing: ENABLED (GT-only on memory bank)")
            else:
                logger.info(
                    "SUNRGBD Learning-dynamics: ENABLED (segmented training; reviewing disabled; no resampling)"
                )
                if not review_fractions:
                    review_fractions = [0.2, 0.4, 0.6, 0.8]
                    logger.info(
                        "  - review_fractions not set; using default "
                        f"{review_fractions} for learning-dynamics segmentation"
                    )
            assert review_fractions, 'review_fractions must be non-empty for SUNRGBD segmented training.'
            logger.info(f"  - review_fractions={review_fractions}")
            logger.info(f"  - stage_epochs={stage_epochs}, data.train.times={original_times}")

            # Reviewing evaluation always reports AP/AR at these IoU thresholds.
            eval_iou_thrs = reviewing_cfg.get('eval_iou_thrs', None)
            if eval_iou_thrs is None:
                eval_iou_thrs = [0.25, 0.5]
            if isinstance(eval_iou_thrs, (int, float)):
                eval_iou_thrs = [float(eval_iou_thrs)]
            eval_iou_thrs = sorted({float(x) for x in (eval_iou_thrs or [])})
            assert eval_iou_thrs, 'reviewing.eval_iou_thrs must be non-empty.'

            # Which IoU is used to derive reviewing sampling weights.
            # - Legacy policy derives weights from AP@thr drops.
            # - LD policy derives weights from q drops at IoU τ.
            weight_iou_thr = reviewing_cfg.get('weight_iou_thr', None)
            if weight_iou_thr is None:
                weight_iou_thr = reviewing_cfg.get('eval_iou_thr', 0.25)  # legacy alias
            weight_iou_key = f"{float(weight_iou_thr):.2f}"
            eval_iou_keys = [f"{float(x):.2f}" for x in eval_iou_thrs]
            assert weight_iou_key in eval_iou_keys, (
                f"reviewing.weight_iou_thr={float(weight_iou_thr):.2f} must be in "
                f"reviewing.eval_iou_thrs={eval_iou_thrs}."
            )
            weight_iou_thr = float(weight_iou_key)

            # Weight policy (tunable via config).
            wp = dict(reviewing_cfg.get('weight_policy', {}) or {})
            wp_type_raw = wp.get('type', 'drop_dominant_sum')
            wp_type = str(wp_type_raw).strip().lower()
            if wp_type in ('', 'drop_dominant_sum', 'ap_drop', 'legacy'):
                reviewing_weight_policy = 'ap_drop'
            elif wp_type in ('ld_f1_drop', 'ld_drop', 'f1_drop'):
                reviewing_weight_policy = 'ld_drop'
            elif wp_type in ('fixed', 'constant'):
                reviewing_weight_policy = 'fixed'
            else:
                raise ValueError(
                    "Invalid reviewing.weight_policy.type. "
                    "Supported: ['drop_dominant_sum' (legacy AP-drop), "
                    "'ld_drop' (LD q drop), 'fixed' (constant per-seat weight); "
                    "legacy aliases: 'ld_f1_drop', 'f1_drop', 'constant']. "
                    f"Got: {wp_type_raw}"
                )

            if reviewing_weight_policy == 'ap_drop':
                derive_note = "derive weights from AP"
            elif reviewing_weight_policy == 'ld_drop':
                derive_note = "derive weights from LD q drop"
            else:
                derive_note = "use constant per-seat reviewing weight"
            logger.info(
                f"  - eval_iou_thrs={eval_iou_thrs} (report AP/AR), "
                f"weight_iou_thr={weight_iou_thr:.2f} ({derive_note})"
            )

            # Legacy AP-drop parameters.
            alpha_drop = float(wp.get('alpha_drop', 1.0))
            beta_ap = float(wp.get('beta_ap', 0.1))
            gamma = float(wp.get('gamma', 5.0))
            w_max = float(wp.get('w_max', 10.0))
            drop_clamp_min = float(reviewing_cfg.get('drop_clamp_min', 0.0))
            fixed_review_weight = None
            if reviewing_weight_policy == 'fixed':
                fixed_review_weight = float(wp.get('fixed_value', None))
                if not np.isfinite(float(fixed_review_weight)):
                    raise ValueError(
                        "reviewing.weight_policy.fixed_value must be finite when "
                        "reviewing.weight_policy.type='fixed'."
                    )
                if float(fixed_review_weight) < 1.0:
                    raise ValueError(
                        "reviewing.weight_policy.fixed_value must be >= 1.0 when "
                        "reviewing.weight_policy.type='fixed'."
                    )

            # Shared parameter name: eta (entry weight scale).
            default_eta = 5.0 if reviewing_weight_policy == 'ld_drop' else 1.0
            eta = float(wp.get('eta', default_eta))

            # LD q-drop parameters (resolved later if LD scoring is enabled).
            reviewing_ld_normalize_by_gt_weight = bool(wp.get('normalize_by_gt_weight', True))
            reviewing_ld_object_count_cap = wp.get('object_count_cap', None)
            reviewing_ld_w_entry_max = wp.get('w_entry_max', None)

            if reviewing_weight_policy == 'ap_drop':
                logger.info(
                    "  - weight_policy(drop_dominant_sum): "
                    f"alpha_drop={alpha_drop}, beta_ap={beta_ap}, gamma={gamma}, "
                    f"w_max={w_max}, eta={eta}, drop_clamp_min={drop_clamp_min}"
                )
            elif reviewing_weight_policy == 'ld_drop':
                logger.info(
                    "  - weight_policy(ld_drop): "
                    f"q_metric=resolved_from_ld_config(required, canonical default=recall), eta={eta}, "
                    f"normalize_by_gt_weight={reviewing_ld_normalize_by_gt_weight}, "
                    f"object_count_cap={reviewing_ld_object_count_cap}, w_entry_max={reviewing_ld_w_entry_max}"
                )
            else:
                logger.info(
                    "  - weight_policy(fixed): "
                    f"fixed_value={float(fixed_review_weight):.3f}"
                )

            samp = dict(reviewing_cfg.get('sampling', {}) or {})
            # Coverage-preserving reviewing is the only supported mode.
            sampling_mode = 'coverage_preserving'
            memory_share_max = float(samp.get('memory_share_max', 0.9))
            assert 0.0 < memory_share_max <= 1.0, memory_share_max
            seed_offset = int(samp.get('seed_offset', 9000))
            strict_memory_coverage = bool(
                samp.get('strict_memory_coverage', True)
            )

            # Build stage->class_indices map.
            stage_to_classes = {
                int(sd.get('stage_id')): [int(x) for x in sd.get('class_indices', [])]
                for sd in stage_definitions
                if sd.get('stage_id') is not None
            }

            # Collect memory seats once (fixed within the stage).
            if (object_memory_bank is not None and sunrgbd_reviewing_active
                    and not isinstance(getattr(
                        getattr(object_memory_bank, 'dataset_ref', None),
                        '_object_scene_info_by_id', None), dict)):
                # A bank loaded with --object-memory-path deliberately does not
                # persist its source dataset. Reattach read-only metadata from
                # the current dataset config before evaluating carrier scenes.
                source_cfg = copy.deepcopy(train_dataset_cfg)
                source_cfg.times = 1
                source_cfg.dataset.scene_memory_bank = None
                _detach_object_memory_insertion(source_cfg.dataset)
                source_cfg.dataset.object_memory_bank = object_memory_bank
                source_cfg.dataset.reviewing_sampling = dict(enabled=False)
                source_ds = build_dataset(source_cfg)
                source_inner = get_innermost_dataset(source_ds)
                object_memory_bank.dataset_ref = source_inner
            mem_entries = (
                scene_memory_bank.list_memory_entries(
                    max_save_stage=int(stage_id) - 1)
                if scene_memory_bank is not None else
                object_memory_bank.list_source_scene_entries(
                    max_save_stage=int(stage_id) - 1)
                if (object_memory_bank is not None
                    and sunrgbd_reviewing_active) else [])
            mem_entries = [e for e in mem_entries if int(e.get('save_stage', 0)) < int(stage_id)]
            if sunrgbd_reviewing_active:
                assert mem_entries, (
                    f"SUNRGBD reviewing is enabled at stage {stage_id}, but memory bank is empty."
                )
            elif (learning_dynamics_enabled and scene_memory_bank is not None
                  and int(stage_id) >= 2):
                assert mem_entries, (
                    "selection_strategy in ['learning_dynamics', "
                    "'learning_dynamics_design1', 'learning_dynamics_design2'] "
                    f"requires previous-stage memory seats at stage {stage_id}, "
                    "but memory bank is empty."
                )
            elif learning_dynamics_enabled and not mem_entries:
                logger.info(
                    "SUNRGBD Learning-dynamics: no scene-memory seats to evaluate "
                    "(expected for object selection); continuing with natural-pool-only tracking."
                )

            mem_infos_by_stage = {}
            # Memory "seats" are (scene_id, save_stage) pairs (structured; no string keys).
            mem_seat_keys_by_stage = {}
            for e in mem_entries:
                s = int(e.get('save_stage', 0))
                snap = e.get('snapshot', {}) or {}
                info = snap.get('data_info', None)
                if not isinstance(info, dict):
                    continue
                mem_infos_by_stage.setdefault(s, []).append(info)
                mem_seat_keys_by_stage.setdefault(s, []).append(dict(
                    scene_id=str(e.get('scene_id')),
                    save_stage=int(s),
                ))

            # Build class name list aligned with current head size.
            try:
                target_model = model.module if hasattr(model, 'module') else model
                n_cls = int(getattr(getattr(target_model, 'head', None), 'n_classes', stage_cfg.model.head.n_classes))
            except Exception:
                n_cls = int(stage_cfg.model.head.n_classes)
            class_names = [mappings['model_idx_to_name'][i] for i in range(int(n_cls))]

            # Create output directories for reviewing artifacts.
            # - `review_actions_dir`: replay-resampling actions (counts, composition).
            # - `review_weights_dir`: per-seat weights used for resampling.
            review_actions_dir = incremental_cfg.paths.reviewing_actions_dir() / f"stage_{stage_id}"
            review_actions_dir.mkdir(parents=True, exist_ok=True)
            review_weights_dir = incremental_cfg.paths.reviewing_weights_dir() / f"stage_{stage_id}"
            review_weights_dir.mkdir(parents=True, exist_ok=True)

            # Learning-dynamics scoring artifacts (optional; config-gated).
            ld_dir = None
            ld_cfg = {}
            ld_iou_mode = None
            ld_iou_thr = 0.50
            ld_eps = 1e-9
            ld_alpha = 1.0
            ld_beta = 1.0
            ld_object_count_cap = 20
            ld_report_topk = 30
            ld_slope_k_start_cfg = None
            ld_slope_k_end_cfg = None
            ld_design1_q_metric = 'f1'
            ld_stats_q_metric = 'f1'
            ld_stats_q_formula = '2TP/(2TP+FP+FN+eps)'
            ld_replay_priority_policy = dict(
                type='slow_saturation',
                delta=0.002,
                tau_q=0.02,
                use_competence=True,
                slow_factor='centroid',
            )
            ld_replay_priority_policy_type = 'slow_saturation'

            # Precompute class splits for this stage.
            ld_new_classes = [int(x) for x in stage_definition.get('class_indices', [])]
            ld_old_classes = []
            for sd in stage_definitions:
                try:
                    if int(sd.get('stage_id', 0)) < int(stage_id):
                        ld_old_classes.extend([int(x) for x in sd.get('class_indices', [])])
                except Exception:
                    continue
            ld_old_classes = sorted(set(ld_old_classes))

            if learning_dynamics_enabled:
                ld_dir = incremental_cfg.paths.learning_dynamics_dir() / f"stage_{stage_id}"
                ld_dir.mkdir(parents=True, exist_ok=True)
                ld_cfg = getattr(
                    learning_dynamics_bank, 'learning_dynamics_update', {}) or {}
                # Disallow legacy/aliased knobs (kept explicit to avoid confusion).
                unsupported_ld_keys = [
                    # Legacy slope window knobs (replaced by (k_start,k_end)).
                    'slope_window',
                    'slope_window_L',
                    # Legacy pseudo/weighting knobs (learning-dynamics scoring is GT-only for now).
                    'label_source_policy',
                    'pseudo_conf_thresh',
                    'pseudo_confidence_threshold',
                    'pseudo_weight_gamma',
                    'pseudo_gamma',
                    # Legacy naming.
                    'topk',
                ]
                for k in unsupported_ld_keys:
                    if k in ld_cfg:
                        raise ValueError(
                            f"learning_dynamics_update.{k} is not supported. "
                            "Use learning_dynamics_update.slope_k_start/slope_k_end and GT-only scoring."
                        )

                # IoU threshold for learning-dynamics scoring (single threshold).
                scoring_cfg = {}
                try:
                    scoring_cfg = stage_cfg.get('SCORING', {}) or {}
                except Exception:
                    scoring_cfg = {}
                ld_iou_mode = (
                    scoring_cfg.get('LD_IOU_MODE', None)
                    if isinstance(scoring_cfg, dict) else None
                )
                if ld_iou_mode is None:
                    ld_iou_mode = ld_cfg.get('iou_mode', ld_cfg.get('LD_IOU_MODE', None))

                # Legacy: ld_cfg.iou_thr used to be a float (deprecated but supported).
                if ld_iou_mode is None and 'iou_thr' in ld_cfg:
                    try:
                        ld_iou_mode = f"{float(ld_cfg.get('iou_thr')):.2f}"
                    except Exception as e:
                        raise ValueError(
                            "learning_dynamics_update.iou_thr must be a float IoU threshold."
                        ) from e
                    if is_main_process:
                        logger.warning(
                            "learning_dynamics_update.iou_thr is deprecated. "
                            "Please set SCORING.LD_IOU_MODE instead."
                        )

                if ld_iou_mode is None:
                    ld_iou_mode = '0.50'
                ld_iou_mode = str(ld_iou_mode).strip()

                # Normalize to a supported IoU threshold.
                allowed_ld_iou_thrs = (0.25, 0.50, 0.75, 0.80, 0.90)
                if 'avg' in ld_iou_mode.lower():
                    raise ValueError(
                        "SCORING.LD_IOU_MODE averaging (e.g. 'avg_0.25_0.50') is no longer supported. "
                        "Use a single IoU threshold in "
                        "['0.25', '0.50', '0.75', '0.80', '0.90']."
                    )
                try:
                    ld_iou_thr = float(ld_iou_mode.replace('_', '.'))
                except Exception as e:
                    raise ValueError(
                        "Invalid SCORING.LD_IOU_MODE. Expected one of "
                        "['0.25', '0.50', '0.75', '0.80', '0.90'], "
                        f"got '{ld_iou_mode}'."
                    ) from e
                matched = None
                for a in allowed_ld_iou_thrs:
                    if abs(float(ld_iou_thr) - float(a)) < 1e-6:
                        matched = float(a)
                        break
                if matched is None:
                    raise ValueError(
                        "Invalid SCORING.LD_IOU_MODE. Expected one of "
                        "['0.25', '0.50', '0.75', '0.80', '0.90'], "
                        f"got '{ld_iou_mode}'."
                    )
                ld_iou_thr = float(matched)
                ld_iou_mode = f"{float(ld_iou_thr):.2f}"

                ld_eps = float(ld_cfg.get('eps', 1e-9))
                if ld_eps <= 0.0:
                    ld_eps = 1e-9

                # alpha/beta were used by the legacy recall q; keep for config compatibility only.
                ld_alpha = float(ld_cfg.get('alpha', 1.0))
                ld_beta = float(ld_cfg.get('beta', 1.0))
                ld_object_count_cap = int(ld_cfg.get('object_count_cap', 20))
                assert ld_object_count_cap > 0, ld_object_count_cap
                ld_report_topk = int(ld_cfg.get('report_topk', 30))
                ld_report_topk = max(5, min(500, ld_report_topk))
                if learning_dynamics_design_strategy:
                    ld_block_key = (
                        'learning_dynamics_design2'
                        if learning_dynamics_strategy_key == LD_DESIGN2_STRATEGY
                        else 'learning_dynamics_design1'
                    )
                    if not hasattr(learning_dynamics_bank, 'learning_dynamics_design1_q_metric'):
                        raise RuntimeError(
                            f"{ld_block_key}.q_metric missing in memory bank. "
                            "Configure q_metric explicitly in the active memory config."
                        )
                    ld_design1_q_metric = str(
                        getattr(learning_dynamics_bank, 'learning_dynamics_design1_q_metric')
                    ).strip().lower()
                    if ld_design1_q_metric not in ('f1', 'recall'):
                        raise ValueError(
                            f"{ld_block_key}.q_metric must be one of "
                            "['f1', 'recall']."
                        )
                    ld_stats_q_metric = str(ld_design1_q_metric)
                    # Design version (1 or 2) for class-need aggregation.
                    ld_design_version = int(
                        getattr(learning_dynamics_bank, 'learning_dynamics_design_version', 1)
                    )
                    ld_stats_q_formula = (
                        'TP/(TP+FN+eps)'
                        if str(ld_stats_q_metric) == 'recall'
                        else '2TP/(2TP+FP+FN+eps)'
                    )
                    if is_main_process:
                        logger.info(
                            f"Learning-dynamics {learning_dynamics_strategy_key} scoring: ENABLED "
                            f"(iou_thr={ld_iou_mode}, eps={ld_eps:.1e}, "
                            f"object_count_cap={ld_object_count_cap}, "
                            f"q_metric={ld_design1_q_metric})"
                        )
                        if int(ld_design_version) >= 2:
                            logger.info(
                                "Design-2 overrides: "
                                f"supply_scaling={getattr(learning_dynamics_bank, 'learning_dynamics_design1_supply_scaling_mode', 'raw')}, "
                                f"w_max={getattr(learning_dynamics_bank, 'learning_dynamics_design2_w_max', 10.0)}, "
                                f"redundancy_lambda={getattr(learning_dynamics_bank, 'design2_redundancy_lambda', getattr(learning_dynamics_bank, 'learning_dynamics_design2_redundancy_lambda', 0.3))}, "
                                f"redundancy_topk={getattr(learning_dynamics_bank, 'design2_redundancy_topk', getattr(learning_dynamics_bank, 'learning_dynamics_design2_redundancy_topk', 5))}"
                            )
                else:
                    ld_stats_q_metric = 'f1'
                    ld_stats_q_formula = '2TP/(2TP+FP+FN+eps)'
                    # Replay-priority policy for new-class natural seats.
                    replay_policy_cfg = ld_cfg.get('replay_priority_policy', {}) or {}
                    if not isinstance(replay_policy_cfg, dict):
                        raise ValueError(
                            "learning_dynamics_update.replay_priority_policy must be a dict."
                        )
                    from mmdet3d.utils.learning_dynamics_scoring import (
                        normalize_replay_priority_policy_type,
                    )
                    ld_replay_priority_policy_type = normalize_replay_priority_policy_type(
                        replay_policy_cfg.get('type', 'slow_saturation')
                    )
                    if ld_replay_priority_policy_type == 'slow_saturation':
                        rp_delta = replay_policy_cfg.get('delta', 0.002)
                        rp_tau_q = replay_policy_cfg.get('tau_q', 0.02)
                        rp_use_competence_raw = replay_policy_cfg.get(
                            'use_competence', True
                        )
                        if isinstance(rp_use_competence_raw, bool):
                            rp_use_competence = bool(rp_use_competence_raw)
                        elif (
                            isinstance(rp_use_competence_raw, (int, float))
                            and float(rp_use_competence_raw) in (0.0, 1.0)
                        ):
                            rp_use_competence = bool(int(rp_use_competence_raw))
                        else:
                            raise ValueError(
                                "learning_dynamics_update.replay_priority_policy."
                                "use_competence must be bool."
                            )
                        rp_slow_factor = str(
                            replay_policy_cfg.get('slow_factor', 'centroid')
                        ).strip().lower()
                        if rp_slow_factor != 'centroid':
                            raise ValueError(
                                "learning_dynamics_update.replay_priority_policy.slow_factor "
                                "must be 'centroid' (current supported option)."
                            )
                        ld_replay_priority_policy = dict(
                            type='slow_saturation',
                            delta=float(rp_delta),
                            tau_q=float(rp_tau_q),
                            use_competence=bool(rp_use_competence),
                            slow_factor=str(rp_slow_factor),
                        )
                        if float(ld_replay_priority_policy['delta']) < 0.0:
                            raise ValueError(
                                "learning_dynamics_update.replay_priority_policy."
                                f"delta must be >= 0, got {ld_replay_priority_policy['delta']}."
                            )
                        if not (0.0 <= float(ld_replay_priority_policy['tau_q']) <= 1.0):
                            raise ValueError(
                                "learning_dynamics_update.replay_priority_policy."
                                f"tau_q must be in [0,1], got {ld_replay_priority_policy['tau_q']}."
                            )
                    else:
                        ld_replay_priority_policy = dict(type='legacy_between')

                    # Optional ablations: choose the slope interval by k-indices.
                    # k=0 is stage-start; k=1..K are in-training evaluations.
                    ld_slope_k_start_cfg = ld_cfg.get('slope_k_start', None)
                    ld_slope_k_end_cfg = ld_cfg.get('slope_k_end', None)
                    if (
                        ld_replay_priority_policy_type == 'legacy_between'
                        and ((ld_slope_k_start_cfg is None) ^
                             (ld_slope_k_end_cfg is None))
                    ):
                        raise ValueError(
                            "learning_dynamics_update.slope_k_start and slope_k_end must be set together."
                        )
                    if (
                        ld_replay_priority_policy_type == 'legacy_between'
                        and ld_slope_k_start_cfg is not None
                    ):
                        ld_slope_k_start_cfg = int(ld_slope_k_start_cfg)
                        ld_slope_k_end_cfg = int(ld_slope_k_end_cfg)

                    if is_main_process:
                        logger.info(
                            "Learning-dynamics scoring: ENABLED "
                            f"(iou_thr={ld_iou_mode}, eps={ld_eps:.1e}, "
                            f"object_count_cap={ld_object_count_cap}, "
                            f"report_topk={ld_report_topk}, "
                            f"replay_priority_policy={ld_replay_priority_policy})"
                        )

            # Consistency guard: avoid mixing LD memory updates with legacy AP-drop reviewing.
            if learning_dynamics_enabled and sunrgbd_reviewing_active:
                from mmdet3d.utils.learning_dynamics_scoring import (
                    validate_sunrgbd_ld_reviewing_design_consistency,
                )
                validate_sunrgbd_ld_reviewing_design_consistency(
                    learning_dynamics_selection=True,
                    reviewing_enabled=True,
                    reviewing_weight_policy_type=str(reviewing_weight_policy),
                    ld_iou_mode=str(ld_iou_mode),
                    reviewing_weight_iou_thr=float(weight_iou_thr),
                )
                if reviewing_ld_object_count_cap is not None:
                    try:
                        cap_override = int(reviewing_ld_object_count_cap)
                    except Exception:
                        cap_override = None
                    if cap_override is not None and cap_override != int(ld_object_count_cap):
                        logger.warning(
                            "LD/reviewing consistency: "
                            f"reviewing.weight_policy.object_count_cap={cap_override} differs from "
                            f"learning_dynamics_update.object_count_cap={int(ld_object_count_cap)}."
                        )

            # Reviewing LD policy tracks the LD q_metric for consistent weighting metadata.
            reviewing_ld_q_metric = 'f1'
            if learning_dynamics_design_strategy:
                reviewing_ld_q_metric = str(ld_design1_q_metric)
            reviewing_ld_q_formula = (
                'TP/(TP+FN+eps)'
                if str(reviewing_ld_q_metric) == 'recall'
                else '2TP/(2TP+FP+FN+eps)'
            )

            # Build natural pool snapshot for scoring (rank0 only; fixed across k).
            ld_natural_infos = None
            ld_natural_seat_keys = None

            def _ld_scene_id_from_info(info: Dict[str, Any]) -> Optional[str]:
                if not isinstance(info, dict):
                    return None
                if 'point_cloud' in info and isinstance(info.get('point_cloud'), dict):
                    lidar_idx = info['point_cloud'].get('lidar_idx', None)
                    if lidar_idx is not None:
                        return str(lidar_idx)
                if 'sample_idx' in info and info.get('sample_idx', None) is not None:
                    return str(info['sample_idx'])
                if 'scene_id' in info and info.get('scene_id', None) is not None:
                    return str(info['scene_id'])
                return None

            if learning_dynamics_enabled and is_main_process:
                try:
                    natural_cfg = copy.deepcopy(train_dataset_cfg)
                    natural_cfg.times = 1
                    natural_cfg.dataset.scene_memory_bank = None
                    _detach_object_memory_insertion(natural_cfg.dataset)
                    natural_cfg.dataset.scene_dedup_strategy = 'keep_both'
                    natural_cfg.dataset.reviewing_sampling = dict(enabled=False)
                    # Learning-dynamics scoring uses GT-only (no pseudo labels).
                    natural_cfg.dataset.use_pseudo_labels = False
                    natural_cfg.dataset.pseudo_label_config = None
                except Exception:
                    natural_cfg = None

                if natural_cfg is None:
                    raise RuntimeError("Learning-dynamics scoring: failed to build natural_cfg for scoring.")

                natural_ds = build_dataset(natural_cfg)
                natural_inner = get_innermost_dataset(natural_ds)
                infos_all = list(getattr(natural_inner, 'data_infos', []) or [])
                infos_all = [
                    info for info in infos_all
                    if isinstance(info, dict) and not bool(info.get('is_replay', False)) and not bool(info.get('is_merged', False))
                ]
                ld_natural_infos = infos_all
                ld_natural_seat_keys = []
                for info in ld_natural_infos:
                    sid = _ld_scene_id_from_info(info)
                    if sid is None:
                        sid = str(info.get('scene_id', ''))
                    ld_natural_seat_keys.append(dict(scene_id=str(sid), save_stage=int(stage_id)))

                # Persist the evaluated natural pool seats for reproducibility/debugging.
                if not ld_path_only_logging:
                    try:
                        with open(ld_dir / "natural_seats.json", "w") as f:
                            json.dump(list(ld_natural_seat_keys), f, indent=2)
                    except Exception:
                        pass

            def _compute_ld_match_stats(
                    raw_gt_annos: List[Dict[str, Any]],
                    raw_dt_annos: List[Dict[str, Any]],
                    *,
                    seat_keys: List[Dict[str, Any]],
                    eval_class_indices: List[int],
                    box_type_3d,
                    box_mode_3d):
                """Compute per-seat per-class TP/FP/FN stats for LD scoring.

                Matching uses IoU >= SCORING.LD_IOU_MODE (single threshold),
                and emits per-seat `q` using the resolved LD q metric.
                """
                from mmdet3d.core.evaluation.scene_class_stats import (
                    compute_scene_class_match_stats,
                )

                stats = compute_scene_class_match_stats(
                    list(raw_gt_annos),
                    list(raw_dt_annos),
                    seat_keys=list(seat_keys),
                    eval_class_indices=list(eval_class_indices),
                    iou_thr=float(ld_iou_thr),
                    box_type_3d=box_type_3d,
                    box_mode_3d=box_mode_3d,
                    q_metric=str(ld_stats_q_metric),
                    eps=float(ld_eps),
                )
                return list(stats or [])

            # --- Stage-start baseline evaluation on memory bank (rank0 only) ---
            stage_start_ap_forgetness = None  # class_idx -> AP@weight_iou_thr (stage start)
            forgetness_drop_file = (
                incremental_cfg.paths.memory_bank_scores_dir()
                / f"forgetness_start_end_class_drops_stage_{stage_id}.json"
            )
            prev_ap = {}  # class_idx -> AP@weight_iou_thr
            prev_ld_mem_stats_for_reviewing = None  # seat stats snapshot at k-1 (for ld_drop)
            if is_main_process:
                if sunrgbd_reviewing_active:
                    if reviewing_weight_policy == 'ld_drop':
                        assert learning_dynamics_enabled, (
                            "reviewing.weight_policy.type='ld_drop' requires "
                            "scene_memory_config.selection_strategy in "
                            "['learning_dynamics', 'learning_dynamics_design1', "
                            "'learning_dynamics_design2'] (LD enabled)."
                        )
                    logger.info("SUNRGBD Reviewing: stage-start evaluation on memory bank")
                    effective_ld_cap, effective_w_entry_max = _resolve_effective_ld_reviewing_params_impl(
                        reviewing_ld_object_count_cap=reviewing_ld_object_count_cap,
                        reviewing_ld_w_entry_max=reviewing_ld_w_entry_max,
                        ld_object_count_cap=ld_object_count_cap,
                        learning_dynamics_enabled=bool(learning_dynamics_enabled),
                    )
                    weight_metric, weight_policy_desc = _build_review_weight_policy_impl(
                        reviewing_weight_policy=str(reviewing_weight_policy),
                        alpha_drop=float(alpha_drop),
                        beta_ap=float(beta_ap),
                        gamma=float(gamma),
                        w_max=float(w_max),
                        eta=float(eta),
                        fixed_review_weight=fixed_review_weight,
                        drop_clamp_min=float(drop_clamp_min),
                        reviewing_ld_q_metric=str(reviewing_ld_q_metric),
                        reviewing_ld_q_formula=str(reviewing_ld_q_formula),
                        ld_iou_thr=float(ld_iou_thr),
                        ld_iou_mode=str(ld_iou_mode),
                        ld_eps=float(ld_eps),
                        effective_ld_cap=int(effective_ld_cap),
                        reviewing_ld_normalize_by_gt_weight=bool(
                            reviewing_ld_normalize_by_gt_weight
                        ),
                        effective_w_entry_max=effective_w_entry_max,
                    )
                    review0 = _build_reviewing_eval_payload_impl(
                        stage_id=int(stage_id),
                        review_k=0,
                        eval_iou_thrs=[float(x) for x in eval_iou_thrs],
                        weight_iou_thr=float(weight_iou_thr),
                        reviewing_weight_policy=str(reviewing_weight_policy),
                        weight_metric=str(weight_metric),
                        weight_policy_desc=dict(weight_policy_desc),
                        sampling_mode=str(sampling_mode),
                        memory_share_max=float(memory_share_max),
                        seed_offset=int(seed_offset),
                    )
                else:
                    logger.info(
                        "SUNRGBD Learning-dynamics: stage-start evaluation on memory bank "
                        "(reviewing disabled; no resampling)"
                    )
                    review0 = None

                ld_mem_stats = None
                if learning_dynamics_enabled:
                    ld_mem_stats = []

                for s in sorted(mem_infos_by_stage.keys()):
                    if s >= int(stage_id):
                        continue
                    cls_s = stage_to_classes.get(int(s), [])
                    if not cls_s:
                        continue
                    logger.info(
                        f"{seg_mode_label}: "
                        f"evaluating memory seats for intro_stage={int(s)} "
                        f"(split=train(memory_bank_subset), scenes={len(mem_infos_by_stage[s])})"
                    )
                    raw_results = []
                    raw_gt = []
                    raw_box_type = []
                    raw_box_mode = []
                    if learning_dynamics_enabled:
                        metrics_s = _sunrgbd_eval_memory_subset(
                            model=model,
                            stage_cfg=stage_cfg,
                            data_infos=mem_infos_by_stage[s],
                            eval_class_indices=cls_s,
                            class_names=class_names,
                            iou_thrs=eval_iou_thrs,
                            stage_idx=max(0, int(s) - 1),
                            split_name=f"train(memory_bank_subset,intro_stage={int(s)})",
                            eval_purpose='reviewing' if sunrgbd_reviewing_active else 'learning_dynamics',
                            review_k=0,
                            logger=logger,
                            raw_results_out=raw_results,
                            raw_gt_annos_out=raw_gt,
                            raw_box_type_3d_out=raw_box_type,
                            raw_box_mode_3d_out=raw_box_mode,
                        )
                        if isinstance(ld_mem_stats, list):
                            seat_keys = list(mem_seat_keys_by_stage.get(int(s), []) or [])
                            assert len(seat_keys) == len(raw_gt) == len(raw_results), (
                                "Learning-dynamics scoring: seat/raw output length mismatch "
                                f"for intro_stage={int(s)}: seats={len(seat_keys)}, "
                                f"gt={len(raw_gt)}, dt={len(raw_results)}"
                            )

                            stats_s = _compute_ld_match_stats(
                                list(raw_gt),
                                list(raw_results),
                                seat_keys=list(seat_keys),
                                eval_class_indices=list(ld_old_classes),
                                box_type_3d=raw_box_type[0],
                                box_mode_3d=raw_box_mode[0],
                            )
                            ld_mem_stats.extend(list(stats_s or []))
                    else:
                        metrics_s = _sunrgbd_eval_memory_subset(
                            model=model,
                            stage_cfg=stage_cfg,
                            data_infos=mem_infos_by_stage[s],
                            eval_class_indices=cls_s,
                            class_names=class_names,
                            iou_thrs=eval_iou_thrs,
                            stage_idx=max(0, int(s) - 1),
                            split_name=f"train(memory_bank_subset,intro_stage={int(s)})",
                            eval_purpose='reviewing' if sunrgbd_reviewing_active else 'learning_dynamics',
                            review_k=0,
                            logger=logger,
                        )
                    if sunrgbd_reviewing_active and isinstance(review0, dict):
                        stage_pack = {
                            'num_scenes': int(len(mem_infos_by_stage[s])),
                            'classes': {},
                        }
                        for thr in eval_iou_thrs:
                            key = f"{float(thr):.2f}"
                            stage_pack[f"mAP_{key}"] = float(metrics_s.get(f"mAP_{key}", 0.0))
                            stage_pack[f"mAR_{key}"] = float(metrics_s.get(f"mAR_{key}", 0.0))
                        for c in cls_s:
                            c = int(c)
                            name = mappings['model_idx_to_name'].get(c, f"class_{c}")
                            ap_weight = float(metrics_s.get(f"{name}_AP_{weight_iou_key}", 0.0))
                            prev_ap[c] = ap_weight
                            per_cls = {'name': name}
                            for thr in eval_iou_thrs:
                                key = f"{float(thr):.2f}"
                                per_cls[f"AP_{key}"] = float(metrics_s.get(f"{name}_AP_{key}", 0.0))
                                per_cls[f"AR_{key}"] = float(metrics_s.get(f"{name}_rec_{key}", 0.0))
                            stage_pack['classes'][str(c)] = per_cls
                        review0['by_intro_stage'][str(s)] = stage_pack

                # Save for stage-level forgetness (start vs end) if enabled.
                if forgetness_eviction_enabled:
                    stage_start_ap_forgetness = dict(prev_ap)

                # Learning-dynamics per-seat stats dumps at k=0.
                if learning_dynamics_enabled and isinstance(ld_mem_stats, list):
                    try:
                        out_path = ld_dir / "memory_stats_k0.json"
                        payload = {
                            'stage_id': int(stage_id),
                            'review_k': 0,
                            'split': 'train(memory_bank_subset)',
                            'iou_thr': float(ld_iou_thr),
                            'iou_mode': str(ld_iou_mode),
                            'iou_thrs': [float(ld_iou_thr)],
                            'eps': float(ld_eps),
                            'q_metric': str(ld_stats_q_metric),
                            'q_formula': str(ld_stats_q_formula),
                            'old_classes': [int(x) for x in ld_old_classes],
                            'num_seats': int(len(ld_mem_stats)),
                            'seats': list(ld_mem_stats),
                        }
                        with open(out_path, "w") as f:
                            json.dump(payload, f, indent=2)
                        logger.info(f"Learning-dynamics scoring: saved k=0 memory stats to {out_path}")
                    except Exception as e:
                        logger.warning(f"Learning-dynamics scoring: failed to write k=0 memory stats: {e}")

                    # Seed ld_drop reviewing baseline at k=0.
                    if reviewing_weight_policy == 'ld_drop':
                        prev_ld_mem_stats_for_reviewing = list(ld_mem_stats)

                    # Natural pool stats at k=0 (current-stage scenes; new classes).
                    try:
                        assert ld_natural_infos is not None and ld_natural_seat_keys is not None, (
                            "Learning-dynamics scoring: natural pool snapshot missing."
                        )
                        raw_results_n = []
                        raw_gt_n = []
                        raw_box_type_n = []
                        raw_box_mode_n = []
                        _sunrgbd_eval_memory_subset(
                            model=model,
                            stage_cfg=stage_cfg,
                            data_infos=list(ld_natural_infos),
                            eval_class_indices=list(ld_new_classes),
                            class_names=class_names,
                            iou_thrs=(float(ld_iou_thr),),
                            stage_idx=max(0, int(stage_id) - 1),
                            split_name=f"train(stage_{int(stage_id)}_natural)",
                            eval_purpose='learning_dynamics',
                            review_k=0,
                            logger=logger,
                            raw_results_out=raw_results_n,
                            raw_gt_annos_out=raw_gt_n,
                            raw_box_type_3d_out=raw_box_type_n,
                            raw_box_mode_3d_out=raw_box_mode_n,
                        )

                        stats_n = _compute_ld_match_stats(
                            list(raw_gt_n),
                            list(raw_results_n),
                            seat_keys=list(ld_natural_seat_keys),
                            eval_class_indices=list(ld_new_classes),
                            box_type_3d=raw_box_type_n[0],
                            box_mode_3d=raw_box_mode_n[0],
                        )
                        out_path = ld_dir / "natural_stats_k0.json"
                        payload = {
                            'stage_id': int(stage_id),
                            'review_k': 0,
                            'split': 'train(natural)',
                            'iou_thr': float(ld_iou_thr),
                            'iou_mode': str(ld_iou_mode),
                            'iou_thrs': [float(ld_iou_thr)],
                            'eps': float(ld_eps),
                            'q_metric': str(ld_stats_q_metric),
                            'q_formula': str(ld_stats_q_formula),
                            'new_classes': [int(x) for x in ld_new_classes],
                            'num_seats': int(len(stats_n)),
                            'seats': list(stats_n or []),
                        }
                        with open(out_path, "w") as f:
                            json.dump(payload, f, indent=2)
                        logger.info(f"Learning-dynamics scoring: saved k=0 natural stats to {out_path}")
                    except Exception as e:
                        logger.warning(f"Learning-dynamics scoring: failed to compute/write k=0 natural stats: {e}")

            _dist_barrier()

            # --- Segment schedule in RepeatDataset units ---
            seg_times_list = _compute_review_segment_times(
                stage_epochs=stage_epochs,
                repeat_times=original_times,
                review_fractions=review_fractions,
            )
            num_segments = int(len(seg_times_list))
            logger.info(
                f"{seg_mode_label}: "
                f"segment_times={seg_times_list} (sum={sum(seg_times_list)})"
            )

            # Train segment-by-segment (resume optimizer state).
            last_ckpt = None
            baseline_inner_len = None
            weights_by_uid = None  # seat weights for resampling next segment

            for seg_idx, seg_times in enumerate(seg_times_list, start=1):
                seg_times = int(seg_times)
                if seg_times <= 0:
                    continue

                seg_train_cfg = copy.deepcopy(train_dataset_cfg)
                seg_train_cfg.times = seg_times
                # IMPORTANT: `copy.deepcopy()` will clone `SceneMemoryBank`, which
                # breaks cross-segment/stage memory updates by updating a detached
                # copy held inside the dataset. Always re-attach the shared
                # `scene_memory_bank` instance used by the trainer.
                try:
                    seg_train_cfg.dataset.scene_memory_bank = scene_memory_bank
                    seg_train_cfg.dataset.object_memory_bank = object_memory_bank
                    _attach_object_memory_insertion(
                        seg_train_cfg.dataset,
                        object_memory_bank,
                        (object_memory_config or {}).get('insertion', {}))
                except Exception:
                    pass

                # Disable ratio-based downsampling only when reviewing does oversampling explicitly.
                if sunrgbd_reviewing_active:
                    try:
                        seg_train_cfg.dataset.target_memory_ratio = None
                    except Exception:
                        pass

                # Apply reviewing resampling for segments after the first review.
                reviewing_sampling_seed = None
                if weights_by_uid is not None:
                    if object_memory_bank is not None:
                        report = object_memory_bank.apply_source_seat_replay_weights(
                            weights_by_uid, strict=True)
                        seg_train_cfg.dataset.reviewing_sampling = dict(enabled=False)
                        logger.info(
                            'SUNRGBD Object Reviewing: transferred source-seat '
                            f"weights to {report['applied']} object exemplars")
                    else:
                        assert baseline_inner_len is not None, (
                            "reviewing_sampling requires baseline_inner_len to be set."
                        )
                        reviewing_sampling_seed = (
                            int(seed) + seed_offset + 100 * int(stage_id) + int(seg_idx)
                        )
                        seg_train_cfg.dataset.reviewing_sampling = dict(
                            enabled=True,
                            target_length=int(baseline_inner_len),
                            weights_by_replay_unique_id=weights_by_uid,
                            memory_share_max=float(memory_share_max),
                            strict_memory_coverage=bool(strict_memory_coverage),
                            seed=int(reviewing_sampling_seed),
                        )
                else:
                    try:
                        seg_train_cfg.dataset.reviewing_sampling = dict(enabled=False)
                    except Exception:
                        pass

                datasets = [build_dataset(seg_train_cfg)]
                inner_ds = get_innermost_dataset(datasets[0])
                if is_main_process and bool(use_pseudo_labels):
                    try:
                        injected_scenes = getattr(inner_ds, 'pseudo_injected_scene_count', None)
                        injected_boxes = getattr(inner_ds, 'pseudo_injected_box_count', None)
                        if injected_scenes is not None and injected_boxes is not None:
                            logger.info(
                                f"Pseudo labels injected (segment {seg_idx} dataset init): "
                                f"scenes={int(injected_scenes)}, boxes={int(injected_boxes)}"
                            )
                    except Exception as _e:
                        logger.warning(f"Pseudo label injection summary unavailable: {_e}")
                if baseline_inner_len is None:
                    baseline_inner_len = int(len(getattr(inner_ds, 'data_infos', [])))
                    assert baseline_inner_len > 0

                # Persist replay sampling / composition summary for this segment.
                # Convention:
                # - review_0_* describes segment-1 composition (no reviewing oversampling applied yet)
                # - review_k_* describes segment-(k+1) composition built using weights from review_k
                if is_main_process:
                    try:
                        from collections import Counter

                        data_infos_list = list(getattr(inner_ds, 'data_infos', []) or [])
                        total_len = int(len(data_infos_list))
                        mem_count = 0
                        uid_counts = Counter()
                        for info in data_infos_list:
                            if not isinstance(info, dict):
                                continue
                            if bool(info.get('is_replay', False)):
                                mem_count += 1
                                uid = str(info.get('replay_unique_id', ''))
                                if uid:
                                    uid_counts[uid] += 1
                                continue
                            if bool(info.get('is_merged', False)):
                                mem_count += 1
                                uids = info.get('replay_unique_ids', None) or []
                                uids = [str(x) for x in uids if str(x)]
                                if not uids:
                                    continue
                                if isinstance(weights_by_uid, dict) and weights_by_uid:
                                    uid = max(
                                        uids,
                                        key=lambda u: float(weights_by_uid.get(str(u), 1.0)),
                                    )
                                else:
                                    uid = sorted(uids)[0]
                                uid_counts[str(uid)] += 1

                        sampling_debug = getattr(inner_ds, 'reviewing_sampling_debug', {}) or {}
                        if not isinstance(sampling_debug, dict):
                            sampling_debug = {}
                        debug_cov_ratio = sampling_debug.get('memory_coverage_ratio', None)
                        debug_mem_total = sampling_debug.get('memory_candidate_count', None)
                        debug_mem_seen = sampling_debug.get('memory_seen_count', None)
                        debug_mem_never = sampling_debug.get('memory_never_seen_count', None)
                        debug_strict_cov = sampling_debug.get('strict_memory_coverage', None)

                        if isinstance(weights_by_uid, dict) and weights_by_uid:
                            memory_uid_universe = set(str(k) for k in weights_by_uid.keys())
                        else:
                            memory_uid_universe = set(str(k) for k in uid_counts.keys())
                        memory_uid_seen = set(str(k) for k in uid_counts.keys())
                        memory_uid_unseen = memory_uid_universe - memory_uid_seen

                        if sunrgbd_reviewing_active:
                            applied_review_k = int(seg_idx - 1) if weights_by_uid is not None else 0
                            summary = {
                                'stage_id': int(stage_id),
                                'segment_idx': int(seg_idx),
                                'applied_review_k': int(applied_review_k),
                                'split': 'train(stage_dataset_with_replay)',
                                'eval_iou_thrs': [float(x) for x in eval_iou_thrs],
                                'weight_iou_thr': float(weight_iou_thr),
                                'target_length': int(baseline_inner_len),
                                'actual_length': int(total_len),
                                'num_memory_samples': int(mem_count),
                                'num_natural_samples': int(max(0, total_len - mem_count)),
                                'memory_ratio': float(mem_count / max(1, total_len)),
                                'reviewing_sampling_enabled': bool(weights_by_uid is not None),
                                'reviewing_sampling_seed': int(reviewing_sampling_seed) if reviewing_sampling_seed is not None else None,
                                'reviewing_sampling_mode': str(sampling_mode),
                                'reviewing_sampling_strict_memory_coverage': (
                                    bool(debug_strict_cov)
                                    if debug_strict_cov is not None else bool(strict_memory_coverage)
                                ),
                                'memory_uid_counts': {str(k): int(v) for k, v in uid_counts.items()},
                                'memory_uid_universe_size': int(len(memory_uid_universe)),
                                'memory_uid_seen_size': int(len(memory_uid_seen)),
                                'memory_uid_never_seen_count': int(len(memory_uid_unseen)),
                                'memory_uid_coverage_ratio': (
                                    float(len(memory_uid_seen) / max(1, len(memory_uid_universe)))
                                ),
                                'memory_candidate_count': (
                                    int(debug_mem_total)
                                    if debug_mem_total is not None else None
                                ),
                                'memory_candidate_seen_count': (
                                    int(debug_mem_seen)
                                    if debug_mem_seen is not None else None
                                ),
                                'memory_candidate_never_seen_count': (
                                    int(debug_mem_never)
                                    if debug_mem_never is not None else None
                                ),
                                'memory_candidate_coverage_ratio': (
                                    float(debug_cov_ratio)
                                    if debug_cov_ratio is not None else None
                                ),
                                'top_sampled_memory_uids': [
                                    {
                                        'replay_unique_id': str(uid),
                                        'count': int(count),
                                        'weight': float(weights_by_uid.get(str(uid), 1.0)) if isinstance(weights_by_uid, dict) else 1.0,
                                    }
                                    for uid, count in uid_counts.most_common(50)
                                ],
                            }
                            summary_path = review_actions_dir / f"review_{applied_review_k}_replay_sampling_summary.json"
                            with open(summary_path, "w") as f:
                                json.dump(summary, f, indent=2)

                            logger.info(
                                f"SUNRGBD Reviewing: segment {seg_idx} dataset composition: "
                                f"len={total_len}, memory={mem_count} ({mem_count/max(1,total_len):.1%}), "
                                f"natural={max(0,total_len-mem_count)}"
                            )
                            logger.info(
                                f"SUNRGBD Reviewing: saved replay sampling summary to {summary_path}"
                            )
                            if uid_counts:
                                top = uid_counts.most_common(5)
                                top_s = ", ".join(
                                    f"{uid}×{cnt}" for uid, cnt in top
                                )
                                logger.info(
                                    f"SUNRGBD Reviewing: most-sampled memory entries: {top_s}"
                                )
                        else:
                            logger.info(
                                f"SUNRGBD Learning-dynamics: segment {seg_idx} dataset composition (no resampling): "
                                f"len={total_len}, memory={mem_count} ({mem_count/max(1,total_len):.1%}), "
                                f"natural={max(0,total_len-mem_count)}"
                            )
                    except Exception as e:
                        logger.warning(
                            f"{seg_mode_label}: "
                            f"failed to write sampling summary: {e}"
                        )

                stage_cfg_seg = copy.deepcopy(stage_cfg)
                stage_cfg_seg.data.train = seg_train_cfg
                stage_cfg_seg.runner.max_epochs = int(seg_idx)
                if seg_idx == 1:
                    stage_cfg_seg.resume_from = None
                    stage_cfg_seg.load_from = None
                else:
                    stage_cfg_seg.resume_from = last_ckpt
                    stage_cfg_seg.load_from = None

                logger.info(
                    f"{seg_mode_label}: "
                    f"training segment {seg_idx}/{num_segments} "
                    f"(times={seg_times}, max_epochs={stage_cfg_seg.runner.max_epochs})"
                )
                train_model(
                    model,
                    datasets,
                    stage_cfg_seg,
                    distributed=distributed,
                    validate=True,
                    timestamp=timestamp,
                    meta={
                        'stage_id': stage_id,
                        'stage_name': stage_name,
                        'segment': f"{'review' if sunrgbd_reviewing_active else 'ld'}_{seg_idx}",
                        'reviewing': bool(sunrgbd_reviewing_active),
                        'learning_dynamics': bool(learning_dynamics_enabled),
                    },
                )
                _dist_barrier()

                # Resolve checkpoint to resume from.
                checkpoint_dir = stage_cfg.work_dir
                ckpt = osp.join(checkpoint_dir, 'latest.pth')
                if osp.exists(ckpt):
                    last_ckpt = ckpt
                else:
                    cand = osp.join(checkpoint_dir, f'epoch_{seg_idx}.pth')
                    last_ckpt = cand if osp.exists(cand) else None

                # At review points (after each segment except the last), evaluate + update weights.
                if seg_idx >= num_segments:
                    continue

                if is_main_process and not sunrgbd_reviewing_active:
                    assert learning_dynamics_enabled, (
                        "SUNRGBD learning-dynamics segmented training requires learning_dynamics_enabled."
                    )
                    logger.info(
                        f"SUNRGBD Learning-dynamics: evaluation at review point k={seg_idx} (after segment {seg_idx}) "
                        "(reviewing disabled; no resampling)"
                    )

                    ld_mem_stats_k = []
                    for s in sorted(mem_infos_by_stage.keys()):
                        if s >= int(stage_id):
                            continue
                        cls_s = stage_to_classes.get(int(s), [])
                        if not cls_s:
                            continue
                        logger.info(
                            f"SUNRGBD Learning-dynamics: evaluating memory seats for intro_stage={int(s)} "
                            f"(split=train(memory_bank_subset), scenes={len(mem_infos_by_stage[s])})"
                        )
                        raw_results = []
                        raw_gt = []
                        raw_box_type = []
                        raw_box_mode = []
                        _sunrgbd_eval_memory_subset(
                            model=model,
                            stage_cfg=stage_cfg,
                            data_infos=mem_infos_by_stage[s],
                            eval_class_indices=cls_s,
                            class_names=class_names,
                            iou_thrs=eval_iou_thrs,
                            stage_idx=max(0, int(s) - 1),
                            split_name=f"train(memory_bank_subset,intro_stage={int(s)})",
                            eval_purpose='learning_dynamics',
                            review_k=int(seg_idx),
                            logger=logger,
                            raw_results_out=raw_results,
                            raw_gt_annos_out=raw_gt,
                            raw_box_type_3d_out=raw_box_type,
                            raw_box_mode_3d_out=raw_box_mode,
                        )

                        seat_keys = list(mem_seat_keys_by_stage.get(int(s), []) or [])
                        assert len(seat_keys) == len(raw_gt) == len(raw_results), (
                            "Learning-dynamics scoring: seat/raw output length mismatch "
                            f"for intro_stage={int(s)} at k={int(seg_idx)}: seats={len(seat_keys)}, "
                            f"gt={len(raw_gt)}, dt={len(raw_results)}"
                        )
                        stats_s = _compute_ld_match_stats(
                            list(raw_gt),
                            list(raw_results),
                            seat_keys=list(seat_keys),
                            eval_class_indices=list(ld_old_classes),
                            box_type_3d=raw_box_type[0],
                            box_mode_3d=raw_box_mode[0],
                        )
                        ld_mem_stats_k.extend(list(stats_s or []))

                    # Persist per-seat stats for this review point.
                    try:
                        out_path = ld_dir / f"memory_stats_k{int(seg_idx)}.json"
                        payload = {
                            'stage_id': int(stage_id),
                            'review_k': int(seg_idx),
                            'split': 'train(memory_bank_subset)',
                            'iou_thr': float(ld_iou_thr),
                            'iou_mode': str(ld_iou_mode),
                            'iou_thrs': [float(ld_iou_thr)],
                            'eps': float(ld_eps),
                            'q_metric': str(ld_stats_q_metric),
                            'q_formula': str(ld_stats_q_formula),
                            'old_classes': [int(x) for x in ld_old_classes],
                            'num_seats': int(len(ld_mem_stats_k)),
                            'seats': list(ld_mem_stats_k),
                        }
                        with open(out_path, "w") as f:
                            json.dump(payload, f, indent=2)
                        logger.info(
                            f"Learning-dynamics scoring: saved k={int(seg_idx)} memory stats to {out_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Learning-dynamics scoring: failed to write memory stats at k={int(seg_idx)}: {e}"
                        )

                    # Natural pool stats at this k (current-stage scenes; new classes).
                    try:
                        assert ld_natural_infos is not None and ld_natural_seat_keys is not None, (
                            "Learning-dynamics scoring: natural pool snapshot missing."
                        )
                        raw_results_n = []
                        raw_gt_n = []
                        raw_box_type_n = []
                        raw_box_mode_n = []
                        _sunrgbd_eval_memory_subset(
                            model=model,
                            stage_cfg=stage_cfg,
                            data_infos=list(ld_natural_infos),
                            eval_class_indices=list(ld_new_classes),
                            class_names=class_names,
                            iou_thrs=(float(ld_iou_thr),),
                            stage_idx=max(0, int(stage_id) - 1),
                            split_name=f"train(stage_{int(stage_id)}_natural)",
                            eval_purpose='learning_dynamics',
                            review_k=int(seg_idx),
                            logger=logger,
                            raw_results_out=raw_results_n,
                            raw_gt_annos_out=raw_gt_n,
                            raw_box_type_3d_out=raw_box_type_n,
                            raw_box_mode_3d_out=raw_box_mode_n,
                        )

                        stats_n = _compute_ld_match_stats(
                            list(raw_gt_n),
                            list(raw_results_n),
                            seat_keys=list(ld_natural_seat_keys),
                            eval_class_indices=list(ld_new_classes),
                            box_type_3d=raw_box_type_n[0],
                            box_mode_3d=raw_box_mode_n[0],
                        )
                        out_path = ld_dir / f"natural_stats_k{int(seg_idx)}.json"
                        payload = {
                            'stage_id': int(stage_id),
                            'review_k': int(seg_idx),
                            'split': 'train(natural)',
                            'iou_thr': float(ld_iou_thr),
                            'iou_mode': str(ld_iou_mode),
                            'iou_thrs': [float(ld_iou_thr)],
                            'eps': float(ld_eps),
                            'q_metric': str(ld_stats_q_metric),
                            'q_formula': str(ld_stats_q_formula),
                            'new_classes': [int(x) for x in ld_new_classes],
                            'num_seats': int(len(stats_n)),
                            'seats': list(stats_n or []),
                        }
                        with open(out_path, "w") as f:
                            json.dump(payload, f, indent=2)
                        logger.info(
                            f"Learning-dynamics scoring: saved k={int(seg_idx)} natural stats to {out_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Learning-dynamics scoring: failed to compute/write natural stats at k={int(seg_idx)}: {e}"
                        )

                if is_main_process and sunrgbd_reviewing_active:
                    logger.info(
                        f"SUNRGBD Reviewing: evaluation at review point k={seg_idx} (after segment {seg_idx})"
                    )

                    curr_ap = {}  # class_idx -> AP@weight_iou_thr
                    effective_ld_cap, effective_w_entry_max = _resolve_effective_ld_reviewing_params_impl(
                        reviewing_ld_object_count_cap=reviewing_ld_object_count_cap,
                        reviewing_ld_w_entry_max=reviewing_ld_w_entry_max,
                        ld_object_count_cap=ld_object_count_cap,
                        learning_dynamics_enabled=bool(learning_dynamics_enabled),
                    )
                    weight_metric, weight_policy_desc = _build_review_weight_policy_impl(
                        reviewing_weight_policy=str(reviewing_weight_policy),
                        alpha_drop=float(alpha_drop),
                        beta_ap=float(beta_ap),
                        gamma=float(gamma),
                        w_max=float(w_max),
                        eta=float(eta),
                        fixed_review_weight=fixed_review_weight,
                        drop_clamp_min=float(drop_clamp_min),
                        reviewing_ld_q_metric=str(reviewing_ld_q_metric),
                        reviewing_ld_q_formula=str(reviewing_ld_q_formula),
                        ld_iou_thr=float(ld_iou_thr),
                        ld_iou_mode=str(ld_iou_mode),
                        ld_eps=float(ld_eps),
                        effective_ld_cap=int(effective_ld_cap),
                        reviewing_ld_normalize_by_gt_weight=bool(
                            reviewing_ld_normalize_by_gt_weight
                        ),
                        effective_w_entry_max=effective_w_entry_max,
                    )
                    reviewk = _build_reviewing_eval_payload_impl(
                        stage_id=int(stage_id),
                        review_k=int(seg_idx),
                        eval_iou_thrs=[float(x) for x in eval_iou_thrs],
                        weight_iou_thr=float(weight_iou_thr),
                        reviewing_weight_policy=str(reviewing_weight_policy),
                        weight_metric=str(weight_metric),
                        weight_policy_desc=dict(weight_policy_desc),
                        sampling_mode=str(sampling_mode),
                        memory_share_max=float(memory_share_max),
                        seed_offset=int(seed_offset),
                    )

                    ld_mem_stats_k = None
                    if learning_dynamics_enabled:
                        ld_mem_stats_k = []

                    for s in sorted(mem_infos_by_stage.keys()):
                        if s >= int(stage_id):
                            continue
                        cls_s = stage_to_classes.get(int(s), [])
                        if not cls_s:
                            continue
                        logger.info(
                            f"SUNRGBD Reviewing: evaluating memory seats for intro_stage={int(s)} "
                            f"(split=train(memory_bank_subset), scenes={len(mem_infos_by_stage[s])})"
                        )
                        raw_results = []
                        raw_gt = []
                        raw_box_type = []
                        raw_box_mode = []
                        if learning_dynamics_enabled:
                            metrics_s = _sunrgbd_eval_memory_subset(
                                model=model,
                                stage_cfg=stage_cfg,
                                data_infos=mem_infos_by_stage[s],
                                eval_class_indices=cls_s,
                                class_names=class_names,
                                iou_thrs=eval_iou_thrs,
                                stage_idx=max(0, int(s) - 1),
                                split_name=f"train(memory_bank_subset,intro_stage={int(s)})",
                                eval_purpose='reviewing',
                                review_k=int(seg_idx),
                                logger=logger,
                                raw_results_out=raw_results,
                                raw_gt_annos_out=raw_gt,
                                raw_box_type_3d_out=raw_box_type,
                                raw_box_mode_3d_out=raw_box_mode,
                            )
                            if isinstance(ld_mem_stats_k, list):
                                seat_keys = list(mem_seat_keys_by_stage.get(int(s), []) or [])
                                assert len(seat_keys) == len(raw_gt) == len(raw_results), (
                                    "Learning-dynamics scoring: seat/raw output length mismatch "
                                    f"for intro_stage={int(s)} at k={int(seg_idx)}: seats={len(seat_keys)}, "
                                    f"gt={len(raw_gt)}, dt={len(raw_results)}"
                                )

                                stats_s = _compute_ld_match_stats(
                                    list(raw_gt),
                                    list(raw_results),
                                    seat_keys=list(seat_keys),
                                    eval_class_indices=list(ld_old_classes),
                                    box_type_3d=raw_box_type[0],
                                    box_mode_3d=raw_box_mode[0],
                                )
                                ld_mem_stats_k.extend(list(stats_s or []))
                        else:
                            metrics_s = _sunrgbd_eval_memory_subset(
                                model=model,
                                stage_cfg=stage_cfg,
                                data_infos=mem_infos_by_stage[s],
                                eval_class_indices=cls_s,
                                class_names=class_names,
                                iou_thrs=eval_iou_thrs,
                                stage_idx=max(0, int(s) - 1),
                                split_name=f"train(memory_bank_subset,intro_stage={int(s)})",
                                eval_purpose='reviewing',
                                review_k=int(seg_idx),
                                logger=logger,
                            )
                        stage_pack = {
                            'num_scenes': int(len(mem_infos_by_stage[s])),
                            'classes': {},
                        }
                        for thr in eval_iou_thrs:
                            key = f"{float(thr):.2f}"
                            stage_pack[f"mAP_{key}"] = float(metrics_s.get(f"mAP_{key}", 0.0))
                            stage_pack[f"mAR_{key}"] = float(metrics_s.get(f"mAR_{key}", 0.0))
                        for c in cls_s:
                            c = int(c)
                            name = mappings['model_idx_to_name'].get(c, f"class_{c}")
                            ap_weight = float(metrics_s.get(f"{name}_AP_{weight_iou_key}", 0.0))
                            curr_ap[c] = ap_weight
                            per_cls = {'name': name}
                            for thr in eval_iou_thrs:
                                key = f"{float(thr):.2f}"
                                per_cls[f"AP_{key}"] = float(metrics_s.get(f"{name}_AP_{key}", 0.0))
                                per_cls[f"AR_{key}"] = float(metrics_s.get(f"{name}_rec_{key}", 0.0))
                            stage_pack['classes'][str(c)] = per_cls
                        reviewk['by_intro_stage'][str(s)] = stage_pack

                    # Intentionally do not persist the full per-stage reviewing eval dict
                    # to reduce artifact volume; keep only weights + actions.

                    if learning_dynamics_enabled and isinstance(ld_mem_stats_k, list):
                        # Persist per-seat stats for this review point.
                        try:
                            out_path = ld_dir / f"memory_stats_k{int(seg_idx)}.json"
                            payload = {
                                'stage_id': int(stage_id),
                                'review_k': int(seg_idx),
                                'split': 'train(memory_bank_subset)',
                                'iou_thr': float(ld_iou_thr),
                                'iou_mode': str(ld_iou_mode),
                                'iou_thrs': [float(ld_iou_thr)],
                                'eps': float(ld_eps),
                                'q_metric': str(ld_stats_q_metric),
                                'q_formula': str(ld_stats_q_formula),
                                'old_classes': [int(x) for x in ld_old_classes],
                                'num_seats': int(len(ld_mem_stats_k)),
                                'seats': list(ld_mem_stats_k),
                            }
                            with open(out_path, "w") as f:
                                json.dump(payload, f, indent=2)
                            logger.info(
                                f"Learning-dynamics scoring: saved k={int(seg_idx)} memory stats to {out_path}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Learning-dynamics scoring: failed to write memory stats at k={int(seg_idx)}: {e}"
                            )

                        # Natural pool stats at this k (current-stage scenes; new classes).
                        try:
                            assert ld_natural_infos is not None and ld_natural_seat_keys is not None, (
                                "Learning-dynamics scoring: natural pool snapshot missing."
                            )
                            raw_results_n = []
                            raw_gt_n = []
                            raw_box_type_n = []
                            raw_box_mode_n = []
                            _sunrgbd_eval_memory_subset(
                                model=model,
                                stage_cfg=stage_cfg,
                                data_infos=list(ld_natural_infos),
                                eval_class_indices=list(ld_new_classes),
                                class_names=class_names,
                                iou_thrs=(float(ld_iou_thr),),
                                stage_idx=max(0, int(stage_id) - 1),
                                split_name=f"train(stage_{int(stage_id)}_natural)",
                                eval_purpose='learning_dynamics',
                                review_k=int(seg_idx),
                                logger=logger,
                                raw_results_out=raw_results_n,
                                raw_gt_annos_out=raw_gt_n,
                                raw_box_type_3d_out=raw_box_type_n,
                                raw_box_mode_3d_out=raw_box_mode_n,
                            )

                            stats_n = _compute_ld_match_stats(
                                list(raw_gt_n),
                                list(raw_results_n),
                                seat_keys=list(ld_natural_seat_keys),
                                eval_class_indices=list(ld_new_classes),
                                box_type_3d=raw_box_type_n[0],
                                box_mode_3d=raw_box_mode_n[0],
                            )
                            out_path = ld_dir / f"natural_stats_k{int(seg_idx)}.json"
                            payload = {
                                'stage_id': int(stage_id),
                                'review_k': int(seg_idx),
                                'split': 'train(natural)',
                                'iou_thr': float(ld_iou_thr),
                                'iou_mode': str(ld_iou_mode),
                                'iou_thrs': [float(ld_iou_thr)],
                                'eps': float(ld_eps),
                                'q_metric': str(ld_stats_q_metric),
                                'q_formula': str(ld_stats_q_formula),
                                'new_classes': [int(x) for x in ld_new_classes],
                                'num_seats': int(len(stats_n)),
                                'seats': list(stats_n or []),
                            }
                            with open(out_path, "w") as f:
                                json.dump(payload, f, indent=2)
                            logger.info(
                                f"Learning-dynamics scoring: saved k={int(seg_idx)} natural stats to {out_path}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Learning-dynamics scoring: failed to compute/write natural stats at k={int(seg_idx)}: {e}"
                            )

                    if reviewing_weight_policy == 'ap_drop':
                        # Compute drop (clamped) and class weights.
                        prev_ap_snapshot = dict(prev_ap)
                        drop = {}
                        class_w = {}
                        for c, ap_prev in prev_ap.items():
                            ap_prev = float(ap_prev)
                            ap_now = float(curr_ap.get(int(c), 0.0))
                            d = max(float(drop_clamp_min), ap_prev - ap_now)
                            d = max(0.0, d)
                            drop[int(c)] = float(d)
                            u = alpha_drop * float(d) + beta_ap * (1.0 - float(ap_now))
                            w = 1.0 + gamma * float(u)
                            w = min(float(w_max), float(w))
                            class_w[int(c)] = float(max(1.0, w))

                        with open(review_weights_dir / f"review_{seg_idx}_drop.json", "w") as f:
                            json.dump(drop, f, indent=2)
                        with open(review_weights_dir / f"review_{seg_idx}_class_weights.json", "w") as f:
                            json.dump(class_w, f, indent=2)

                        # Compute seat weights (presence-only) and use them for next segment resampling.
                        logger.info(
                            f"SUNRGBD Reviewing: deriving class/entry weights from AP@{weight_iou_thr:.2f} "
                            f"(drop=clamp(AP_prev-AP_now, min={drop_clamp_min}, 0), "
                            f"w_class=clamp(1+gamma*u, 1, w_max), "
                            f"w_entry=1+eta*sum(w_class-1 for present_classes))"
                        )
                        try:
                            vals = [float(v) for v in class_w.values()]
                            if vals:
                                logger.info(
                                    f"SUNRGBD Reviewing: class_weights stats: "
                                    f"min={min(vals):.3f}, max={max(vals):.3f}, mean={sum(vals)/len(vals):.3f}"
                                )
                                top = sorted(class_w.items(), key=lambda kv: kv[1], reverse=True)[:8]
                                for cid, wv in top:
                                    cid = int(cid)
                                    name = mappings['model_idx_to_name'].get(cid, f"class_{cid}")
                                    logger.info(
                                        f"  class {cid} ({name}): "
                                        f"w={float(wv):.3f}, drop={float(drop.get(cid, 0.0)):.4f}, "
                                        f"AP_prev={float(prev_ap_snapshot.get(cid, 0.0)):.4f}, "
                                        f"AP_now={float(curr_ap.get(cid, 0.0)):.4f}"
                                    )
                        except Exception:
                            pass

                        weights_by_uid = {}
                        boosted_by_uid = {}
                        for e in mem_entries:
                            sid = str(e.get('scene_id'))
                            save_stage = int(e.get('save_stage', 0))
                            uid = f"{sid}_stage{save_stage}"
                            snap = e.get('snapshot', {}) or {}
                            present = snap.get('present_classes', []) or []
                            inc = 0.0
                            boosted = []
                            for cid in present:
                                cid = int(cid)
                                wc = float(class_w.get(cid, 1.0))
                                inc += float(wc - 1.0)
                                if wc > 1.0:
                                    boosted.append((cid, wc))
                            w_entry = 1.0 + eta * float(inc)
                            weights_by_uid[str(uid)] = float(max(1.0, w_entry))
                            boosted.sort(key=lambda x: x[1], reverse=True)
                            boosted_by_uid[str(uid)] = boosted

                        with open(review_weights_dir / f"review_{seg_idx}_entry_weights.json", "w") as f:
                            json.dump(weights_by_uid, f, indent=2)
                        try:
                            vals = [float(v) for v in weights_by_uid.values()]
                            if vals:
                                logger.info(
                                    f"SUNRGBD Reviewing: entry_weights stats: "
                                    f"min={min(vals):.3f}, max={max(vals):.3f}, mean={sum(vals)/len(vals):.3f}"
                                )
                                top = sorted(weights_by_uid.items(), key=lambda kv: kv[1], reverse=True)[:8]
                                for uid, wv in top:
                                    boosted = boosted_by_uid.get(str(uid), []) or []
                                    boosted_s = ", ".join(
                                        f"{mappings['model_idx_to_name'].get(int(cid), cid)}:{float(cw):.2f}"
                                        for cid, cw in boosted[:6]
                                    )
                                    logger.info(
                                        f"  entry {uid}: w={float(wv):.3f}, boosted=[{boosted_s}]"
                                    )
                        except Exception:
                            pass

                        # Update baseline for next round (compare to the last eval).
                        prev_ap = dict(curr_ap)
                    elif reviewing_weight_policy == 'ld_drop':
                        from mmdet3d.utils.learning_dynamics_scoring import (
                            compute_reviewing_entry_weights_ld_drop,
                        )

                        assert learning_dynamics_enabled, (
                            "reviewing.weight_policy.type='ld_drop' requires "
                            "scene_memory_config.selection_strategy in "
                            "['learning_dynamics', 'learning_dynamics_design1', "
                            "'learning_dynamics_design2'] (LD enabled)."
                        )
                        assert isinstance(ld_mem_stats_k, list), (
                            "reviewing.weight_policy.type='ld_drop' requires LD memory stats at this k."
                        )
                        assert prev_ld_mem_stats_for_reviewing is not None, (
                            "reviewing.weight_policy.type='ld_drop' requires a previous k-1 stats snapshot."
                        )

                        logger.info(
                            "SUNRGBD Reviewing: deriving entry weights via "
                            f"LD q({str(reviewing_ld_q_metric).upper()}) drop at IoU={float(ld_iou_thr):.2f} "
                            f"(eta={float(eta):.3f}, normalize_by_gt_weight={bool(reviewing_ld_normalize_by_gt_weight)}, "
                            f"object_count_cap={int(effective_ld_cap)})"
                        )

                        out = compute_reviewing_entry_weights_ld_drop(
                            prev_ld_mem_stats_for_reviewing,
                            ld_mem_stats_k,
                            old_classes=list(ld_old_classes),
                            iou_mode=str(ld_iou_mode),
                            q_metric=str(reviewing_ld_q_metric),
                            object_count_cap=int(effective_ld_cap),
                            eps=float(ld_eps),
                            eta=float(eta),
                            normalize_by_gt_weight=bool(reviewing_ld_normalize_by_gt_weight),
                            w_entry_max=effective_w_entry_max,
                        )
                        computed = out.get('weights_by_uid', {}) or {}

                        # Provide default weight=1.0 for all memory entries (missing seats stay neutral).
                        weights_by_uid = {}
                        for e in mem_entries:
                            sid = str(e.get('scene_id'))
                            save_stage = int(e.get('save_stage', 0))
                            uid = f"{sid}_stage{save_stage}"
                            weights_by_uid[str(uid)] = 1.0
                        for uid, wv in computed.items():
                            try:
                                weights_by_uid[str(uid)] = float(wv)
                            except Exception:
                                continue

                        with open(review_weights_dir / f"review_{seg_idx}_entry_weights.json", "w") as f:
                            json.dump(weights_by_uid, f, indent=2)
                        meta = dict(out) if isinstance(out, dict) else {}
                        if not ld_path_only_logging:
                            try:
                                meta['stage_id'] = int(stage_id)
                                meta['review_k'] = int(seg_idx)
                                meta['policy'] = str(reviewing_weight_policy)
                                meta['weights_by_uid_final'] = dict(weights_by_uid)
                                meta_path = review_weights_dir / f"review_{seg_idx}_entry_weights_meta.json"
                                with open(meta_path, "w") as f:
                                    json.dump(meta, f, indent=2)
                                logger.info(
                                    f"SUNRGBD Reviewing: saved ld_drop meta to {meta_path}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"SUNRGBD Reviewing: failed to write ld_drop meta for k={int(seg_idx)}: {e}"
                                )

                        try:
                            vals = [float(v) for v in weights_by_uid.values()]
                            if vals:
                                logger.info(
                                    f"SUNRGBD Reviewing: entry_weights(ld_drop) stats: "
                                    f"min={min(vals):.3f}, max={max(vals):.3f}, mean={sum(vals)/len(vals):.3f}"
                                )
                                top = sorted(weights_by_uid.items(), key=lambda kv: kv[1], reverse=True)[:8]
                                seat_drop = meta.get('seat_drop_by_uid', {}) if isinstance(meta, dict) else {}
                                seat_denom = meta.get('seat_denom_by_uid', {}) if isinstance(meta, dict) else {}
                                for uid, wv in top:
                                    logger.info(
                                        f"  entry {uid}: w={float(wv):.3f}, "
                                        f"raw_drop={float(seat_drop.get(str(uid), 0.0)):.4f}, "
                                        f"gt_weight_sum={float(seat_denom.get(str(uid), 0.0)):.1f}"
                                    )
                        except Exception:
                            pass

                        # Update baselines for next round (compare to the last eval).
                        prev_ld_mem_stats_for_reviewing = list(ld_mem_stats_k)
                        prev_ap = dict(curr_ap)
                    elif reviewing_weight_policy == 'fixed':
                        assert fixed_review_weight is not None, (
                            "reviewing.weight_policy.type='fixed' requires "
                            "reviewing.weight_policy.fixed_value."
                        )
                        logger.info(
                            "SUNRGBD Reviewing: deriving entry weights via fixed policy "
                            f"(fixed_value={float(fixed_review_weight):.3f})"
                        )
                        weights_by_uid = {}
                        for e in mem_entries:
                            sid = str(e.get('scene_id'))
                            save_stage = int(e.get('save_stage', 0))
                            uid = f"{sid}_stage{save_stage}"
                            weights_by_uid[str(uid)] = float(fixed_review_weight)

                        with open(review_weights_dir / f"review_{seg_idx}_entry_weights.json", "w") as f:
                            json.dump(weights_by_uid, f, indent=2)

                        fixed_meta = dict(
                            stage_id=int(stage_id),
                            review_k=int(seg_idx),
                            policy=str(reviewing_weight_policy),
                            fixed_value=float(fixed_review_weight),
                            weights_by_uid_final=dict(weights_by_uid),
                        )
                        try:
                            meta_path = review_weights_dir / f"review_{seg_idx}_entry_weights_meta.json"
                            with open(meta_path, "w") as f:
                                json.dump(fixed_meta, f, indent=2)
                            logger.info(
                                f"SUNRGBD Reviewing: saved fixed-weight meta to {meta_path}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"SUNRGBD Reviewing: failed to write fixed-weight meta for k={int(seg_idx)}: {e}"
                            )

                        try:
                            vals = [float(v) for v in weights_by_uid.values()]
                            if vals:
                                logger.info(
                                    "SUNRGBD Reviewing: entry_weights(fixed) stats: "
                                    f"min={min(vals):.3f}, max={max(vals):.3f}, mean={sum(vals)/len(vals):.3f}"
                                )
                                top = sorted(weights_by_uid.items(), key=lambda kv: kv[1], reverse=True)[:8]
                                for uid, wv in top:
                                    logger.info(f"  entry {uid}: w={float(wv):.3f}")
                        except Exception:
                            pass

                        prev_ap = dict(curr_ap)
                    else:
                        raise ValueError(
                            f"Unsupported reviewing_weight_policy: {reviewing_weight_policy}"
                        )

                _dist_barrier()
                if (not is_main_process) and sunrgbd_reviewing_active:
                    # Load weights computed by rank0 so all ranks build identical
                    # resampled datasets in the next segment.
                    try:
                        with open(review_weights_dir / f"review_{seg_idx}_entry_weights.json", "r") as f:
                            weights_by_uid = json.load(f)
                    except Exception as e:
                        logger.warning(
                            f"Failed to load reviewing entry weights for segment {seg_idx}: {e}"
                        )
                        weights_by_uid = None

            # Stage-end evaluation for reviewing / learning-dynamics (rank0 only).
            #
            # This is intentionally performed AFTER the final segment so LD trajectories include
            # the true end-of-stage model (not just the last in-training review point).
            if is_main_process and (sunrgbd_reviewing_active or learning_dynamics_enabled):
                k_final = int(num_segments)
                assert k_final >= 1, k_final

                if sunrgbd_reviewing_active:
                    logger.info("SUNRGBD Reviewing: stage-end evaluation on memory bank")
                    effective_ld_cap, effective_w_entry_max = _resolve_effective_ld_reviewing_params_impl(
                        reviewing_ld_object_count_cap=reviewing_ld_object_count_cap,
                        reviewing_ld_w_entry_max=reviewing_ld_w_entry_max,
                        ld_object_count_cap=ld_object_count_cap,
                        learning_dynamics_enabled=bool(learning_dynamics_enabled),
                    )
                    weight_metric, weight_policy_desc = _build_review_weight_policy_impl(
                        reviewing_weight_policy=str(reviewing_weight_policy),
                        alpha_drop=float(alpha_drop),
                        beta_ap=float(beta_ap),
                        gamma=float(gamma),
                        w_max=float(w_max),
                        eta=float(eta),
                        fixed_review_weight=fixed_review_weight,
                        drop_clamp_min=float(drop_clamp_min),
                        reviewing_ld_q_metric=str(reviewing_ld_q_metric),
                        reviewing_ld_q_formula=str(reviewing_ld_q_formula),
                        ld_iou_thr=float(ld_iou_thr),
                        ld_iou_mode=str(ld_iou_mode),
                        ld_eps=float(ld_eps),
                        effective_ld_cap=int(effective_ld_cap),
                        reviewing_ld_normalize_by_gt_weight=bool(
                            reviewing_ld_normalize_by_gt_weight
                        ),
                        effective_w_entry_max=effective_w_entry_max,
                    )
                    reviewk_final = _build_reviewing_eval_payload_impl(
                        stage_id=int(stage_id),
                        review_k=int(k_final),
                        eval_iou_thrs=[float(x) for x in eval_iou_thrs],
                        weight_iou_thr=float(weight_iou_thr),
                        reviewing_weight_policy=str(reviewing_weight_policy),
                        weight_metric=str(weight_metric),
                        weight_policy_desc=dict(weight_policy_desc),
                        sampling_mode=str(sampling_mode),
                        memory_share_max=float(memory_share_max),
                        seed_offset=int(seed_offset),
                    )
                    curr_ap_final = {}
                else:
                    logger.info(
                        "SUNRGBD Learning-dynamics: stage-end evaluation on memory bank "
                        "(reviewing disabled; no resampling)"
                    )
                    reviewk_final = None
                    curr_ap_final = None

                ld_mem_stats_final = []
                if not learning_dynamics_enabled:
                    ld_mem_stats_final = None

                for s in sorted(mem_infos_by_stage.keys()):
                    if s >= int(stage_id):
                        continue
                    cls_s = stage_to_classes.get(int(s), [])
                    if not cls_s:
                        continue
                    logger.info(
                        f"{seg_mode_label}: "
                        f"evaluating memory seats for intro_stage={int(s)} "
                        f"(split=train(memory_bank_subset), scenes={len(mem_infos_by_stage[s])})"
                    )

                    raw_results = []
                    raw_gt = []
                    raw_box_type = []
                    raw_box_mode = []
                    if learning_dynamics_enabled:
                        metrics_s = _sunrgbd_eval_memory_subset(
                            model=model,
                            stage_cfg=stage_cfg,
                            data_infos=mem_infos_by_stage[s],
                            eval_class_indices=cls_s,
                            class_names=class_names,
                            iou_thrs=eval_iou_thrs,
                            stage_idx=max(0, int(s) - 1),
                            split_name=f"train(memory_bank_subset,intro_stage={int(s)})",
                            eval_purpose='reviewing' if sunrgbd_reviewing_active else 'learning_dynamics',
                            review_k=int(k_final),
                            logger=logger,
                            raw_results_out=raw_results,
                            raw_gt_annos_out=raw_gt,
                            raw_box_type_3d_out=raw_box_type,
                            raw_box_mode_3d_out=raw_box_mode,
                        )
                        if isinstance(ld_mem_stats_final, list):
                            seat_keys = list(mem_seat_keys_by_stage.get(int(s), []) or [])
                            assert len(seat_keys) == len(raw_gt) == len(raw_results), (
                                "Learning-dynamics scoring: seat/raw output length mismatch "
                                f"for intro_stage={int(s)} at k={int(k_final)}: seats={len(seat_keys)}, "
                                f"gt={len(raw_gt)}, dt={len(raw_results)}"
                            )

                            stats_s = _compute_ld_match_stats(
                                list(raw_gt),
                                list(raw_results),
                                seat_keys=list(seat_keys),
                                eval_class_indices=list(ld_old_classes),
                                box_type_3d=raw_box_type[0],
                                box_mode_3d=raw_box_mode[0],
                            )
                            ld_mem_stats_final.extend(list(stats_s or []))
                    else:
                        metrics_s = _sunrgbd_eval_memory_subset(
                            model=model,
                            stage_cfg=stage_cfg,
                            data_infos=mem_infos_by_stage[s],
                            eval_class_indices=cls_s,
                            class_names=class_names,
                            iou_thrs=eval_iou_thrs,
                            stage_idx=max(0, int(s) - 1),
                            split_name=f"train(memory_bank_subset,intro_stage={int(s)})",
                            eval_purpose='reviewing' if sunrgbd_reviewing_active else 'learning_dynamics',
                            review_k=int(k_final),
                            logger=logger,
                        )

                    if sunrgbd_reviewing_active and isinstance(reviewk_final, dict):
                        stage_pack = {
                            'num_scenes': int(len(mem_infos_by_stage[s])),
                            'classes': {},
                        }
                        for thr in eval_iou_thrs:
                            key = f"{float(thr):.2f}"
                            stage_pack[f"mAP_{key}"] = float(metrics_s.get(f"mAP_{key}", 0.0))
                            stage_pack[f"mAR_{key}"] = float(metrics_s.get(f"mAR_{key}", 0.0))
                        for c in cls_s:
                            c = int(c)
                            name = mappings['model_idx_to_name'].get(c, f"class_{c}")
                            ap_weight = float(metrics_s.get(f"{name}_AP_{weight_iou_key}", 0.0))
                            curr_ap_final[int(c)] = float(ap_weight)
                            per_cls = {'name': name}
                            for thr in eval_iou_thrs:
                                key = f"{float(thr):.2f}"
                                per_cls[f"AP_{key}"] = float(metrics_s.get(f"{name}_AP_{key}", 0.0))
                                per_cls[f"AR_{key}"] = float(metrics_s.get(f"{name}_rec_{key}", 0.0))
                            stage_pack['classes'][str(c)] = per_cls
                        reviewk_final['by_intro_stage'][str(s)] = stage_pack

                if sunrgbd_reviewing_active and isinstance(reviewk_final, dict):
                    if isinstance(curr_ap_final, dict) and curr_ap_final:
                        prev_ap = dict(curr_ap_final)

                # Learning-dynamics per-seat stats dumps at k=stage_end.
                if learning_dynamics_enabled and isinstance(ld_mem_stats_final, list):
                    try:
                        assert ld_dir is not None, "Learning-dynamics scoring: ld_dir is None."
                        out_path = ld_dir / f"memory_stats_k{int(k_final)}.json"
                        payload = {
                            'stage_id': int(stage_id),
                            'review_k': int(k_final),
                            'split': 'train(memory_bank_subset)',
                            'iou_thr': float(ld_iou_thr),
                            'iou_mode': str(ld_iou_mode),
                            'iou_thrs': [float(ld_iou_thr)],
                            'eps': float(ld_eps),
                            'q_metric': str(ld_stats_q_metric),
                            'q_formula': str(ld_stats_q_formula),
                            'old_classes': [int(x) for x in ld_old_classes],
                            'num_seats': int(len(ld_mem_stats_final)),
                            'seats': list(ld_mem_stats_final),
                        }
                        with open(out_path, "w") as f:
                            json.dump(payload, f, indent=2)
                        logger.info(
                            f"Learning-dynamics scoring: saved k={int(k_final)} memory stats to {out_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Learning-dynamics scoring: failed to write memory stats at k={int(k_final)}: {e}"
                        )

                    # Natural pool stats at k=stage_end (current-stage scenes; new classes).
                    try:
                        assert ld_natural_infos is not None and ld_natural_seat_keys is not None, (
                            "Learning-dynamics scoring: natural pool snapshot missing."
                        )
                        raw_results_n = []
                        raw_gt_n = []
                        raw_box_type_n = []
                        raw_box_mode_n = []
                        _sunrgbd_eval_memory_subset(
                            model=model,
                            stage_cfg=stage_cfg,
                            data_infos=list(ld_natural_infos),
                            eval_class_indices=list(ld_new_classes),
                            class_names=class_names,
                            iou_thrs=(float(ld_iou_thr),),
                            stage_idx=max(0, int(stage_id) - 1),
                            split_name=f"train(stage_{int(stage_id)}_natural)",
                            eval_purpose='learning_dynamics',
                            review_k=int(k_final),
                            logger=logger,
                            raw_results_out=raw_results_n,
                            raw_gt_annos_out=raw_gt_n,
                            raw_box_type_3d_out=raw_box_type_n,
                            raw_box_mode_3d_out=raw_box_mode_n,
                        )

                        stats_n = _compute_ld_match_stats(
                            list(raw_gt_n),
                            list(raw_results_n),
                            seat_keys=list(ld_natural_seat_keys),
                            eval_class_indices=list(ld_new_classes),
                            box_type_3d=raw_box_type_n[0],
                            box_mode_3d=raw_box_mode_n[0],
                        )
                        out_path = ld_dir / f"natural_stats_k{int(k_final)}.json"
                        payload = {
                            'stage_id': int(stage_id),
                            'review_k': int(k_final),
                            'split': 'train(natural)',
                            'iou_thr': float(ld_iou_thr),
                            'iou_mode': str(ld_iou_mode),
                            'iou_thrs': [float(ld_iou_thr)],
                            'eps': float(ld_eps),
                            'q_metric': str(ld_stats_q_metric),
                            'q_formula': str(ld_stats_q_formula),
                            'new_classes': [int(x) for x in ld_new_classes],
                            'num_seats': int(len(stats_n)),
                            'seats': list(stats_n or []),
                        }
                        with open(out_path, "w") as f:
                            json.dump(payload, f, indent=2)
                        logger.info(
                            f"Learning-dynamics scoring: saved k={int(k_final)} natural stats to {out_path}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Learning-dynamics scoring: failed to compute/write natural stats at k={int(k_final)}: {e}"
                        )

            # Ensure downstream code sees the correct final "epoch" count.
            stage_cfg.runner.max_epochs = int(num_segments)

            # Learning-dynamics scoring: aggregate per-seat trajectories and
            # prepare scores for the memory bank update (rank0 writes, others load).
            if learning_dynamics_enabled:
                assert ld_dir is not None, "Learning-dynamics scoring: ld_dir is None."
                K_review = int(num_segments) - 1
                K = int(num_segments)
                assert K_review >= 1, (
                    "selection_strategy in ['learning_dynamics', "
                    "'learning_dynamics_design1', 'learning_dynamics_design2'] "
                    "requires at least one review point. "
                    f"Got num_segments={int(num_segments)} (K_review={K_review})."
                )
                assert K >= 2, (K, K_review)
                ld_design_stage1_files = (
                    get_ld_design_stage1_filenames(learning_dynamics_strategy_key)
                    if learning_dynamics_design_strategy else
                    None
                )
                if learning_dynamics_design_strategy:
                    assert isinstance(ld_design_stage1_files, dict), (
                        "LD design strategy requires stage1 filename profile."
                    )
                    scores_path = ld_dir / str(ld_design_stage1_files.get('score_filename'))
                    traj_path = ld_dir / str(ld_design_stage1_files.get('trajectories_filename'))
                else:
                    scores_path = ld_dir / "learning_dynamics_scores.json"
                    traj_path = ld_dir / "learning_dynamics_q_trajectories.json"

                if is_main_process:
                    try:
                        from mmdet3d.utils.learning_dynamics_scoring import (
                            compute_learning_dynamics_scores,
                            compute_learning_dynamics_design1_scores,
                            topk_seats,
                        )

                        def _load_seats(p: Path) -> List[Dict[str, Any]]:
                            with open(p, "r") as f:
                                payload = json.load(f)
                            seats = payload.get('seats', []) or []
                            if not isinstance(seats, list):
                                raise TypeError(
                                    f"Expected seats=list in {p}, got {type(seats)}"
                                )
                            return list(seats)

                        def _scores_map_to_records(score_by_seat: Dict[str, Dict[int, float]]) -> List[Dict[str, Any]]:
                            records = []
                            for sid, by_stage in (score_by_seat or {}).items():
                                if not isinstance(by_stage, dict):
                                    continue
                                for st, sc in by_stage.items():
                                    try:
                                        st_i = int(st)
                                        sc_f = float(sc)
                                    except Exception:
                                        continue
                                    if not np.isfinite(sc_f):
                                        sc_f = 0.0
                                    records.append(dict(scene_id=str(sid), save_stage=int(st_i), score=float(sc_f)))
                            records.sort(key=lambda d: (str(d.get('scene_id')), int(d.get('save_stage', -1))))
                            return records

                        # Load per-k seat stats (produced during stage-start/review/stage-end evaluations).
                        mem_metrics_by_k = {}
                        nat_metrics_by_k = {}
                        for k in range(0, K + 1):
                            mem_path = ld_dir / f"memory_stats_k{int(k)}.json"
                            nat_path = ld_dir / f"natural_stats_k{int(k)}.json"
                            if not mem_path.exists():
                                raise FileNotFoundError(f"Missing memory stats: {mem_path}")
                            if not nat_path.exists():
                                raise FileNotFoundError(f"Missing natural stats: {nat_path}")
                            mem_metrics_by_k[int(k)] = _load_seats(mem_path)
                            nat_metrics_by_k[int(k)] = _load_seats(nat_path)

                        # Determine the slope interval for legacy replay priority.
                        # slow_saturation ignores slope and always uses full k=0..K.
                        if str(ld_replay_priority_policy_type) == 'legacy_between':
                            if ld_slope_k_start_cfg is not None:
                                slope_k_start = int(ld_slope_k_start_cfg)
                                slope_k_end = int(ld_slope_k_end_cfg)
                                slope_policy = 'config'
                            else:
                                if int(K) >= 2:
                                    # Default: first→last in-training eval (k=1..K).
                                    slope_k_start = 1
                                    slope_k_end = int(K)
                                    slope_policy = 'default_first_last_in_training'
                                else:
                                    # Edge case: K=1 has one in-training eval. Use k=0→1.
                                    slope_k_start = 0
                                    slope_k_end = 1
                                    slope_policy = 'default_stage_start_to_first_update'

                            assert 0 <= int(slope_k_start) < int(slope_k_end) <= int(K), (
                                f"Invalid slope interval: (k_start={int(slope_k_start)}, "
                                f"k_end={int(slope_k_end)}) for K={int(K)}."
                            )
                        else:
                            # Keep metadata deterministic while explicitly marking it ignored.
                            slope_k_start = 0
                            slope_k_end = int(K)
                            slope_policy = 'ignored_full_trajectory'

                        mem_scores = compute_learning_dynamics_scores(
                            mem_metrics_by_k,
                            old_classes=list(ld_old_classes),
                            new_classes=[],
                            iou_mode=str(ld_iou_mode),
                            alpha=float(ld_alpha),
                            beta=float(ld_beta),
                            slope_k_start=int(slope_k_start),
                            slope_k_end=int(slope_k_end),
                            object_count_cap=int(ld_object_count_cap),
                            eps=float(ld_eps),
                            replay_priority_policy=dict(ld_replay_priority_policy),
                            return_trajectories=True,
                        )
                        nat_scores = compute_learning_dynamics_scores(
                            nat_metrics_by_k,
                            old_classes=[],
                            new_classes=list(ld_new_classes),
                            iou_mode=str(ld_iou_mode),
                            alpha=float(ld_alpha),
                            beta=float(ld_beta),
                            slope_k_start=int(slope_k_start),
                            slope_k_end=int(slope_k_end),
                            object_count_cap=int(ld_object_count_cap),
                            eps=float(ld_eps),
                            replay_priority_policy=dict(ld_replay_priority_policy),
                            return_trajectories=True,
                        )

                        forgetness_by_seat = mem_scores.get('forgetness_by_seat', {}) or {}
                        replay_priority_by_seat = nat_scores.get('replay_priority_by_seat', {}) or {}

                        # Convert to structured records for JSON summaries.
                        forgetness_records = _scores_map_to_records(forgetness_by_seat)
                        replay_priority_records = _scores_map_to_records(replay_priority_by_seat)

                        seed_base = int(seed) if seed is not None else 0
                        top_forgetness = topk_seats(
                            forgetness_by_seat,
                            int(ld_report_topk),
                            seed=seed_base + 65000 + 10 * int(stage_id),
                        )
                        top_replay_priority = topk_seats(
                            replay_priority_by_seat,
                            int(ld_report_topk),
                            seed=seed_base + 66000 + 10 * int(stage_id),
                        )

                        summary = dict(
                            stage_id=int(stage_id),
                            K=int(K),
                            K_review=int(K_review),
                            iou_thr=float(ld_iou_thr),
                            iou_mode=str(ld_iou_mode),
                            iou_thrs=[float(ld_iou_thr)],
                            eps=float(ld_eps),
                            object_count_cap=int(ld_object_count_cap),
                            q_metric=str(ld_stats_q_metric),
                            q_formula=str(ld_stats_q_formula),
                            slope_k_start=int(slope_k_start),
                            slope_k_end=int(slope_k_end),
                            slope_window=int(slope_k_end) - int(slope_k_start),
                            slope_policy=str(slope_policy),
                            replay_priority_policy=dict(ld_replay_priority_policy),
                            old_classes=[int(x) for x in ld_old_classes],
                            new_classes=[int(x) for x in ld_new_classes],
                            learning_dynamics_forgetness_by_seat=dict(
                                forgetness_by_seat
                            ),
                            learning_dynamics_replay_priority_by_seat=dict(
                                replay_priority_by_seat
                            ),
                            forgetness_seat_scores=forgetness_records,
                            replay_priority_seat_scores=replay_priority_records,
                        )
                        if not ld_path_only_logging and not learning_dynamics_design_strategy:
                            summary.update(
                                forgetness_by_class=(
                                    mem_scores.get('forgetness_by_class', {}) or {}
                                ),
                                replay_priority_by_class=(
                                    nat_scores.get('replay_priority_by_class', {}) or {}
                                ),
                                top_forgetness_seats=top_forgetness,
                                top_replay_priority_seats=top_replay_priority,
                            )
                        with open(scores_path, "w") as f:
                            json.dump(summary, f, indent=2)

                        if not ld_path_only_logging and not learning_dynamics_design_strategy:
                            # Also write a compact per-seat score artifact under
                            # memory_bank/scores/ to avoid confusion with per-class AP metrics.
                            try:
                                score_dir = incremental_cfg.paths.memory_bank_scores_dir()
                                score_dir.mkdir(parents=True, exist_ok=True)
                                per_seat_path = (
                                    score_dir / f"learning_dynamics_stage_{int(stage_id)}_seat_scores.json"
                                )
                                per_seat_payload = dict(
                                    stage_id=int(stage_id),
                                    iou_thr=float(ld_iou_thr),
                                    iou_mode=str(ld_iou_mode),
                                    iou_thrs=[float(ld_iou_thr)],
                                    eps=float(ld_eps),
                                    object_count_cap=int(ld_object_count_cap),
                                    q_metric=str(ld_stats_q_metric),
                                    q_formula=str(ld_stats_q_formula),
                                    slope_k_start=int(slope_k_start),
                                    slope_k_end=int(slope_k_end),
                                    slope_window=int(slope_k_end) - int(slope_k_start),
                                    slope_policy=str(slope_policy),
                                    replay_priority_policy=dict(
                                        ld_replay_priority_policy
                                    ),
                                    old_classes=[int(x) for x in ld_old_classes],
                                    new_classes=[int(x) for x in ld_new_classes],
                                    learning_dynamics_forgetness_by_seat=dict(
                                        forgetness_by_seat
                                    ),
                                    learning_dynamics_replay_priority_by_seat=dict(
                                        replay_priority_by_seat
                                    ),
                                    forgetness_seat_scores=list(forgetness_records),
                                    replay_priority_seat_scores=list(replay_priority_records),
                                    top_forgetness_seats=list(top_forgetness),
                                    top_replay_priority_seats=list(top_replay_priority),
                                    source=str(scores_path),
                                    note=(
                                        "Per-seat aggregated learning-dynamics scores used for memory-bank actions "
                                        "(prune old seats by forgetness; admit current seats by replay priority). "
                                        "Per-class AP metrics live in memory_bank/scores/stage_{t}_metrics.json."
                                    ),
                                )
                                with open(per_seat_path, "w") as f:
                                    json.dump(per_seat_payload, f, indent=2)
                                logger.info(
                                    f"Learning-dynamics scoring: wrote per-seat seat-scores to {per_seat_path}"
                                )
                            except Exception:
                                pass

                            # Raw q trajectories (for analysis; can be large).
                            traj = dict(
                                stage_id=int(stage_id),
                                K=int(K),
                                K_review=int(K_review),
                                slope_k_start=int(slope_k_start),
                                slope_k_end=int(slope_k_end),
                                slope_policy=str(slope_policy),
                                replay_priority_policy=dict(
                                    ld_replay_priority_policy
                                ),
                                object_count_cap=int(ld_object_count_cap),
                                memory_q_trajectories=mem_scores.get('q_trajectories', {}) or {},
                                natural_q_trajectories=nat_scores.get('q_trajectories', {}) or {},
                            )
                            with open(traj_path, "w") as f:
                                json.dump(traj, f)

                        learning_dynamics_scores_file_for_memory_update = str(scores_path)
                        learning_dynamics_scores_for_memory_update = dict(
                            stage_id=int(stage_id),
                            iou_thr=float(ld_iou_thr),
                            iou_mode=str(ld_iou_mode),
                            iou_thrs=[float(ld_iou_thr)],
                            eps=float(ld_eps),
                            object_count_cap=int(ld_object_count_cap),
                            q_metric=str(ld_stats_q_metric),
                            q_formula=str(ld_stats_q_formula),
                            slope_k_start=int(slope_k_start),
                            slope_k_end=int(slope_k_end),
                            replay_priority_policy=dict(ld_replay_priority_policy),
                            old_classes=[int(x) for x in ld_old_classes],
                            new_classes=[int(x) for x in ld_new_classes],
                            forgetness_by_seat=forgetness_by_seat,
                            replay_priority_by_seat=replay_priority_by_seat,
                        )

                        logger.info(f"Learning-dynamics scoring: saved summary to {scores_path}")
                        try:
                            top_f = top_forgetness[:5]
                            top_u = top_replay_priority[:5]
                            logger.info(
                                "Learning-dynamics scoring: top forgetness seats: " +
                                ", ".join(
                                    f"{x.get('scene_id')}@{int(x.get('save_stage', -1))}:{float(x.get('score', 0.0)):.3f}"
                                    for x in top_f
                                )
                            )
                            logger.info(
                                "Learning-dynamics scoring: top replay-priority seats: " +
                                ", ".join(
                                    f"{x.get('scene_id')}@{int(x.get('save_stage', -1))}:{float(x.get('score', 0.0)):.3f}"
                                    for x in top_u
                                )
                            )
                        except Exception:
                            pass

                        if learning_dynamics_design_strategy:
                            # LD design payload (Design-1/Design-2): class need + per-seat class terms.
                            merged_metrics_by_k = {}
                            for k in range(0, K + 1):
                                merged_metrics_by_k[int(k)] = (
                                    list(mem_metrics_by_k.get(int(k), []) or [])
                                    + list(nat_metrics_by_k.get(int(k), []) or [])
                                )
                            design_class_ids = sorted(
                                set([int(x) for x in ld_old_classes] + [int(x) for x in ld_new_classes])
                            )
                            design_scores = compute_learning_dynamics_design1_scores(
                                merged_metrics_by_k,
                                class_ids=list(design_class_ids),
                                new_classes=list(ld_new_classes),
                                q_metric=str(ld_design1_q_metric),
                                eps=float(ld_eps),
                                design_version=int(ld_design_version),
                            )
                            seat_terms = design_scores.get('seat_class_terms', {}) or {}
                            if not isinstance(seat_terms, dict) or not seat_terms:
                                raise RuntimeError(
                                    f"{learning_dynamics_strategy_key} scoring produced empty seat_class_terms; "
                                    f"stage_id={int(stage_id)}."
                                )
                            summary_design = dict(
                                stage_id=int(stage_id),
                                K=int(K),
                                K_review=int(K_review),
                                iou_thr=float(ld_iou_thr),
                                iou_mode=str(ld_iou_mode),
                                iou_thrs=[float(ld_iou_thr)],
                                eps=float(ld_eps),
                                q_metric=str(ld_design1_q_metric),
                                q_formula=str(ld_stats_q_formula),
                                object_count_cap=int(ld_object_count_cap),
                                class_ids=[int(x) for x in design_scores.get('class_ids', [])],
                                new_classes=[int(x) for x in design_scores.get('new_classes', [])],
                                class_need={
                                    str(int(k)): float(v)
                                    for k, v in (design_scores.get('class_need', {}) or {}).items()
                                },
                                class_q_current={
                                    str(int(k)): float(v)
                                    for k, v in (design_scores.get('class_q_current', {}) or {}).items()
                                },
                                class_q_best={
                                    str(int(k)): float(v)
                                    for k, v in (design_scores.get('class_q_best', {}) or {}).items()
                                },
                                seat_class_terms=seat_terms,
                            )
                            with open(scores_path, "w") as f:
                                json.dump(summary_design, f, indent=2)

                            learning_dynamics_scores_file_for_memory_update = str(scores_path)
                            learning_dynamics_design_payload_for_memory_update = dict(
                                stage_id=int(stage_id),
                                iou_thr=float(ld_iou_thr),
                                iou_mode=str(ld_iou_mode),
                                iou_thrs=[float(ld_iou_thr)],
                                eps=float(ld_eps),
                                q_metric=str(ld_design1_q_metric),
                                q_formula=str(ld_stats_q_formula),
                                object_count_cap=int(ld_object_count_cap),
                                class_ids=[int(x) for x in design_scores.get('class_ids', [])],
                                new_classes=[int(x) for x in design_scores.get('new_classes', [])],
                                class_need=dict(design_scores.get('class_need', {}) or {}),
                                class_q_current=dict(design_scores.get('class_q_current', {}) or {}),
                                class_q_best=dict(design_scores.get('class_q_best', {}) or {}),
                                seat_class_terms=seat_terms,
                            )
                            ld_design_label = (
                                "Design-2"
                                if learning_dynamics_strategy_key == LD_DESIGN2_STRATEGY
                                else "Design-1"
                            )
                            logger.info(
                                f"Learning-dynamics {ld_design_label} scoring: saved summary to "
                                f"{scores_path} (q_metric={ld_design1_q_metric}, "
                                f"classes={len(design_class_ids)}, seats={len(seat_terms)})"
                            )
                    except Exception as e:
                        raise RuntimeError(
                            f"Learning-dynamics scoring aggregation failed: {e}"
                        ) from e

                _dist_barrier()
                if learning_dynamics_design_strategy:
                    try:
                        loaded_ld_design = _load_learning_dynamics_design1_scores_for_memory_update(
                            scores_path,
                            require_stage_id=int(stage_id),
                            strategy_name=str(learning_dynamics_strategy_key),
                            score_file_label=(
                                str(ld_design_stage1_files.get('score_filename'))
                                if isinstance(ld_design_stage1_files, dict) else
                                'learning_dynamics_design_scores.json'
                            ),
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load {learning_dynamics_strategy_key} scores from {scores_path}: {e}"
                        ) from e
                    learning_dynamics_scores_file_for_memory_update = str(scores_path)
                    learning_dynamics_design_payload_for_memory_update = dict(loaded_ld_design)
                else:
                    try:
                        loaded_ld_scores = _load_learning_dynamics_scores_for_memory_update(
                            scores_path,
                            require_stage_id=int(stage_id),
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load learning-dynamics scores from {scores_path}: {e}"
                        ) from e
                    learning_dynamics_scores_file_for_memory_update = str(scores_path)
                    learning_dynamics_scores_for_memory_update = dict(loaded_ld_scores)

            # Stage-level forgetness for memory bank eviction:
            # use old-class AP drops between stage-start and stage-end eval on train(memory).
            if forgetness_eviction_enabled:
                class_drops_out = None
                if is_main_process:
                    if stage_start_ap_forgetness is None:
                        raise RuntimeError(
                            "Forgetness eviction is enabled, but stage-start AP snapshot "
                            "was not captured (stage_start_ap_forgetness is None)."
                        )
                    stage_end_ap_forgetness = dict(prev_ap)
                    class_drops = {}
                    keys = set(stage_start_ap_forgetness.keys()) | set(stage_end_ap_forgetness.keys())
                    for cid in keys:
                        cid = int(cid)
                        ap0 = float(stage_start_ap_forgetness.get(cid, 0.0))
                        ap1 = float(stage_end_ap_forgetness.get(cid, 0.0))
                        class_drops[cid] = float(max(0.0, ap0 - ap1))

                    with open(forgetness_drop_file, "w") as f:
                        json.dump(class_drops, f, indent=2)
                    class_drops_out = dict(class_drops)

                    try:
                        vals = [float(v) for v in class_drops.values()]
                        if vals:
                            logger.info(
                                f"SUNRGBD Forgetness: saved start/end class drops to {forgetness_drop_file}"
                            )
                            logger.info(
                                f"SUNRGBD Forgetness: class_drop stats: "
                                f"min={min(vals):.4f}, max={max(vals):.4f}, mean={sum(vals)/len(vals):.4f}"
                            )
                            top = sorted(class_drops.items(), key=lambda kv: kv[1], reverse=True)[:8]
                            for cid, dv in top:
                                name = mappings['model_idx_to_name'].get(int(cid), f"class_{cid}")
                                logger.info(f"  class {int(cid)} ({name}): drop={float(dv):.4f}")
                    except Exception:
                        pass

                _dist_barrier()
                if not is_main_process:
                    try:
                        with open(forgetness_drop_file, "r") as f:
                            loaded = json.load(f)
                        class_drops_out = loaded
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load SUNRGBD forgetness class drops from {forgetness_drop_file}: {e}"
                        ) from e

                assert class_drops_out is not None, (
                    "forgetness_eviction_enabled=True, but class_drops_out is None."
                )
                forgetness_class_drops_for_memory_update = class_drops_out

        elif do_segment:
            # Segment A: always train 1 epoch first
            logger.info(
                f"Segment A training: Stage {stage_id}, epochs=1, times={times_a}"
            )
            stage_cfg_A = copy.deepcopy(stage_cfg)
            stage_cfg_A.runner.max_epochs = 1
            # Ensure loader reflects Segment A repeats
            stage_cfg_A.data.train.times = times_a
            train_model(
                model,
                datasets,
                stage_cfg_A,
                distributed=distributed,
                validate=True,
                timestamp=timestamp,
                meta={'stage_id': stage_id, 'stage_name': stage_name, 'segment': 'A'})
            _dist_barrier()

            # Locate mid checkpoint
            checkpoint_dir = stage_cfg.work_dir
            mid_checkpoint = osp.join(checkpoint_dir, 'epoch_1.pth')
            if not osp.exists(mid_checkpoint):
                mid_checkpoint = osp.join(checkpoint_dir, 'latest.pth')

            # Pseudo-to-pseudo consistency measurement on a subsample (rank0 only)
            if is_main_process:
                try:
                    logger.info(
                        f"Pseudo consistency: computing prev vs mid for Stage {stage_id}"
                    )
                    # Allowed classes: all old classes (up to stage_id-1)
                    old_classes = []
                    for sdef in stage_definitions[:stage_idx]:
                        old_classes.extend(sdef['class_indices'])
                    old_classes = sorted(list(set(int(x) for x in old_classes)))

                    # Prepare Segment-B dataset CFG upfront (used even on fallback).
                    # Keep a clean base copy so fallback is not affected by any
                    # adaptation mutations below.
                    # Use the already-prepared training dataset cfg (has
                    # stage_definition/mappings/memory settings applied) as the
                    # base for Segment B.
                    train_dataset_cfg_B_base = copy.deepcopy(train_dataset_cfg)
                    train_dataset_cfg_B_base.times = times_b if (stage_epochs == 1 and times_b > 0) else getattr(train_dataset_cfg_B_base, 'times', 1)

                    # Sample scenes: mix natural and replay
                    sample_size = int(getattr(incremental_cfg, 'retention_eval', {}).get('sample_size', 80)) if hasattr(incremental_cfg, 'retention_eval') else 80
                    sample_mode = getattr(incremental_cfg, 'retention_eval', {}).get('sample_mode', 'proportional') if hasattr(incremental_cfg, 'retention_eval') else 'proportional'
                    rng = np.random.RandomState(42 + stage_id)
                    # Access innermost dataset (RepeatDataset wrapper has no data_infos)
                    inner_ds = get_innermost_dataset(datasets[0])
                    nat_indices = []
                    rep_indices = []
                    for i, info in enumerate(inner_ds.data_infos):
                        if isinstance(info, dict):
                            sid_meta = info.get('scene_identity', None)
                            if isinstance(sid_meta, dict):
                                t = sid_meta.get('type', None)
                                if t == 'natural_only':
                                    nat_indices.append(i)
                                elif t == 'memory_only':
                                    rep_indices.append(i)
                                else:
                                    if info.get('is_replay', False):
                                        rep_indices.append(i)
                                    else:
                                        nat_indices.append(i)
                            else:
                                if info.get('is_replay', False):
                                    rep_indices.append(i)
                                else:
                                    nat_indices.append(i)
                        else:
                            # Unknown structure: default to natural bucket
                            nat_indices.append(i)
                    chosen = []
                    if sample_mode == 'balanced' and nat_indices and rep_indices:
                        half = sample_size // 2
                        k_nat = min(half, len(nat_indices))
                        k_rep = min(sample_size - k_nat, len(rep_indices))
                        chosen = rng.choice(nat_indices, size=k_nat, replace=False).tolist() + \
                                 rng.choice(rep_indices, size=k_rep, replace=False).tolist()
                    else:
                        pool = nat_indices + rep_indices
                        k = min(sample_size, len(pool))
                        if k > 0:
                            chosen = rng.choice(pool, size=k, replace=False).tolist()
                        else:
                            logger.warning(
                                "No scenes available for pseudo-consistency sampling; "
                                "skipping adaptation"
                            )
                            chosen = []

                    # Default thresholds and per-class overrides start empty for measurement
                    default_thr = float(getattr(stage_cfg.data.train, 'pseudo_label_config', {}).get('confidence_threshold', 0.45)) if hasattr(stage_cfg.data.train, 'pseudo_label_config') else 0.45
                    class_thrs = None

                    # Resolve prev checkpoint for Stage t-1
                    prev_ckpt = None
                    # Prefer saved previous stage checkpoint in experiment
                    prev_stage_dir = incremental_cfg.paths.checkpoints_dir(stage_id - 1)
                    cand = prev_stage_dir / 'latest.pth'
                    if cand.exists():
                        prev_ckpt = str(cand)
                    elif args.start_stage == stage_id and args.checkpoint_path:
                        prev_ckpt = args.checkpoint_path
                    if prev_ckpt is None or not osp.exists(prev_ckpt):
                        logger.warning(
                            f"Prev checkpoint not found for Stage {stage_id}; "
                            f"skipping consistency eval"
                        )
                    else:
                        # Use model.cfg assigned to TR3D inference earlier
                        base_infer_cfg = getattr(model, 'cfg', None)
                        if base_infer_cfg is None:
                            # Fallback to stage_cfg (works with init_model)
                            base_infer_cfg = stage_cfg
                        focus_classes = []
                        focus_scenes = []
                        if chosen:
                            # Generate prev and mid pseudo sets
                            prev_pseudo = generate_pseudo_set_for_indices(
                                base_infer_cfg, prev_ckpt, inner_ds, chosen, old_classes, class_thresholds=class_thrs, default_thr=default_thr)
                            mid_pseudo = generate_pseudo_set_for_indices(
                                base_infer_cfg, mid_checkpoint, inner_ds, chosen, old_classes, class_thresholds=class_thrs, default_thr=default_thr)

                            # Persist pseudo sets
                            prev_out = incremental_cfg.paths.pseudo_set_file(stage_id, 'prev')
                            mid_out = incremental_cfg.paths.pseudo_set_file(stage_id, 'mid')
                            save_jsonl([{ 'scene_id': k, **v } for k, v in prev_pseudo.items()], prev_out)
                            save_jsonl([{ 'scene_id': k, **v } for k, v in mid_pseudo.items()], mid_out)

                            # Compute drop
                            drop_metrics = compute_consistency_drop(prev_pseudo, mid_pseudo, old_classes, iou_thr=0.25)
                            ret_out = incremental_cfg.paths.retention_scores_file(stage_id, 'mid')
                            os.makedirs(ret_out.parent, exist_ok=True)
                            with open(ret_out, 'w') as f:
                                json.dump(drop_metrics, f, indent=2)
                            logger.info(f"Retention (pseudo) scores saved: {ret_out}")

                            # Prepare adaptation sets
                            # Top-K classes by drop weighted by prev counts
                            drop_by_class = drop_metrics['drop_by_class']
                            # Sort by drop desc, then prev count desc
                            sorted_classes = sorted(drop_by_class.items(), key=lambda kv: (kv[1], drop_metrics['per_class_prev_counts'].get(int(kv[0]), 0)), reverse=True)
                            topk = int(getattr(incremental_cfg, 'retention_eval', {}).get('topk_classes', 5)) if hasattr(incremental_cfg, 'retention_eval') else 5
                            focus_classes = [int(c) for c, _ in sorted_classes[:topk]]
                            drop_by_scene = drop_metrics['drop_by_scene']
                            # Focus scenes: top by drop among chosen
                            sorted_scenes = sorted(drop_by_scene.items(), key=lambda kv: kv[1], reverse=True)
                            focus_scenes = [s for s, _ in sorted_scenes[:max(10, len(sorted_scenes)//4)]]

                    # Build Segment B dataset config with adaptation
                    train_dataset_cfg_B = copy.deepcopy(train_dataset_cfg_B_base)
                    # Adjust times for Segment B (if split)
                    train_dataset_cfg_B.times = times_b if (stage_epochs == 1 and times_b > 0) else getattr(train_dataset_cfg_B, 'times', 1)
                    # Push replay focus and adaptive memory ratio
                    mem_ratio_cfg = getattr(incremental_cfg, 'retention_policy', {}) if hasattr(incremental_cfg, 'retention_policy') else {}
                    scene_focus_share = float(mem_ratio_cfg.get('scene_focus_share', 0.7))
                    mem_step = float(mem_ratio_cfg.get('memory_ratio_step', 0.10))
                    mem_cap = float(mem_ratio_cfg.get('memory_ratio_max', 0.50))
                    # Increase target_memory_ratio slightly within cap
                    current_mem_ratio = float(getattr(train_dataset_cfg_B.dataset, 'target_memory_ratio', 0.0)) if hasattr(train_dataset_cfg_B, 'dataset') else float(getattr(train_dataset_cfg_B, 'target_memory_ratio', 0.0))
                    new_mem_ratio = min(mem_cap, current_mem_ratio + mem_step) if current_mem_ratio > 0 else min(mem_cap, mem_step)
                    # Attach replay focus
                    if hasattr(train_dataset_cfg_B, 'dataset'):
                        train_dataset_cfg_B.dataset.target_memory_ratio = new_mem_ratio
                        train_dataset_cfg_B.dataset.replay_focus = dict(
                            class_ids=focus_classes if prev_ckpt and chosen else [],
                            scene_ids=focus_scenes if prev_ckpt and chosen else [],
                            focus_share=scene_focus_share,
                        )
                        # Per-class pseudo thresholds for focus classes.
                        # Only applicable when pseudo labels are explicitly enabled
                        # by config (e.g., ScanNet pipelines). SUNRGBD currently
                        # asserts on pseudo_label_config to fail fast.
                        if bool(getattr(train_dataset_cfg_B.dataset, 'use_pseudo_labels', False)):
                            pl_cfg = getattr(train_dataset_cfg_B.dataset, 'pseudo_label_config', None)
                            if pl_cfg is None:
                                train_dataset_cfg_B.dataset.pseudo_label_config = dict()
                                pl_cfg = train_dataset_cfg_B.dataset.pseudo_label_config
                            base_thr = float(pl_cfg.get('confidence_threshold', 0.45))
                            thr_delta = float(mem_ratio_cfg.get('pseudo_threshold_delta', -0.05))
                            class_thrs_map = {
                                int(c): max(0.05, base_thr + thr_delta)
                                for c in (focus_classes if prev_ckpt and chosen else [])
                            }
                            if class_thrs_map:
                                pl_cfg['class_thresholds'] = class_thrs_map
                    else:
                        # Legacy path if dataset attr not nested
                        train_dataset_cfg_B.target_memory_ratio = new_mem_ratio

                    # Rebuild Segment B dataset
                    datasets_B = [build_dataset(train_dataset_cfg_B)]

                    # Train Segment B (continue from mid checkpoint)
                    logger.info(
                        f"Segment B training: Stage {stage_id}, epochs=1, "
                        f"times={getattr(train_dataset_cfg_B, 'times', 1)}"
                    )
                    stage_cfg_B = copy.deepcopy(stage_cfg)
                    # Run exactly one additional epoch instead of resuming runner state
                    stage_cfg_B.runner.max_epochs = 1
                    stage_cfg_B.resume_from = None
                    stage_cfg_B.load_from = mid_checkpoint
                    # Apply dataset override
                    stage_cfg_B.data.train = train_dataset_cfg_B
                    train_model(
                        model,
                        datasets_B,
                        stage_cfg_B,
                        distributed=distributed,
                        validate=True,
                        timestamp=timestamp,
                        meta={'stage_id': stage_id, 'stage_name': stage_name, 'segment': 'B'})
                    _dist_barrier()
                except Exception as e:
                    logger.warning(f"Pseudo consistency/adaptation failed: {e}")
                    # Fallback: continue with normal training for remaining epoch(s)
                    logger.info("Continuing with standard training path for one additional epoch...")
                    stage_cfg_F = copy.deepcopy(stage_cfg)
                    stage_cfg_F.runner.max_epochs = 1
                    stage_cfg_F.resume_from = None
                    stage_cfg_F.load_from = mid_checkpoint
                    stage_cfg_F.data.train = train_dataset_cfg_B_base
                    train_model(
                        model,
                        [build_dataset(train_dataset_cfg_B_base)],
                        stage_cfg_F,
                        distributed=distributed,
                        validate=True,
                        timestamp=timestamp,
                        meta={'stage_id': stage_id, 'stage_name': stage_name, 'segment': 'B_fallback'})
                    _dist_barrier()
        else:
            # Non-segmented: standard training
            train_model(
                model,
                datasets,
                stage_cfg,
                distributed=distributed,
                validate=True,
                timestamp=timestamp,
                meta={'stage_id': stage_id, 'stage_name': stage_name})
            _dist_barrier()

        # CRITICAL FIX: Unwrap model from MMDataParallel after training
        # This ensures clean model.head access for the next stage
        if hasattr(model, 'module'):
            model = model.module
            if log_debug:
                logger.info("Unwrapped model from MMDataParallel for next stage")

        # FORCE FINAL EVALUATION: Ensure we evaluate at the last epoch regardless of interval (rank0 only)
        final_epoch = stage_cfg.runner.max_epochs
        eval_interval = stage_cfg.evaluation.get('interval', 1)
        should_force_eval = (final_epoch % eval_interval) != 0

        if is_main_process and should_force_eval:
            logger.info(
                f"Force final evaluation: last epoch {final_epoch} not evaluated by "
                f"interval {eval_interval} (stage {stage_id})"
            )
            logger.info(
                f"Running final evaluation to ensure we have metrics for stage {stage_id}"
            )

            try:
                from mmdet3d.apis import init_model, single_gpu_test
                from mmdet.datasets import build_dataloader

                # Find the latest checkpoint (stage_cfg.work_dir already points to the stage directory)
                checkpoint_dir = stage_cfg.work_dir
                latest_checkpoint = osp.join(checkpoint_dir, f'epoch_{final_epoch}.pth')
                if not osp.exists(latest_checkpoint):
                    latest_checkpoint = osp.join(checkpoint_dir, 'latest.pth')

                if not osp.exists(latest_checkpoint):
                    logger.warning(
                        f"No checkpoint found for evaluation at: {latest_checkpoint}"
                    )
                    raise FileNotFoundError(f"Checkpoint not found: {latest_checkpoint}")

                # Load model from checkpoint for clean initialization
                logger.info(f"Loading checkpoint: {latest_checkpoint}")
                eval_model = init_model(stage_cfg, latest_checkpoint, device='cuda:0')
                eval_model.eval()

                # Build validation dataloader
                val_dataset_cfg = stage_cfg.data.val
                val_dataset = build_dataset(val_dataset_cfg)
                val_dataloader = build_dataloader(
                    val_dataset,
                    samples_per_gpu=1,
                    workers_per_gpu=stage_cfg.data.workers_per_gpu,
                    dist=False,
                    shuffle=False)

                # Run evaluation
                logger.info(
                    f"Running final evaluation on {len(val_dataset)} validation scenes..."
                )
                outputs = single_gpu_test(eval_model, val_dataloader, show=False)

                # Get evaluation results
                eval_kwargs = stage_cfg.evaluation.copy()
                eval_kwargs.pop('interval', None)  # Remove interval for direct evaluation
                if 'save_best' in eval_kwargs:
                    eval_kwargs.pop('save_best', None)  # Remove save_best for direct evaluation

                eval_results = val_dataset.evaluate(outputs, **eval_kwargs)

                # Log results
                if eval_results:
                    logger.info(f"Final evaluation results for Stage {stage_id}:")
                    if log_debug:
                        for metric, value in eval_results.items():
                            if isinstance(value, (int, float)):
                                logger.info(f"  {metric}: {value:.4f}")
                            else:
                                logger.info(f"  {metric}: {value}")
                    else:
                        key_metrics = []
                        for k in ('mAP_0.25', 'mAP_0.50', 'mAR_0.25', 'mAR_0.50'):
                            v = eval_results.get(k, None)
                            if isinstance(v, (int, float)):
                                key_metrics.append(f"{k}={float(v):.4f}")
                        if key_metrics:
                            logger.info(
                                "Final evaluation metrics: " + ", ".join(key_metrics)
                            )
                else:
                    logger.warning("No evaluation results returned")

            except Exception as e:
                logger.error(f"Force final evaluation failed: {e}")
                logger.info(
                    "This may not affect training, but final metrics are missing"
                )
        else:
            if log_debug:
                logger.info(
                    f"Final epoch {final_epoch} will be evaluated by normal "
                    f"interval {eval_interval}"
                )

        # PSEUDO LABEL GENERATION: Generate pseudo labels for NEXT stage using current trained model
        is_last_stage = (stage_idx == len(stage_definitions) - 1)
        next_stage_id = stage_id + 1

        if is_main_process and hasattr(datasets[0], 'use_pseudo_labels') and datasets[0].use_pseudo_labels and not is_last_stage:
            logger.info(
                f"Generating pseudo labels for next stage {next_stage_id}"
            )

            try:
                # Set model to evaluation mode
                model.eval()

                # Generate pseudo labels for next stage using current trained model
                next_pseudo_labels_file = str(incremental_cfg.paths.pseudo_label_file(next_stage_id))

                # Check if already exists
                if os.path.exists(next_pseudo_labels_file):
                    logger.info(
                        f"Pseudo labels for stage {next_stage_id} already exist at: "
                        f"{next_pseudo_labels_file}"
                    )
                else:
                    current_classes = stage_cfg.evaluation_classes_mask.sum().item()
                    logger.info(
                        f"Generating pseudo labels for next stage using current "
                        f"{current_classes}-class model"
                    )

                    # Ensure model has a consistent dynamic-head config for inference
                    # Use the stage config (already dynamic_head and sequential GCI) to avoid class order mismatch
                    model.cfg = stage_cfg
                    if log_debug:
                        logger.info(
                            "Model config set to stage config for pseudo-label generation "
                            "(dynamic_head)"
                        )

                    # Generate pseudo labels using the trained model
                    # No need for work_dir swapping - dataset uses unified paths now
                    pseudo_labels = datasets[0].generate_pseudo_labels_for_training_scenes(
                        model=model,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )

                    if pseudo_labels:
                        logger.info(
                            f"Generated pseudo labels for {len(pseudo_labels)} scenes"
                        )
                        # Save for next stage (directory created automatically by IncrementalPaths)
                        with open(next_pseudo_labels_file, 'wb') as f:
                            pickle.dump(pseudo_labels, f)
                        logger.info(
                            f"Pseudo labels saved to {next_pseudo_labels_file}"
                        )

                        # VALIDATE GENERATED PSEUDO LABELS FOR NEXT STAGE
                        try:
                            from mmdet3d.utils.validate_pseudo_labels import quick_validate_on_generation
                            # Use default confidence threshold for post-training generation
                            default_confidence = 0.45
                            if log_debug:
                                logger.info(
                                    f"Validating post-training pseudo labels for Stage {next_stage_id}..."
                                )
                            quick_validate_on_generation(pseudo_labels, next_stage_id, default_confidence)
                        except Exception as e:
                            logger.warning(
                                f"Post-training pseudo label validation failed: {e}"
                            )
                            logger.warning("   This is non-critical - pseudo labels saved successfully")

                        # Verify file was saved successfully
                        if os.path.exists(next_pseudo_labels_file):
                            logger.info(
                                f"Confirmed: pseudo labels file exists and is ready for "
                                f"stage {next_stage_id}"
                            )
                        else:
                            logger.warning(
                                f"Failed to save pseudo labels file at {next_pseudo_labels_file}"
                            )
                    else:
                        logger.warning(
                            "No pseudo labels generated - check model predictions and confidence threshold"
                        )

            except Exception as e:
                logger.error(f"Pseudo label generation failed: {e}")
                logger.info("Next stage will train without pseudo labels")
        elif is_last_stage and is_main_process:
            if log_debug:
                logger.info(
                    f"Stage {stage_id} is the last stage - skipping pseudo label generation"
                )

        # Synchronize so all ranks see next-stage pseudo labels before building next stage
        _dist_barrier()

        # UPDATE OBJECT MEMORY BANK: crop this stage's objects after training.
        # Persist the final cohort too. Although the current protocol has no
        # next consumer, a complete final bank is needed for auditability,
        # exact budget accounting, and extension with future stages.
        if object_memory_bank is not None:
            train_inner = get_innermost_dataset(datasets[0])
            if not hasattr(train_inner, 'update_memory_bank_from_stage'):
                raise RuntimeError(
                    f'{type(train_inner).__name__} does not implement '
                    'update_memory_bank_from_stage required by object replay.')
            logger.info(f'Updating object memory bank from stage {stage_id}')
            object_ld_payload = None
            if object_memory_bank.selection_strategy_name == LD_DESIGN2_STRATEGY:
                if not isinstance(
                        learning_dynamics_design_payload_for_memory_update, dict):
                    raise RuntimeError(
                        'Object Design-2 selection requires the current stage '
                        'learning-dynamics payload before bank population.')
                object_ld_payload = dict(
                    learning_dynamics_design_payload_for_memory_update)
            train_inner.update_memory_bank_from_stage(
                learning_dynamics_design2_payload=object_ld_payload)
            if is_main_process:
                state_path = osp.join(
                    str(work_dir), 'object_memory_bank',
                    f'object_memory_bank_stage_{stage_id}.pkl')
                object_memory_bank.save_state(state_path, stage_id=stage_id)
                logger.info(f'Object memory bank state saved to {state_path}')
            _dist_barrier()

        # UPDATE SCENE MEMORY BANK: Add scenes from completed stage
        # Skip update after last stage since those scenes won't be used

        if scene_memory_bank is not None and not is_last_stage:
            logger.info(f"Updating scene memory bank from stage {stage_id}")
            # Under-learning insertion (train-only): compute new-class AP
            # on the natural pool and pass into the memory bank update.
            underlearning_class_ap_for_memory_update = None
            underlearning_new_classes_for_memory_update = None
            ld_forgetness_by_seat_for_memory_update = None
            ld_replay_priority_by_seat_for_memory_update = None
            ld_design_payload_for_memory_update = None
            if learning_dynamics_design_strategy:
                if not isinstance(learning_dynamics_design_payload_for_memory_update, dict):
                    raise RuntimeError(
                        f"selection_strategy='{learning_dynamics_strategy_key}' requires per-stage "
                        "design scores before memory-bank update, but no payload was loaded."
                    )
                loaded_stage_id = learning_dynamics_design_payload_for_memory_update.get(
                    'stage_id', None
                )
                if loaded_stage_id is None or int(loaded_stage_id) != int(stage_id):
                    raise RuntimeError(
                        f"{learning_dynamics_strategy_key} score stage mismatch before memory-bank update: "
                        f"expected stage_id={int(stage_id)}, got {loaded_stage_id!r}. "
                        f"source={learning_dynamics_scores_file_for_memory_update}"
                    )
                ld_design_payload_for_memory_update = dict(
                    learning_dynamics_design_payload_for_memory_update
                )
                seat_terms = ld_design_payload_for_memory_update.get('seat_class_terms', None)
                if not isinstance(seat_terms, dict) or not seat_terms:
                    raise RuntimeError(
                        f"selection_strategy='{learning_dynamics_strategy_key}' requires non-empty "
                        f"seat_class_terms for stage {int(stage_id)} memory update. "
                        f"source={learning_dynamics_scores_file_for_memory_update}"
                    )
            elif learning_dynamics_strategy:
                if not isinstance(learning_dynamics_scores_for_memory_update, dict):
                    raise RuntimeError(
                        "selection_strategy='learning_dynamics' requires per-stage LD scores "
                        "before memory-bank update, but no scores were loaded."
                    )
                loaded_stage_id = learning_dynamics_scores_for_memory_update.get('stage_id', None)
                if loaded_stage_id is None or int(loaded_stage_id) != int(stage_id):
                    raise RuntimeError(
                        "learning-dynamics score stage mismatch before memory-bank update: "
                        f"expected stage_id={int(stage_id)}, got {loaded_stage_id!r}. "
                        f"source={learning_dynamics_scores_file_for_memory_update}"
                    )
                ld_forgetness_by_seat_for_memory_update = (
                    learning_dynamics_scores_for_memory_update.get('forgetness_by_seat', None)
                )
                ld_replay_priority_by_seat_for_memory_update = (
                    learning_dynamics_scores_for_memory_update.get('replay_priority_by_seat', None)
                )
                if (not isinstance(ld_replay_priority_by_seat_for_memory_update, dict)
                        or not ld_replay_priority_by_seat_for_memory_update):
                    raise RuntimeError(
                        "selection_strategy='learning_dynamics' requires non-empty replay-priority "
                        f"seat scores for stage {int(stage_id)} memory update. "
                        f"source={learning_dynamics_scores_file_for_memory_update}"
                    )

            underlearning_enabled = bool(
                scene_memory_bank is not None and
                getattr(scene_memory_bank, 'underlearning_insertion_enabled', False)
            )

            # When reviewing is enabled, the stage training dataset may be a resampled
            # subset (duplicates, fewer natural scenes). Memory bank updates must use
            # the full natural pool for this stage to avoid selection bias.
            if (reviewing_enabled and incremental_dataset_type in (
                        'IncrementalSUNRGBDDataset', 'IncrementalScanNetDataset')
                    and stage_id >= 2):
                try:
                    logger.info(
                        f"Updating {incremental_dataset_type} memory bank using full natural pool "
                        "(reviewing enabled)"
                    )
                    natural_cfg = copy.deepcopy(train_dataset_cfg)
                    natural_cfg.times = 1
                    # Build a natural-only dataset snapshot (no replay, no reviewing resampling).
                    natural_cfg.dataset.scene_memory_bank = None
                    natural_cfg.dataset.scene_dedup_strategy = 'keep_both'
                    natural_cfg.dataset.reviewing_sampling = dict(enabled=False)
                    natural_ds = build_dataset(natural_cfg)
                    natural_inner = get_innermost_dataset(natural_ds)
                    natural_infos = list(getattr(natural_inner, 'data_infos', []))

                    # Compute under-learning weights from train split (natural pool).
                    if underlearning_enabled:
                        try:
                            # New classes are the current stage classes.
                            new_classes = [int(x) for x in stage_definition.get('class_indices', [])]
                            underlearning_new_classes_for_memory_update = list(new_classes)

                            # Read settings from config (stored on the memory bank).
                            ul_cfg = getattr(scene_memory_bank, 'underlearning_insertion', {}) or {}
                            ap_iou_thr = float(ul_cfg.get('ap_iou_thr', 0.25))
                            max_eval = ul_cfg.get('eval_max_scenes', None)
                            seed_offset = int(ul_cfg.get('eval_seed_offset', 11000))
                            if max_eval is not None:
                                max_eval = int(max_eval)
                                assert max_eval > 0, max_eval

                            eval_infos = natural_infos
                            if max_eval is not None and len(eval_infos) > max_eval:
                                rng = np.random.RandomState(int(seed) + seed_offset + 100 * int(stage_id))
                                idx = rng.choice(len(eval_infos), size=int(max_eval), replace=False)
                                eval_infos = [eval_infos[i] for i in sorted(idx.tolist())]

                            # Build class name list aligned with current head size.
                            try:
                                target_model = model.module if hasattr(model, 'module') else model
                                n_cls = int(getattr(getattr(target_model, 'head', None), 'n_classes', stage_cfg.model.head.n_classes))
                            except Exception:
                                n_cls = int(stage_cfg.model.head.n_classes)
                            class_names = [mappings['model_idx_to_name'][i] for i in range(int(n_cls))]

                            iou_key = f"{float(ap_iou_thr):.2f}"
                            if is_main_process:
                                logger.info(
                                    f"Under-learning: evaluating train(natural) for stage {stage_id} "
                                    f"new_classes={new_classes}, iou_thr={float(ap_iou_thr):.2f}, "
                                    f"scenes={len(eval_infos)}/{len(natural_infos)}"
                                )
                                metrics_ul = _sunrgbd_eval_memory_subset(
                                    model=model,
                                    stage_cfg=stage_cfg,
                                    data_infos=eval_infos,
                                    eval_class_indices=new_classes,
                                    class_names=class_names,
                                    iou_thrs=(float(ap_iou_thr),),
                                    stage_idx=max(0, int(stage_id) - 1),
                                    split_name=f"train(stage_{stage_id}_natural)",
                                    eval_purpose='underlearning',
                                    logger=logger,
                                )

                                ap_by_class = {}
                                weights = {}
                                for cid in new_classes:
                                    cid = int(cid)
                                    name = mappings['model_idx_to_name'].get(cid, f"class_{cid}")
                                    ap = float(metrics_ul.get(f"{name}_AP_{iou_key}", 0.0))
                                    if not np.isfinite(ap):
                                        ap = 0.0
                                    ap = float(max(0.0, min(1.0, ap)))
                                    ap_by_class[cid] = ap
                                    weights[cid] = float(max(0.0, min(1.0, 1.0 - ap)))

                                # Persist for reproducibility and for other ranks to load.
                                score_dir = incremental_cfg.paths.memory_bank_scores_dir()
                                score_dir.mkdir(parents=True, exist_ok=True)
                                out_path = score_dir / f"underlearning_stage_{stage_id}_train_ap.json"
                                payload = {
                                    'stage_id': int(stage_id),
                                    'split': 'train(natural)',
                                    'iou_thr': float(ap_iou_thr),
                                    'new_classes': [int(x) for x in new_classes],
                                    'num_scenes_total': int(len(natural_infos)),
                                    'num_scenes_evaluated': int(len(eval_infos)),
                                    'ap_by_class': {str(int(k)): float(v) for k, v in ap_by_class.items()},
                                    'underlearning_weight_by_class': {str(int(k)): float(v) for k, v in weights.items()},
                                    'score_mode': str(ul_cfg.get('score_mode', 'object_count_sum')),
                                }
                                with open(out_path, 'w') as f:
                                    json.dump(payload, f, indent=2)
                                underlearning_class_ap_for_memory_update = ap_by_class
                                logger.info(f"Under-learning: saved train AP to {out_path}")

                            _dist_barrier()
                            if not is_main_process:
                                in_path = (
                                    incremental_cfg.paths.memory_bank_scores_dir()
                                    / f"underlearning_stage_{stage_id}_train_ap.json"
                                )
                                with open(in_path, 'r') as f:
                                    loaded = json.load(f)
                                ap_loaded = loaded.get('ap_by_class', {}) or {}
                                ap_by_class = {}
                                for k, v in ap_loaded.items():
                                    try:
                                        ap_by_class[int(k)] = float(v)
                                    except Exception:
                                        continue
                                underlearning_class_ap_for_memory_update = ap_by_class
                        except Exception as e:
                            raise RuntimeError(
                                f"Under-learning train evaluation failed at stage {stage_id}: {e}"
                            ) from e

                    # Cumulative seen classes up to current stage.
                    seen_classes = []
                    for sd in stage_definitions:
                        if int(sd.get('stage_id', 0)) <= int(stage_id):
                            seen_classes.extend([int(x) for x in sd.get('class_indices', [])])
                    seen_classes = sorted(set(seen_classes))

                    scene_memory_bank.add_stage_scenes(
                        stage_id=int(stage_id),
                        scene_infos=natural_infos,
                        seen_classes=seen_classes,
                        mappings=mappings,
                        dataset_ref=natural_inner if is_main_process else None,
                        scene_metrics=None,
                        forgetness_class_drops=forgetness_class_drops_for_memory_update,
                        underlearning_class_ap=underlearning_class_ap_for_memory_update,
                        underlearning_new_classes=underlearning_new_classes_for_memory_update,
                        learning_dynamics_forgetness_by_seat=ld_forgetness_by_seat_for_memory_update,
                        learning_dynamics_replay_priority_by_seat=ld_replay_priority_by_seat_for_memory_update,
                        learning_dynamics_design1_payload=(
                            ld_design_payload_for_memory_update
                            if learning_dynamics_strategy_key == LD_DESIGN1_STRATEGY
                            else None
                        ),
                        learning_dynamics_design2_payload=(
                            ld_design_payload_for_memory_update
                            if learning_dynamics_strategy_key == LD_DESIGN2_STRATEGY
                            else None
                        ),
                    )
                except Exception as e:
                    if learning_dynamics_enabled:
                        raise RuntimeError(
                            f"{incremental_dataset_type} memory bank update failed in strict LD mode "
                            f"at stage {stage_id}: {e}"
                        ) from e
                    logger.warning(
                        f"{incremental_dataset_type} reviewing memory bank update failed: {e}. "
                        "Falling back to updating from the training dataset."
                    )
                    train_dataset = datasets[0]
                    innermost_dataset = get_innermost_dataset(train_dataset)
                    if hasattr(innermost_dataset, 'update_scene_memory_bank_from_stage'):
                        innermost_dataset.update_scene_memory_bank_from_stage(
                            model=model,
                            forgetness_class_drops=forgetness_class_drops_for_memory_update,
                            underlearning_class_ap=underlearning_class_ap_for_memory_update,
                            underlearning_new_classes=underlearning_new_classes_for_memory_update,
                            learning_dynamics_forgetness_by_seat=ld_forgetness_by_seat_for_memory_update,
                            learning_dynamics_replay_priority_by_seat=ld_replay_priority_by_seat_for_memory_update,
                            learning_dynamics_design1_payload=(
                                ld_design_payload_for_memory_update
                                if learning_dynamics_strategy_key == LD_DESIGN1_STRATEGY
                                else None
                            ),
                            learning_dynamics_design2_payload=(
                                ld_design_payload_for_memory_update
                                if learning_dynamics_strategy_key == LD_DESIGN2_STRATEGY
                                else None
                            ),
                            dataset_ref=innermost_dataset if is_main_process else None,
                        )
                    else:
                        logger.warning(
                            f"Inner dataset {type(innermost_dataset)} does not "
                            f"implement update_scene_memory_bank_from_stage; "
                            f"skipping memory bank update for this stage."
                        )
            else:
                train_dataset = datasets[0]  # Get the training dataset
                # Update scene memory bank with scenes from this stage
                # Handle RepeatDataset wrapper consistently
                innermost_dataset = get_innermost_dataset(train_dataset)
                # Provide the trained model so uncertainty/diversity-based
                # policies can use prediction statistics when configured.
                if hasattr(innermost_dataset, 'update_scene_memory_bank_from_stage'):
                    # Compute under-learning weights from train split (natural pool)
                    # in the non-reviewing path (rank0 only; others load from file).
                    if underlearning_enabled:
                        try:
                            new_classes = [int(x) for x in stage_definition.get('class_indices', [])]
                            underlearning_new_classes_for_memory_update = list(new_classes)

                            ul_cfg = getattr(scene_memory_bank, 'underlearning_insertion', {}) or {}
                            ap_iou_thr = float(ul_cfg.get('ap_iou_thr', 0.25))
                            max_eval = ul_cfg.get('eval_max_scenes', None)
                            seed_offset = int(ul_cfg.get('eval_seed_offset', 11000))
                            if max_eval is not None:
                                max_eval = int(max_eval)
                                assert max_eval > 0, max_eval

                            # Natural pool from the stage dataset (exclude replay/merged seats).
                            infos_all = list(getattr(innermost_dataset, 'data_infos', []) or [])
                            natural_infos = [
                                info for info in infos_all
                                if isinstance(info, dict)
                                and not bool(info.get('is_replay', False))
                                and not bool(info.get('is_merged', False))
                            ]
                            eval_infos = natural_infos
                            if max_eval is not None and len(eval_infos) > max_eval:
                                rng = np.random.RandomState(int(seed) + seed_offset + 100 * int(stage_id))
                                idx = rng.choice(len(eval_infos), size=int(max_eval), replace=False)
                                eval_infos = [eval_infos[i] for i in sorted(idx.tolist())]

                            # Build class name list aligned with current head size.
                            try:
                                target_model = model.module if hasattr(model, 'module') else model
                                n_cls = int(getattr(getattr(target_model, 'head', None), 'n_classes', stage_cfg.model.head.n_classes))
                            except Exception:
                                n_cls = int(stage_cfg.model.head.n_classes)
                            class_names = [mappings['model_idx_to_name'][i] for i in range(int(n_cls))]

                            iou_key = f"{float(ap_iou_thr):.2f}"
                            if is_main_process:
                                logger.info(
                                    f"Under-learning: evaluating train(natural) for stage {stage_id} "
                                    f"new_classes={new_classes}, iou_thr={float(ap_iou_thr):.2f}, "
                                    f"scenes={len(eval_infos)}/{len(natural_infos)}"
                                )
                                metrics_ul = _sunrgbd_eval_memory_subset(
                                    model=model,
                                    stage_cfg=stage_cfg,
                                    data_infos=eval_infos,
                                    eval_class_indices=new_classes,
                                    class_names=class_names,
                                    iou_thrs=(float(ap_iou_thr),),
                                    stage_idx=max(0, int(stage_id) - 1),
                                    split_name=f"train(stage_{stage_id}_natural)",
                                    eval_purpose='underlearning',
                                    logger=logger,
                                )

                                ap_by_class = {}
                                weights = {}
                                for cid in new_classes:
                                    cid = int(cid)
                                    name = mappings['model_idx_to_name'].get(cid, f"class_{cid}")
                                    ap = float(metrics_ul.get(f"{name}_AP_{iou_key}", 0.0))
                                    if not np.isfinite(ap):
                                        ap = 0.0
                                    ap = float(max(0.0, min(1.0, ap)))
                                    ap_by_class[cid] = ap
                                    weights[cid] = float(max(0.0, min(1.0, 1.0 - ap)))

                                score_dir = incremental_cfg.paths.memory_bank_scores_dir()
                                score_dir.mkdir(parents=True, exist_ok=True)
                                out_path = score_dir / f"underlearning_stage_{stage_id}_train_ap.json"
                                payload = {
                                    'stage_id': int(stage_id),
                                    'split': 'train(natural)',
                                    'iou_thr': float(ap_iou_thr),
                                    'new_classes': [int(x) for x in new_classes],
                                    'num_scenes_total': int(len(natural_infos)),
                                    'num_scenes_evaluated': int(len(eval_infos)),
                                    'ap_by_class': {str(int(k)): float(v) for k, v in ap_by_class.items()},
                                    'underlearning_weight_by_class': {str(int(k)): float(v) for k, v in weights.items()},
                                    'score_mode': str(ul_cfg.get('score_mode', 'object_count_sum')),
                                }
                                with open(out_path, 'w') as f:
                                    json.dump(payload, f, indent=2)
                                underlearning_class_ap_for_memory_update = ap_by_class
                                logger.info(f"Under-learning: saved train AP to {out_path}")

                            _dist_barrier()
                            if not is_main_process:
                                in_path = (
                                    incremental_cfg.paths.memory_bank_scores_dir()
                                    / f"underlearning_stage_{stage_id}_train_ap.json"
                                )
                                with open(in_path, 'r') as f:
                                    loaded = json.load(f)
                                ap_loaded = loaded.get('ap_by_class', {}) or {}
                                ap_by_class = {}
                                for k, v in ap_loaded.items():
                                    try:
                                        ap_by_class[int(k)] = float(v)
                                    except Exception:
                                        continue
                                underlearning_class_ap_for_memory_update = ap_by_class
                        except Exception as e:
                            raise RuntimeError(
                                f"Under-learning train evaluation failed at stage {stage_id}: {e}"
                            ) from e
                    innermost_dataset.update_scene_memory_bank_from_stage(
                        model=model,
                        forgetness_class_drops=forgetness_class_drops_for_memory_update,
                        underlearning_class_ap=underlearning_class_ap_for_memory_update,
                        underlearning_new_classes=underlearning_new_classes_for_memory_update,
                        learning_dynamics_forgetness_by_seat=ld_forgetness_by_seat_for_memory_update,
                        learning_dynamics_replay_priority_by_seat=ld_replay_priority_by_seat_for_memory_update,
                        learning_dynamics_design1_payload=(
                            ld_design_payload_for_memory_update
                            if learning_dynamics_strategy_key == LD_DESIGN1_STRATEGY
                            else None
                        ),
                        learning_dynamics_design2_payload=(
                            ld_design_payload_for_memory_update
                            if learning_dynamics_strategy_key == LD_DESIGN2_STRATEGY
                            else None
                        ),
                        dataset_ref=innermost_dataset if is_main_process else None,
                    )
                else:
                    logger.warning(
                        f"Inner dataset {type(innermost_dataset)} does not "
                        f"implement update_scene_memory_bank_from_stage; "
                        f"skipping memory bank update for this stage."
                    )

            logger.info(f"Scene memory bank updated with scenes from stage {stage_id}")
            if log_debug and is_main_process:
                scene_memory_bank.print_summary()

            # Unified replay-scene pseudo behavior is now controlled only by
            # pseudo_label_config.apply_to_memory_scenes and stage-start pseudo files.
            # Legacy MEMORY.ENRICH_PSEUDO_* runtime path is removed.

            _dist_barrier()

            # Save state for debugging
            if is_main_process:
                state_path = str(incremental_cfg.paths.scene_memory_file(stage_id))
                scene_memory_bank.save_state(state_path)
                logger.info(f"Scene memory bank state saved to {state_path}")
        elif is_last_stage:
            if log_debug:
                logger.info(
                    f"Stage {stage_id} is the last stage - skipping scene memory bank update"
                )
        elif scene_memory_bank is None:
            if log_debug:
                logger.info("Scene memory bank disabled - skipping memory bank update")

        # INCREMENTAL LEARNING POST-PROCESSING: Filter results to seen classes only
        if log_debug:
            logger.info("Post-processing evaluation results for incremental learning")

        # Get class names dynamically from mappings (works for any config)
        class_names = [mappings['model_idx_to_name'][i] for i in range(num_classes)]

        # Try to extract the most recent evaluation results from logs
        stage_work_dir_for_logs = stage_cfg.work_dir
        log_files = []
        if os.path.exists(stage_work_dir_for_logs):
            for f in os.listdir(stage_work_dir_for_logs):
                if f.endswith('.log.json'):
                    log_files.append(os.path.join(stage_work_dir_for_logs, f))

        # Find the most recent log file
        if log_files:
            latest_log = max(log_files, key=os.path.getmtime)
            if log_debug:
                logger.info(f"Processing evaluation results from: {latest_log}")

            # Try to extract results and apply incremental learning filtering
            try:
                # use top-level json module (avoid shadowing in function scope)
                with open(latest_log, 'r') as f:
                    lines = f.readlines()

                # Look for the most recent evaluation line (mode = "val")
                original_mAP = None
                current_stage_results = None
                stripped_prefix = ''
                map_key_025 = f"stage_{stage_idx}_mAP_0.25"
                for line in reversed(lines):  # Start from the end to find most recent
                    try:
                        data = json.loads(line.strip())
                        if data.get('mode') != 'val':
                            continue
                        if map_key_025 in data:
                            original_mAP = data[map_key_025]
                            current_stage_results, stripped_prefix = _strip_any_stage_prefix(data)
                            break
                        if 'mAP_0.25' in data:
                            original_mAP = data['mAP_0.25']
                            current_stage_results, stripped_prefix = _strip_any_stage_prefix(data)
                            break
                    except (json.JSONDecodeError, KeyError):
                        continue

                if original_mAP is not None and current_stage_results is not None:
                    if log_debug:
                        logger.info(f"Found evaluation mAP@0.25: {float(original_mAP):.4f}")
                        if stripped_prefix:
                            logger.info(
                                f"Normalized eval metrics by stripping prefix: {stripped_prefix}"
                            )

                    # Store current results for next stage's forgetting analysis
                    stage_results_history[stage_id] = current_stage_results

                    # Persist structured per-class metrics (no overwrite):
                    # 1) Current stage metrics → memory_bank/scores/stage_{stage_id}_metrics.json
                    # 2) Previous cohort evaluated at current stage → memory_bank/scores/cohort_{prev}_evaluated_at_{stage_id}.json
                    try:
                        from datetime import datetime
                        scores_dir = incremental_cfg.paths.memory_bank_scores_dir()
                        if not ld_path_only_logging:
                            scores_dir.mkdir(parents=True, exist_ok=True)

                        # Normalized via `_strip_any_stage_prefix(...)` above.
                        pref = ""

                        def _get_float(key: str, default: float = 0.0) -> float:
                            try:
                                val = current_stage_results.get(key, default)
                            except Exception:
                                val = default
                            try:
                                return float(val)
                            except Exception:
                                return float(default)

                        def _mean(arr):
                            return float(sum(arr) / len(arr)) if arr else float('nan')

                        def build_classes_section_for_indices(indices):
                            out = []
                            for c_idx in indices:
                                c_idx = int(c_idx)
                                name = mappings['model_idx_to_name'].get(c_idx, f"class_{c_idx}")
                                ap25 = _get_float(f"{pref}{name}_AP_0.25", 0.0)
                                ap50 = _get_float(f"{pref}{name}_AP_0.50", 0.0)
                                out.append({
                                    'model_idx': c_idx,
                                    'name': name,
                                    'AP_0.25': float(ap25),
                                    'AP_0.50': float(ap50)
                                })
                            return out

                        # (1) Current stage metrics (all evaluated/seen classes)
                        cls_list = current_stage_results.get(f"{pref}class_list", None)
                        if not isinstance(cls_list, list) or not cls_list:
                            cls_list = sorted({
                                int(x)
                                for sdef in stage_definitions[:stage_idx + 1]
                                for x in sdef.get('class_indices', [])
                            })
                        current_classes_section = build_classes_section_for_indices(cls_list)
                        map25 = _get_float(f"{pref}mAP_0.25", _get_float('mAP_0.25', 0.0))
                        map50 = _get_float(f"{pref}mAP_0.50", _get_float('mAP_0.50', 0.0))
                        current_stage_metrics = {
                            'stage_id': stage_id,
                            'evaluated_at_stage': stage_id,
                            'timestamp': datetime.now().isoformat(),
                            'mAP_0.25': float(map25),
                            'mAP_0.50': float(map50),
                            'classes': current_classes_section
                        }
                        stage_metrics_file_current = incremental_cfg.paths.stage_metrics_file(stage_id)
                        if not ld_path_only_logging:
                            with open(stage_metrics_file_current, 'w') as f:
                                json.dump(current_stage_metrics, f, indent=2)
                            logger.info(f"Saved current stage metrics to: {stage_metrics_file_current}")

                        # (2) Previous cohort evaluated at current stage (unique name to avoid overwrite)
                        if stage_idx > 0:
                            prev_stage_def = stage_definitions[stage_idx - 1]
                            prev_stage_id = int(prev_stage_def.get('stage_id', stage_id - 1))
                            prev_indices = [int(x) for x in prev_stage_def.get('class_indices', [])]
                            prev_classes_section = build_classes_section_for_indices(prev_indices)
                            prev_map25_calc = _mean([float(x.get('AP_0.25', 0.0)) for x in prev_classes_section])
                            prev_map50_calc = _mean([float(x.get('AP_0.50', 0.0)) for x in prev_classes_section])
                            prev_map25 = _get_float(
                                f"{pref}cohort_stage_{prev_stage_id}_mAP_0.25",
                                prev_map25_calc,
                            )
                            prev_map50 = _get_float(
                                f"{pref}cohort_stage_{prev_stage_id}_mAP_0.50",
                                prev_map50_calc,
                            )
                            prev_evaluated_metrics = {
                                'stage_id': prev_stage_id,
                                'evaluated_at_stage': stage_id,
                                'timestamp': datetime.now().isoformat(),
                                'mAP_0.25': float(prev_map25),
                                'mAP_0.50': float(prev_map50),
                                'mAP_0.25_calculated': float(prev_map25_calc),
                                'mAP_0.50_calculated': float(prev_map50_calc),
                                'classes': prev_classes_section
                            }
                            cohort_file = scores_dir / f"cohort_{prev_stage_id}_evaluated_at_{stage_id}.json"
                            if not ld_path_only_logging:
                                with open(cohort_file, 'w') as f:
                                    json.dump(prev_evaluated_metrics, f, indent=2)
                                logger.info(
                                    f"Saved previous cohort evaluated metrics to: {cohort_file}"
                                )

                        # === Stage-wise Base/Novel/Overall aggregation (AP_0.25 / AP_0.50) ===
                        try:
                            # Overall across all evaluated/seen classes
                            overall_ap025_calc = _mean([float(x.get('AP_0.25', 0.0)) for x in current_classes_section])
                            overall_ap050_calc = _mean([float(x.get('AP_0.50', 0.0)) for x in current_classes_section])

                            # Novel cohort is the current stage classes
                            novel_indices = [int(x) for x in stage_definition.get('class_indices', [])]
                            novel_sec = build_classes_section_for_indices(novel_indices)
                            novel_ap025 = _mean([float(x.get('AP_0.25', 0.0)) for x in novel_sec])
                            novel_ap050 = _mean([float(x.get('AP_0.50', 0.0)) for x in novel_sec])

                            # Base cohorts are previous stages
                            counts = {}
                            base_ap025 = {}
                            base_ap050 = {}
                            for sdef in stage_definitions[:stage_idx]:
                                sid = int(sdef.get('stage_id', 0))
                                idxs = [int(x) for x in sdef.get('class_indices', [])]
                                sec = build_classes_section_for_indices(idxs)
                                counts[sid] = len(idxs)
                                base_ap025[sid] = _mean([float(x.get('AP_0.25', 0.0)) for x in sec])
                                base_ap050[sid] = _mean([float(x.get('AP_0.50', 0.0)) for x in sec])

                            # Reported overall from evaluator (if present)
                            overall_ap025_rep = float(map25)
                            overall_ap050_rep = float(map50)

                            # Console summary
                            if base_ap025:
                                base_parts_25 = "; ".join(
                                    [f"Base(s{b})={base_ap025[b]:.4f}" for b in sorted(base_ap025)]
                                )
                            else:
                                base_parts_25 = ""
                            if base_ap050:
                                base_parts_50 = "; ".join(
                                    [f"Base(s{b})={base_ap050[b]:.4f}" for b in sorted(base_ap050)]
                                )
                            else:
                                base_parts_50 = ""
                            summary_25 = f"Overall={overall_ap025_calc:.4f}; Novel={novel_ap025:.4f}"
                            if base_parts_25:
                                summary_25 += f"; {base_parts_25}"
                            summary_50 = f"Overall={overall_ap050_calc:.4f}; Novel={novel_ap050:.4f}"
                            if base_parts_50:
                                summary_50 += f"; {base_parts_50}"
                            logger.info(
                                f"Stage {stage_id} Base/Novel/Overall (AP_0.25): {summary_25}"
                            )
                            logger.info(
                                f"Stage {stage_id} Base/Novel/Overall (AP_0.50): {summary_50}"
                            )

                            # Persist JSON
                            base_novel_overall = {
                                'stage_id': stage_id,
                                'base_ap025': {str(k): float(v) for k, v in base_ap025.items()},
                                'novel_ap025': float(novel_ap025),
                                'overall_ap025': float(overall_ap025_calc),
                                'base_ap050': {str(k): float(v) for k, v in base_ap050.items()},
                                'novel_ap050': float(novel_ap050),
                                'overall_ap050': float(overall_ap050_calc),
                                'counts': {str(k): int(v) for k, v in counts.items()},
                                'overall_ap025_reported': float(overall_ap025_rep),
                                'overall_ap050_reported': float(overall_ap050_rep)
                            }
                            out_json = scores_dir / f"stage_{stage_id}_base_novel_overall.json"
                            if not ld_path_only_logging:
                                with open(out_json, 'w') as f:
                                    json.dump(base_novel_overall, f, indent=2)
                                logger.info(f"Saved base/novel/overall summary to: {out_json}")

                            # Append concise per-stage line (root-level summary log).
                            try:
                                summary_path = Path(work_dir) / f"eval_summary_{timestamp}.log"
                                base25 = ";".join(
                                    f"s{int(k)}={float(v):.4f}" for k, v in sorted(base_ap025.items())
                                )
                                base50 = ";".join(
                                    f"s{int(k)}={float(v):.4f}" for k, v in sorted(base_ap050.items())
                                )
                                line = (
                                    f"{datetime.now().isoformat()} "
                                    f"stage={int(stage_id)} "
                                    f"mAP25={float(overall_ap025_rep):.4f} mAP50={float(overall_ap050_rep):.4f} "
                                    f"overall25={float(overall_ap025_calc):.4f} overall50={float(overall_ap050_calc):.4f} "
                                    f"novel25={float(novel_ap025):.4f} novel50={float(novel_ap050):.4f} "
                                    f"base25={base25} base50={base50}\n"
                                )
                                if is_main_process:
                                    with open(summary_path, "a") as f:
                                        f.write(line)
                            except Exception as e:
                                logger.warning(f"Failed to append eval summary: {e}")
                        except Exception as e:
                            logger.warning(
                                f"Failed to compute/save base/novel/overall summary: {e}"
                            )
                    except Exception as e:
                        logger.warning(f"Failed to persist structured metrics: {e}")

                    # Calculate forgetting metrics if this is not the first stage.
                    # In LD path-only profile, we intentionally skip these
                    # memory_bank/scores diagnostics.
                    if stage_idx > 0 and not ld_path_only_logging:
                        # Get all previous stage classes
                        all_previous_classes = []
                        for prev_idx in range(stage_idx):
                            all_previous_classes.extend(
                                stage_definitions[prev_idx]['class_indices']
                            )

                        # Prefer the stable JSON artifacts (stage_{t}_metrics.json) over
                        # raw log dicts. This avoids any dependence on stage-prefixed keys.
                        prev_stage_id = int(stage_definitions[stage_idx - 1]['stage_id'])
                        prev_stage_metrics_file = Path(
                            incremental_cfg.paths.stage_metrics_file(prev_stage_id)
                        )
                        curr_stage_metrics_file = Path(
                            incremental_cfg.paths.stage_metrics_file(stage_id)
                        )

                        forgetting_metrics = None
                        source = None

                        if (prev_stage_metrics_file.exists()
                                and curr_stage_metrics_file.exists()):
                            try:
                                with open(prev_stage_metrics_file, 'r') as f:
                                    prev_stage_metrics_json = json.load(f)
                                with open(curr_stage_metrics_file, 'r') as f:
                                    curr_stage_metrics_json = json.load(f)
                                forgetting_metrics = (
                                    calculate_forgetting_metrics_from_stage_metrics_json(
                                        prev_stage_metrics_json,
                                        curr_stage_metrics_json,
                                        all_previous_classes,
                                        mappings,
                                        logger,
                                        previous_stage_id=prev_stage_id,
                                        current_stage_id=stage_id,
                                        verbose=log_debug,
                                    )
                                )
                                source = 'stage_metrics_json'
                            except Exception as e:
                                logger.warning(
                                    f"Failed to load stage metrics JSON for forgetting: {e}"
                                )
                                forgetting_metrics = None

                        if not forgetting_metrics:
                            # Fallback: use in-memory normalized eval dicts (less stable).
                            if prev_stage_id in stage_results_history:
                                previous_stage_results = stage_results_history[prev_stage_id]
                                if isinstance(previous_stage_results, dict):
                                    previous_stage_results, _ = _strip_any_stage_prefix(
                                        previous_stage_results)
                                current_results_for_forgetting = current_stage_results
                                forgetting_metrics = calculate_forgetting_metrics(
                                    previous_stage_results,
                                    current_results_for_forgetting,
                                    all_previous_classes,
                                    mappings,
                                    logger,
                                    previous_stage_id=prev_stage_id,
                                    current_stage_id=stage_id,
                                    verbose=log_debug,
                                )
                                source = 'log_dict'

                        if forgetting_metrics:
                            forgetting_metrics['source'] = source

                            forgetting_file = str(
                                incremental_cfg.paths.forgetting_metrics_file(stage_id)
                            )
                            with open(forgetting_file, 'w') as f:
                                json.dump(forgetting_metrics, f, indent=2)
                            logger.info(
                                f"Forgetting metrics saved to: {forgetting_file}")
                        else:
                            logger.warning(
                                f"Skipping forgetting metrics for stage {stage_id}: "
                                f"missing previous stage metrics for stage {prev_stage_id}. "
                                f"(If resuming with --start-stage {args.start_stage}, you may need "
                                f"a val baseline for stage {prev_stage_id}.)"
                            )
                    elif stage_idx > 0 and log_debug:
                        logger.info(
                            "LD artifact profile ('ld_path_only'): skipping forgetting_metrics "
                            "JSON artifacts."
                        )

                else:
                    logger.warning("Could not find mAP results in log file")

            except Exception as e:
                logger.warning(f"Error processing log file: {e}")
        else:
            logger.warning("No log files found for post-processing")

        # Note: Memory bank updates are handled differently for scene-based approach
        # Scene-based updates are done via train_dataset.update_scene_memory_bank_from_stage()
        # which was already called earlier if scene_memory_bank is present

        # Create empty pseudo label file for next stage if this is the last stage
        if is_last_stage and is_main_process and hasattr(datasets[0], 'use_pseudo_labels') and datasets[0].use_pseudo_labels:
            next_stage_id = stage_id + 1
            pseudo_label_dir = Path(work_dir) / "pseudo_labels"
            pseudo_label_dir.mkdir(exist_ok=True)

            # Create empty pseudo label file for consistency
            empty_pseudo_file = pseudo_label_dir / f"stage_{next_stage_id}_no_next_stage.pkl"
            if not empty_pseudo_file.exists():
                with open(empty_pseudo_file, 'wb') as f:
                    pickle.dump({}, f)
                logger.info(
                    f"Created empty pseudo label file for consistency: {empty_pseudo_file}"
                )

                # Update metadata
                metadata_file = pseudo_label_dir / "pseudo_labels_metadata.json"
                _create_pseudo_labels_metadata(
                    metadata_file=metadata_file,
                    stage_id=next_stage_id,
                    pseudo_label_file=empty_pseudo_file,
                    config_suffix="no_next_stage",
                    confidence_threshold=0.0,
                    source_global=False,
                    global_source=None,
                    generation_checkpoint="N/A - last stage",
                    logger=logger
                )

        logger.info(f"Stage {stage_id} ({stage_name}) completed!")

    logger.info(f"{'='*20} INCREMENTAL TRAINING COMPLETED {'='*20}")
    logger.info(f"All {len(stage_definitions)} stages completed successfully!")
    logger.info(f"Final model saved in: {work_dir}")

    # Final forgetting summary across all stages.
    # In LD path-only profile, skip writing memory_bank/scores forgetting artifacts.
    if len(stage_results_history) > 1 and not ld_path_only_experiment:
        logger.info("="*80)
        logger.info("FINAL FORGETTING SUMMARY ACROSS ALL STAGES")
        logger.info("="*80)

        # Calculate overall forgetting from first to last stage
        first_stage_id = stage_definitions[0]['stage_id']
        last_stage_id = stage_definitions[-1]['stage_id']

        if first_stage_id in stage_results_history and last_stage_id in stage_results_history:
            # use top-level json module for final forgetting save
            first_stage_results = stage_results_history[first_stage_id]
            last_stage_results = stage_results_history[last_stage_id]
            first_stage_classes = stage_definitions[0]['class_indices']

            overall_forgetting = calculate_forgetting_metrics(
                first_stage_results,
                last_stage_results,
                first_stage_classes,
                mappings,
                logger,
                verbose=log_debug,
            )

            # Save overall forgetting metrics
            overall_forgetting_file = str(incremental_cfg.paths.forgetting_metrics_file())
            with open(overall_forgetting_file, 'w') as f:
                json.dump(overall_forgetting, f, indent=2)
            logger.info(
                f"Overall forgetting metrics saved to: {overall_forgetting_file}"
            )
    elif len(stage_results_history) > 1 and log_debug:
        logger.info(
            "LD artifact profile ('ld_path_only'): skipping overall_forgetting_metrics.json."
        )

    logger.info("Explicit mapping incremental learning completed!")


if __name__ == '__main__':
    main()
