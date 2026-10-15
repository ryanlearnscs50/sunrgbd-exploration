"""Pseudo-label preparation/validation helpers for incremental scene training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def resolve_previous_stage_checkpoint(checkpoint_dir: Path) -> Path:
    """Use the completed stage checkpoint, regardless of retention policy.

    MMCV writes latest.pth at each save. If that alias is unavailable, select
    the highest numbered epoch, never the first epoch of a multi-epoch stage.
    """
    checkpoint_dir = Path(checkpoint_dir)
    latest = checkpoint_dir / 'latest.pth'
    if latest.is_file():
        return latest
    import re
    candidates = []
    for path in checkpoint_dir.glob('epoch_*.pth'):
        match = re.fullmatch(r'epoch_([1-9][0-9]*)\.pth', path.name)
        if match and path.is_file():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f'No previous-stage checkpoint in {checkpoint_dir}')
    return max(candidates)[1]


def build_pseudo_config_suffix(*,
                               incremental_dataset_type: str,
                               use_scene_memory: bool,
                               scene_memory_config: Optional[Dict[str, Any]],
                               pseudo_label_config: Dict[str, Any]) -> str:
    """Build deterministic pseudo-label filename suffix used across branches."""
    if use_scene_memory and scene_memory_config:
        memory_id = scene_memory_config.get('score_criteria', 'default')
        base = f"pseudo_memory_{memory_id}"
    else:
        base = 'pseudo_only'
    default_conf = 0.5 if incremental_dataset_type == 'IncrementalSUNRGBDDataset' else 0.45
    conf = float(pseudo_label_config.get('confidence_threshold', default_conf))
    conf_str = f"conf{int(conf * 100):02d}"
    return f"{base}_{conf_str}"


def resolve_stage_pseudo_file(*,
                              work_dir: str,
                              stage_id: int,
                              config_suffix: str) -> Path:
    pseudo_label_dir = Path(work_dir) / 'pseudo_labels'
    pseudo_label_dir.mkdir(parents=True, exist_ok=True)
    if config_suffix:
        return pseudo_label_dir / f"stage_{int(stage_id)}_{config_suffix}_pseudo_labels.pkl"
    return pseudo_label_dir / f"stage_{int(stage_id)}_pseudo_labels.pkl"


def validate_pseudo_labels_nonfatal(*,
                                    incremental_dataset_type: str,
                                    pseudo_file: str,
                                    stage_id: int,
                                    pseudo_label_config: Dict[str, Any],
                                    ann_file: Optional[str],
                                    logger,
                                    is_main_process: bool,
                                    log_debug: bool,
                                    source_label: str) -> None:
    """Run dataset-specific pseudo-label validation; never raise to caller."""
    if not is_main_process:
        return

    try:
        if incremental_dataset_type == 'IncrementalScanNetDataset':
            from mmdet3d.utils.validate_pseudo_labels import validate_pseudo_labels_from_file

            confidence_thresh = float(pseudo_label_config.get('confidence_threshold', 0.45))
            if log_debug:
                logger.info(f"Validating {source_label} pseudo labels (ScanNet)...")
            validation_passed = validate_pseudo_labels_from_file(
                str(pseudo_file),
                int(stage_id),
                confidence_thresh,
            )
            if validation_passed:
                if log_debug:
                    logger.info(f"Pseudo label validation passed for Stage {int(stage_id)}")
            else:
                logger.warning(
                    f"Pseudo label validation found issues for Stage {int(stage_id)} - check logs above"
                )
            return

        if incremental_dataset_type == 'IncrementalSUNRGBDDataset':
            try:
                from tools.pseudo_labels.validate_sunrgbd_pseudo_labels import (
                    validate_sunrgbd_pseudo_labels_from_file,
                )

                if log_debug:
                    logger.info(f"Validating {source_label} pseudo labels (SUNRGBD)...")
                validate_sunrgbd_pseudo_labels_from_file(
                    pseudo_file=str(pseudo_file),
                    ann_file=str(ann_file) if ann_file else '',
                    stage_id=int(stage_id),
                )
            except Exception as sun_err:
                logger.warning(f"SUNRGBD pseudo label validation skipped: {sun_err}")
    except Exception as err:
        logger.warning(f"Pseudo label validation failed: {err}")
        logger.warning('   This is non-critical - training will continue')
