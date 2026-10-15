"""Stage-setting inference/validation helpers for incremental training."""

from __future__ import annotations

import hashlib
import json
from os import path as osp
from typing import Any, Dict, List, Optional, Tuple
import sys


def fingerprint_stage_definitions(stage_definitions: List[Dict[str, Any]]) -> str:
    """Stable fingerprint for stage definitions (ids + indices + names)."""
    payload = []
    for sd in stage_definitions:
        payload.append(
            dict(
                stage_id=int(sd.get('stage_id', 0)),
                class_indices=[int(x) for x in sd.get('class_indices', [])],
                class_names=list(sd.get('class_names', [])),
            )
        )
    blob = json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha1(blob).hexdigest()


def infer_sunrgbd_stage_setting(stage_definitions: List[Dict[str, Any]],
                                num_classes: int) -> Optional[str]:
    """Infer known SUNRGBD stage setting labels from stage definitions."""
    try:
        num_classes = int(num_classes)
    except Exception:
        return None
    if num_classes != 40:
        return None

    try:
        stage_ids = [int(sd.get('stage_id', i + 1)) for i, sd in enumerate(stage_definitions)]
    except Exception:
        return None
    if stage_ids != list(range(1, len(stage_definitions) + 1)):
        return None

    sizes: List[int] = []
    concat_indices: List[int] = []
    for sd in stage_definitions:
        indices = [int(x) for x in sd.get('class_indices', [])]
        sizes.append(len(indices))
        concat_indices.extend(indices)
    if concat_indices != list(range(40)):
        return None

    if len(stage_definitions) == 5 and all(s == 8 for s in sizes):
        return 'sunrgbd40_s5_freqorder'
    if len(stage_definitions) == 10 and all(s == 4 for s in sizes):
        return 'sunrgbd40_s10_freqorder_split'
    if len(stage_definitions) == 3 and sizes == [20, 10, 10]:
        return 'sunrgbd40_s3_20_10_10_freqorder'
    return None


def resolve_stage_setting(incremental_cfg,
                          stage_definitions: List[Dict[str, Any]],
                          num_classes: int) -> Tuple[Optional[str], str]:
    """Resolve stage setting from config or infer for known SUNRGBD splits."""
    explicit = None

    try:
        inc_block = incremental_cfg.get('incremental', None)
    except Exception:
        inc_block = getattr(incremental_cfg, 'incremental', None)

    try:
        if inc_block is not None and hasattr(inc_block, 'get'):
            explicit = inc_block.get('stage_setting', None)
    except Exception:
        explicit = None

    if explicit is None:
        try:
            explicit = incremental_cfg.get('stage_setting', None)
        except Exception:
            explicit = getattr(incremental_cfg, 'stage_setting', None)

    if explicit is not None:
        return str(explicit), 'explicit'

    inferred = infer_sunrgbd_stage_setting(stage_definitions, num_classes)
    if inferred is not None:
        return str(inferred), 'inferred'

    return None, 'unset'


def validate_stage_setting_or_raise(*,
                                    stage_setting: Optional[str],
                                    stage_definitions: List[Dict[str, Any]],
                                    repo_root: Optional[str] = None) -> None:
    """Fail-fast sanity checks for known SUNRGBD stage settings."""
    expected_sizes_map = {
        'sunrgbd40_s5_freqorder': [8, 8, 8, 8, 8],
        'sunrgbd40_s10_freqorder_split': [4] * 10,
        'sunrgbd40_s3_20_10_10_freqorder': [20, 10, 10],
    }
    if stage_setting not in expected_sizes_map:
        return

    expected_sizes = list(expected_sizes_map[str(stage_setting)])
    expected_num_stages = len(expected_sizes)

    assert len(stage_definitions) == expected_num_stages, (
        f"{stage_setting}: expected {expected_num_stages} stages, got {len(stage_definitions)}"
    )

    stage_ids = [int(sd.get('stage_id', i + 1)) for i, sd in enumerate(stage_definitions)]
    assert stage_ids == list(range(1, expected_num_stages + 1)), (
        f"{stage_setting}: expected stage_id=1..{expected_num_stages}, got {stage_ids}"
    )

    concat_indices: List[int] = []
    concat_names: List[str] = []
    for idx, sd in enumerate(stage_definitions):
        expected_stage_size = int(expected_sizes[idx])
        indices = [int(x) for x in sd.get('class_indices', [])]
        names = list(sd.get('class_names', []))
        assert len(indices) == expected_stage_size, (
            f"{stage_setting}: stage {sd.get('stage_id')} expected "
            f"{expected_stage_size} classes, got {len(indices)}"
        )
        assert len(names) == expected_stage_size, (
            f"{stage_setting}: stage {sd.get('stage_id')} expected "
            f"{expected_stage_size} class_names, got {len(names)}"
        )
        concat_indices.extend(indices)
        concat_names.extend(names)

    assert concat_indices == list(range(40)), (
        f"{stage_setting}: expected concatenated class_indices to be 0..39, "
        f"got {concat_indices[:10]}... (len={len(concat_indices)})"
    )
    assert len(set(concat_indices)) == 40, f"{stage_setting}: class_indices not disjoint"
    assert len(set(concat_names)) == 40, f"{stage_setting}: class_names not disjoint"

    # Best-effort strict check against canonical SUNRGBD order.
    if repo_root is not None:
        try:
            mapping_dir = osp.join(str(repo_root), 'configs', '_base_', 'class_mappings')
            if mapping_dir not in sys.path:
                sys.path.append(mapping_dir)
            from sunrgbd_40class_mapping import SUNRGBD_40_RAW_TOP40_CLASSES  # type: ignore
            assert concat_names == list(SUNRGBD_40_RAW_TOP40_CLASSES), (
                f"{stage_setting}: class_names concatenation does not match "
                f"SUNRGBD_40_RAW_TOP40_CLASSES"
            )
        except Exception:
            # Keep structural checks even if import is unavailable.
            pass


def log_stage_groups(*,
                     logger,
                     stage_setting: Optional[str],
                     stage_setting_source: str,
                     stage_definitions: List[Dict[str, Any]]) -> None:
    if logger is None:
        return

    label = stage_setting if stage_setting is not None else 'unset'
    logger.info(
        f"Stage setting: {label} (source={stage_setting_source}, "
        f"num_stages={len(stage_definitions)})"
    )
    for sd in stage_definitions:
        sid = int(sd.get('stage_id', -1))
        indices = [int(x) for x in sd.get('class_indices', [])]
        names = list(sd.get('class_names', []))
        if indices:
            idx_str = f"{indices[0]}-{indices[-1]}"
        else:
            idx_str = 'empty'
        logger.info(f"  - Stage {sid}: idx={idx_str}, names={names}")
