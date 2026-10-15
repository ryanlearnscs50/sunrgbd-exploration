"""Helpers for writing resolved runtime config snapshots."""

from __future__ import annotations

import os
import pprint
import subprocess
import types
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from mmcv import Config

CFG_SNAPSHOT_SKIP = object()


def sanitize_for_cfg_snapshot(value: Any) -> Any:
    """Convert config values to plain Python types for snapshotting."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # NumPy types.
    try:
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass

    # Torch tensors.
    try:
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                return float(value.detach().cpu().item())
            return value.detach().cpu().tolist()
    except Exception:
        pass

    if isinstance(value, Path):
        return str(value)

    # Drop modules/functions/classes from snapshot.
    if isinstance(value, (types.ModuleType, types.FunctionType, type)):
        return CFG_SNAPSHOT_SKIP

    # mmcv Config objects: snapshot dict content.
    try:
        if isinstance(value, Config):
            return sanitize_for_cfg_snapshot(dict(value))
    except Exception:
        pass

    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            sv = sanitize_for_cfg_snapshot(v)
            if sv is CFG_SNAPSHOT_SKIP:
                continue
            out[str(k)] = sv
        return out

    if isinstance(value, list):
        out_list = []
        for v in value:
            sv = sanitize_for_cfg_snapshot(v)
            if sv is CFG_SNAPSHOT_SKIP:
                continue
            out_list.append(sv)
        return out_list

    if isinstance(value, tuple):
        out_list = []
        for v in value:
            sv = sanitize_for_cfg_snapshot(v)
            if sv is CFG_SNAPSHOT_SKIP:
                continue
            out_list.append(sv)
        return tuple(out_list)

    if isinstance(value, set):
        out_list = []
        for v in value:
            sv = sanitize_for_cfg_snapshot(v)
            if sv is CFG_SNAPSHOT_SKIP:
                continue
            out_list.append(sv)
        try:
            return sorted(out_list)
        except Exception:
            return out_list

    try:
        return repr(value)
    except Exception:
        return CFG_SNAPSHOT_SKIP


def try_get_git_info(repo_root: str) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode('utf-8', errors='replace').strip()
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
        ).decode('utf-8', errors='replace').strip()
        return {
            'commit': str(commit),
            'is_dirty': bool(status),
        }
    except Exception:
        return {}


def write_resolved_config_snapshot(*,
                                   cfg: Config,
                                   dest_path: str,
                                   run_meta: Dict[str, Any]) -> None:
    """Write a resolved, loadable config snapshot to ``dest_path``."""
    base_cfg_obj = cfg.get('base_config', None)
    base_cfg_dict = sanitize_for_cfg_snapshot(base_cfg_obj)
    if base_cfg_dict is CFG_SNAPSHOT_SKIP:
        base_cfg_dict = None

    resolved = {}
    for k, v in cfg.items():
        if str(k) in {'Config', 'base_config', 'paths'}:
            continue
        sv = sanitize_for_cfg_snapshot(v)
        if sv is CFG_SNAPSHOT_SKIP:
            continue
        resolved[str(k)] = sv

    resolved = {k: resolved[k] for k in sorted(resolved.keys())}

    header = (
        "# Auto-generated resolved config snapshot (TR3D)\n"
        "# - Includes cfg_options and CLI args in `run_meta`\n"
        "# - `base_config` is reconstructed as an mmcv Config\n"
        f"# - Generated at: {run_meta.get('timestamp', '')}\n\n"
    )

    run_meta_clean = sanitize_for_cfg_snapshot(run_meta)
    if run_meta_clean is CFG_SNAPSHOT_SKIP:
        run_meta_clean = {}

    run_meta_text = pprint.pformat(run_meta_clean, width=120, sort_dicts=True)
    base_cfg_text = pprint.pformat(base_cfg_dict, width=120, sort_dicts=True)
    resolved_text = pprint.pformat(resolved, width=120, sort_dicts=True)

    body = []
    body.append('run_meta = ' + run_meta_text)
    body.append('')
    body.append('from mmcv import Config as __Config')
    if isinstance(base_cfg_dict, dict):
        body.append(f"base_config = __Config(cfg_dict={base_cfg_text})")
    else:
        body.append('base_config = None')
    body.append('del __Config')
    body.append('')
    body.append(f"__cfg__ = {resolved_text}")
    body.append('globals().update(__cfg__)')
    body.append('del __cfg__')
    body.append('')

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'w') as f:
        f.write(header + '\n'.join(body))
