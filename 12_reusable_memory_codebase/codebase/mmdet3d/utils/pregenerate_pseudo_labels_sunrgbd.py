"""Pre-generate pseudo labels for SUN RGB-D incremental learning.

This module is SUN RGB-D specific. It generates pseudo labels once at the
beginning of each incremental stage (stage >= 2) using the previous-stage
checkpoint, saves them to a pickle file, and optionally writes a compact JSON
summary for quick inspection.

Pseudo boxes are saved in the same convention as SUNRGBD info PKLs:
`annos['gt_boxes_upright_depth']` is gravity-centred (origin=(0.5, 0.5, 0.5)).

Key correctness detail:
MMDet3D DepthInstance3DBoxes store `tensor` as *bottom-centred* by default.
When saving pseudo boxes to match SUNRGBD GT arrays, we convert:
  z_center := z_bottom + h/2
This is handled explicitly in this module (we keep yaw).
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mmcv import Config

from mmdet3d.apis import inference_detector, init_model
from mmdet3d.datasets.pseudo_label_utils import (
    canonicalize_bottom_center_boxes,
    nms_indices_iou,
)


def _extract_scene_id(info: Dict[str, Any]) -> Optional[str]:
    if 'point_cloud' in info and isinstance(info['point_cloud'], dict):
        if 'lidar_idx' in info['point_cloud']:
            return str(info['point_cloud']['lidar_idx'])
    if 'sample_idx' in info:
        return str(info['sample_idx'])
    if 'scene_id' in info:
        return str(info['scene_id'])
    return None


def _resolve_data_root(cfg: Config) -> Path:
    # Supervised configs typically use RepeatDataset: cfg.data.train.dataset.dataset.data_root
    candidates = []
    try:
        candidates.append(getattr(cfg.data.train.dataset.dataset, 'data_root', None))
    except Exception:
        pass
    try:
        candidates.append(getattr(cfg.data.train.dataset, 'data_root', None))
    except Exception:
        pass
    for c in candidates:
        if c:
            return Path(str(c))
    raise ValueError("Failed to resolve SUNRGBD data_root from config.")


def _filter_infos_for_stage(
    infos: Sequence[Dict[str, Any]],
    stage_classes: Sequence[int],
    *,
    filter_empty_gt: bool,
) -> List[Dict[str, Any]]:
    stage_set = {int(x) for x in stage_classes}
    selected: List[Dict[str, Any]] = []
    for info in infos:
        if not isinstance(info, dict):
            continue
        annos = info.get('annos', None)
        if not isinstance(annos, dict):
            continue
        cls = np.asarray(annos.get('class', []), dtype=np.int64).reshape(-1)
        if cls.size == 0:
            if not filter_empty_gt:
                selected.append(info)
            continue
        keep = np.isin(cls, np.array(sorted(stage_set), dtype=np.int64))
        if bool(keep.any()) or not filter_empty_gt:
            selected.append(info)
    return selected


def _compute_previous_seen_classes(
    stage_definitions: Sequence[Dict[str, Any]],
    stage_id: int,
) -> List[int]:
    prev: List[int] = []
    for sd in stage_definitions:
        sid = int(sd.get('stage_id', 0) or 0)
        if sid < int(stage_id):
            prev.extend([int(x) for x in sd.get('class_indices', [])])
    return sorted(set(prev))


def _to_numpy(x: Any, *, dtype=None) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=dtype if dtype is not None else np.float32)
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    if hasattr(x, 'numpy'):
        arr = x.numpy()
    else:
        arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _bottom_center_to_gravity_center_keep_yaw(boxes: np.ndarray) -> np.ndarray:
    """Convert bottom-centred boxes to gravity-centred (SUNRGBD GT convention).

    Args:
        boxes: (N, 6|7[+]) array in bottom-centred Depth box convention.

    Returns:
        (N, 7) float32 [x, y, z_center, dx, dy, dz, yaw]
        If yaw is missing, yaw=0 is appended.
    """
    if boxes.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Boxes must be 2D with >=6 dims; got {arr.shape}")

    if arr.shape[1] >= 7:
        out = arr[:, :7].copy()
    else:
        out = np.concatenate([arr[:, :6], np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)

    out[:, 2] = out[:, 2] + out[:, 5] * 0.5
    return out.astype(np.float32, copy=False)


def _generate_scene_pseudo(
    *,
    model,
    pts_path: Path,
    previous_seen_classes: Sequence[int],
    confidence_threshold: float,
    nms_iou_thr: float,
    max_pseudo_per_scene: int,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not pts_path.exists():
        return None

    result, _ = inference_detector(model, str(pts_path))
    if not isinstance(result, list) or len(result) == 0 or not isinstance(result[0], dict):
        return None
    pred = result[0]

    boxes_3d = pred.get('boxes_3d', None)
    scores_3d = pred.get('scores_3d', None)
    labels_3d = pred.get('labels_3d', None)
    if boxes_3d is None or scores_3d is None or labels_3d is None:
        return None

    if hasattr(boxes_3d, 'tensor'):
        boxes_np = _to_numpy(boxes_3d.tensor, dtype=np.float32)
    else:
        boxes_np = _to_numpy(boxes_3d, dtype=np.float32)
    scores_np = _to_numpy(scores_3d, dtype=np.float32).reshape(-1)
    labels_np = _to_numpy(labels_3d, dtype=np.int64).reshape(-1)

    if boxes_np.ndim != 2 or boxes_np.shape[0] != scores_np.shape[0] or labels_np.shape[0] != scores_np.shape[0]:
        raise ValueError(
            f"Invalid prediction shapes: boxes={boxes_np.shape}, scores={scores_np.shape}, labels={labels_np.shape}"
        )
    if boxes_np.shape[0] == 0:
        return None

    prev_set = {int(x) for x in previous_seen_classes}
    keep = scores_np >= float(confidence_threshold)
    if prev_set:
        keep &= np.isin(labels_np, np.array(sorted(prev_set), dtype=np.int64))
    if not bool(keep.any()):
        return None

    boxes_gc7 = _bottom_center_to_gravity_center_keep_yaw(boxes_np[keep])
    boxes_aabb6 = canonicalize_bottom_center_boxes(boxes_np[keep])
    scores_f = scores_np[keep]
    labels_f = labels_np[keep]

    # Per-class axis-aligned NMS.
    kept_global: List[int] = []
    for cls in np.unique(labels_f):
        cls = int(cls)
        idxs = np.where(labels_f == cls)[0]
        if idxs.size == 0:
            continue
        keep_local = nms_indices_iou(boxes_aabb6[idxs], scores_f[idxs], iou_thr=float(nms_iou_thr))
        kept_global.extend(idxs[keep_local].tolist())

    if not kept_global:
        return None

    kept_idx = np.array(kept_global, dtype=np.int64)
    # Global top-K by score (deterministic).
    order = np.argsort(scores_f[kept_idx])[::-1]
    kept_idx = kept_idx[order]
    if max_pseudo_per_scene is not None and int(max_pseudo_per_scene) > 0:
        kept_idx = kept_idx[:int(max_pseudo_per_scene)]

    return (
        boxes_gc7[kept_idx].astype(np.float32, copy=False),
        labels_f[kept_idx].astype(np.int64, copy=False),
        scores_f[kept_idx].astype(np.float32, copy=False),
    )


def pregenerate_sunrgbd_pseudo_labels_for_stage(
    *,
    stage_id: int,
    checkpoint_path: str,
    train_ann_file: str,
    stage_definition: Dict[str, Any],
    all_stage_definitions: Sequence[Dict[str, Any]],
    confidence_threshold: float = 0.5,
    nms_iou_thr: float = 0.3,
    max_pseudo_per_scene: int = 100,
    max_scenes: Optional[int] = None,
    seed: int = 0,
    output_dir: str,
    config_suffix: str = "",
    device: str = "cuda:0",
    base_config_path: str = "configs/tr3d/tr3d_sunrgbd-3d-40class.py",
) -> str:
    """Pre-generate pseudo labels for SUNRGBD stage >= 2.

    Returns:
        Path to the saved pseudo label pickle file (string).
    """
    stage_id = int(stage_id)
    if stage_id < 2:
        raise ValueError(f"SUNRGBD pseudo labels are only defined for stage>=2, got stage_id={stage_id}.")

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ann_path = Path(train_ann_file)
    if not ann_path.exists():
        raise FileNotFoundError(f"Train ann_file not found: {train_ann_file}")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Determine file names.
    suffix = str(config_suffix).strip()
    stem = f"stage_{stage_id}_pseudo_labels" if not suffix else f"stage_{stage_id}_{suffix}_pseudo_labels"
    out_pkl = output_root / f"{stem}.pkl"
    out_json = output_root / f"{stem}_stats.json"

    previous_seen = _compute_previous_seen_classes(all_stage_definitions, stage_id)
    if not previous_seen:
        payload: Dict[str, Any] = {
            "__meta__": {
                "dataset": "sunrgbd",
                "stage_id": stage_id,
                "previous_seen_classes": [],
                "checkpoint_used": str(ckpt),
                "confidence_threshold": float(confidence_threshold),
                "nms_iou_thr": float(nms_iou_thr),
                "max_pseudo_per_scene": int(max_pseudo_per_scene),
                "timestamp": int(time.time()),
                "note": "No previous classes (unexpected for stage>=2).",
            }
        }
        with out_pkl.open("wb") as f:
            pickle.dump(payload, f)
        with out_json.open("w") as f:
            json.dump({"stage_id": stage_id, "total_scenes": 0, "total_detections": 0}, f, indent=2)
        return str(out_pkl)

    prev_n_classes = int(max(previous_seen) + 1)

    cfg = Config.fromfile(str(base_config_path))
    # Ensure head size matches the checkpoint (previous stage head).
    cfg.model.head.n_classes = prev_n_classes
    # Keep score_thr low; we filter using `confidence_threshold` below.
    try:
        cfg.model.test_cfg.score_thr = float(min(getattr(cfg.model.test_cfg, "score_thr", 0.01), 0.01))
    except Exception:
        pass

    model = init_model(cfg, str(ckpt), device=str(device))
    model.eval()

    with ann_path.open("rb") as f:
        infos = pickle.load(f)
    if not isinstance(infos, list):
        raise ValueError(f"Expected a list in {train_ann_file}, got {type(infos)}.")

    stage_classes = [int(x) for x in stage_definition.get("class_indices", [])]
    if not stage_classes:
        raise ValueError(f"stage_definition.class_indices is empty for stage_id={stage_id}.")

    # Mirror training: process only scenes that would appear after stage GT filtering.
    filter_empty_gt = bool(stage_definition.get("filter_empty_gt", True))
    # If the stage config explicitly sets filter_empty_gt, prefer it.
    # (The caller can inject it into stage_definition for exactness.)
    stage_infos = _filter_infos_for_stage(infos, stage_classes, filter_empty_gt=filter_empty_gt)
    stage_infos = sorted(
        stage_infos,
        key=lambda x: str(_extract_scene_id(x) or ""),
    )
    if max_scenes is not None and int(max_scenes) > 0 and len(stage_infos) > int(max_scenes):
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(len(stage_infos), size=int(max_scenes), replace=False)
        idx.sort()
        stage_infos = [stage_infos[int(i)] for i in idx.tolist()]

    data_root = _resolve_data_root(cfg)

    pseudo: Dict[str, Any] = {
        "__meta__": {
            "dataset": "sunrgbd",
            "stage_id": stage_id,
            "previous_stage_id": stage_id - 1,
            "previous_seen_classes": previous_seen,
            "prev_head_n_classes": prev_n_classes,
            "checkpoint_used": str(ckpt),
            "train_ann_file": str(ann_path),
            "confidence_threshold": float(confidence_threshold),
            "nms_iou_thr": float(nms_iou_thr),
            "max_pseudo_per_scene": int(max_pseudo_per_scene),
            "max_scenes": int(max_scenes) if max_scenes is not None else None,
            "seed": int(seed),
            "box_type": "upright_depth_7d",
            "center_type": "gravity",
            "axis_aligned": False,
            "timestamp": int(time.time()),
        }
    }

    total_dets = 0
    per_class: Dict[int, int] = {}
    processed = 0
    kept_scenes = 0

    for info in stage_infos:
        sid = _extract_scene_id(info)
        if sid is None:
            continue
        pts_rel = info.get("pts_path", None)
        if not pts_rel:
            continue
        pts_path = data_root / str(pts_rel)

        generated = _generate_scene_pseudo(
            model=model,
            pts_path=pts_path,
            previous_seen_classes=previous_seen,
            confidence_threshold=float(confidence_threshold),
            nms_iou_thr=float(nms_iou_thr),
            max_pseudo_per_scene=int(max_pseudo_per_scene),
        )
        processed += 1
        if generated is None:
            continue
        boxes, labels, scores = generated
        if boxes.size == 0:
            continue
        kept_scenes += 1
        total_dets += int(boxes.shape[0])
        for lbl in labels.tolist():
            per_class[int(lbl)] = per_class.get(int(lbl), 0) + 1

        pseudo[str(sid)] = {
            "gt_boxes_upright_depth": boxes.astype(np.float32, copy=False),
            "class": labels.astype(np.int64, copy=False),
            "scores": scores.astype(np.float32, copy=False),
            "gt_num": int(boxes.shape[0]),
            "center_type": "gravity",
            "axis_aligned": False,
        }

    with out_pkl.open("wb") as f:
        pickle.dump(pseudo, f)

    stats = {
        "dataset": "sunrgbd",
        "stage_id": stage_id,
        "previous_stage_id": stage_id - 1,
        "checkpoint_used": str(ckpt),
        "train_ann_file": str(ann_path),
        "confidence_threshold": float(confidence_threshold),
        "nms_iou_thr": float(nms_iou_thr),
        "max_pseudo_per_scene": int(max_pseudo_per_scene),
        "max_scenes": int(max_scenes) if max_scenes is not None else None,
        "seed": int(seed),
        "scenes_considered": int(len(stage_infos)),
        "scenes_processed": int(processed),
        "scenes_with_pseudo": int(kept_scenes),
        "total_detections": int(total_dets),
        "avg_dets_per_scene_with_pseudo": float(total_dets / kept_scenes) if kept_scenes else 0.0,
        "per_class_counts": {str(k): int(v) for k, v in sorted(per_class.items())},
    }
    with out_json.open("w") as f:
        json.dump(stats, f, indent=2)

    return str(out_pkl)
