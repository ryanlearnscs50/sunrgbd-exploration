"""Memory-bank pseudo-label enrichment utilities (SUN RGB-D).

This module generates pseudo labels for *current-stage classes* on scenes that
remain in the scene memory bank, to provide supervision for newer classes on
old replay scenes in future stages.

Important:
- This is **not** used for learning-dynamics (LD) scoring: LD remains GT-only.
- Pseudo labels are stored separately from GT and can be injected into replay
  scenes behind a dataset flag.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from mmdet3d.datasets.pseudo_label_utils import (
    canonicalize_bottom_center_boxes,
    nms_indices_iou,
)


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
    """Convert bottom-centred boxes to gravity-centred (SUNRGBD GT convention)."""
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


def _filter_and_nms(
    *,
    boxes_bottom: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    target_classes: Sequence[int],
    confidence_threshold: float,
    nms_iou_thr: float,
    max_pseudo_per_scene: int = 200,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    boxes_bottom = np.asarray(boxes_bottom, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if boxes_bottom.ndim != 2 or boxes_bottom.shape[1] < 6:
        raise ValueError(f"boxes_bottom must be (N,6/7[+]); got {boxes_bottom.shape}")
    if boxes_bottom.shape[0] != scores.shape[0] or labels.shape[0] != scores.shape[0]:
        raise ValueError(
            "Prediction length mismatch: "
            f"boxes={boxes_bottom.shape}, scores={scores.shape}, labels={labels.shape}"
        )
    if boxes_bottom.shape[0] == 0:
        return None

    cls_set = {int(x) for x in (target_classes or [])}
    if not cls_set:
        return None

    keep = scores >= float(confidence_threshold)
    keep &= np.isin(labels, np.array(sorted(cls_set), dtype=np.int64))
    if not bool(keep.any()):
        return None

    boxes_keep = boxes_bottom[keep]
    scores_keep = scores[keep]
    labels_keep = labels[keep]

    boxes_gc7 = _bottom_center_to_gravity_center_keep_yaw(boxes_keep)
    boxes_aabb6 = canonicalize_bottom_center_boxes(boxes_keep)

    kept_global = []
    for cls in np.unique(labels_keep):
        cls = int(cls)
        idxs = np.where(labels_keep == cls)[0]
        if idxs.size == 0:
            continue
        keep_local = nms_indices_iou(boxes_aabb6[idxs], scores_keep[idxs], iou_thr=float(nms_iou_thr))
        kept_global.extend(idxs[keep_local].tolist())

    if not kept_global:
        return None

    kept_idx = np.array(sorted(set(kept_global)), dtype=np.int64)
    order = np.argsort(scores_keep[kept_idx])[::-1]
    kept_idx = kept_idx[order]
    if max_pseudo_per_scene is not None and int(max_pseudo_per_scene) > 0:
        kept_idx = kept_idx[:int(max_pseudo_per_scene)]

    return (
        boxes_gc7[kept_idx].astype(np.float32, copy=False),
        labels_keep[kept_idx].astype(np.int64, copy=False),
        scores_keep[kept_idx].astype(np.float32, copy=False),
    )


def build_memory_pseudo_record(
    *,
    boxes_bottom: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    target_classes: Sequence[int],
    stage_generated: int,
    confidence_threshold: float,
    nms_iou_thr: float,
    max_pseudo_per_scene: int = 200,
) -> Optional[Dict[str, Any]]:
    """Create a pseudo label record for a single scene (SUNRGBD format)."""
    out = _filter_and_nms(
        boxes_bottom=boxes_bottom,
        scores=scores,
        labels=labels,
        target_classes=target_classes,
        confidence_threshold=float(confidence_threshold),
        nms_iou_thr=float(nms_iou_thr),
        max_pseudo_per_scene=int(max_pseudo_per_scene),
    )
    if out is None:
        return None
    boxes_gc7, labels_f, scores_f = out
    return {
        'gt_boxes_upright_depth': boxes_gc7,
        'class': labels_f,
        'scores': scores_f,
        'gt_num': int(boxes_gc7.shape[0]),
        'label_source': 'pseudo',
        'stage_generated': int(stage_generated),
        'center_type': 'gravity',
        'confidence_threshold': float(confidence_threshold),
        'nms_iou_thr': float(nms_iou_thr),
        'is_pseudo': True,
    }


def _resolve_pts_path(info: Mapping[str, Any]) -> Optional[str]:
    if not isinstance(info, Mapping):
        return None
    pts_path = info.get('pts_path', None)
    if pts_path:
        return str(pts_path)
    # Legacy fallback
    pts_filename = info.get('pts_filename', None)
    if pts_filename:
        return str(pts_filename)
    return None


def generate_sunrgbd_memory_pseudo_labels(
    *,
    model,
    data_root: Path,
    memory_entries: Sequence[Mapping[str, Any]],
    stage_id: int,
    new_classes: Sequence[int],
    confidence_threshold: float,
    nms_iou_thr: float,
    output_file: Path,
    ckpt_tag: str = 'final',
    max_pseudo_per_scene: int = 200,
    logger=None,
) -> Dict[str, Any]:
    """Generate memory-enrichment pseudo labels for SUNRGBD replay scenes.

    Args:
        memory_entries: list entries from SceneMemoryBank.list_memory_entries().
            These should be **kept** memory seats (typically save_stage < stage_id).
        new_classes: current-stage class indices C_new(stage_id) in model label space.
        output_file: pickle file destination.
    """
    from mmdet3d.apis import inference_detector

    stage_id = int(stage_id)
    data_root = Path(data_root)
    out: Dict[str, Any] = {}
    out['__meta__'] = dict(
        label_source='pseudo',
        stage_generated=int(stage_id),
        classes=list(int(x) for x in new_classes),
        confidence_threshold=float(confidence_threshold),
        nms_iou_thr=float(nms_iou_thr),
        max_pseudo_per_scene=int(max_pseudo_per_scene),
        ckpt_tag=str(ckpt_tag),
    )

    if not memory_entries:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        import pickle
        with output_file.open('wb') as f:
            pickle.dump(out, f)
        return out

    model.eval()

    total = 0
    kept = 0
    for entry in memory_entries:
        if not isinstance(entry, Mapping):
            continue
        scene_id = str(entry.get('scene_id', ''))
        if not scene_id:
            continue
        snap = entry.get('snapshot', {}) or {}
        info = snap.get('data_info', None)
        if not isinstance(info, Mapping):
            continue

        rel = _resolve_pts_path(info)
        if not rel:
            continue
        pts_path = data_root / str(rel)
        if not pts_path.exists():
            if logger is not None:
                try:
                    logger.warning(f"Memory pseudo enrichment: missing pts file for {scene_id}: {pts_path}")
                except Exception:
                    pass
            continue

        total += 1
        result, _ = inference_detector(model, str(pts_path))
        if not isinstance(result, list) or not result or not isinstance(result[0], Mapping):
            continue
        pred = result[0]
        boxes_3d = pred.get('boxes_3d', None)
        scores_3d = pred.get('scores_3d', None)
        labels_3d = pred.get('labels_3d', None)
        if boxes_3d is None or scores_3d is None or labels_3d is None:
            continue

        if hasattr(boxes_3d, 'tensor'):
            boxes_np = _to_numpy(boxes_3d.tensor, dtype=np.float32)
        else:
            boxes_np = _to_numpy(boxes_3d, dtype=np.float32)
        scores_np = _to_numpy(scores_3d, dtype=np.float32).reshape(-1)
        labels_np = _to_numpy(labels_3d, dtype=np.int64).reshape(-1)

        rec = build_memory_pseudo_record(
            boxes_bottom=boxes_np,
            scores=scores_np,
            labels=labels_np,
            target_classes=new_classes,
            stage_generated=stage_id,
            confidence_threshold=float(confidence_threshold),
            nms_iou_thr=float(nms_iou_thr),
            max_pseudo_per_scene=int(max_pseudo_per_scene),
        )
        if rec is None:
            continue
        out[str(scene_id)] = rec
        kept += 1

    out['__meta__']['num_scenes_considered'] = int(total)
    out['__meta__']['num_scenes_with_pseudo'] = int(kept)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with output_file.open('wb') as f:
        pickle.dump(out, f)

    return out
