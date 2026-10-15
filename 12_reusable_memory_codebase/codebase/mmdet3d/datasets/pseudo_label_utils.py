"""Utility helpers for pseudo label canonicalization and diagnostics."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def canonicalize_bottom_center_boxes(
    boxes: np.ndarray,
    axis_align_matrix: Optional[np.ndarray] = None,
    assume_aligned: bool = True,
) -> np.ndarray:
    """Convert detector outputs to canonical upright-depth boxes.

    Args:
        boxes: Array shaped (N, 6|7) with bottom-centered depths.
        axis_align_matrix: Optional ScanNet 4x4 axis alignment matrix.
        assume_aligned: When True, skip applying ``axis_align_matrix`` because
            the detector already operated in the aligned frame.

    Returns:
        (N, 6) float32 array with gravity-centred upright-depth boxes.
    """
    if boxes.size == 0:
        return boxes.astype(np.float32).reshape(0, 6)

    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Boxes must be 2D with >=6 dims; got {arr.shape}")

    canonical = arr.copy()

    # Convert oriented width/height to axis-aligned extents when yaw present.
    if canonical.shape[1] >= 7:
        yaw = canonical[:, 6].astype(np.float32)
        dx = canonical[:, 3].astype(np.float32)
        dy = canonical[:, 4].astype(np.float32)
        c = np.abs(np.cos(yaw))
        s = np.abs(np.sin(yaw))
        canonical[:, 3] = c * dx + s * dy
        canonical[:, 4] = s * dx + c * dy

    # Shift from bottom centre to gravity centre (z := z + h/2).
    canonical[:, 2] = canonical[:, 2] + canonical[:, 5] * 0.5

    # Optionally apply axis alignment if predictions are in raw frame.
    if axis_align_matrix is not None and not assume_aligned:
        mat = np.asarray(axis_align_matrix, dtype=np.float32)
        if mat.shape != (4, 4):
            raise ValueError(f"axis_align_matrix must be 4x4; got {mat.shape}")
        rot = mat[:3, :3]
        trans = mat[:3, 3]
        canonical[:, :3] = canonical[:, :3] @ rot.T + trans

    # Drop yaw/extra dims.
    if canonical.shape[1] > 6:
        canonical = canonical[:, :6]

    return canonical.astype(np.float32, copy=False)


def build_canonical_pseudo_record(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
) -> dict:
    """Package canonical pseudo label arrays with metadata."""
    boxes = np.asarray(boxes, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)

    if labels.shape[0] != boxes.shape[0] or scores.shape[0] != boxes.shape[0]:
        raise ValueError("boxes, labels, and scores must share length")

    return {
        'boxes': boxes,
        'labels': labels,
        'scores': scores,
        'num_detections': int(boxes.shape[0]),
        'label_space': 'nyu40',
        'center_type': 'gravity',
        'axis_aligned': True,
        'box_type': 'upright_depth_6d',
    }


# ---------------------------------------------------------------------------
# Pseudo label diagnostics
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=4)
def load_stage_definitions(strategy: str = 'frequency') -> List[Dict]:
    import sys

    mappings_path = _repo_root() / 'configs' / '_base_' / 'class_mappings'
    sys.path.append(str(mappings_path))
    from scannet_dynamic_head_mappings import get_stage_definitions  # type: ignore

    return get_stage_definitions(strategy)


def get_previous_stage_nyu40(stage_id: int, strategy: str = 'frequency') -> List[int]:
    """Return NYU40 IDs for all stages strictly before ``stage_id``."""

    if stage_id <= 1:
        return []

    nyu40: List[int] = []
    for stage_def in load_stage_definitions(strategy):
        sid = int(stage_def.get('stage_id', 0))
        if sid >= stage_id:
            break
        nyu40.extend(int(x) for x in stage_def.get('nyu40_ids', []))
    return sorted(set(nyu40))


@lru_cache(maxsize=2)
def load_annotation_map(ann_file: Path) -> Dict[str, Dict[str, np.ndarray]]:
    import pickle

    with ann_file.open('rb') as f:
        infos = pickle.load(f)

    mapping: Dict[str, Dict[str, np.ndarray]] = {}
    for info in infos:
        sid = None
        if 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
            sid = str(info['point_cloud']['lidar_idx'])
        elif 'sample_idx' in info:
            sid = str(info['sample_idx'])
        if sid is None:
            continue
        annos = info.get('annos', {})
        boxes = np.asarray(annos.get('gt_boxes_upright_depth', np.zeros((0, 6))), dtype=np.float32)
        labels = np.asarray(annos.get('class', np.zeros((0,))), dtype=np.int64)
        mapping[sid] = {'boxes': boxes, 'labels': labels}
    return mapping


def pairwise_aligned_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    a_min = boxes_a[:, :3] - boxes_a[:, 3:] / 2.0
    a_max = boxes_a[:, :3] + boxes_a[:, 3:] / 2.0
    b_min = boxes_b[:, :3] - boxes_b[:, 3:] / 2.0
    b_max = boxes_b[:, :3] + boxes_b[:, 3:] / 2.0

    inter_min = np.maximum(a_min[:, None, :], b_min[None, :, :])
    inter_max = np.minimum(a_max[:, None, :], b_max[None, :, :])
    inter_dims = np.clip(inter_max - inter_min, 0.0, None)
    inter_vol = inter_dims.prod(axis=2)

    vol_a = boxes_a[:, 3:].prod(axis=1)[:, None]
    vol_b = boxes_b[:, 3:].prod(axis=1)[None, :]
    union = vol_a + vol_b - inter_vol
    union = np.maximum(union, 1e-9)
    return inter_vol / union


def filter_pseudo_by_iou_against_gt(
    gt_boxes: np.ndarray,
    pseudo_boxes: np.ndarray,
    iou_thr: float = 0.25,
) -> np.ndarray:
    """Compute a boolean mask of pseudo boxes to keep by suppressing those
    that overlap any GT box with IoU >= ``iou_thr`` (class-agnostic).

    Args:
        gt_boxes: (G, 6) GT boxes [x,y,z,dx,dy,dz]
        pseudo_boxes: (P, 6[+]) pseudo boxes (only first 6 dims are used)
        iou_thr: IoU threshold to consider a pseudo as conflicting with GT

    Returns:
        keep_mask: (P,) boolean array, True for pseudo boxes to keep
    """
    if pseudo_boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    p = np.asarray(pseudo_boxes, dtype=np.float32)
    p = p[:, :6]
    if gt_boxes.size == 0:
        return np.ones((p.shape[0],), dtype=bool)
    g = np.asarray(gt_boxes, dtype=np.float32)
    iou = pairwise_aligned_iou(p, g)
    max_iou = iou.max(axis=1) if iou.size else np.zeros((p.shape[0],), dtype=np.float32)
    return max_iou < float(iou_thr)


def dedup_same_class_by_iou(
    boxes: np.ndarray,
    labels: np.ndarray,
    iou_thr: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove duplicates among boxes of the same class using IoU-based NMS.

    Args:
        boxes: (N, 6[+]) boxes [x,y,z,dx,dy,dz,...]
        labels: (N,) class labels (int)
        iou_thr: IoU threshold for suppression (higher is stricter dedup)

    Returns:
        dedup_boxes, dedup_labels
    """
    import numpy as _np
    if boxes.size == 0:
        return boxes, labels
    boxes = _np.asarray(boxes, dtype=_np.float32)
    labels = _np.asarray(labels, dtype=_np.int64)
    keep_mask = _np.zeros((boxes.shape[0],), dtype=bool)
    # Process per class for deterministic behavior
    for cls in _np.unique(labels):
        idxs = _np.where(labels == cls)[0]
        if idxs.size == 0:
            continue
        b = boxes[idxs][:, :6]
        # Greedy NMS without scores: keep earlier instances, suppress later overlaps
        kept: List[int] = []
        suppressed = _np.zeros((idxs.size,), dtype=bool)
        for i in range(idxs.size):
            if suppressed[i]:
                continue
            kept.append(i)
            if i + 1 < idxs.size:
                iou = pairwise_aligned_iou(b[i:i+1], b[i+1:])  # (1, M)
                sup = (iou[0] >= float(iou_thr))
                suppressed[i+1:][sup] = True
        keep_mask[idxs[_np.array(kept, dtype=_np.int64)]] = True
    return boxes[keep_mask], labels[keep_mask]


def nms_indices_iou(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thr: float = 0.3,
) -> np.ndarray:
    """Greedy axis-aligned IoU NMS returning kept indices.

    Args:
        boxes: (N, 6[+]) boxes [x,y,z,dx,dy,dz,...]
        scores: (N,) confidence scores (float)
        iou_thr: IoU threshold for suppression

    Returns:
        kept indices (np.int64)
    """
    import numpy as _np
    if boxes.size == 0:
        return _np.zeros((0,), dtype=_np.int64)
    if boxes.shape[0] == 1:
        return _np.array([0], dtype=_np.int64)
    boxes = _np.asarray(boxes, dtype=_np.float32)[:, :6]
    scores = _np.asarray(scores, dtype=_np.float32)
    order = _np.argsort(scores)[::-1]
    kept: List[int] = []
    suppressed = _np.zeros((boxes.shape[0],), dtype=bool)
    for o in order:
        if suppressed[o]:
            continue
        kept.append(o)
        if len(kept) == boxes.shape[0]:
            break
        # Compute IoU with remaining unsuppressed boxes
        rem = _np.where(~suppressed & (_np.arange(boxes.shape[0]) != o))[0]
        if rem.size == 0:
            continue
        iou = pairwise_aligned_iou(boxes[o:o+1], boxes[rem])  # (1, M)
        suppressed[rem] |= (iou[0] >= float(iou_thr))
    return _np.array(kept, dtype=_np.int64)


def compute_pseudo_label_hit_metrics(
    pseudo_labels: Dict[str, Dict[str, np.ndarray]],
    annotations: Dict[str, Dict[str, np.ndarray]],
    stage_nyu40: Iterable[int],
    thresholds: Tuple[float, ...] = (0.25, 0.5),
) -> Dict[float, Dict[str, float]]:
    stage_nyu40 = set(stage_nyu40)
    results: Dict[float, Dict[str, float]] = {}

    for thr in thresholds:
        pseudo_total = 0
        pseudo_hits = 0
        gt_total = 0
        gt_hits = 0
        per_class_hits: Dict[int, int] = {}
        per_class_total: Dict[int, int] = {}

        for sid, rec in pseudo_labels.items():
            if sid == '__meta__':
                continue
            pseudo_boxes = np.asarray(rec.get('boxes', np.zeros((0, 6))), dtype=np.float32)
            pseudo_labels_arr = np.asarray(rec.get('labels', np.zeros((0,))), dtype=np.int64)

            ann = annotations.get(str(sid))
            if ann is None:
                continue
            gt_boxes = ann['boxes']
            gt_labels = ann['labels']

            pseudo_mask = np.isin(pseudo_labels_arr, list(stage_nyu40))
            gt_mask = np.isin(gt_labels, list(stage_nyu40))

            pseudo_stage = pseudo_boxes[pseudo_mask]
            pseudo_stage_labels = pseudo_labels_arr[pseudo_mask]
            gt_stage = gt_boxes[gt_mask]

            if pseudo_stage.size:
                pseudo_total += pseudo_stage_labels.size
            if gt_stage.size:
                gt_total += gt_stage.shape[0]

            iou_mat = pairwise_aligned_iou(pseudo_stage, gt_stage)
            if iou_mat.size:
                max_pseudo = iou_mat.max(axis=1)
                max_gt = iou_mat.max(axis=0)
            else:
                max_pseudo = np.zeros(pseudo_stage_labels.size, dtype=np.float32)
                max_gt = np.zeros(gt_stage.shape[0], dtype=np.float32)

            pseudo_hit_mask = max_pseudo >= thr
            gt_hit_mask = max_gt >= thr

            pseudo_hits += int(pseudo_hit_mask.sum())
            gt_hits += int(gt_hit_mask.sum())

            for label, hit in zip(pseudo_stage_labels, pseudo_hit_mask):
                lbl = int(label)
                per_class_total[lbl] = per_class_total.get(lbl, 0) + 1
                if hit:
                    per_class_hits[lbl] = per_class_hits.get(lbl, 0) + 1

        results[thr] = {
            'pseudo_total': float(pseudo_total),
            'pseudo_hits': float(pseudo_hits),
            'pseudo_hit_rate': float(pseudo_hits / pseudo_total) if pseudo_total else 0.0,
            'gt_total': float(gt_total),
            'gt_hits': float(gt_hits),
            'gt_recall': float(gt_hits / gt_total) if gt_total else 0.0,
            'per_class_hits': per_class_hits,
            'per_class_totals': per_class_total,
        }

    return results


def evaluate_pseudo_label_file_hits(
    pseudo_file: Path,
    ann_file: Path,
    stage_id: int,
    strategy: str = 'frequency',
    thresholds: Tuple[float, ...] = (0.25, 0.5),
) -> Dict[float, Dict[str, float]]:
    import pickle

    with pseudo_file.open('rb') as f:
        pseudo = pickle.load(f)

    ann_map = load_annotation_map(ann_file)
    stage_nyu40 = get_previous_stage_nyu40(stage_id, strategy)

    if not stage_nyu40:
        return {}

    return compute_pseudo_label_hit_metrics(pseudo, ann_map, stage_nyu40, thresholds)


def log_pseudo_hit_metrics(
    metrics: Dict[float, Dict[str, float]],
    stage_id: int,
    logger: Optional[object] = None,
) -> None:
    if not metrics:
        message = f"No previous-stage classes for Stage {stage_id}; hit metrics skipped."
        if logger:
            logger.info(message)
        else:
            print(message)
        return

    for thr, vals in metrics.items():
        summary = (
            f"Pseudo hits @ IoU>={thr:.2f}: {int(vals['pseudo_hits'])}/"
            f"{int(vals['pseudo_total'])} ({vals['pseudo_hit_rate']*100:.2f}%), "
            f"GT recall: {int(vals['gt_hits'])}/"
            f"{int(vals['gt_total'])} ({vals['gt_recall']*100:.2f}%)"
        )
        if logger:
            logger.info(summary)
        else:
            print(summary)

        per_hits = vals['per_class_hits']
        per_total = vals['per_class_totals']
        if logger:
            logger.info("Per-class pseudo hit rates:")
        else:
            print("Per-class pseudo hit rates:")
        for cls in sorted(per_total):
            hits = per_hits.get(cls, 0)
            total = per_total[cls]
            rate = hits / total if total else 0.0
            line = f"  NYU40 {cls:2d}: {hits}/{total} ({rate*100:.1f}%)"
            if logger:
                logger.info(line)
            else:
                print(line)
