"""Per-seat, per-class matching stats for indoor 3D detection.

This helper supports learning-dynamics scoring (memory bank management) and
computes per-(seat,class) TP/FP/FN/denom at a fixed IoU τ.

Matching intentionally mirrors the greedy logic used in `indoor_eval.py`
(descending score; each GT box can be matched at most once), except that the
thresholding rule for learning-dynamics scoring uses IoU >= τ (as documented).

q definition is selected by `q_metric`:
  - 'f1':     q = 2*TP / (2*TP + FP + FN + eps)
  - 'recall': q = TP / (TP + FN + eps)

IMPORTANT: q is only defined for (s,c) where the class has GT for the seat
(gt_count > 0). When gt_count == 0, we set q=None so downstream scoring can
skip missing labels without treating them as q=0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch


def _to_numpy_1d(x: Any, *, dtype: np.dtype) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError('LD labels and scores must be one-dimensional')
    if not np.isfinite(arr).all():
        raise ValueError('LD labels and scores must be finite')
    if arr.size and np.issubdtype(dtype, np.integer) and (arr.dtype.kind not in 'iu' or np.any(arr < 0)):
        raise ValueError('LD labels must be nonnegative integers')
    return np.asarray(arr, dtype=dtype)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if not np.isfinite(v):
            return float(default)
        return float(v)
    except Exception:
        return float(default)


SeatKey = Tuple[str, int]  # (scene_id, save_stage)


def _normalize_seat_key(seat_key: Any) -> Optional[SeatKey]:
    if isinstance(seat_key, tuple) and len(seat_key) == 2:
        sid, stage = seat_key
    elif isinstance(seat_key, Mapping):
        sid, stage = seat_key.get('scene_id'), seat_key.get('save_stage')
    elif isinstance(seat_key, str) and '_stage' in seat_key:
        sid, suffix = seat_key.rsplit('_stage', 1)
        if not suffix.isascii() or not suffix.isdecimal():
            return None
        stage = int(suffix)
        if str(stage) != suffix:
            return None
    else:
        return None
    # Casting first aliases fractional stages, booleans and missing scene IDs.
    if (not isinstance(sid, str) or not sid
            or isinstance(stage, (bool, np.bool_))
            or not isinstance(stage, (int, np.integer)) or stage < 1):
        return None
    return sid, int(stage)


def compute_scene_class_match_stats(
        gt_annos: Sequence[Mapping[str, Any]],
        dt_annos: Sequence[Mapping[str, Any]],
        *,
        seat_keys: Sequence[Union[SeatKey, Mapping[str, Any], str]],
        eval_class_indices: Sequence[int],
        iou_thr: float,
        box_type_3d,
        box_mode_3d,
        q_metric: str = 'f1',
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-9,
) -> List[Dict[str, Any]]:
    """Compute per-(seat,class) TP/FP/FN stats at IoU τ.

    Args:
      gt_annos / dt_annos: aligned lists as in `indoor_eval`.
      seat_keys: aligned identifiers for each seat as either:
        - tuple: (scene_id, save_stage)
        - dict: {'scene_id': str, 'save_stage': int}
        - legacy str: "{scene_id}_stage{save_stage}"
      eval_class_indices: classes to compute stats for.
      q_metric: 'f1' or 'recall' for the saved per-seat `q` field.
      alpha/beta: Deprecated (kept for call-site compatibility; ignored).
      eps: Small epsilon for numerical stability in q.

    Returns:
      List of seat records:
        [{'scene_id': str, 'save_stage': int, 'classes': {class_id: stats_dict}}, ...]
    """
    if len(gt_annos) != len(dt_annos) or len(seat_keys) != len(gt_annos):
        raise ValueError('LD annotations and seat keys must have equal lengths')

    iou_thr = float(iou_thr)
    if not 0.0 < iou_thr < 1.0:
        raise ValueError('LD IoU threshold must be finite and between zero and one')

    class_indices = list(eval_class_indices)
    if any(isinstance(c, (bool, np.bool_)) or not isinstance(c, (int, np.integer))
           or c < 0 for c in class_indices):
        raise ValueError('LD class indices must be nonnegative integers')
    classes = sorted(int(c) for c in class_indices)
    if len(classes) != len(set(classes)):
        raise ValueError('Duplicate LD class indices')
    normalized_keys = [_normalize_seat_key(key) for key in seat_keys]
    if any(key is None for key in normalized_keys):
        raise ValueError('Invalid LD seat identity')
    if len(normalized_keys) != len(set(normalized_keys)):
        raise ValueError('Duplicate LD seat identity')
    q_metric_norm = str(q_metric).strip().lower()
    if q_metric_norm not in ('f1', 'recall'):
        raise ValueError(
            "scene_class_stats q_metric must be one of ['f1', 'recall'], "
            f"got {q_metric!r}."
        )
    # alpha/beta were used by the legacy recall-based q. Keep them in the
    # signature for backward compatibility, but do not use them.
    eps = float(eps)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError('LD epsilon must be finite and positive')

    out: List[Dict[str, Any]] = []

    for idx in range(len(gt_annos)):
        sid, save_stage = normalized_keys[idx]
        gt_anno = gt_annos[idx]
        dt_anno = dt_annos[idx]

        # GT boxes (indoor format).
        gt_num = gt_anno.get('gt_num', 0)
        if isinstance(gt_num, (bool, np.bool_)) or not isinstance(gt_num, (int, np.integer)) or gt_num < 0:
            raise ValueError('LD gt_num must be a nonnegative integer')
        gt_boxes_np = np.asarray(gt_anno.get('gt_boxes_upright_depth', np.empty((0, 6))), dtype=np.float32)
        if gt_boxes_np.shape == (0,):
            gt_boxes_np = gt_boxes_np.reshape(0, 6)
        gt_labels_np = _to_numpy_1d(gt_anno.get('class', np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        if (gt_boxes_np.ndim != 2 or gt_boxes_np.shape[1] not in (6, 7)
                or len(gt_boxes_np) != gt_num or len(gt_labels_np) != gt_num
                or not np.isfinite(gt_boxes_np).all() or np.any(gt_boxes_np[:, 3:6] < 0)):
            raise ValueError('LD GT count, boxes and labels must agree and have finite nonnegative dimensions')
        # Original SUN training metadata contains two zero-height annotations.
        # Keep them in the reference GT denominator (zero IoU => false negative)
        # rather than aborting a real LD sweep or silently changing its targets.
        # Replay crop admission still requires positive-volume geometry.
        if gt_num > 0:
            gt_boxes = box_type_3d(
                gt_boxes_np,
                box_dim=gt_boxes_np.shape[-1],
                origin=(0.5, 0.5, 0.5),
            ).convert_to(box_mode_3d)
        else:
            gt_boxes = box_type_3d(np.array([], dtype=np.float32))
            gt_labels_np = np.zeros((0,), dtype=np.int64)

        # Detections (mmdet3d outputs).
        det_labels = _to_numpy_1d(dt_anno.get('labels_3d', np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        if len(det_labels) and 'scores_3d' not in dt_anno:
            raise ValueError('LD nonempty detections require scores')
        det_scores = _to_numpy_1d(dt_anno.get('scores_3d', np.zeros((det_labels.shape[0],), dtype=np.float32)),
                                  dtype=np.float32)
        det_boxes = dt_anno.get('boxes_3d', None)
        if det_boxes is None:
            if len(det_labels) or len(det_scores):
                raise ValueError('LD nonempty detections require boxes')
            det_scores = np.zeros((0,), dtype=np.float32)
            det_labels = np.zeros((0,), dtype=np.int64)
            det_boxes = box_type_3d(np.array([], dtype=np.float32))
        det_boxes = det_boxes.convert_to(box_mode_3d)
        if len(det_boxes) != len(det_labels) or len(det_scores) != len(det_labels):
            raise ValueError('LD detection boxes, labels and scores must have equal lengths')
        tensor = det_boxes.tensor
        if not torch.isfinite(tensor).all() or torch.any(tensor[:, 3:6] <= 0):
            raise ValueError('LD detection boxes must have finite positive dimensions')

        per_scene: Dict[int, Dict[str, Any]] = {}

        for c in classes:
            # GT indices for this class.
            gt_mask = (gt_labels_np == int(c))
            gt_idx = np.nonzero(gt_mask)[0].astype(np.int64)
            gt_count = int(gt_idx.shape[0])
            if gt_count > 0:
                gt_sub = gt_boxes[gt_idx.tolist()]
            else:
                gt_sub = gt_boxes[:0]

            # Detection indices for this class.
            det_mask = (det_labels == int(c))
            det_idx = np.nonzero(det_mask)[0].astype(np.int64)
            dt_count = int(det_idx.shape[0])

            tp_weight = 0.0
            tp_count = 0
            fp_count = 0

            # Learning-dynamics scoring uses GT-only: each GT box has weight 1.
            if gt_count > 0:
                gt_w = np.ones((gt_count,), dtype=np.float32)
                denom = float(gt_w.sum())
            else:
                gt_w = np.zeros((0,), dtype=np.float32)
                denom = 0.0

            if dt_count > 0 and gt_count > 0:
                # Greedy match by descending score.
                order = det_idx[np.argsort(det_scores[det_idx])[::-1]]
                det_sub = det_boxes[order.tolist()]

                iou = det_sub.overlaps(det_sub, gt_sub)
                if isinstance(iou, torch.Tensor):
                    iou = iou.detach().cpu().numpy()
                iou = np.asarray(iou, dtype=np.float32)

                matched = np.zeros((gt_count,), dtype=bool)
                for di in range(iou.shape[0]):
                    if iou.shape[1] == 0:
                        fp_count += 1
                        continue
                    j = int(iou[di].argmax())
                    iou_max = float(iou[di, j])
                    # Learning-dynamics scoring uses IoU >= τ.
                    if iou_max >= iou_thr and not bool(matched[j]):
                        matched[j] = True
                        tp_count += 1
                        tp_weight += float(gt_w[j])
                    else:
                        fp_count += 1
            else:
                # No GT or no det → TP=0. FP are all detections if no GT.
                fp_count = int(dt_count) if gt_count == 0 else 0

            fn_weight = float(max(0.0, denom - tp_weight))
            # Learning-dynamics q is only defined when the class is
            # valid/present for the seat (gt_count > 0). Do not treat missing GT
            # as q=0; downstream scoring should ignore such entries.
            if gt_count <= 0:
                q = None
            else:
                if q_metric_norm == 'recall':
                    denom_rec = float(tp_weight + fn_weight + eps)
                    if denom_rec <= 0.0:
                        q = 0.0
                    else:
                        q = float(tp_weight / denom_rec)
                        q = float(max(0.0, min(1.0, q)))
                else:
                    denom_f1 = float(2.0 * tp_weight + float(fp_count) + fn_weight + eps)
                    if denom_f1 <= 0.0:
                        q = 0.0
                    else:
                        q = float((2.0 * tp_weight) / denom_f1)
                        q = float(max(0.0, min(1.0, q)))

            per_scene[int(c)] = dict(
                tp=float(tp_weight),
                fp=int(fp_count),
                fn=float(fn_weight),
                denom=float(denom),
                gt_count=int(gt_count),
                tp_count=int(tp_count),
                dt_count=int(dt_count),
                q=q if q is None else float(q),
                count=int(gt_count),
            )

        out.append(dict(scene_id=str(sid), save_stage=int(save_stage), classes=per_scene))

    return out
