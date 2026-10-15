"""
Pseudo-to-pseudo consistency evaluation on training scenes.

Generates pseudo predictions for a sampled subset of training scenes using two
checkpoints (prev and mid) and computes per-class and per-scene drop based on
IoU matching at a fixed threshold (default 0.25).

This avoids reliance on GT labels for old classes (which are not present on the
current stage's natural scenes) and instead measures consistency of the model
across checkpoints on the same inputs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from mmcv import Config
from mmdet3d.apis import init_model, inference_detector
from mmdet3d.datasets.pseudo_label_utils import pairwise_aligned_iou


def _scene_id_from_info(info: Dict) -> Optional[str]:
    if 'point_cloud' in info and 'lidar_idx' in info['point_cloud']:
        return str(info['point_cloud']['lidar_idx'])
    return str(info.get('sample_idx') or info.get('scene_id') or '')


def _points_file_from_info(info: Dict, data_root: Optional[str]) -> Optional[str]:
    scene_id = _scene_id_from_info(info)
    if not scene_id:
        return None
    if data_root:
        path = os.path.join(data_root, 'points', f'{scene_id}.bin')
    else:
        path = os.path.abspath(os.path.join('data', 'scannet', 'points', f'{scene_id}.bin'))
    return path


def _filter_predictions(predictions: Dict[str, Any],
                        allowed_classes: List[int],
                        class_thresholds: Optional[Dict[int, float]] = None,
                        default_thr: float = 0.45) -> Optional[Dict[str, Any]]:
    if not predictions:
        return None
    boxes_3d = predictions.get('boxes_3d')
    scores_3d = predictions.get('scores_3d')
    labels_3d = predictions.get('labels_3d')
    if boxes_3d is None or labels_3d is None or scores_3d is None:
        return None
    # Convert tensors to numpy
    if hasattr(boxes_3d, 'tensor'):
        boxes_np = boxes_3d.tensor.detach().cpu().numpy()
    else:
        boxes_np = np.asarray(boxes_3d)
    scores_np = scores_3d.detach().cpu().numpy() if hasattr(scores_3d, 'detach') else np.asarray(scores_3d)
    labels_np = labels_3d.detach().cpu().numpy() if hasattr(labels_3d, 'detach') else np.asarray(labels_3d)

    # Class filter
    mask_cls = np.isin(labels_np, np.asarray(allowed_classes, dtype=labels_np.dtype))
    if not mask_cls.any():
        return None
    boxes_np = boxes_np[mask_cls]
    scores_np = scores_np[mask_cls]
    labels_np = labels_np[mask_cls]

    # Thresholds (per-class optional)
    if class_thresholds:
        thr_arr = np.full_like(scores_np, fill_value=float(default_thr), dtype=np.float32)
        for cls_id in np.unique(labels_np):
            if int(cls_id) in class_thresholds:
                thr_arr[labels_np == cls_id] = float(class_thresholds[int(cls_id)])
        mask_thr = scores_np >= thr_arr
    else:
        mask_thr = scores_np >= float(default_thr)
    if not mask_thr.any():
        return None
    boxes_np = boxes_np[mask_thr]
    scores_np = scores_np[mask_thr]
    labels_np = labels_np[mask_thr]

    return dict(boxes=boxes_np, scores=scores_np, labels=labels_np)


def generate_pseudo_set_for_indices(cfg: Config,
                                    checkpoint: str,
                                    dataset,
                                    scene_indices: List[int],
                                    allowed_classes: List[int],
                                    class_thresholds: Optional[Dict[int, float]] = None,
                                    default_thr: float = 0.45) -> Dict[str, Dict[str, Any]]:
    """Run inference on selected scenes and return filtered pseudo sets.

    Returns a dict scene_id -> {boxes: Nx6, scores: N, labels: N}.
    """
    model = init_model(cfg, checkpoint, device='cuda:0')
    model.eval()

    pseudo = {}
    for idx in scene_indices:
        if idx < 0 or idx >= len(dataset.data_infos):
            continue
        info = dataset.data_infos[idx]
        scene_id = _scene_id_from_info(info)
        pts = _points_file_from_info(info, getattr(dataset, 'data_root', None))
        if not pts or not os.path.exists(pts):
            continue
        try:
            results = inference_detector(model, pts)
            # results structure: tuple(list(dict)) – follow existing usage
            if isinstance(results, tuple) and len(results) > 0:
                predictions_list = results[0]
                if isinstance(predictions_list, list) and len(predictions_list) > 0:
                    predictions = predictions_list[0]
                else:
                    predictions = {}
            elif isinstance(results, list) and len(results) > 0:
                predictions = results[0]
            else:
                predictions = {}
        except Exception:
            continue

        filtered = _filter_predictions(predictions, allowed_classes, class_thresholds, default_thr)
        if filtered is not None:
            pseudo[scene_id] = {
                'boxes': filtered['boxes'].tolist(),
                'scores': filtered['scores'].tolist(),
                'labels': [int(x) for x in filtered['labels'].tolist()],
            }

    return pseudo


def save_jsonl(records: List[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')


def compute_consistency_drop(prev: Dict[str, Dict[str, Any]],
                             mid: Dict[str, Dict[str, Any]],
                             allowed_classes: List[int],
                             iou_thr: float = 0.25) -> Dict[str, Any]:
    """Compute drop as 1 - recall_prev->mid with IoU matching per scene/class."""
    # Aggregates
    per_class_prev_counts = {int(c): 0 for c in allowed_classes}
    per_class_matched = {int(c): 0 for c in allowed_classes}
    per_scene_prev_counts: Dict[str, int] = {}
    per_scene_matched: Dict[str, int] = {}

    def _match_count(b1: np.ndarray, b2: np.ndarray) -> int:
        # greedy IoU matching
        if b1.size == 0 or b2.size == 0:
            return 0
        ious = pairwise_aligned_iou(b1, b2)  # (N1, N2)
        matched = 0
        used_rows = set()
        used_cols = set()
        # Greedy: keep choosing max IoU until below thr
        while True:
            max_idx = np.unravel_index(np.argmax(ious, axis=None), ious.shape)
            i = int(max_idx[0])
            j = int(max_idx[1])
            if ious[i, j] < iou_thr:
                break
            if i in used_rows or j in used_cols:
                ious[i, j] = -1.0
                continue
            matched += 1
            used_rows.add(i)
            used_cols.add(j)
            ious[i, :] = -1.0
            ious[:, j] = -1.0
        return matched

    # Iterate scenes present in prev
    for sid, prev_rec in prev.items():
        labels_prev = np.asarray(prev_rec.get('labels', []), dtype=np.int64)
        boxes_prev = np.asarray(prev_rec.get('boxes', []), dtype=np.float32)
        scene_prev = 0
        scene_matched = 0
        mid_rec = mid.get(sid, {'labels': [], 'boxes': []})
        labels_mid = np.asarray(mid_rec.get('labels', []), dtype=np.int64)
        boxes_mid = np.asarray(mid_rec.get('boxes', []), dtype=np.float32)

        for c in allowed_classes:
            m = labels_prev == c
            n_prev = int(m.sum())
            if n_prev == 0:
                continue
            per_class_prev_counts[c] += n_prev
            scene_prev += n_prev
            b1 = boxes_prev[m]
            b2 = boxes_mid[labels_mid == c]
            matched = _match_count(b1, b2)
            per_class_matched[c] += matched
            scene_matched += matched

        per_scene_prev_counts[sid] = scene_prev
        per_scene_matched[sid] = scene_matched

    # Compute drops
    drop_by_class = {}
    for c in allowed_classes:
        n_prev = per_class_prev_counts.get(c, 0)
        if n_prev > 0:
            rec = per_class_matched.get(c, 0) / float(n_prev)
            drop_by_class[int(c)] = float(max(0.0, 1.0 - rec))
        else:
            drop_by_class[int(c)] = 0.0

    drop_by_scene = {}
    for sid, n_prev in per_scene_prev_counts.items():
        if n_prev > 0:
            rec = per_scene_matched.get(sid, 0) / float(n_prev)
            drop_by_scene[sid] = float(max(0.0, 1.0 - rec))
        else:
            drop_by_scene[sid] = 0.0

    # Top-k helpers are left to caller based on thresholds; return basics + counts
    return {
        'drop_by_class': drop_by_class,
        'drop_by_scene': drop_by_scene,
        'per_class_prev_counts': per_class_prev_counts,
        'per_scene_prev_counts': per_scene_prev_counts,
    }

