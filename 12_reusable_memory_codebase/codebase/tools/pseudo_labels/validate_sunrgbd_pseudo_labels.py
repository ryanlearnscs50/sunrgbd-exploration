#!/usr/bin/env python3
"""Validate SUN RGB-D pseudo label files against GT annotations.

This validator is lightweight and intentionally does *not* run model inference.
It checks:
- file format sanity (keys/shapes/dtypes)
- basic quality diagnostics via class-aware IoU matching to GT on previous classes

The main entrypoint is `validate_sunrgbd_pseudo_labels_from_file`, which is also
importable by training scripts.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from mmdet3d.datasets.pseudo_label_utils import pairwise_aligned_iou


def _aabb6_from_centered_depth_boxes(boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0, 6), dtype=np.float32)
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError(f"Boxes must be 2D with >=6 dims; got {arr.shape}")

    if arr.shape[1] < 7:
        return arr[:, :6].astype(np.float32, copy=False)

    yaw = arr[:, 6].astype(np.float32)
    dx = arr[:, 3].astype(np.float32)
    dy = arr[:, 4].astype(np.float32)
    c = np.abs(np.cos(yaw))
    s = np.abs(np.sin(yaw))
    aabb_dx = c * dx + s * dy
    aabb_dy = s * dx + c * dy
    return np.stack([arr[:, 0], arr[:, 1], arr[:, 2], aabb_dx, aabb_dy, arr[:, 5]], axis=1).astype(
        np.float32, copy=False
    )


def _load_pseudo(pseudo_file: str) -> Dict[str, Any]:
    path = Path(pseudo_file)
    if not path.exists():
        raise FileNotFoundError(f"Pseudo file not found: {pseudo_file}")
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Pseudo file must contain a dict, got {type(data)}")
    return data


def _load_ann_map(ann_file: str) -> Dict[str, Dict[str, np.ndarray]]:
    if not ann_file:
        raise ValueError("ann_file is empty")
    ann_path = Path(ann_file)
    if not ann_path.exists():
        raise FileNotFoundError(f"ann_file not found: {ann_file}")

    import pickle as _pickle

    with ann_path.open("rb") as f:
        infos = _pickle.load(f)
    if not isinstance(infos, list):
        raise ValueError(f"ann_file must contain a list, got {type(infos)}")

    mapping: Dict[str, Dict[str, np.ndarray]] = {}
    for info in infos:
        if not isinstance(info, dict):
            continue
        sid = None
        if "point_cloud" in info and isinstance(info["point_cloud"], dict) and "lidar_idx" in info["point_cloud"]:
            sid = str(info["point_cloud"]["lidar_idx"])
        elif "sample_idx" in info:
            sid = str(info["sample_idx"])
        if sid is None:
            continue
        annos = info.get("annos", {}) or {}
        if not isinstance(annos, dict):
            continue
        boxes = np.asarray(annos.get("gt_boxes_upright_depth", np.zeros((0, 7))), dtype=np.float32)
        labels = np.asarray(annos.get("class", np.zeros((0,))), dtype=np.int64).reshape(-1)
        mapping[sid] = {"boxes": boxes, "labels": labels}
    return mapping


def _infer_previous_seen_classes(stage_id: int) -> List[int]:
    # Prefer explicit mapping file (repo source-of-truth for SUNRGBD 40-class 8x5).
    sys.path.append(str(Path(__file__).resolve().parents[2] / "configs" / "_base_" / "class_mappings"))
    try:
        from sunrgbd_40class_mapping import get_stage_definitions  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Failed to import sunrgbd_40class_mapping: {e}") from e

    prev: List[int] = []
    for sd in get_stage_definitions():
        sid = int(sd.get("stage_id", 0) or 0)
        if sid < int(stage_id):
            prev.extend(int(x) for x in sd.get("class_indices", []))
    return sorted(set(prev))


def _extract_arrays(record: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = record.get("gt_boxes_upright_depth", record.get("boxes", np.zeros((0, 7), dtype=np.float32)))
    labels = record.get("class", record.get("labels", np.zeros((0,), dtype=np.int64)))
    scores = record.get("scores", record.get("scores_3d", None))

    boxes = np.asarray(boxes, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores is None:
        scores = np.ones((labels.shape[0],), dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)

    if boxes.ndim != 2 or boxes.shape[1] < 6:
        raise ValueError(f"Pseudo boxes must be (N,6/7[+]), got {boxes.shape}")
    if boxes.shape[0] != labels.shape[0] or labels.shape[0] != scores.shape[0]:
        raise ValueError(f"Length mismatch: boxes={boxes.shape}, labels={labels.shape}, scores={scores.shape}")
    return boxes, labels, scores


def compute_class_aware_iou_metrics(
    pseudo: Dict[str, Any],
    ann_map: Dict[str, Dict[str, np.ndarray]],
    target_classes: Sequence[int],
    iou_thrs: Sequence[float] = (0.25, 0.5),
    *,
    max_scenes: Optional[int] = None,
    seed: int = 0,
) -> Dict[float, Dict[str, Any]]:
    cls_set = {int(x) for x in target_classes}
    scene_ids = [sid for sid in pseudo.keys() if sid != "__meta__"]
    scene_ids = [str(s) for s in scene_ids if str(s) in ann_map]

    scene_ids.sort()
    if max_scenes is not None and int(max_scenes) > 0 and len(scene_ids) > int(max_scenes):
        rng = np.random.RandomState(int(seed))
        idx = rng.choice(len(scene_ids), size=int(max_scenes), replace=False)
        idx.sort()
        scene_ids = [scene_ids[int(i)] for i in idx.tolist()]

    results: Dict[float, Dict[str, Any]] = {}
    for thr in iou_thrs:
        pseudo_total = 0
        pseudo_hits = 0
        gt_total = 0
        gt_hits = 0
        per_class_hits: Dict[int, int] = {}
        per_class_totals: Dict[int, int] = {}

        for sid in scene_ids:
            rec = pseudo.get(sid, None)
            if not isinstance(rec, dict):
                continue
            p_boxes, p_labels, _ = _extract_arrays(rec)
            g = ann_map.get(str(sid))
            if g is None:
                continue
            g_boxes = g["boxes"]
            g_labels = g["labels"]

            p_mask = np.isin(p_labels, np.array(sorted(cls_set), dtype=np.int64))
            g_mask = np.isin(g_labels, np.array(sorted(cls_set), dtype=np.int64))
            if not bool(p_mask.any()) and not bool(g_mask.any()):
                continue

            p_boxes = p_boxes[p_mask]
            p_labels = p_labels[p_mask]
            g_boxes = g_boxes[g_mask]
            g_labels = g_labels[g_mask]

            if p_boxes.size:
                pseudo_total += int(p_labels.shape[0])
            if g_boxes.size:
                gt_total += int(g_labels.shape[0])

            if p_boxes.size == 0 or g_boxes.size == 0:
                for lbl in p_labels.tolist():
                    per_class_totals[int(lbl)] = per_class_totals.get(int(lbl), 0) + 1
                continue

            p_aabb6 = _aabb6_from_centered_depth_boxes(p_boxes)
            g_aabb6 = _aabb6_from_centered_depth_boxes(g_boxes)

            for cls in sorted(set(p_labels.tolist()) | set(g_labels.tolist())):
                cls = int(cls)
                p_idx = np.where(p_labels == cls)[0]
                g_idx = np.where(g_labels == cls)[0]
                if p_idx.size:
                    per_class_totals[cls] = per_class_totals.get(cls, 0) + int(p_idx.size)
                if p_idx.size == 0 or g_idx.size == 0:
                    continue

                iou = pairwise_aligned_iou(p_aabb6[p_idx], g_aabb6[g_idx])
                max_p = iou.max(axis=1) if iou.size else np.zeros((p_idx.size,), dtype=np.float32)
                max_g = iou.max(axis=0) if iou.size else np.zeros((g_idx.size,), dtype=np.float32)

                hit_p = (max_p >= float(thr))
                hit_g = (max_g >= float(thr))
                pseudo_hits += int(hit_p.sum())
                gt_hits += int(hit_g.sum())
                per_class_hits[cls] = per_class_hits.get(cls, 0) + int(hit_p.sum())

        results[float(thr)] = {
            "pseudo_total": int(pseudo_total),
            "pseudo_hits": int(pseudo_hits),
            "pseudo_hit_rate": float(pseudo_hits / pseudo_total) if pseudo_total else 0.0,
            "gt_total": int(gt_total),
            "gt_hits": int(gt_hits),
            "gt_recall": float(gt_hits / gt_total) if gt_total else 0.0,
            "per_class_hits": {int(k): int(v) for k, v in per_class_hits.items()},
            "per_class_totals": {int(k): int(v) for k, v in per_class_totals.items()},
            "scenes_evaluated": int(len(scene_ids)),
        }
    return results


def validate_sunrgbd_pseudo_labels_from_file(
    *,
    pseudo_file: str,
    ann_file: str,
    stage_id: int,
    iou_thrs: Sequence[float] = (0.25, 0.5),
    max_scenes: Optional[int] = 200,
    seed: int = 0,
    verbose: bool = True,
) -> Dict[float, Dict[str, Any]]:
    pseudo = _load_pseudo(pseudo_file)
    ann_map = _load_ann_map(ann_file)

    meta = pseudo.get("__meta__", {}) if isinstance(pseudo.get("__meta__", None), dict) else {}
    target = meta.get("previous_seen_classes", None)
    if target is None:
        target = _infer_previous_seen_classes(int(stage_id))
    target = [int(x) for x in (target or [])]
    if not target:
        raise ValueError(f"No target previous classes found for stage_id={stage_id}.")

    metrics = compute_class_aware_iou_metrics(
        pseudo=pseudo,
        ann_map=ann_map,
        target_classes=target,
        iou_thrs=iou_thrs,
        max_scenes=max_scenes,
        seed=seed,
    )

    if verbose:
        for thr in sorted(metrics.keys()):
            m = metrics[thr]
            print(
                f"[SUNRGBD pseudo] stage={stage_id} IoU>={thr:.2f} "
                f"pseudo_hit_rate={m['pseudo_hit_rate']*100:.2f}% "
                f"gt_recall={m['gt_recall']*100:.2f}% "
                f"(pseudo {m['pseudo_hits']}/{m['pseudo_total']}, "
                f"gt {m['gt_hits']}/{m['gt_total']}, scenes={m['scenes_evaluated']})"
            )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo-file", required=True, help="Path to stage_{k}_*_pseudo_labels.pkl")
    parser.add_argument("--ann-file", required=True, help="Path to SUNRGBD infos train/val PKL")
    parser.add_argument("--stage-id", type=int, required=True)
    parser.add_argument("--iou-thrs", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--max-scenes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    validate_sunrgbd_pseudo_labels_from_file(
        pseudo_file=args.pseudo_file,
        ann_file=args.ann_file,
        stage_id=int(args.stage_id),
        iou_thrs=tuple(float(x) for x in args.iou_thrs),
        max_scenes=int(args.max_scenes) if args.max_scenes else None,
        seed=int(args.seed),
        verbose=True,
    )


if __name__ == "__main__":
    main()

