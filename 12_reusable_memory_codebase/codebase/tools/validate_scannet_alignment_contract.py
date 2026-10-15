#!/usr/bin/env python3
"""Validate ScanNet alignment contract for TR3D ScanNet-35 pipelines.

Contract this validator enforces:
1) `annos['gt_boxes_upright_depth']` must match `<scene>_aligned_bbox.npy` (first 6 dims).
2) After applying `axis_align_matrix` to `<scene>_vert.npy`, bbox centers must lie
   within the aligned point bounds (ratio >= threshold).
3) Raw unaligned points should not satisfy the aligned-center check at the same
   threshold (sanity guard against silent frame mixing).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_DATA_ROOT = "data/scannet"
DEFAULT_ANN_FILE = "scannet_infos_train_40class_corrected.pkl"
DEFAULT_INSTANCE_DATA_SUBDIR = "scannet_instance_data_40class"


@dataclass
class SceneValidationResult:
    scene_id: str
    n_boxes: int
    box_match_ok: bool
    aligned_center_ratio: float
    raw_center_ratio: float
    reason: str = ""


def _load_infos(ann_file: Path) -> List[Dict[str, Any]]:
    with ann_file.open("rb") as f:
        infos = pickle.load(f)
    if isinstance(infos, dict) and "data_list" in infos:
        infos = infos["data_list"]
    if not isinstance(infos, list):
        raise ValueError(f"Unsupported ann file payload type: {type(infos)!r}")
    return infos


def _resolve_ann_file(data_root: Path, ann_file: str) -> Path:
    p = Path(str(ann_file))
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return data_root / p


def _scene_id_from_info(info: Dict[str, Any]) -> str:
    pc = info.get("point_cloud", {})
    sid = pc.get("lidar_idx", None)
    if sid is None:
        sid = info.get("sample_idx", None)
    if sid is None:
        raise ValueError("Cannot resolve scene id from info entry")
    return str(sid)


def _ratio_centers_in_bounds(points_xyz: np.ndarray, boxes6: np.ndarray) -> float:
    if boxes6.size == 0:
        return 1.0
    centers = np.asarray(boxes6[:, :3], dtype=np.float64)
    mins = np.min(points_xyz[:, :3], axis=0)
    maxs = np.max(points_xyz[:, :3], axis=0)
    inside = np.logical_and(centers >= mins[None, :], centers <= maxs[None, :]).all(axis=1)
    return float(np.mean(inside)) if inside.size else 1.0


def _apply_axis_align(points_xyz: np.ndarray, axis_align_matrix: np.ndarray) -> np.ndarray:
    mat = np.asarray(axis_align_matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"axis_align_matrix must be 4x4, got {mat.shape}")
    rot = mat[:3, :3]
    trans = mat[:3, 3]
    return points_xyz @ rot.T + trans


def _validate_one_scene(
    *,
    scene_id: str,
    annos: Dict[str, Any],
    instance_data_dir: Path,
    atol: float,
) -> SceneValidationResult:
    gt_boxes = np.asarray(annos.get("gt_boxes_upright_depth", []), dtype=np.float64)
    if gt_boxes.size == 0:
        return SceneValidationResult(
            scene_id=scene_id,
            n_boxes=0,
            box_match_ok=True,
            aligned_center_ratio=1.0,
            raw_center_ratio=1.0,
            reason="empty_gt",
        )
    if gt_boxes.ndim != 2 or gt_boxes.shape[1] < 6:
        return SceneValidationResult(
            scene_id=scene_id,
            n_boxes=0,
            box_match_ok=False,
            aligned_center_ratio=0.0,
            raw_center_ratio=0.0,
            reason="invalid_gt_boxes_shape",
        )
    gt_boxes6 = gt_boxes[:, :6]

    aligned_bbox_file = instance_data_dir / f"{scene_id}_aligned_bbox.npy"
    vert_file = instance_data_dir / f"{scene_id}_vert.npy"
    axis_file = instance_data_dir / f"{scene_id}_axis_align_matrix.npy"
    if not aligned_bbox_file.exists() or not vert_file.exists() or not axis_file.exists():
        missing = []
        if not aligned_bbox_file.exists():
            missing.append("aligned_bbox")
        if not vert_file.exists():
            missing.append("vert")
        if not axis_file.exists():
            missing.append("axis_align_matrix")
        return SceneValidationResult(
            scene_id=scene_id,
            n_boxes=int(gt_boxes6.shape[0]),
            box_match_ok=False,
            aligned_center_ratio=0.0,
            raw_center_ratio=0.0,
            reason=f"missing_files:{','.join(missing)}",
        )

    aligned_arr = np.asarray(np.load(aligned_bbox_file), dtype=np.float64)
    if aligned_arr.ndim != 2 or aligned_arr.shape[1] < 6:
        return SceneValidationResult(
            scene_id=scene_id,
            n_boxes=int(gt_boxes6.shape[0]),
            box_match_ok=False,
            aligned_center_ratio=0.0,
            raw_center_ratio=0.0,
            reason="invalid_aligned_bbox_shape",
        )
    aligned_boxes6 = aligned_arr[:, :6]

    box_match_ok = bool(
        gt_boxes6.shape == aligned_boxes6.shape
        and np.allclose(gt_boxes6, aligned_boxes6, atol=float(atol), rtol=0.0)
    )

    points = np.asarray(np.load(vert_file), dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        return SceneValidationResult(
            scene_id=scene_id,
            n_boxes=int(gt_boxes6.shape[0]),
            box_match_ok=box_match_ok,
            aligned_center_ratio=0.0,
            raw_center_ratio=0.0,
            reason="invalid_vert_shape",
        )
    points_xyz = points[:, :3]
    axis_align_matrix = np.asarray(np.load(axis_file), dtype=np.float64)

    raw_ratio = _ratio_centers_in_bounds(points_xyz, gt_boxes6)
    aligned_points = _apply_axis_align(points_xyz, axis_align_matrix)
    aligned_ratio = _ratio_centers_in_bounds(aligned_points, gt_boxes6)

    return SceneValidationResult(
        scene_id=scene_id,
        n_boxes=int(gt_boxes6.shape[0]),
        box_match_ok=box_match_ok,
        aligned_center_ratio=float(aligned_ratio),
        raw_center_ratio=float(raw_ratio),
    )


def validate_scannet_alignment_contract(
    *,
    data_root: str = DEFAULT_DATA_ROOT,
    ann_file: str = DEFAULT_ANN_FILE,
    instance_data_subdir: str = DEFAULT_INSTANCE_DATA_SUBDIR,
    sample_scenes: int = 256,
    min_aligned_center_ratio: float = 0.99,
    atol: float = 1e-6,
    fail_on_mismatch: bool = False,
) -> Dict[str, Any]:
    data_root_path = Path(str(data_root)).resolve()
    ann_file_path = _resolve_ann_file(data_root_path, str(ann_file)).resolve()
    instance_data_dir = (data_root_path / str(instance_data_subdir)).resolve()

    if not ann_file_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file_path}")
    if not instance_data_dir.exists():
        raise FileNotFoundError(f"Instance data directory not found: {instance_data_dir}")
    if int(sample_scenes) <= 0:
        raise ValueError(f"sample_scenes must be > 0, got {sample_scenes}")

    infos = _load_infos(ann_file_path)
    checked_infos = infos[: int(sample_scenes)]

    scene_results: List[SceneValidationResult] = []
    for info in checked_infos:
        sid = _scene_id_from_info(info)
        annos = info.get("annos", None)
        if not isinstance(annos, dict):
            # test split may not have annos; skip but keep traceability
            scene_results.append(
                SceneValidationResult(
                    scene_id=sid,
                    n_boxes=0,
                    box_match_ok=True,
                    aligned_center_ratio=1.0,
                    raw_center_ratio=1.0,
                    reason="missing_annos",
                )
            )
            continue
        scene_results.append(
            _validate_one_scene(
                scene_id=sid,
                annos=annos,
                instance_data_dir=instance_data_dir,
                atol=float(atol),
            )
        )

    with_boxes = [r for r in scene_results if r.n_boxes > 0]
    missing_annos = [r.scene_id for r in scene_results if r.reason == "missing_annos"]
    missing_files = [r.scene_id for r in scene_results if r.reason.startswith("missing_files")]
    invalid_shapes = [
        r.scene_id
        for r in scene_results
        if r.reason.startswith("invalid_")
    ]

    box_mismatch = [r.scene_id for r in with_boxes if not r.box_match_ok]
    low_aligned = [
        r.scene_id
        for r in with_boxes
        if float(r.aligned_center_ratio) < float(min_aligned_center_ratio)
    ]
    raw_false_pass = [
        r.scene_id
        for r in with_boxes
        if float(r.raw_center_ratio) >= float(min_aligned_center_ratio)
    ]

    aligned_ratios = [float(r.aligned_center_ratio) for r in with_boxes] or [1.0]
    raw_ratios = [float(r.raw_center_ratio) for r in with_boxes] or [1.0]

    report: Dict[str, Any] = {
        "ok": bool(
            len(box_mismatch) == 0
            and len(low_aligned) == 0
            and len(raw_false_pass) == 0
            and len(missing_files) == 0
            and len(invalid_shapes) == 0
        ),
        "data_root": str(data_root_path),
        "ann_file": str(ann_file_path),
        "instance_data_dir": str(instance_data_dir),
        "scenes_checked": int(len(scene_results)),
        "scenes_with_boxes": int(len(with_boxes)),
        "total_boxes_checked": int(sum(int(r.n_boxes) for r in with_boxes)),
        "min_aligned_center_ratio": float(min_aligned_center_ratio),
        "aligned_center_ratio_mean": float(np.mean(aligned_ratios)),
        "aligned_center_ratio_median": float(np.median(aligned_ratios)),
        "aligned_center_ratio_min": float(np.min(aligned_ratios)),
        "raw_center_ratio_mean": float(np.mean(raw_ratios)),
        "raw_center_ratio_median": float(np.median(raw_ratios)),
        "raw_center_ratio_min": float(np.min(raw_ratios)),
        "box_mismatch_scene_count": int(len(box_mismatch)),
        "box_mismatch_scene_ids": box_mismatch[:50],
        "low_aligned_ratio_scene_count": int(len(low_aligned)),
        "low_aligned_ratio_scene_ids": low_aligned[:50],
        "raw_false_pass_scene_count": int(len(raw_false_pass)),
        "raw_false_pass_scene_ids": raw_false_pass[:50],
        "missing_files_scene_count": int(len(missing_files)),
        "missing_files_scene_ids": missing_files[:50],
        "invalid_shape_scene_count": int(len(invalid_shapes)),
        "invalid_shape_scene_ids": invalid_shapes[:50],
        "missing_annos_scene_count": int(len(missing_annos)),
        "missing_annos_scene_ids": missing_annos[:50],
    }

    if fail_on_mismatch and not report["ok"]:
        raise RuntimeError(
            "ScanNet alignment contract validation failed. "
            f"box_mismatch={report['box_mismatch_scene_count']}, "
            f"low_aligned_ratio={report['low_aligned_ratio_scene_count']}, "
            f"raw_false_pass={report['raw_false_pass_scene_count']}, "
            f"missing_files={report['missing_files_scene_count']}, "
            f"invalid_shape={report['invalid_shape_scene_count']}"
        )

    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate ScanNet alignment contract for TR3D ScanNet-35 data."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=DEFAULT_DATA_ROOT,
        help=f"ScanNet data root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--ann-file",
        type=str,
        default=DEFAULT_ANN_FILE,
        help=(
            "Annotation pkl path. Relative paths are resolved against --data-root. "
            f"(default: {DEFAULT_ANN_FILE})"
        ),
    )
    parser.add_argument(
        "--instance-data-subdir",
        type=str,
        default=DEFAULT_INSTANCE_DATA_SUBDIR,
        help=f"Instance-data subdir under data-root (default: {DEFAULT_INSTANCE_DATA_SUBDIR})",
    )
    parser.add_argument(
        "--sample-scenes",
        type=int,
        default=256,
        help="Number of scenes to validate from the annotation file (default: 256)",
    )
    parser.add_argument(
        "--min-aligned-center-ratio",
        type=float,
        default=0.99,
        help="Minimum acceptable aligned center-in-range ratio per scene (default: 0.99)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance used for GT-vs-aligned bbox match (default: 1e-6)",
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero when contract invariants fail.",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default="",
        help="Optional JSON file path to write the full report.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = validate_scannet_alignment_contract(
        data_root=args.data_root,
        ann_file=args.ann_file,
        instance_data_subdir=args.instance_data_subdir,
        sample_scenes=args.sample_scenes,
        min_aligned_center_ratio=args.min_aligned_center_ratio,
        atol=args.atol,
        fail_on_mismatch=args.fail_on_mismatch,
    )

    compact = {
        "ok": report["ok"],
        "scenes_checked": report["scenes_checked"],
        "scenes_with_boxes": report["scenes_with_boxes"],
        "total_boxes_checked": report["total_boxes_checked"],
        "aligned_center_ratio_mean": report["aligned_center_ratio_mean"],
        "raw_center_ratio_mean": report["raw_center_ratio_mean"],
        "box_mismatch_scene_count": report["box_mismatch_scene_count"],
        "low_aligned_ratio_scene_count": report["low_aligned_ratio_scene_count"],
        "raw_false_pass_scene_count": report["raw_false_pass_scene_count"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))

    if args.report_file:
        report_path = Path(args.report_file).resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    return 0 if report["ok"] else (2 if args.fail_on_mismatch else 1)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        raise SystemExit(2)
