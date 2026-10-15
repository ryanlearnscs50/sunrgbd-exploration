#!/usr/bin/env python3
"""Create a compact audit dashboard for a saved object-memory bank.

The command is intentionally offline: it reads the self-contained bank pickle
and does not reopen SUN RGB-D scenes or initialize MMDetection3D.

Example:
  MPLBACKEND=Agg ./venv/bin/python \
    tools/analysis_tools/visualize_object_memory_bank.py \
    ../incremental_logs/<run>/object_memory_bank/object_memory_bank_stage_5.pkl \
    --output-dir ../visualizations/object_memory_random_stage5
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np


def load_class_names(mapping_path: Path) -> List[str]:
    spec = importlib.util.spec_from_file_location(
        "sunrgbd_40class_mapping", str(mapping_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load class mapping: {mapping_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    names = getattr(module, "SUNRGBD_40_RAW_TOP40_CLASSES", None)
    if not isinstance(names, list):
        raise RuntimeError(f"No SUNRGBD_40_RAW_TOP40_CLASSES in {mapping_path}")
    return [str(name) for name in names]


def load_bank(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or payload.get("memory_level") != "object":
        raise ValueError(f"Not an object-memory state: {path}")
    exemplars = payload.get("exemplars")
    if not isinstance(exemplars, dict) or not exemplars:
        raise ValueError(f"Object-memory state has no exemplars: {path}")
    payload["exemplars"] = {
        int(class_id): list(rows) for class_id, rows in exemplars.items()
    }
    return payload


def _finite(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def summarize(payload: Dict[str, Any], class_names: Sequence[str]) -> Dict[str, Any]:
    exemplars = payload["exemplars"]
    rows = [row for class_rows in exemplars.values() for row in class_rows]
    class_counts = {int(cid): len(class_rows)
                    for cid, class_rows in sorted(exemplars.items())}
    source_counts = {
        int(cid): len({str(row.get("scene_id")) for row in class_rows})
        for cid, class_rows in sorted(exemplars.items())
    }
    distinct_sources = len({str(row.get("scene_id")) for row in rows})
    point_counts = _finite([row.get("point_count", 0) for row in rows])
    floor_offsets = _finite([
        row.get("source_floor_offset", np.nan) for row in rows
    ])
    stage_counts = Counter(int(row.get("stage_id", -1)) for row in rows)
    design_rows = [row.get("learning_dynamics_design2") for row in rows
                   if isinstance(row.get("learning_dynamics_design2"), dict)]
    return {
        "format_version": payload.get("format_version"),
        "stage_id": payload.get("stage_id"),
        "selection_strategy": payload.get("config", {}).get(
            "selection_strategy", "unknown"),
        "total_exemplars": len(rows),
        "stored_classes": len(exemplars),
        "class_counts": {str(cid): count for cid, count in class_counts.items()},
        "class_names": {
            str(cid): class_names[cid] if cid < len(class_names) else f"class_{cid}"
            for cid in class_counts
        },
        "distinct_source_scenes_per_class": {
            str(cid): count for cid, count in source_counts.items()
        },
        "source_scene_diversity": {
            "distinct_scenes": int(distinct_sources),
            "duplicate_assignments": int(len(rows) - distinct_sources),
        },
        "stage_counts": {str(stage): count
                         for stage, count in sorted(stage_counts.items())},
        "point_count": {
            "total": int(point_counts.sum()),
            "min": int(point_counts.min()),
            "mean": float(point_counts.mean()),
            "median": float(np.median(point_counts)),
            "max": int(point_counts.max()),
            "below_20": int(np.sum(point_counts < 20)),
            "below_50": int(np.sum(point_counts < 50)),
            "below_100": int(np.sum(point_counts < 100)),
            "below_500": int(np.sum(point_counts < 500)),
        },
        "source_floor_offset": {
            "count": int(len(floor_offsets)),
            "min": float(floor_offsets.min()) if len(floor_offsets) else None,
            "median": float(np.median(floor_offsets)) if len(floor_offsets) else None,
            "max": float(floor_offsets.max()) if len(floor_offsets) else None,
        },
        "design2_annotated_exemplars": len(design_rows),
    }


def plot_overview(payload: Dict[str, Any], class_names: Sequence[str],
                  output_path: Path) -> None:
    exemplars = payload["exemplars"]
    class_ids = sorted(exemplars)
    labels = [class_names[cid] if cid < len(class_names) else str(cid)
              for cid in class_ids]
    counts = [len(exemplars[cid]) for cid in class_ids]
    median_points = [np.median([row["point_count"] for row in exemplars[cid]])
                     for cid in class_ids]
    distinct_sources = [len({str(row.get("scene_id"))
                             for row in exemplars[cid]}) for cid in class_ids]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11), constrained_layout=True)
    x = np.arange(len(class_ids))
    axes[0, 0].bar(x, counts, color="#4477AA")
    axes[0, 0].set_title("Stored exemplars per class")
    axes[0, 0].set_ylabel("objects")

    axes[0, 1].bar(x, median_points, color="#228833")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("Median cropped points per exemplar")
    axes[0, 1].set_ylabel("points (log scale)")

    axes[1, 0].bar(x, distinct_sources, color="#CCBB44")
    axes[1, 0].set_title("Distinct source scenes per class")
    axes[1, 0].set_ylabel("scenes")

    offsets_by_stage: Dict[int, List[float]] = {}
    for class_rows in exemplars.values():
        for row in class_rows:
            offset = float(row.get("source_floor_offset", np.nan))
            if np.isfinite(offset):
                offsets_by_stage.setdefault(int(row.get("stage_id", -1)), []).append(offset)
    stages = sorted(offsets_by_stage)
    if stages:
        axes[1, 1].boxplot([offsets_by_stage[stage] for stage in stages],
                           tick_labels=[str(stage) for stage in stages],
                           showfliers=False)
    axes[1, 1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1, 1].set_title("Source floor-relative box-bottom offset")
    axes[1, 1].set_xlabel("source cohort")
    axes[1, 1].set_ylabel("metres")

    for axis in axes[:, :].flat:
        if axis is axes[1, 1]:
            continue
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2)

    config = payload.get("config", {})
    fig.suptitle(
        "Object-memory audit — "
        f"stage {payload.get('stage_id')}, {sum(counts)} objects, "
        f"selection={config.get('selection_strategy', 'unknown')}",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _sample_row(rows: Sequence[Dict[str, Any]], rng: np.random.RandomState,
                sample_index: int) -> Dict[str, Any]:
    if sample_index >= 0:
        return rows[min(sample_index, len(rows) - 1)]
    return rows[int(rng.randint(len(rows)))]


def plot_gallery(payload: Dict[str, Any], class_names: Sequence[str],
                 output_path: Path, seed: int, sample_index: int,
                 max_points: int) -> None:
    exemplars = payload["exemplars"]
    class_ids = sorted(exemplars)
    ncols = 5
    nrows = int(np.ceil(len(class_ids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.8 * nrows),
                             constrained_layout=True)
    axes_flat = np.asarray(axes).reshape(-1)
    rng = np.random.RandomState(seed)
    scatter = None
    for axis, class_id in zip(axes_flat, class_ids):
        row = _sample_row(exemplars[class_id], rng, sample_index)
        points = np.asarray(row["points"])
        if len(points) > max_points:
            keep = rng.choice(len(points), size=max_points, replace=False)
            points = points[keep]
        xyz = points[:, :3]
        scatter = axis.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], s=2,
                               cmap="viridis", linewidths=0, rasterized=True)
        box_size = np.asarray(row.get("box_size", row.get("bbox", [0] * 6)[3:6]))
        if box_size.size >= 2:
            dx, dy = box_size[:2]
            axis.add_patch(plt.Rectangle((-dx / 2, -dy / 2), dx, dy,
                                         fill=False, color="#EE6677", linewidth=0.8))
        name = class_names[class_id] if class_id < len(class_names) else str(class_id)
        axis.set_title(f"{class_id}: {name}\n{len(row['points'])} pts", fontsize=9)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes_flat[len(class_ids):]:
        axis.axis("off")
    if scatter is not None:
        fig.colorbar(scatter, ax=axes_flat.tolist(), shrink=0.35,
                     label="local Z (m)")
    fig.suptitle("One self-contained object crop per class (local XY view)",
                 fontsize=15)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path(
        "configs/_base_/class_mappings/sunrgbd_40class_mapping.py"))
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--sample-index", type=int, default=-1,
                        help="Fixed exemplar index; negative chooses deterministically at random.")
    parser.add_argument("--max-points", type=int, default=2500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")
    payload = load_bank(args.bank)
    class_names = load_class_names(args.mapping)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(payload, class_names)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    plot_overview(payload, class_names, args.output_dir / "bank_overview.png")
    plot_gallery(payload, class_names, args.output_dir / "exemplar_gallery.png",
                 args.seed, args.sample_index, args.max_points)
    print(json.dumps({
        "bank": str(args.bank),
        "output_dir": str(args.output_dir),
        "total_exemplars": summary["total_exemplars"],
        "stored_classes": summary["stored_classes"],
    }, indent=2))


if __name__ == "__main__":
    main()
