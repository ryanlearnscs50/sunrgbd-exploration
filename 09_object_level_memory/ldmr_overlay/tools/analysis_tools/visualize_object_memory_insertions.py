#!/usr/bin/env python3
"""Render deterministic before/after views of real object-memory insertion.

This is an offline diagnostic: it loads SUN RGB-D point clouds and a saved
object bank, then calls the same placement routine used by the training
pipeline.  No detector checkpoint or GPU is required.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import mmcv
import numpy as np
from matplotlib.patches import Polygon

from mmdet3d.datasets.pipelines.exemplar_insertion import InsertExemplarObjects


def load_class_names(mapping_path: Path) -> List[str]:
    spec = importlib.util.spec_from_file_location("sunrgbd_mapping", mapping_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load class mapping: {mapping_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return list(module.SUNRGBD_40_RAW_TOP40_CLASSES)


def box_xy_corners(box: Sequence[float], bottom_center: bool = False) -> np.ndarray:
    """Return the four oriented XY corners of a seven-dimensional depth box."""
    del bottom_center  # XY geometry is independent of the Z convention.
    box = np.asarray(box, dtype=np.float32)
    dx, dy = box[3:5] / 2.0
    corners = np.asarray([[-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]])
    yaw = float(box[6]) if len(box) >= 7 else 0.0
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray([[c, s], [-s, c]], dtype=np.float32)
    return corners @ rotation + box[:2]


def _load_bank(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("memory_level") != "object":
        raise ValueError(f"Not an object-memory bank: {path}")
    payload["exemplars"] = {
        int(class_id): list(rows)
        for class_id, rows in payload["exemplars"].items()
    }
    return payload


def _pick_exemplar(rows: Sequence[Dict[str, Any]], rank: int) -> Dict[str, Any]:
    """Choose a reasonably supported crop while remaining deterministic."""
    eligible = [row for row in rows
                if float(row.get("source_floor_offset", 0.0)) >= 0.0]
    ordered = sorted(eligible or rows,
                     key=lambda row: (-int(row["point_count"]), str(row["scene_id"])))
    return ordered[min(rank, len(ordered) - 1)]


def make_examples(bank: Dict[str, Any], infos: Sequence[Dict[str, Any]],
                  points_root: Path, scene_indices: Sequence[int],
                  class_ids: Sequence[int], seed: int) -> List[Dict[str, Any]]:
    if len(scene_indices) != len(class_ids):
        raise ValueError("scene indices and class IDs must have equal lengths")
    transform = InsertExemplarObjects(
        collision_threshold=0.0, placement_jitter=0.35,
        max_placement_attempts=100, floor_percentile=1.0,
        preserve_source_height=True)
    examples = []
    for row_index, (scene_index, class_id) in enumerate(zip(scene_indices, class_ids)):
        info = infos[scene_index]
        points_path = points_root / info["pts_path"]
        points = np.fromfile(points_path, dtype=np.float32).reshape(-1, 6)
        boxes = np.asarray(
            info["annos"].get("gt_boxes_upright_depth", np.empty((0, 7))),
            dtype=np.float32).reshape(-1, 7)
        exemplar = _pick_exemplar(bank["exemplars"][class_id], row_index + 2)
        random.seed(seed + row_index)
        placed_points, placed_box = transform._find_valid_placement(
            np.asarray(exemplar["points"], dtype=np.float32),
            np.asarray(exemplar["bbox"], dtype=np.float32), boxes, points,
            source_floor_offset=float(exemplar.get("source_floor_offset", 0.0)))
        if placed_points is None:
            raise RuntimeError(
                f"Could not place class {class_id} in scene index {scene_index}")
        examples.append({
            "scene_index": int(scene_index),
            "scene_id": str(info["point_cloud"]["lidar_idx"]),
            "points": points,
            "boxes": boxes,
            "class_id": int(class_id),
            "exemplar": exemplar,
            "placed_points": placed_points,
            "placed_box": placed_box,
        })
    return examples


def _scatter_scene(axis: Any, points: np.ndarray, keep: np.ndarray) -> None:
    axis.set_facecolor("#101820")
    axis.scatter(points[keep, 0], points[keep, 1], c=points[keep, 2],
                 cmap="cividis", s=0.55, linewidths=0, rasterized=True,
                 alpha=0.72)


def plot_examples(examples: Sequence[Dict[str, Any]], class_names: Sequence[str],
                  output_path: Path, seed: int, max_scene_points: int) -> None:
    fig, axes = plt.subplots(len(examples), 3, figsize=(16, 4.6 * len(examples)),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)
    for row_index, (example, row_axes) in enumerate(zip(examples, axes)):
        points = example["points"]
        placed = example["placed_points"]
        rng = np.random.RandomState(seed + row_index)
        keep = rng.choice(len(points), min(max_scene_points, len(points)), replace=False)
        for axis in row_axes[:2]:
            _scatter_scene(axis, points, keep)
            for box in example["boxes"]:
                axis.add_patch(Polygon(box_xy_corners(box), fill=False,
                                       edgecolor="#56B4E9", linewidth=0.6, alpha=0.7))
            axis.set_aspect("equal", adjustable="box")
            axis.set_xticks([])
            axis.set_yticks([])
        row_axes[0].set_title(f"Scene {example['scene_id']} — natural input")
        row_axes[1].scatter(placed[:, 0], placed[:, 1], c="#D81B60", s=3.0,
                            linewidths=0, rasterized=True, label="pasted crop")
        row_axes[1].add_patch(Polygon(box_xy_corners(example["placed_box"]),
                                     fill=False, edgecolor="#D81B60", linewidth=2.0))
        name = class_names[example["class_id"]]
        row_axes[1].set_title(
            f"After replay — pasted {name} ({len(placed):,} points)")
        row_axes[1].legend(loc="upper right", frameon=True, markerscale=2.5)

        side = row_axes[2]
        side.scatter(points[keep, 0], points[keep, 2], c="0.65", s=0.4,
                     linewidths=0, rasterized=True, alpha=0.35)
        side.scatter(placed[:, 0], placed[:, 2], c="#D81B60", s=3.0,
                     linewidths=0, rasterized=True)
        box = example["placed_box"]
        side.add_patch(plt.Rectangle(
            (box[0] - box[3] / 2.0, box[2]), box[3], box[5], fill=False,
            edgecolor="#D81B60", linewidth=2.0))
        floor_z = float(np.percentile(points[:, 2], 1.0))
        side.axhline(floor_z, color="#1B9E77", linestyle="--", linewidth=1.0,
                     label="scene floor (1st percentile)")
        side.set_xlabel("X (m)")
        side.set_ylabel("Z (m)")
        side.set_title(
            f"Height-aware placement — saved offset "
            f"{example['exemplar'].get('source_floor_offset', 0.0):.2f} m")
        side.legend(loc="upper right", fontsize=8)
        side.grid(alpha=0.15)
    fig.suptitle(
        "Object-level memory replay in real SUN RGB-D scenes\n"
        "Magenta points/box are inserted; black boxes are natural ground truth",
        fontsize=16)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--infos", type=Path, default=Path(
        "data/sunrgbd/sunrgbd_infos_train_40class.pkl"))
    parser.add_argument("--points-root", type=Path, default=Path("data/sunrgbd"))
    parser.add_argument("--mapping", type=Path, default=Path(
        "configs/_base_/class_mappings/sunrgbd_40class_mapping.py"))
    parser.add_argument("--scene-indices", type=int, nargs="+", default=[0, 9, 23])
    parser.add_argument("--class-ids", type=int, nargs="+", default=[0, 8, 27])
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--max-scene-points", type=int, default=18000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_scene_points <= 0:
        raise ValueError("--max-scene-points must be positive")
    bank = _load_bank(args.bank)
    infos = mmcv.load(str(args.infos))
    names = load_class_names(args.mapping)
    examples = make_examples(bank, infos, args.points_root, args.scene_indices,
                             args.class_ids, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_examples(examples, names, args.output_dir / "pasted_scene_examples.png",
                  args.seed, args.max_scene_points)
    manifest = [{
        "scene_index": row["scene_index"],
        "scene_id": row["scene_id"],
        "class_id": row["class_id"],
        "class_name": names[row["class_id"]],
        "source_scene_id": str(row["exemplar"]["scene_id"]),
        "source_object_idx": int(row["exemplar"]["object_idx"]),
        "point_count": int(len(row["placed_points"])),
        "source_floor_offset": float(row["exemplar"].get("source_floor_offset", 0.0)),
        "placed_box_bottom_center": np.asarray(row["placed_box"]).tolist(),
    } for row in examples]
    (args.output_dir / "pasted_scene_examples.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "examples": manifest}, indent=2))


if __name__ == "__main__":
    main()
