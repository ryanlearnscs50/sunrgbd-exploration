import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from refine_sunrgbd_result import depth_to_pointcloud, corners_xy


SUNRGBD_CLASSES = {
    "bed", "table", "sofa", "chair", "toilet",
    "desk", "dresser", "night_stand", "bookshelf", "bathtub",
}
PRED_COLOR = "#d62728"
GT_COLOR = "#1f77b4"


def load_gt_boxes(scene_dir):
    path = os.path.join(scene_dir, "annotation3Dfinal", "index.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    boxes = []
    for obj in data.get("objects", []):
        if not obj or not obj.get("name"):
            continue
        class_name = obj["name"].split(":")[0].strip().lower()
        if class_name not in SUNRGBD_CLASSES:
            continue
        polygons = obj.get("polygon") or []
        if not polygons:
            continue
        poly = polygons[0]
        xs = np.array(poly["X"], dtype=np.float32)
        zs = np.array(poly["Z"], dtype=np.float32)
        ymin = float(poly["Ymin"])
        ymax = float(poly["Ymax"])

        edge1 = np.array([xs[1] - xs[0], zs[1] - zs[0]])
        edge2 = np.array([xs[2] - xs[1], zs[2] - zs[1]])
        dx = float(np.linalg.norm(edge1))
        dz = float(np.linalg.norm(edge2))
        yaw = float(math.atan2(edge1[1], edge1[0]))
        boxes.append({
            "class_name": class_name,
            "bbox_3d": [
                float(xs.mean()),
                float((ymin + ymax) / 2),
                float(zs.mean()),
                dx,
                float(abs(ymax - ymin)),
                dz,
                yaw,
            ],
        })
    return boxes


def box_xz_corners(box):
    cx, _, cz, dx, _, dz, yaw = box
    local = np.array([
        [-dx / 2, -dz / 2],
        [dx / 2, -dz / 2],
        [dx / 2, dz / 2],
        [-dx / 2, dz / 2],
    ])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cz])


def axis_bounds_xz(box):
    pts = box_xz_corners(box)
    y_min = box[1] - box[4] / 2
    y_max = box[1] + box[4] / 2
    return np.array([pts[:, 0].min(), y_min, pts[:, 1].min()]), np.array([
        pts[:, 0].max(), y_max, pts[:, 1].max()
    ])


def iou3d_axis_approx(a, b):
    min_a, max_a = axis_bounds_xz(a)
    min_b, max_b = axis_bounds_xz(b)
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_size = np.maximum(0, inter_max - inter_min)
    inter = float(np.prod(inter_size))
    vol_a = float(np.prod(np.maximum(0, max_a - min_a)))
    vol_b = float(np.prod(np.maximum(0, max_b - min_b)))
    union = vol_a + vol_b - inter
    return inter / union if union else 0.0


def draw_xz_box(ax, box, color, label, linestyle="-"):
    pts = box_xz_corners(np.array(box, dtype=np.float32))
    ax.add_patch(patches.Polygon(
        pts, fill=False, edgecolor=color, linewidth=1.8, linestyle=linestyle))
    ax.text(pts[:, 0].mean(), pts[:, 1].max() + 0.035, label,
            color=color, fontsize=7, ha="center", va="bottom")


def draw_xy_box(ax, box, color, linestyle="-"):
    cx, cy, _, dx, dy, _, _ = box
    ax.add_patch(patches.Rectangle(
        (cx - dx / 2, cy - dy / 2), dx, dy, fill=False,
        edgecolor=color, linewidth=1.6, linestyle=linestyle))


def compare_scene(scene_dir, refined_summary_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    scene = os.path.basename(scene_dir.rstrip("/\\"))
    xyzrgb, _, _ = depth_to_pointcloud(scene_dir)
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:6] / 255.0
    step = max(1, len(xyz) // 35000)

    with open(refined_summary_path, encoding="utf-8") as f:
        refined = json.load(f)
    preds = refined["kept_detections"]
    gt = load_gt_boxes(scene_dir)

    for pred in preds:
        best = 0.0
        best_gt = None
        for gt_box in gt:
            if gt_box["class_name"] != pred["class_name"]:
                continue
            iou = iou3d_axis_approx(pred["bbox_3d"], gt_box["bbox_3d"])
            if iou > best:
                best = iou
                best_gt = gt_box["class_name"]
        pred["best_same_class_gt_iou_approx"] = best
        pred["matched_gt_class"] = best_gt

    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharex=False, sharey=False)
    pts = xyz[::step]
    col = rgb[::step]

    for ax, title, show_preds, show_gt in [
        (axes[0], "Ground truth boxes", False, True),
        (axes[1], "TR3D refined predictions", True, False),
    ]:
        ax.scatter(pts[:, 0], pts[:, 2], c=col, s=0.35, linewidths=0)
        if show_gt:
            for box in gt:
                draw_xz_box(ax, box["bbox_3d"], GT_COLOR, box["class_name"])
        if show_preds:
            for pred in preds:
                label = f"{pred['class_name']} {pred['score']:.2f}"
                draw_xz_box(ax, pred["bbox_3d"], PRED_COLOR, label)
        ax.set_title(f"{scene}: {title} (X/Z floor plane)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z depth (m)")
        ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{scene}_gt_vs_pred_side_by_side.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(pts[:, 0], pts[:, 2], c=col, s=0.35, linewidths=0)
    for box in gt:
        draw_xz_box(ax, box["bbox_3d"], GT_COLOR, f"GT {box['class_name']}")
    for pred in preds:
        draw_xz_box(ax, pred["bbox_3d"], PRED_COLOR,
                    f"P {pred['class_name']} {pred['score']:.2f}", "--")
    ax.set_title(f"{scene}: GT blue vs predictions dashed red (X/Z)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z depth (m)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{scene}_gt_pred_overlay.png"), dpi=180)
    plt.close(fig)

    comparison = {
        "scene_dir": scene_dir,
        "num_gt_supported_class_boxes": len(gt),
        "num_refined_predictions": len(preds),
        "gt_boxes": gt,
        "predictions_with_best_same_class_iou": preds,
    }
    with open(os.path.join(out_dir, f"{scene}_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", action="append", required=True,
                        help="scene_dir|refined_summary.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    all_results = {}
    for item in args.scene:
        scene_dir, summary = item.split("|", 1)
        result = compare_scene(scene_dir, summary, args.out)
        scene_name = os.path.basename(scene_dir.rstrip("/\\"))
        all_results[scene_name] = result
        print(f"{scene_name}: GT={result['num_gt_supported_class_boxes']} refined_pred={result['num_refined_predictions']}")

    with open(os.path.join(args.out, "two_scene_comparison_index.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved comparison outputs to {args.out}")


if __name__ == "__main__":
    main()
