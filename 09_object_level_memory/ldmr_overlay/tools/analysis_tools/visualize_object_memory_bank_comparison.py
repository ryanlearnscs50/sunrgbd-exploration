#!/usr/bin/env python3
"""Compare two complete object-memory banks in one summary figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np

from tools.analysis_tools.visualize_object_memory_bank import (
    load_bank,
    load_class_names,
)


def bank_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    exemplars = payload["exemplars"]
    rows = [row for class_rows in exemplars.values() for row in class_rows]
    counts = np.asarray([int(row["point_count"]) for row in rows])
    sources = [str(row["scene_id"]) for row in rows]
    return {
        "point_counts": counts,
        "distinct_sources": len(set(sources)),
        "duplicate_assignments": len(sources) - len(set(sources)),
        "per_class_median": {
            int(class_id): float(np.median([
                int(row["point_count"]) for row in class_rows
            ])) for class_id, class_rows in exemplars.items()
        },
        "sparse": {
            threshold: int(np.sum(counts < threshold))
            for threshold in (20, 50, 100, 500)
        },
    }


def comparison_summary(random_metrics: Dict[str, Any],
                       design_metrics: Dict[str, Any]) -> Dict[str, Any]:
    def serializable(metrics: Dict[str, Any]) -> Dict[str, Any]:
        counts = metrics["point_counts"]
        return {
            "objects": int(len(counts)),
            "total_points": int(counts.sum()),
            "median_points": float(np.median(counts)),
            "distinct_sources": int(metrics["distinct_sources"]),
            "duplicate_assignments": int(metrics["duplicate_assignments"]),
            "sparse": {str(k): int(v) for k, v in metrics["sparse"].items()},
        }
    return {
        "random": serializable(random_metrics),
        "design2": serializable(design_metrics),
    }


def plot_comparison(random_metrics: Dict[str, Any], design_metrics: Dict[str, Any],
                    class_names: Sequence[str], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    colors = {"Random": "#4477AA", "Design-2": "#EE6677"}
    metrics_by_name = {"Random": random_metrics, "Design-2": design_metrics}

    for name, metrics in metrics_by_name.items():
        values = np.sort(metrics["point_counts"])
        percentile = np.arange(1, len(values) + 1) / len(values)
        axes[0, 0].plot(values, percentile, linewidth=2.4,
                        color=colors[name], label=name)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Points in crop (log scale)")
    axes[0, 0].set_ylabel("Fraction of bank")
    axes[0, 0].set_title("Crop-support distribution")
    axes[0, 0].legend(frameon=False)
    axes[0, 0].grid(alpha=.2)

    thresholds = [20, 50, 100, 500]
    x = np.arange(len(thresholds))
    width = .35
    for offset, name in zip((-.5, .5), metrics_by_name):
        values = [metrics_by_name[name]["sparse"][threshold]
                  for threshold in thresholds]
        bars = axes[0, 1].bar(x + offset * width, values, width,
                              label=name, color=colors[name])
        axes[0, 1].bar_label(bars, fontsize=9, padding=2)
    axes[0, 1].set_xticks(x, [f"< {threshold}" for threshold in thresholds])
    axes[0, 1].set_ylabel("Crops")
    axes[0, 1].set_title("Sparse crops (lower is better)")
    axes[0, 1].legend(frameon=False)
    axes[0, 1].grid(axis="y", alpha=.2)

    categories = ["Distinct source\nscenes", "Duplicate source\nassignments"]
    values_random = [random_metrics["distinct_sources"],
                     random_metrics["duplicate_assignments"]]
    values_design = [design_metrics["distinct_sources"],
                     design_metrics["duplicate_assignments"]]
    bars_r = axes[1, 0].bar(np.arange(2) - width / 2, values_random, width,
                            color=colors["Random"], label="Random")
    bars_d = axes[1, 0].bar(np.arange(2) + width / 2, values_design, width,
                            color=colors["Design-2"], label="Design-2")
    axes[1, 0].bar_label(bars_r, padding=2)
    axes[1, 0].bar_label(bars_d, padding=2)
    axes[1, 0].set_xticks(np.arange(2), categories)
    axes[1, 0].set_title("Source-scene diversity")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].grid(axis="y", alpha=.2)

    class_ids = sorted(set(random_metrics["per_class_median"]) &
                       set(design_metrics["per_class_median"]))
    random_medians = np.asarray([
        random_metrics["per_class_median"][class_id] for class_id in class_ids])
    design_medians = np.asarray([
        design_metrics["per_class_median"][class_id] for class_id in class_ids])
    axes[1, 1].scatter(random_medians, design_medians, color="#AA3377", s=32)
    upper = max(float(random_medians.max()), float(design_medians.max())) * 1.06
    axes[1, 1].plot([1, upper], [1, upper], linestyle="--", color="0.5")
    ratios = np.log((design_medians + 1) / (random_medians + 1))
    for index in np.argsort(np.abs(ratios))[-6:]:
        class_id = class_ids[index]
        axes[1, 1].annotate(class_names[class_id],
                            (random_medians[index], design_medians[index]),
                            xytext=(4, 3), textcoords="offset points", fontsize=8)
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlim(10, upper)
    axes[1, 1].set_ylim(10, upper)
    axes[1, 1].set_xlabel("Random median points/crop")
    axes[1, 1].set_ylabel("Design-2 median points/crop")
    axes[1, 1].set_title("Per-class crop support (diagonal = unchanged)")
    axes[1, 1].grid(alpha=.2)

    fig.suptitle(
        "What did Design-2 change in the object bank?\n"
        "More source diversity and fewer extreme sparse crops, but lower overall support",
        fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("random_bank", type=Path)
    parser.add_argument("design2_bank", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, default=Path(
        "configs/_base_/class_mappings/sunrgbd_40class_mapping.py"))
    args = parser.parse_args()
    random_metrics = bank_metrics(load_bank(args.random_bank))
    design_metrics = bank_metrics(load_bank(args.design2_bank))
    class_names = load_class_names(args.mapping)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_comparison(random_metrics, design_metrics, class_names,
                    args.output_dir / "random_vs_design2_bank.png")
    summary = comparison_summary(random_metrics, design_metrics)
    (args.output_dir / "random_vs_design2_bank.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
