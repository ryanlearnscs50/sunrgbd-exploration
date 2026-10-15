#!/usr/bin/env python3
"""Create a summary figure for the object-memory experiments."""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


def plot_progress(output: Path) -> None:
    stages = np.asarray([2, 3, 4, 5])
    trajectories = {
        "No memory": ([.1704, .0982, .0869, .0314], "#999999", "--"),
        "Object replay (preferred)": ([.1957, .1344, .1257, .0808], "#D81B60", "-"),
        "Scene replay (matched policy)": ([.3492, .2676, .2369, .1929], "#1B9E77", "-"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for label, (values, color, linestyle) in trajectories.items():
        axes[0].plot(stages, values, marker="o", linewidth=2.5,
                     linestyle=linestyle, color=color, label=label)
    axes[0].set_xticks(stages)
    axes[0].set_xlabel("Incremental stage")
    axes[0].set_ylabel("Seen-class mAP@0.25")
    axes[0].set_title("Replay prevents forgetting, but object crops trail scenes")
    axes[0].grid(alpha=.2)
    axes[0].legend(frameon=False)

    labels = ["Random object\n(matched stage 1)", "Object Design-2\n(no review)",
              "Object Design-2\n+ reviewing"]
    values = [.0931, .0667, .0632]
    colors = ["#4477AA", "#CCBB44", "#EE6677"]
    bars = axes[1].bar(np.arange(3), values, color=colors, width=.68)
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylim(0, .11)
    axes[1].set_ylabel("Final mAP@0.25")
    axes[1].set_title("LDMR-style selection/review did not improve crop replay")
    axes[1].grid(axis="y", alpha=.2)
    for bar, value in zip(bars, values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + .002,
                     f"{value:.4f}", ha="center", fontweight="bold")
    axes[1].annotate("−0.0299", xy=(2, .0632), xytext=(1.78, .092),
                     arrowprops={"arrowstyle": "->", "color": "#444444"},
                     ha="center", color="#444444")
    fig.suptitle("Object-level memory bank — current experimental conclusion",
                 fontsize=16, fontweight="bold")
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot_progress(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
