#!/usr/bin/env python3
"""Estimate a strict global confidence threshold for SUN RGB-D pseudo labels.

This helper runs a small threshold sweep by:
1) generating pseudo labels (Stage t uses Stage t-1 checkpoint) on a fixed,
   seeded subset of *train* scenes for Stage t,
2) validating pseudo-vs-GT IoU match metrics on previous-seen classes.

It is intentionally standalone and is not called by training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from mmdet3d.utils.pregenerate_pseudo_labels_sunrgbd import (
    pregenerate_sunrgbd_pseudo_labels_for_stage,
)
from tools.pseudo_labels.validate_sunrgbd_pseudo_labels import (
    validate_sunrgbd_pseudo_labels_from_file,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-id", type=int, required=True, help="Current stage id (>=2)")
    parser.add_argument("--checkpoint", required=True, help="Previous-stage checkpoint path")
    parser.add_argument("--train-ann-file", required=True, help="SUNRGBD train infos PKL path")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.30, 0.40, 0.50, 0.60],
        help="Confidence thresholds to sweep",
    )
    parser.add_argument("--nms-iou-thr", type=float, default=0.3)
    parser.add_argument("--max-pseudo-per-scene", type=int, default=100)
    parser.add_argument("--max-scenes", type=int, default=200, help="Number of train scenes to sample")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default="./pseudo_threshold_sweep_sunrgbd",
        help="Directory to write generated pseudo label files",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    stage_id = int(args.stage_id)
    if stage_id < 2:
        raise ValueError("--stage-id must be >=2 for pseudo labeling.")

    # Load SUNRGBD stage definitions (source-of-truth).
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2] / "configs" / "_base_" / "class_mappings"))
    from sunrgbd_40class_mapping import get_stage_definitions  # type: ignore

    stage_definitions = get_stage_definitions()
    stage_definition = None
    for sd in stage_definitions:
        if int(sd.get("stage_id", 0) or 0) == int(stage_id):
            stage_definition = dict(sd)
            break
    if stage_definition is None:
        raise ValueError(f"Unknown stage_id={stage_id}; expected one of {[s['stage_id'] for s in stage_definitions]}")
    stage_definition["filter_empty_gt"] = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds: List[float] = [float(x) for x in args.thresholds]
    thresholds = sorted({max(0.0, min(1.0, t)) for t in thresholds})

    print(f"[SUNRGBD threshold sweep] stage={stage_id}")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  train_ann_file={args.train_ann_file}")
    print(f"  max_scenes={args.max_scenes}, seed={args.seed}")
    print("")

    for thr in thresholds:
        suffix = f"thr_sweep_conf{int(thr * 100):02d}_seed{int(args.seed)}_n{int(args.max_scenes)}"
        pseudo_file = pregenerate_sunrgbd_pseudo_labels_for_stage(
            stage_id=stage_id,
            checkpoint_path=args.checkpoint,
            train_ann_file=args.train_ann_file,
            stage_definition=stage_definition,
            all_stage_definitions=stage_definitions,
            confidence_threshold=float(thr),
            nms_iou_thr=float(args.nms_iou_thr),
            max_pseudo_per_scene=int(args.max_pseudo_per_scene),
            max_scenes=int(args.max_scenes),
            seed=int(args.seed),
            output_dir=str(output_dir),
            config_suffix=suffix,
        )

        metrics = validate_sunrgbd_pseudo_labels_from_file(
            pseudo_file=str(pseudo_file),
            ann_file=str(args.train_ann_file),
            stage_id=stage_id,
            iou_thrs=(0.25, 0.5),
            max_scenes=int(args.max_scenes),
            seed=int(args.seed),
            verbose=False,
        )

        m25 = metrics.get(0.25, {})
        m50 = metrics.get(0.5, {})
        print(
            f"thr={thr:.2f} | "
            f"IoU@0.25 hit={m25.get('pseudo_hit_rate', 0.0)*100:.2f}% "
            f"recall={m25.get('gt_recall', 0.0)*100:.2f}% | "
            f"IoU@0.50 hit={m50.get('pseudo_hit_rate', 0.0)*100:.2f}% "
            f"recall={m50.get('gt_recall', 0.0)*100:.2f}% | "
            f"file={pseudo_file}"
        )


if __name__ == "__main__":
    main()
