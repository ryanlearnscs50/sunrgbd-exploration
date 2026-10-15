#!/usr/bin/env python3
"""Check SUNRGBD 40-class GT labels in an info PKL split.

This script audits class coverage from `annos['class']` in SUNRGBD info files:
  - data/sunrgbd/sunrgbd_infos_train_40class.pkl
  - data/sunrgbd/sunrgbd_infos_val_40class.pkl

It reports per-class:
  - instance count
  - scene count (number of scenes where class appears at least once)

Example:
  ./venv/bin/python tools/analysis_tools/check_sunrgbd40_gt_labels.py \
    --ann-file data/sunrgbd/sunrgbd_infos_val_40class.pkl \
    --focus-classes drawer night_stand bookshelf whiteboard ottoman \
    --audit-name-index --strict --expect-full-range
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np


def _load_infos(path: Path):
    """Load annotation infos via mmcv when available, else pickle."""
    try:
        import mmcv  # type: ignore
        return mmcv.load(str(path))
    except Exception:
        with path.open("rb") as f:
            return pickle.load(f)


def _load_class_names(mapping_path: Path) -> List[str]:
    spec = importlib.util.spec_from_file_location("sunrgbd_mapping", str(mapping_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load mapping file: {mapping_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]

    direct_names = getattr(module, "SUNRGBD_40_RAW_TOP40_CLASSES", None)
    if isinstance(direct_names, list) and len(direct_names) == 40:
        return [str(x) for x in direct_names]

    get_stage_definitions = getattr(module, "get_stage_definitions", None)
    if callable(get_stage_definitions):
        stage_defs = get_stage_definitions(stage_setting="sunrgbd40_s5_freqorder")
        names: List[str] = []
        for sd in stage_defs:
            names.extend([str(x) for x in sd.get("class_names", [])])
        if len(names) == 40:
            return names

    raise RuntimeError(
        "Could not resolve SUNRGBD 40-class names from mapping file: "
        f"{mapping_path}"
    )


def _iter_labels_from_infos(infos: Sequence[dict]) -> Iterable[np.ndarray]:
    for info in infos:
        if not isinstance(info, dict):
            continue
        annos = info.get("annos", {})
        if not isinstance(annos, dict):
            continue
        labels = np.asarray(annos.get("class", []), dtype=np.int64).reshape(-1)
        yield labels


def _iter_name_label_pairs_from_infos(
    infos: Sequence[dict],
) -> Iterable[tuple[np.ndarray, np.ndarray]]:
    for info in infos:
        if not isinstance(info, dict):
            continue
        annos = info.get("annos", {})
        if not isinstance(annos, dict):
            continue
        names = np.asarray(annos.get("name", []), dtype=object).reshape(-1)
        labels = np.asarray(annos.get("class", []), dtype=np.int64).reshape(-1)
        if names.size == 0 or labels.size == 0:
            continue
        n = min(int(names.size), int(labels.size))
        if n <= 0:
            continue
        yield names[:n], labels[:n]


def _format_row(idx: int, name: str, inst: int, scenes: int) -> str:
    return f"{idx:>3}  {name:<16}  {inst:>8}  {scenes:>7}"


def _infer_split_from_ann_file(ann_file: Path) -> Optional[str]:
    name = ann_file.name.lower()
    if "train" in name:
        return "train"
    if "val" in name or "test" in name:
        return "val"
    return None


def _load_reference_csv(csv_path: Path) -> Dict[str, Dict[str, int]]:
    """Load reference CSV in either header or no-header format.

    Supported formats:
    1) Header:
       class,instances_train,samples_train,instances_val,samples_val
    2) No header (Sheet7 historical):
       idx,rank,class,instances_train,samples_train,instances_val,samples_val
    """
    rows: Dict[str, Dict[str, int]] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        raw_rows = [r for r in reader if r]

    if not raw_rows:
        raise RuntimeError(f"CSV is empty: {csv_path}")

    first = [x.strip().lower() for x in raw_rows[0]]
    has_header = "class" in first

    if has_header:
        with csv_path.open(newline="", encoding="utf-8") as f:
            dict_reader = csv.DictReader(f)
            for row in dict_reader:
                if row is None:
                    continue
                cls = str(row.get("class", "")).strip().lower()
                if not cls:
                    continue
                rows[cls] = dict(
                    instances_train=int(float(row.get("instances_train", 0) or 0)),
                    samples_train=int(float(row.get("samples_train", 0) or 0)),
                    instances_val=int(float(row.get("instances_val", 0) or 0)),
                    samples_val=int(float(row.get("samples_val", 0) or 0)),
                )
        return rows

    for row in raw_rows:
        if len(row) < 7:
            continue
        cls = str(row[2]).strip().lower()
        if not cls:
            continue
        rows[cls] = dict(
            instances_train=int(float(row[3])),
            samples_train=int(float(row[4])),
            instances_val=int(float(row[5])),
            samples_val=int(float(row[6])),
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ann-file",
        type=Path,
        default=Path("data/sunrgbd/sunrgbd_infos_val_40class.pkl"),
        help="SUNRGBD info pkl path to inspect (val is commonly used as test split).",
    )
    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=Path("configs/_base_/class_mappings/sunrgbd_40class_mapping.py"),
        help="Mapping file providing SUNRGBD 40-class ordered names.",
    )
    parser.add_argument(
        "--focus-classes",
        nargs="*",
        default=[],
        help="Optional class names to print explicitly (space-separated).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to save full stats as JSON.",
    )
    parser.add_argument(
        "--audit-name-index",
        action="store_true",
        help=(
            "Also audit consistency between annos['name'] and annos['class'] "
            "w.r.t. mapping-file class order."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero status if any selected contract checks fail.",
    )
    parser.add_argument(
        "--expect-full-range",
        action="store_true",
        help="Require all 40 classes to have at least one GT instance.",
    )
    parser.add_argument(
        "--csv-ref",
        type=Path,
        default=None,
        help=(
            "Optional reference CSV for count alignment check "
            "(e.g., TR3D3DCILExperimentResults-Sheet7.csv)."
        ),
    )
    parser.add_argument(
        "--csv-split",
        type=str,
        choices=["train", "val"],
        default=None,
        help=(
            "Split column to compare against CSV reference. "
            "Defaults to inferred split from ann-file name."
        ),
    )
    parser.add_argument(
        "--fail-on-csv-mismatch",
        action="store_true",
        help="When --strict is set, treat CSV alignment mismatches as failures.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {args.ann_file}")
    if not args.mapping_file.exists():
        raise FileNotFoundError(f"Mapping file not found: {args.mapping_file}")
    if args.csv_ref is not None and not args.csv_ref.exists():
        raise FileNotFoundError(f"CSV reference not found: {args.csv_ref}")

    class_names = _load_class_names(args.mapping_file)
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    infos = _load_infos(args.ann_file)
    if not isinstance(infos, list):
        raise RuntimeError(
            f"Unexpected annotation payload type: {type(infos)}. Expected list."
        )

    instance_count: Counter[int] = Counter()
    scene_count: Counter[int] = Counter()
    out_of_range_labels: Counter[int] = Counter()

    total_scenes = len(infos)
    scenes_with_gt = 0
    total_boxes = 0

    for labels in _iter_labels_from_infos(infos):
        if labels.size == 0:
            continue
        scenes_with_gt += 1
        total_boxes += int(labels.size)

        uniq = set()
        for raw in labels.tolist():
            idx = int(raw)
            if 0 <= idx < len(class_names):
                instance_count[idx] += 1
                uniq.add(idx)
            else:
                out_of_range_labels[idx] += 1
        for idx in uniq:
            scene_count[idx] += 1

    nonzero_classes = [i for i in range(len(class_names)) if instance_count[i] > 0]
    zero_classes = [i for i in range(len(class_names)) if instance_count[i] == 0]

    print(f"Annotation file: {args.ann_file}")
    print(f"Total scenes: {total_scenes}")
    print(f"Scenes with >=1 GT box: {scenes_with_gt}")
    print(f"Total GT boxes: {total_boxes}")
    print(f"Classes with GT instances: {len(nonzero_classes)}/40")
    print()

    print("Idx  Class               Instances   Scenes")
    print("---  ----------------  ---------  -------")
    for idx, name in enumerate(class_names):
        print(
            _format_row(
                idx=idx,
                name=name,
                inst=int(instance_count[idx]),
                scenes=int(scene_count[idx]),
            )
        )

    print()
    if zero_classes:
        print("Classes with zero GT instances in this split:")
        print("  " + ", ".join(class_names[i] for i in zero_classes))
    else:
        print("All 40 classes have GT instances in this split.")

    if out_of_range_labels:
        print()
        print("Out-of-range labels detected in annos['class']:")
        for label, cnt in sorted(out_of_range_labels.items()):
            print(f"  label={label}: {cnt} instances")

    if args.focus_classes:
        print()
        print("Focus classes:")
        for name in args.focus_classes:
            if name not in name_to_idx:
                print(f"  {name}: NOT_IN_MAPPING")
                continue
            idx = name_to_idx[name]
            print(
                f"  {name} (idx={idx}): "
                f"instances={int(instance_count[idx])}, scenes={int(scene_count[idx])}"
            )

    mismatch_rows = []
    mismatch_count = 0
    valid_pair_count = 0
    if args.audit_name_index:
        name_label_count = Counter()
        for names, labels in _iter_name_label_pairs_from_infos(infos):
            for raw_name, raw_label in zip(names.tolist(), labels.tolist()):
                name = str(raw_name).strip().lower()
                label = int(raw_label)
                name_label_count[(name, label)] += 1
                if not (0 <= label < len(class_names)):
                    continue
                valid_pair_count += 1
                mapped_name = class_names[label]
                if name != mapped_name:
                    mismatch_count += 1
                    mismatch_rows.append((name, label, mapped_name))

        print()
        print("Name-index consistency audit:")
        print(
            f"  Valid label pairs checked (0..39): {valid_pair_count}, "
            f"mismatches: {mismatch_count}"
        )
        if mismatch_count > 0:
            pair_totals = Counter(mismatch_rows)
            print("  Top mismatched (name, label_idx -> mapped_name, count):")
            for (name, label, mapped), cnt in pair_totals.most_common(20):
                print(f"    {name:>14} , {label:>2} -> {mapped:<14} : {cnt}")

        if args.focus_classes:
            print("  Focus-class name label distribution:")
            for focus in args.focus_classes:
                rows = [
                    (label, cnt)
                    for (name, label), cnt in name_label_count.items()
                    if name == focus
                ]
                rows.sort(key=lambda x: (-x[1], x[0]))
                if not rows:
                    print(f"    {focus}: no name entries")
                    continue
                mapped_desc = ", ".join(
                    f"idx={label}({class_names[label] if 0 <= label < len(class_names) else 'OUT_OF_RANGE'}):{cnt}"
                    for label, cnt in rows
                )
                print(f"    {focus}: {mapped_desc}")

    csv_summary = None
    if args.csv_ref is not None:
        split = args.csv_split or _infer_split_from_ann_file(args.ann_file)
        if split not in {"train", "val"}:
            raise RuntimeError(
                "Could not infer CSV split from ann-file name. "
                "Please provide --csv-split {train,val}."
            )

        ref = _load_reference_csv(args.csv_ref)
        missing_in_csv = [n for n in class_names if n not in ref]
        extra_in_csv = sorted([n for n in ref.keys() if n not in set(class_names)])
        mismatches = []

        for idx, name in enumerate(class_names):
            if name not in ref:
                continue
            row = ref[name]
            exp_inst = int(row[f"instances_{split}"])
            exp_scene = int(row[f"samples_{split}"])
            got_inst = int(instance_count[idx])
            got_scene = int(scene_count[idx])
            if exp_inst != got_inst or exp_scene != got_scene:
                mismatches.append(
                    dict(
                        class_name=name,
                        expected_instances=exp_inst,
                        got_instances=got_inst,
                        expected_scenes=exp_scene,
                        got_scenes=got_scene,
                    )
                )

        print()
        print("CSV reference alignment:")
        print(f"  csv_ref: {args.csv_ref}")
        print(f"  split: {split}")
        print(f"  rows_in_ref: {len(ref)}")
        print(f"  missing_in_ref: {len(missing_in_csv)}")
        print(f"  extra_in_ref: {len(extra_in_csv)}")
        print(f"  mismatch_rows: {len(mismatches)}")
        if mismatches:
            print("  First mismatches:")
            for row in mismatches[:20]:
                print(
                    "    "
                    f"{row['class_name']}: "
                    f"inst {row['got_instances']} != {row['expected_instances']}, "
                    f"scenes {row['got_scenes']} != {row['expected_scenes']}"
                )

        csv_summary = dict(
            csv_ref=str(args.csv_ref),
            split=split,
            rows_in_ref=int(len(ref)),
            missing_in_ref=[str(x) for x in missing_in_csv],
            extra_in_ref=[str(x) for x in extra_in_csv],
            mismatch_rows=mismatches,
        )

    strict_errors = []
    if out_of_range_labels:
        strict_errors.append(
            f"out_of_range_labels={dict(sorted((int(k), int(v)) for k, v in out_of_range_labels.items()))}"
        )
    if args.audit_name_index and mismatch_count > 0:
        strict_errors.append(f"name_index_mismatch_count={int(mismatch_count)}")
    if args.expect_full_range and zero_classes:
        strict_errors.append(
            "missing_classes=" + ",".join(class_names[i] for i in zero_classes)
        )
    if args.fail_on_csv_mismatch and csv_summary is not None:
        if csv_summary["missing_in_ref"] or csv_summary["extra_in_ref"] or csv_summary["mismatch_rows"]:
            strict_errors.append("csv_alignment_failed")

    strict_failed = bool(strict_errors)
    if args.strict and strict_failed:
        print()
        print("STRICT CHECK FAILED:")
        for err in strict_errors:
            print(f"  - {err}")

    if args.json_out is not None:
        payload = {
            "ann_file": str(args.ann_file),
            "total_scenes": int(total_scenes),
            "scenes_with_gt": int(scenes_with_gt),
            "total_gt_boxes": int(total_boxes),
            "class_stats": [
                {
                    "idx": int(i),
                    "name": class_names[i],
                    "instances": int(instance_count[i]),
                    "scenes": int(scene_count[i]),
                }
                for i in range(len(class_names))
            ],
            "zero_gt_classes": [class_names[i] for i in zero_classes],
            "out_of_range_labels": {
                str(int(k)): int(v) for k, v in sorted(out_of_range_labels.items())
            },
            "strict": bool(args.strict),
            "strict_failed": bool(args.strict and strict_failed),
            "strict_errors": strict_errors,
        }
        if args.audit_name_index:
            pair_totals = Counter(mismatch_rows)
            payload["name_index_mismatch_count"] = int(mismatch_count)
            payload["name_index_mismatch_top"] = [
                {
                    "name": name,
                    "label_idx": int(label),
                    "mapped_name": mapped,
                    "count": int(cnt),
                }
                for (name, label, mapped), cnt in pair_totals.most_common(100)
            ]
            payload["name_index_valid_pair_count"] = int(valid_pair_count)
        if csv_summary is not None:
            payload["csv_alignment"] = csv_summary
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print()
        print(f"Wrote JSON: {args.json_out}")

    if args.strict and strict_failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
