"""
incremental_splits_v2.py
------------------------
DATA-DRIVEN replacement for incremental_splits.py.

THE MISTAKE IN V1:
  Stages were defined by manually assigning object classes to room types:
    Stage 1 – Bedroom:  bed, pillow, lamp, curtain, mirror, picture, sofa, light
    Stage 2 – Office:   desk, monitor, keyboard, chair, table, shelf, book, paper
    Stage 3 – Kitchen:  counter, cabinet, bottle, cup, box, bag, window, door
  The class-to-room assignments were hand-coded, motivated post-hoc by Jaccard
  co-occurrence values.  The dataset's own room labels were never consulted.

THE FIX (mentor's direction):
  Every scene in SUN RGB-D contains a scene.txt file with an official room-type
  label (bedroom, office, classroom, kitchen, etc.).  Work from these labels first:
    1. Select stage scene types directly from the dataset's ground-truth labels.
    2. Compute LIFT for every (class, scene_type) pair:
         lift = freq(class | scene_type) / freq(class overall)
       Lift > 1 means the class appears MORE in this scene type than average.
    3. Assign each class to the scene type where its lift is highest.
       Use a balanced greedy algorithm so each stage gets equal class counts.
    4. Apply the same strict filter as v1, but condition (a) is now:
         old: scene contains ≥1 Stage-N class       (class-presence check)
         new: scene.txt label == Stage-N scene type  (ground-truth check)
       Condition (b) — no future-stage classes — is unchanged.

WHY LIFT (not raw frequency)?
  "Chair" appears in 57% of all scenes AND in ~70% of office scenes.  Its
  office frequency looks high, but it is barely above its baseline — it does
  not characterise offices.  "Monitor" appears in only 9% of all scenes but
  ~60% of offices: lift ≈ 6.7.  Lift normalises by baseline frequency so that
  truly characteristic classes score high, not just common ones.

WHY CHANGE CONDITION (a)?
  Using class presence for condition (a) mixes two things: what a scene IS and
  what it CONTAINS.  A furniture store can contain a bed without being a bedroom.
  Ground-truth scene type labels are the honest definition of scene identity.

STAGE SCENE TYPES CHOSEN:
  Selected by count in the working dataset (≥250 scenes) and semantic clarity
  (excluded "idk", "rest_space", "furniture_store" as ambiguous or retail-specific).
    3-stage: bedroom (642), office (893), classroom (858)
    6-stage: bedroom, living_room, office, classroom, library, dining_area

Outputs:
  scene_labels.csv               — scene path → scene_type for all 10k+ scenes
  scene_type_class_lift.csv      — lift table for every (class, scene_type) pair
  class_stage_assignments_v2.csv — which class assigned to which stage and why
  incremental_splits_v2_3stage.csv
  incremental_splits_v2_6stage.csv
  incremental_splits_v2_all.csv
"""

import json
import pathlib
import re
import sys
from itertools import permutations as _permutations

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 1. Paths and constants
# ---------------------------------------------------------------------------
DATASET_ROOT = pathlib.Path(r"C:\Users\ryana\Downloads\SUNRGBD\SUNRGBD")
OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")
SPLITS_DIR   = pathlib.Path(r"C:\sunrgbd-exploration\05_incremental_splits")
FREQ_CSV     = pathlib.Path(r"C:\sunrgbd-exploration\01_class_frequencies\class_frequencies.csv")

EXCLUDE = {"wall", "floor", "ceiling"}
TOP_N   = 24   # kept at 24 for arithmetic divisibility (same as v1)

UNVIABLE_THRESHOLD = 50

# Stage scene types — ground-truth labels from dataset scene.txt files.
# 3-stage: natural home → work → classroom progression.
# 6-stage: home split into private (bedroom) and social (living_room), work
#          split into desk-focused (office) and knowledge-focused (classroom +
#          library), plus eating environment (dining_area).
STAGE_SCENE_TYPES_3 = ["bedroom", "office", "classroom"]
STAGE_SCENE_TYPES_6 = ["bedroom", "living_room", "office", "classroom", "library", "dining_area"]

CLASSES_PER_STAGE_3 = TOP_N // len(STAGE_SCENE_TYPES_3)   # 8
CLASSES_PER_STAGE_6 = TOP_N // len(STAGE_SCENE_TYPES_6)   # 4

# ---------------------------------------------------------------------------
# 2. Synonym normalisation  (identical to v1 — same contract)
# ---------------------------------------------------------------------------
_SYNONYMS: dict[str, str] = {}

def _add(canonical: str, *variants: str) -> None:
    _SYNONYMS[canonical] = canonical
    for v in variants:
        _SYNONYMS[v] = canonical

_add("sofa",    "couch", "sofachair", "sofas", "couches")
_add("chair",   "armchair", "chairs")
_add("shelf",   "shelves", "bookshelf", "bookshelves", "shelve", "bookshelve")
_add("counter", "countertop", "countertops")
_add("desk",    "desktable")
_add("cabinet", "cupboard", "cupboards", "cabinets", "filingcabinet",
     "filecabinet", "storagecabinet")
_add("monitor", "computermonitor", "pcmonitor", "monitors")
_add("pillow",  "pillows", "cushion", "cushions", "throwpillow", "throwpillows")
_add("curtain", "curtains", "windowcurtain")
_add("book",    "books")
_add("lamp",    "tablelamp", "desklamp", "floorlamp", "nightlamp", "walllamp")
_add("window",  "windows")
_add("blinds",  "blind", "windowblinds", "windowblind")
_add("bottle",  "bottles")
_add("bag",     "bags")
_add("mirror",  "mirrors")
_add("door",    "doors")
_add("table",   "tables")
_add("bed",     "beds")
_add("box",     "boxes")

_STRIP_SUFFIX = re.compile(r"^(.+?)\d+$")

def normalise(raw: str) -> str:
    name = raw.strip().lower()
    if name in _SYNONYMS:
        return _SYNONYMS[name]
    m = _STRIP_SUFFIX.match(name)
    if m:
        base = m.group(1)
        return _SYNONYMS.get(base, base)
    return name

# ---------------------------------------------------------------------------
# 3. Load class list
# ---------------------------------------------------------------------------
freq_df = pd.read_csv(FREQ_CSV)
selected_classes: list[str] = (
    freq_df.loc[~freq_df["class"].isin(EXCLUDE), "class"]
    .head(TOP_N)
    .tolist()
)

print("=" * 72)
print(f"TOP {TOP_N} CLASSES  (wall/floor/ceiling excluded; 'sign' dropped for divisibility)")
print("=" * 72)
for i, c in enumerate(selected_classes, 1):
    row = freq_df.loc[freq_df["class"] == c].iloc[0]
    print(f"  {i:2d}. {c:<12s}  {row['scene_frequency']:,} scenes  ({row['pct_of_scenes']:.1f}%)")

# ---------------------------------------------------------------------------
# 4. Build presence matrix + scene_type column in a single annotation scan
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: read scene.txt alongside each annotation file in the
# same pass rather than building a separate lookup dict first.
# - Avoids the 35 "ambiguous folder name" problem found in exploration
#   (same folder name, different sensor path, different label).  Here we use
#   the FULL PATH to read scene.txt, so each scene has exactly one label.
# - Single pass over 10k files is faster than two sequential passes.

selected_set = set(selected_classes)
records      = []
parse_errors = 0
missing_labels = 0

print("\nScanning annotation files (reads scene.txt alongside each annotation)...")
annotation_files = sorted(DATASET_ROOT.rglob("annotation2Dfinal/index.json"))

for ann_path in annotation_files:
    scene_root = ann_path.parent.parent

    scene_txt = scene_root / "scene.txt"
    if scene_txt.exists():
        scene_type = scene_txt.read_text(encoding="utf-8", errors="ignore").strip()
    else:
        scene_type = "unknown"
        missing_labels += 1

    try:
        with ann_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        parse_errors += 1
        continue

    raw_names = {
        normalise(obj["name"])
        for obj in (data.get("objects") or [])
        if obj and isinstance(obj.get("name"), str) and obj["name"].strip()
    }
    present = raw_names & selected_set

    row: dict = {"scene": str(scene_root), "scene_type": scene_type}
    for cls in selected_classes:
        row[cls] = 1 if cls in present else 0
    records.append(row)

presence_df = pd.DataFrame(records)
print(f"  Scenes processed  : {len(presence_df):,}")
print(f"  Parse errors      : {parse_errors:,}")
print(f"  Missing scene.txt : {missing_labels:,}")
print(f"  Unique scene types: {presence_df['scene_type'].nunique()}")

# Distribution of scene types for selected stages
all_stage_types = sorted(set(STAGE_SCENE_TYPES_3 + STAGE_SCENE_TYPES_6))
print("\n  Scene type counts for selected stages:")
for stype in sorted(all_stage_types):
    n = int((presence_df["scene_type"] == stype).sum())
    print(f"    {stype:<18}: {n:,}")

# Save scene labels — useful for every downstream script
scene_labels_path = SPLITS_DIR / "scene_labels.csv"
presence_df[["scene", "scene_type"]].to_csv(scene_labels_path, index=False)
print(f"\n  Saved: scene_labels.csv ({len(presence_df):,} rows)")

# ---------------------------------------------------------------------------
# 5. Compute lift matrix
# ---------------------------------------------------------------------------
# lift(class, scene_type) = P(class | scene_type) / P(class)
#
# Example that motivates lift over raw frequency:
#   "chair" overall: 57% of scenes.  In offices: ~65%.  Lift ≈ 1.14 — barely
#   distinctive of offices.  "monitor" overall: 9%.  In offices: ~60%.
#   Lift ≈ 6.7 — highly distinctive.  Lift reveals real signal; raw frequency
#   just reflects class popularity.
#
# PATTERN: vectorised pandas groupby mean — no per-class loops.

overall_freq = presence_df[selected_classes].mean()   # P(class) over all scenes

lift_records = []
for stype in all_stage_types:
    mask = presence_df["scene_type"] == stype
    n = int(mask.sum())
    if n == 0:
        continue
    freq_in_type = presence_df.loc[mask, selected_classes].mean()   # P(class | type)
    for cls in selected_classes:
        p_cls   = float(overall_freq[cls])
        p_given = float(freq_in_type[cls])
        lift_val = p_given / p_cls if p_cls > 0 else 0.0
        lift_records.append({
            "class":        cls,
            "scene_type":   stype,
            "n_scenes":     n,
            "freq_in_type": round(p_given, 4),
            "overall_freq": round(p_cls,   4),
            "lift":         round(lift_val, 4),
        })

lift_df = pd.DataFrame(lift_records)
lift_df.to_csv(SPLITS_DIR / "scene_type_class_lift.csv", index=False)
print(f"\n  Saved: scene_type_class_lift.csv")

# ---------------------------------------------------------------------------
# 6. Data-driven class-to-stage assignment  (Balanced Greedy by Lift)
# ---------------------------------------------------------------------------
# ALGORITHM:
#   1. For each class, compute its "decisiveness" = (best lift) - (second-best lift).
#      A high decisiveness means the class clearly belongs to one scene type.
#   2. Sort classes from MOST decisive to LEAST decisive.
#   3. Assign each class to its highest-lift stage, subject to capacity.
#      If the top-lift stage is full, fall back to the next-best stage.
#
# WHY sort by decisiveness?
#   Processing clearly-assigned classes first (e.g. "bed" → bedroom, lift 3.2)
#   "fills" each stage with its natural classes.  Ambiguous classes (e.g. "door",
#   low lift everywhere) are assigned last to whichever stage still has room,
#   rather than "stealing" a slot from a class that truly belongs there.
#
# PATTERN: greedy bipartite matching with a custom priority queue.
#   O(n_classes × n_stages) — trivially fast for 24 classes and ≤6 stages.

def assign_classes_balanced(selected_classes, lift_df, stage_scene_types, classes_per_stage):
    """
    Returns:
      stage_classes: {scene_type: [list of classes]}
      class_stage:   {class: scene_type}
    """
    lookup = {
        (r["class"], r["scene_type"]): r["lift"]
        for _, r in lift_df[lift_df["scene_type"].isin(stage_scene_types)].iterrows()
    }

    capacity = {s: classes_per_stage for s in stage_scene_types}

    def decisiveness(cls):
        lifts = sorted([lookup.get((cls, s), 0.0) for s in stage_scene_types], reverse=True)
        return (lifts[0] - lifts[1]) if len(lifts) >= 2 else (lifts[0] if lifts else 0.0)

    ordered = sorted(selected_classes, key=decisiveness, reverse=True)

    class_stage  = {}
    stage_classes = {s: [] for s in stage_scene_types}

    for cls in ordered:
        ranked_stages = sorted(stage_scene_types,
                               key=lambda s: -lookup.get((cls, s), 0.0))
        for s in ranked_stages:
            if capacity[s] > 0:
                class_stage[cls] = s
                stage_classes[s].append(cls)
                capacity[s] -= 1
                break

    return stage_classes, class_stage


stage_classes_3, class_stage_3 = assign_classes_balanced(
    selected_classes, lift_df, STAGE_SCENE_TYPES_3, CLASSES_PER_STAGE_3
)
stage_classes_6, class_stage_6 = assign_classes_balanced(
    selected_classes, lift_df, STAGE_SCENE_TYPES_6, CLASSES_PER_STAGE_6
)

# ---------------------------------------------------------------------------
# 7. Print assignment tables
# ---------------------------------------------------------------------------
def print_assignment_table(selected_classes, lift_df, stage_scene_types, class_stage):
    """Print lift values and assignment for each class — the key output."""
    pivot = (
        lift_df[lift_df["scene_type"].isin(stage_scene_types)]
        .pivot(index="class", columns="scene_type", values="lift")
    )
    col_w = 10
    header = f"  {'Class':<13}" + "".join(f"{s:>{col_w}}" for s in stage_scene_types) + "  Assigned to"
    print(header)
    print("  " + "-" * (13 + col_w * len(stage_scene_types) + 14))
    for cls in selected_classes:
        assigned = class_stage.get(cls, "?")
        lifts = []
        for s in stage_scene_types:
            val = pivot.at[cls, s] if (cls in pivot.index and s in pivot.columns) else 0.0
            marker = "*" if s == assigned else " "
            lifts.append(f"{marker}{val:.3f}   ")
        print(f"  {cls:<13}" + "".join(f"{l:>{col_w}}" for l in lifts) + f"  {assigned}")
    print()
    print("  * = assigned stage  (highest lift among stages with remaining capacity)")


print("\n" + "=" * 72)
print("3-STAGE DATA-DRIVEN CLASS ASSIGNMENT")
print("  bedroom / office / classroom")
print("=" * 72)
print_assignment_table(selected_classes, lift_df, STAGE_SCENE_TYPES_3, class_stage_3)
print("  Stage summary:")
for stype, classes in stage_classes_3.items():
    print(f"    {stype:<18}: {', '.join(classes)}")

print("\n" + "=" * 72)
print("6-STAGE DATA-DRIVEN CLASS ASSIGNMENT")
print("  bedroom / living_room / office / classroom / library / dining_area")
print("=" * 72)
print_assignment_table(selected_classes, lift_df, STAGE_SCENE_TYPES_6, class_stage_6)
print("  Stage summary:")
for stype, classes in stage_classes_6.items():
    print(f"    {stype:<18}: {', '.join(classes)}")

# Save assignment table
assign_rows = []
for cls in selected_classes:
    r = {
        "class":       cls,
        "assigned_3stage": class_stage_3.get(cls, ""),
        "assigned_6stage": class_stage_6.get(cls, ""),
    }
    for stype in all_stage_types:
        sub = lift_df[(lift_df["class"] == cls) & (lift_df["scene_type"] == stype)]
        r[f"lift_{stype}"] = round(float(sub["lift"].values[0]), 4) if len(sub) else 0.0
    assign_rows.append(r)

pd.DataFrame(assign_rows).to_csv(SPLITS_DIR / "class_stage_assignments_v2.csv", index=False)
print(f"\n  Saved: class_stage_assignments_v2.csv")

# ---------------------------------------------------------------------------
# 8. Apply strict filter
# ---------------------------------------------------------------------------
# Condition (a) — NEW: scene_type label matches the stage's scene type.
# Condition (b) — UNCHANGED: no future-stage classes present in the scene.
#
# WHY change condition (a)?
#   In v1, condition (a) tested whether the scene CONTAINS a Stage-N class.
#   This is indirect and fragile: a furniture store contains beds but is not a
#   bedroom.  The ground-truth label is the authoritative definition of scene
#   identity and is used directly here.

def apply_strict_filter(presence_df, split_def):
    """
    split_def: list of (stage_label, scene_type, [classes])

    Returns:
      stats:    list of dicts with survival statistics per stage
      csv_rows: list of dicts for surviving scenes
    """
    n_stages = len(split_def)
    stage_class_lists = [cls_list for _, _, cls_list in split_def]
    stats    = []
    csv_rows = []

    for i, (stage_label, stage_stype, stage_cls) in enumerate(split_def):
        # Condition (a): ground-truth scene type matches
        has_stage = presence_df["scene_type"] == stage_stype

        # Condition (b): no future-stage classes
        future_classes: set[str] = set()
        for j in range(i + 1, n_stages):
            future_classes |= set(stage_class_lists[j])

        if future_classes:
            no_future = presence_df[sorted(future_classes)].sum(axis=1) == 0
        else:
            no_future = pd.Series(True, index=presence_df.index)

        total    = int(has_stage.sum())
        mask     = has_stage & no_future
        survived = int(mask.sum())
        pct_lost = 100.0 * (1 - survived / total) if total > 0 else float("nan")

        stats.append({
            "stage":        stage_label,
            "scene_type":   stage_stype,
            "classes":      ", ".join(stage_cls),
            "future_cls":   ", ".join(sorted(future_classes)) if future_classes else "(none)",
            "total":        total,
            "survived":     survived,
            "pct_lost":     round(pct_lost, 1),
        })

        for _, row in presence_df[mask].iterrows():
            present_here = [c for c in selected_classes if row[c] == 1]
            csv_rows.append({
                "stage":           stage_label,
                "scene_type":      stage_stype,
                "scene":           row["scene"],
                "classes_present": "|".join(present_here),
            })

    return stats, csv_rows


def build_split_def(stage_scene_types, stage_classes_dict):
    result = []
    for i, stype in enumerate(stage_scene_types, 1):
        label = f"Stage {i} – {stype.replace('_', ' ').title()}"
        result.append((label, stype, stage_classes_dict[stype]))
    return result

split_def_3 = build_split_def(STAGE_SCENE_TYPES_3, stage_classes_3)
split_def_6 = build_split_def(STAGE_SCENE_TYPES_6, stage_classes_6)

stats_3, rows_3 = apply_strict_filter(presence_df, split_def_3)
stats_6, rows_6 = apply_strict_filter(presence_df, split_def_6)

# ---------------------------------------------------------------------------
# 9. Print results + comparison with v1
# ---------------------------------------------------------------------------

def print_filter_results(stats, split_name):
    print(f"\n{'=' * 72}")
    print(f"STRICT FILTER — {split_name}")
    print(f"{'=' * 72}")
    for s in stats:
        flag = "  *** UNVIABLE ***" if s["survived"] < UNVIABLE_THRESHOLD else ""
        print(f"\n  {s['stage']}")
        print(f"    Scene type (cond. a) : {s['scene_type']}  ({s['total']:,} scenes of this type)")
        print(f"    Classes learned      : {s['classes']}")
        print(f"    Forbidden (future)   : {s['future_cls']}")
        print(f"    Surviving strict     : {s['survived']:,}{flag}")
        print(f"    % lost               : {s['pct_lost']:.1f}%")

print_filter_results(stats_3, "3-STAGE (bedroom / office / classroom)")
print_filter_results(stats_6, "6-STAGE (bedroom / living_room / office / classroom / library / dining_area)")

# Summary comparison table
print(f"\n{'=' * 72}")
print("SUMMARY COMPARISON: v1 hand-coded  vs  v2 data-driven")
print(f"{'=' * 72}")

V1_SEM_A = [
    ("Stage 1 – Bedroom/Lounge",  4581, 442,  90.4),
    ("Stage 2 – Office/Study",    8144, 2727, 66.5),
    ("Stage 3 – Kitchen/Utility", 6808, 6808,  0.0),
]

print(f"\n  3-STAGE SPLIT")
print(f"  {'Stage':<32} {'V1 survive':>11} {'V2 survive':>11} {'V1 lost%':>9} {'V2 lost%':>9}")
print("  " + "-" * 75)

for (v1_stage, v1_total, v1_surv, v1_lost), s2 in zip(V1_SEM_A, stats_3):
    change = s2["survived"] - v1_surv
    arrow  = f"+{change}" if change >= 0 else str(change)
    print(f"  {v1_stage:<32} {v1_surv:>11,} {s2['survived']:>11,} "
          f"{v1_lost:>8.1f}% {s2['pct_lost']:>8.1f}%   ({arrow})")

print()
print("  NOTE: Stage 3 in v1 is 'Kitchen/Utility'; Stage 3 in v2 is 'Classroom'.")
print("  The scene type choice changes the comparison — see the data directly.")

# ---------------------------------------------------------------------------
# 10. Ordering sweep — find the stage sequence that keeps every stage viable
# ---------------------------------------------------------------------------
# MOTIVATION: condition (b) eliminates scenes that contain "future" classes.
# Classes that are common across ALL room types (chair 57%, table 46%, window,
# door) are catastrophic when assigned to late stages — they become forbidden
# in EVERY earlier stage, collapsing those stages to single-digit counts.
#
# SCORING CRITERION — maximise the MINIMUM surviving stage count.
# We do NOT optimise for total scenes; we optimise so that no individual
# stage collapses.  A split where stages survive [200, 200, 300] is far better
# than [5, 5, 1400] even though the second has more total scenes.
#
# The class-to-scene-type assignments are FIXED (determined by lift in
# section 6).  Only the stage NUMBER (position in the sequence) changes.
#
# PATTERN: itertools.permutations generates all n! orderings.
#   3-stage: 3! =   6 permutations — trivial
#   6-stage: 6! = 720 permutations — fast; apply_strict_filter is just
#            boolean pandas operations on a 10k-row dataframe.

def sweep_orderings(stage_scene_types, stage_classes_dict):
    """
    Try every permutation.  Return list sorted by:
      primary   — min(survived per stage), descending   [no stage collapses]
      tiebreak  — total survived, descending
    """
    rows = []
    for perm in _permutations(stage_scene_types):
        split_def  = build_split_def(list(perm), stage_classes_dict)
        stats, csv = apply_strict_filter(presence_df, split_def)
        min_surv   = min(s["survived"] for s in stats)
        total      = sum(s["survived"] for s in stats)
        rows.append({
            "ordering":       list(perm),
            "min_survived":   min_surv,
            "total_survived": total,
            "stats":          stats,
            "csv_rows":       csv,
        })
    rows.sort(key=lambda r: (-r["min_survived"], -r["total_survived"]))
    return rows


def print_best_ordering(best, default_order, split_name):
    """Print a concise summary of the best ordering vs the default."""
    default_order = list(default_order)
    print(f"\n{'=' * 72}")
    print(f"BEST ORDERING — {split_name}")
    print(f"  Criterion: maximise the minimum surviving stage count")
    print(f"{'=' * 72}")
    print(f"\n  Default ordering : {' → '.join(default_order)}")
    print(f"  Best ordering    : {' → '.join(best['ordering'])}")
    changed = best["ordering"] != default_order
    print(f"  Ordering changed : {'YES' if changed else 'NO — default is already optimal'}")
    print(f"\n  Per-stage survival breakdown (best ordering):")
    print(f"  {'Stage':<35}  {'Total':>7}  {'Survived':>9}  {'% Lost':>7}")
    print("  " + "-" * 65)
    for s in best["stats"]:
        flag = "  *** UNVIABLE ***" if s["survived"] < UNVIABLE_THRESHOLD else ""
        print(f"  {s['stage']:<35}  {s['total']:>7,}  {s['survived']:>9,}  "
              f"{s['pct_lost']:>6.1f}%{flag}")
    print(f"\n  Minimum stage count : {best['min_survived']:,}")
    print(f"  Total surviving     : {best['total_survived']:,}")


print("\nRunning ordering sweeps (testing all permutations)...")
sweep_3 = sweep_orderings(STAGE_SCENE_TYPES_3, stage_classes_3)
sweep_6 = sweep_orderings(STAGE_SCENE_TYPES_6, stage_classes_6)

print_best_ordering(sweep_3[0], STAGE_SCENE_TYPES_3, "3-STAGE")
print_best_ordering(sweep_6[0], STAGE_SCENE_TYPES_6, "6-STAGE")

# Best orderings for downstream use
best_3 = sweep_3[0]
best_6 = sweep_6[0]

# ---------------------------------------------------------------------------
# 11. Save all CSVs
# ---------------------------------------------------------------------------
df_3   = pd.DataFrame(rows_3,  columns=["stage", "scene_type", "scene", "classes_present"])
df_6   = pd.DataFrame(rows_6,  columns=["stage", "scene_type", "scene", "classes_present"])
df_all = pd.concat([
    df_3.assign(split="SceneType-3stage"),
    df_6.assign(split="SceneType-6stage"),
], ignore_index=True)

df_3.to_csv(OUTPUT_DIR  / "incremental_splits_v2_3stage.csv",  index=False)
df_6.to_csv(OUTPUT_DIR  / "incremental_splits_v2_6stage.csv",  index=False)
df_all.to_csv(OUTPUT_DIR / "incremental_splits_v2_all.csv",    index=False)

# Stage stats for default ordering (for v1 comparison in print section)
pd.DataFrame(stats_3).to_csv(SPLITS_DIR / "split_stats_v2_3stage.csv", index=False)
pd.DataFrame(stats_6).to_csv(SPLITS_DIR / "split_stats_v2_6stage.csv", index=False)

# Best-ordering CSVs
cols = ["stage", "scene_type", "scene", "classes_present"]
df_3_best = pd.DataFrame(best_3["csv_rows"], columns=cols)
df_6_best = pd.DataFrame(best_6["csv_rows"], columns=cols)
df_all_best = pd.concat([
    df_3_best.assign(split="SceneType-3stage-bestorder"),
    df_6_best.assign(split="SceneType-6stage-bestorder"),
], ignore_index=True)

df_3_best.to_csv(OUTPUT_DIR  / "incremental_splits_v2_3stage_bestorder.csv",  index=False)
df_6_best.to_csv(OUTPUT_DIR  / "incremental_splits_v2_6stage_bestorder.csv",  index=False)
df_all_best.to_csv(OUTPUT_DIR / "incremental_splits_v2_all_bestorder.csv",    index=False)

stats_3_best = pd.DataFrame(best_3["stats"])
stats_6_best = pd.DataFrame(best_6["stats"])
stats_3_best.to_csv(SPLITS_DIR / "split_stats_v2_3stage_bestorder.csv", index=False)
stats_6_best.to_csv(SPLITS_DIR / "split_stats_v2_6stage_bestorder.csv", index=False)

# ---------------------------------------------------------------------------
# 12. Generate markdown analysis report for the best orderings
# ---------------------------------------------------------------------------

def lift_table_rows(selected_classes, lift_df, stage_scene_types, class_stage):
    """Return list of formatted markdown table rows for the lift section."""
    pivot = (
        lift_df[lift_df["scene_type"].isin(stage_scene_types)]
        .pivot(index="class", columns="scene_type", values="lift")
        .reindex(columns=stage_scene_types, fill_value=0.0)
    )
    rows = []
    for cls in selected_classes:
        assigned = class_stage.get(cls, "?")
        cells = []
        for s in stage_scene_types:
            val = pivot.at[cls, s] if cls in pivot.index else 0.0
            bold = "**" if s == assigned else ""
            cells.append(f"{bold}{val:.3f}{bold}")
        rows.append(f"| {cls} | {' | '.join(cells)} | {assigned} |")
    return rows


def make_best_order_report(best_3, best_6, stage_classes_3, stage_classes_6,
                           selected_classes, lift_df, presence_df, stats_3, stats_6):
    """Return a markdown string for the best-ordering analysis report."""

    ord3 = best_3["ordering"]
    ord6 = best_6["ordering"]

    # Rebuild class_stage dicts for the best orderings (assignments are same,
    # just use the fixed stage_classes dicts)
    cs3 = {cls: stype for stype, clslist in stage_classes_3.items() for cls in clslist}
    cs6 = {cls: stype for stype, clslist in stage_classes_6.items() for cls in clslist}

    # Scene type counts
    type_counts = presence_df["scene_type"].value_counts()

    def stage_defs_table(ordering, stage_classes_dict):
        rows = []
        for i, stype in enumerate(ordering, 1):
            n = int(type_counts.get(stype, 0))
            classes_str = ", ".join(stage_classes_dict[stype])
            rows.append(f"| Stage {i} – {stype.replace('_', ' ').title()} "
                        f"| {stype} ({n:,} scenes) | {classes_str} |")
        return "\n".join(rows)

    def survival_table(stats):
        rows = []
        for s in stats:
            flag = " ***" if s["survived"] < UNVIABLE_THRESHOLD else ""
            rows.append(f"| {s['stage']} | {s['scene_type']} | "
                        f"{s['total']:,} | {s['survived']:,}{flag} | {s['pct_lost']:.1f}% |")
        return "\n".join(rows)

    def comparison_table(default_stats, best_stats):
        rows = []
        for d, b in zip(default_stats, best_stats):
            delta = b["survived"] - d["survived"]
            arrow = f"+{delta}" if delta >= 0 else str(delta)
            rows.append(f"| {d['scene_type']} | {d['survived']:,} | "
                        f"{b['survived']:,} | {arrow} |")
        return "\n".join(rows)

    lt3  = "\n".join(lift_table_rows(selected_classes, lift_df, ord3, cs3))
    hdr3 = " | ".join(s.replace("_", " ").title() for s in ord3)
    lt6  = "\n".join(lift_table_rows(selected_classes, lift_df, ord6, cs6))
    hdr6 = " | ".join(s.replace("_", " ").title() for s in ord6)

    sep3 = "|".join(["---|"] * (len(ord3) + 2))
    sep6 = "|".join(["---|"] * (len(ord6) + 2))

    default_order_3 = STAGE_SCENE_TYPES_3
    default_order_6 = STAGE_SCENE_TYPES_6

    changed_3 = ord3 != list(default_order_3)
    changed_6 = ord6 != list(default_order_6)

    md = f"""# Best-Ordering Split Analysis Report — V2 (Ordering-Optimised)

**Dataset:** SUN RGB-D · 10,295 scenes · Top-24 classes (wall/floor/ceiling excluded)

**Optimisation criterion:** Maximise the minimum surviving stage count.
We do not optimise for total scenes. A split where every stage has a reasonable
number of scenes is far more useful than one where a few stages have thousands
and others have single-digit counts.

---

## What "Best Ordering" Means

The **class-to-scene-type assignments** (determined by lift, section 6 of the script)
are fixed. Only the **stage number** — which scene type appears first, second, etc.
— changes across the ordering sweep.

The strict filter says: Stage-N training data must contain **zero classes from any later
stage**. This makes stage position critical: a class assigned to a late stage becomes
forbidden in *all* earlier stages. Common classes like `chair` (57% of scenes) and
`table` (46%) are catastrophic when placed in a late stage — they contaminate and
eliminate most scenes from every earlier stage.

The ordering sweep tests all n! permutations and picks the one where no single stage
collapses.

---

## 1. Best Ordering — 3-Stage Split

**Default ordering:** {" → ".join(default_order_3)}
**Best ordering:**    {" → ".join(ord3)}
**Ordering changed:** {"YES" if changed_3 else "NO — default is already optimal"}

### Lift Table (best ordering column sequence)

| Class | {hdr3} | Assigned to |
{sep3}
{lt3}

Bold = assigned stage (highest lift subject to capacity).

### Stage Definitions

| Stage | Scene type | Classes assigned by lift |
|---|---|---|
{stage_defs_table(ord3, stage_classes_3)}

### Survival Results

| Stage | Scene type | Total scenes | Surviving | % Lost |
|---|---|---|---|---|
{survival_table(best_3["stats"])}

*** UNVIABLE — fewer than {UNVIABLE_THRESHOLD} scenes.

**Minimum stage count: {best_3["min_survived"]:,}**
**Total surviving: {best_3["total_survived"]:,}**

### Comparison with Default Ordering

| Scene type | Default survived | Best ordering survived | Change |
|---|---|---|---|
{comparison_table(stats_3, best_3["stats"])}

---

## 2. Best Ordering — 6-Stage Split

**Default ordering:** {" → ".join(default_order_6)}
**Best ordering:**    {" → ".join(ord6)}
**Ordering changed:** {"YES" if changed_6 else "NO — default is already optimal"}

### Lift Table (best ordering column sequence)

| Class | {hdr6} | Assigned to |
{sep6}
{lt6}

Bold = assigned stage.

### Stage Definitions

| Stage | Scene type | Classes assigned by lift |
|---|---|---|
{stage_defs_table(ord6, stage_classes_6)}

### Survival Results

| Stage | Scene type | Total scenes | Surviving | % Lost |
|---|---|---|---|---|
{survival_table(best_6["stats"])}

*** UNVIABLE — fewer than {UNVIABLE_THRESHOLD} scenes.

**Minimum stage count: {best_6["min_survived"]:,}**
**Total surviving: {best_6["total_survived"]:,}**

### Comparison with Default Ordering

| Scene type | Default survived | Best ordering survived | Change |
|---|---|---|---|
{comparison_table(stats_6, best_6["stats"])}

---

## 3. Why Common Classes Drive the Ordering

The dominant effect is always the stage position of **generic, ubiquitous classes**:

| Class | Overall frequency | Assigned scene type |
|---|---|---|
| chair | {presence_df["chair"].mean()*100:.1f}% of all scenes | {cs3.get("chair", "?")} |
| table | {presence_df["table"].mean()*100:.1f}% of all scenes | {cs3.get("table", "?")} |
| window | {presence_df["window"].mean()*100:.1f}% of all scenes | {cs3.get("window", "?")} |
| door | {presence_df["door"].mean()*100:.1f}% of all scenes | {cs3.get("door", "?")} |

When the scene type containing `chair` and `table` is placed in Stage 1, those classes
are never future-forbidden for any subsequent stage. Every scene that contains a chair
or table is eligible for later stages. Placing that scene type last is the single biggest
cause of early-stage collapse in the default ordering.

---

## 4. Remaining Limitations

The ordering sweep is the maximum improvement available **within the current
strict-filter framework**. If survival counts are still too low after reordering,
the root cause is the strict filter itself — see Options A–C in
`split_analysis_report_v2.md`.
"""
    return md


report_md = make_best_order_report(
    best_3, best_6, stage_classes_3, stage_classes_6,
    selected_classes, lift_df, presence_df, stats_3, stats_6,
)

report_path = SPLITS_DIR / "split_analysis_report_v2_bestorder.md"
report_path.write_text(report_md, encoding="utf-8")
print(f"\n  Saved: split_analysis_report_v2_bestorder.md")

print(f"\n{'=' * 72}")
print("SAVED FILES")
print(f"{'=' * 72}")
print(f"  scene_labels.csv               ({len(presence_df):,} rows)")
print(f"  scene_type_class_lift.csv")
print(f"  class_stage_assignments_v2.csv")
print(f"  --- default ordering ---")
print(f"  split_stats_v2_3stage.csv")
print(f"  split_stats_v2_6stage.csv")
print(f"  incremental_splits_v2_3stage.csv        ({len(df_3):,} rows)")
print(f"  incremental_splits_v2_6stage.csv        ({len(df_6):,} rows)")
print(f"  incremental_splits_v2_all.csv           ({len(df_all):,} rows)")
print(f"  --- best ordering (from sweep) ---")
print(f"  split_analysis_report_v2_bestorder.md")
print(f"  split_stats_v2_3stage_bestorder.csv")
print(f"  split_stats_v2_6stage_bestorder.csv")
print(f"  incremental_splits_v2_3stage_bestorder.csv  ({len(df_3_best):,} rows)")
print(f"  incremental_splits_v2_6stage_bestorder.csv  ({len(df_6_best):,} rows)")
print(f"  incremental_splits_v2_all_bestorder.csv     ({len(df_all_best):,} rows)")

print("\nDone.")
