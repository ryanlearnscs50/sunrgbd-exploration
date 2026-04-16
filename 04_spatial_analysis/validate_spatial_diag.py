"""
validate_spatial_diag.py
------------------------
Two focused validations:

A) WALL DISTANCE PERCENTILES
   Print 5th / 25th / 50th / 75th / 95th / 99th percentile of
   nearest_edge_dist_to_wall_m for every class.  This lets us confirm
   that the box-plot whiskers (which are set to whis=[5, 95]) end
   exactly where the 95th-percentile column says they should.

B) NEGATIVE BOTTOM-ABOVE-FLOOR  (targeted at "chair")
   Hypothesis: negative values come from inaccurate bounding boxes where
   the annotated Ymax_object (physical bottom in Y-down coords) is
   numerically larger than floor_y, i.e. the bbox extends "through" the
   floor.  We prove this by:
     1. Showing the fraction of chairs with bottom_above_floor < 0.
     2. Pulling the raw annotation for a sample of those scenes and
        printing:  floor_y, Ymax_object, and bottom_above_floor.
     3. Generating a diagnostic chart: histogram of bottom_above_floor
        for chairs, with the negative region highlighted.
"""

import json
import pathlib
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATASET_ROOT = pathlib.Path(r"C:\Users\ryana\Downloads\SUNRGBD\SUNRGBD")
OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")
CSV_PATH     = OUTPUT_DIR / "spatial_features.csv"

TARGET_CLASS   = "chair"
SAMPLE_SIZE    = 10      # how many raw-annotation examples to print
N_SAMPLE_SCENES = 200    # scenes to scan for raw chair annotations (for speed)

# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
print("Loading spatial_features.csv …")
df = pd.read_csv(CSV_PATH)
print(f"  {len(df):,} object records, {df['category'].nunique()} classes\n")

# ===========================================================================
# A) WALL DISTANCE PERCENTILES
# ===========================================================================
print("=" * 72)
print("A) NEAREST-EDGE WALL DISTANCE — percentiles per class")
print("=" * 72)

wall_col = "nearest_edge_dist_to_wall_m"
pcts     = [5, 25, 50, 75, 95, 99]

wall_df = (
    df.dropna(subset=[wall_col])
      .groupby("category")[wall_col]
      .quantile([p / 100 for p in pcts])
      .unstack(level=-1)
)
# Rename columns to readable percentile labels
wall_df.columns = [f"p{p}" for p in pcts]
wall_df = wall_df.sort_values("p50")

header = f"{'class':<18}" + "".join(f"  p{p:>2}" for p in pcts)
print(header)
print("-" * len(header))
for cls, row in wall_df.iterrows():
    vals = "".join(f"  {row[f'p{p}']:>5.2f}" for p in pcts)
    print(f"{cls:<18}{vals}")

print()
print("  ^ The 'p95' column is where each box-plot whisker (whis=[5,95]) ends.")
print("  If the whisker in the PNG ends at ~2.5 m for chairs, p95 should be ~2.5 m here.\n")

# ===========================================================================
# B) NEGATIVE BOTTOM-ABOVE-FLOOR  (chairs)
# ===========================================================================
print("=" * 72)
print(f"B) NEGATIVE BOTTOM-ABOVE-FLOOR  — targeting '{TARGET_CLASS}'")
print("=" * 72)

chair_df = df[df["category"] == TARGET_CLASS].dropna(subset=["bottom_above_floor_m"])

total   = len(chair_df)
neg     = (chair_df["bottom_above_floor_m"] < 0).sum()
pct_neg = 100 * neg / total if total else 0

print(f"\n  Total chair records with valid floor measurement : {total:,}")
print(f"  Records with bottom_above_floor < 0             : {neg:,}  ({pct_neg:.1f} %)")
print()

# --- Quick percentile table for bottom_above_floor
baf_pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
print(f"  bottom_above_floor_m percentiles for '{TARGET_CLASS}':")
for p in baf_pcts:
    v = np.percentile(chair_df["bottom_above_floor_m"], p)
    print(f"    p{p:>2}: {v:+.4f} m")

# ---------------------------------------------------------------------------
# B1) Prove via raw annotation: floor_y, Ymax_object, difference
# ---------------------------------------------------------------------------
# Strategy:
#   - From the CSV get scene names where a chair has bottom_above_floor < 0.
#   - For each such scene, find the raw annotation3Dfinal/index.json and
#     annotation3Dlayout/index.json, extract floor_y and the chair Ymax, and
#     show: floor_y, Ymax_chair, Ymax_chair - floor_y  (should be positive
#     for negative offsets, confirming the bbox overshoots the floor).

neg_scenes = (
    chair_df.loc[chair_df["bottom_above_floor_m"] < 0, "scene"]
    .unique()
    .tolist()
)
print(f"\n  Scenes with at least one negative-offset chair: {len(neg_scenes):,}")
print(f"  Checking up to {SAMPLE_SIZE} raw annotations …\n")

# Scan annotation files in dataset to match scene names
# (scene column was stored as the parent directory name in spatial_analysis.py)
ann3d_files = sorted(DATASET_ROOT.rglob("annotation3Dfinal/index.json"))
scene_map   = {p.parent.parent.name: p for p in ann3d_files}

raw_examples = []
for scene_name in neg_scenes:
    if len(raw_examples) >= SAMPLE_SIZE:
        break
    ann3d_path = scene_map.get(scene_name)
    if ann3d_path is None:
        continue
    scene_dir   = ann3d_path.parent.parent
    layout_path = scene_dir / "annotation3Dlayout" / "index.json"
    if not layout_path.exists():
        continue

    # Read layout to get floor_y
    try:
        with layout_path.open(encoding="utf-8") as f:
            layout = json.load(f)
        floor_y = None
        for obj in (layout.get("objects") or []):
            if obj and obj.get("name", "").lower().startswith("room"):
                polys = obj.get("polygon") or []
                if polys and "Ymax" in polys[0]:
                    floor_y = float(polys[0]["Ymax"])
                    break
        if floor_y is None:
            continue
    except (json.JSONDecodeError, OSError):
        continue

    # Read 3D annotation to find chairs
    try:
        with ann3d_path.open(encoding="utf-8") as f:
            ann3d = json.load(f)
    except (json.JSONDecodeError, OSError):
        continue

    for obj in (ann3d.get("objects") or []):
        if not obj:
            continue
        raw_name = obj.get("name", "")
        if not isinstance(raw_name, str):
            continue
        base = raw_name.split(":")[0].strip().lower()
        if base not in {"chair", "armchair", "chairs"}:
            continue
        polys = obj.get("polygon") or []
        if not polys or "Ymax" not in polys[0]:
            continue
        ymax_obj = float(polys[0]["Ymax"])
        bottom_above_floor = floor_y - ymax_obj
        if bottom_above_floor < 0:
            raw_examples.append({
                "scene"              : scene_name,
                "raw_name"           : raw_name,
                "floor_y (room Ymax)": round(floor_y, 4),
                "Ymax_obj"           : round(ymax_obj, 4),
                # Ymax_obj > floor_y means bbox pokes through the floor
                "Ymax_obj > floor_y?": ymax_obj > floor_y,
                "bottom_above_floor" : round(bottom_above_floor, 4),
            })
            if len(raw_examples) >= SAMPLE_SIZE:
                break

ex_df = pd.DataFrame(raw_examples)
if not ex_df.empty:
    print("  Raw annotation evidence — chairs with negative bottom_above_floor:")
    print()
    # Pretty-print the table
    col_order = [
        "scene", "raw_name",
        "floor_y (room Ymax)", "Ymax_obj", "Ymax_obj > floor_y?",
        "bottom_above_floor",
    ]
    print(ex_df[col_order].to_string(index=False))
    print()
    print(textwrap.dedent("""
    Reading guide
    -------------
    In SUN RGB-D's Y-down coordinate system:
      - floor_y = room Ymax  (large Y = physically low = floor level)
      - Ymax_obj = object's own Ymax = its physical BOTTOM
      - bottom_above_floor = floor_y - Ymax_obj

    When Ymax_obj > floor_y the object's bbox bottom is numerically
    BELOW the floor level. That is a geometric impossibility for a
    real chair and indicates the bounding box is too tall / shifted
    downward in the annotation. This is the annotation noise that
    produces the negative bottom_above_floor values we see in the plot.
    """))
else:
    print("  [Could not retrieve raw examples — check DATASET_ROOT path]")

# ---------------------------------------------------------------------------
# B2) Diagnostic chart: histogram of bottom_above_floor for chairs
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))

values = chair_df["bottom_above_floor_m"].clip(-0.6, 2.5)

# Split into negative and non-negative for colour-coding
neg_vals = values[values < 0]
pos_vals = values[values >= 0]

bins = np.linspace(-0.6, 2.5, 65)

ax.hist(neg_vals, bins=bins, color="#d62728", alpha=0.85, label="< 0  (bbox below floor)")
ax.hist(pos_vals, bins=bins, color="#1f77b4", alpha=0.7,  label="≥ 0  (bbox above / at floor)")

ax.axvline(0, color="black", linewidth=1.2, linestyle="--", label="floor level")

ax.set_xlabel("bottom_above_floor_m  (clipped to [−0.6, 2.5])", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.set_title(
    f"'{TARGET_CLASS}' — bottom_above_floor distribution\n"
    f"Red region = bbox extends below floor  ({neg:,} / {total:,} = {pct_neg:.1f} %)",
    fontsize=10,
)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

plt.tight_layout()
out_path = OUTPUT_DIR / "diag_chair_floor_offset.png"
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"  Chart saved -> {out_path}")
print("\nDone.")
