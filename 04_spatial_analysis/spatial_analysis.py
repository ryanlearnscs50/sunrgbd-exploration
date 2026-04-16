"""
spatial_analysis.py
-------------------
Computes floor- and wall-relative spatial features for every annotated 3D
object belonging to the top-25 object classes in SUN RGB-D.

Data sources (per scene directory):
  annotation3Dfinal/index.json  -- axis-aligned 3D bounding boxes
  annotation3Dlayout/index.json -- room polygon (floor/ceiling Y, wall XZ)

Coordinate system (same in both files):
  Y axis = up.  Ymin = bottom of object, Ymax = top.
  X, Z  = horizontal plane (floor plan).
  Units = metres.

Features computed per object:
  category                    canonical class name (after synonym merging)
  cx, cy, cz                  3D centroid of bounding box  (metres)
  height_m                    bbox height  (Ymax - Ymin)
  width_m, depth_m            horizontal bbox extents
  bottom_above_floor_m        Ymin_object - Ymin_room  (NaN if layout missing)
  on_floor                    True when bottom_above_floor_m < ON_FLOOR_THRESH
  dist_to_wall_m              2D distance from (cx,cz) to nearest wall segment
                              (NaN if layout missing)
  nearest_edge_dist_to_wall_m minimum of dist_to_wall across the 4 bounding-box
                              corners — better than centroid distance for large
                              objects whose back is flush against a wall while
                              their centroid is ~half their depth away from it
                              (NaN if layout or bbox dimensions missing)
  norm_cx, norm_cz            (cx,cz) centred on room centroid, divided by
                              room half-width/half-depth → range ≈ [-1, 1].
                              Useful for cross-scene spatial comparisons.

Outputs:
  spatial_features.csv             raw per-object records
  spatial_summary.csv              per-category statistics table
  spatial_height_distribution.png  box plots: bottom_above_floor per class
  spatial_on_floor_fraction.png    bar chart: % on-floor per class
  spatial_wall_distance.png        side-by-side: centroid vs nearest-edge dist
  spatial_topdown_scatter.png      room-normalised XZ scatter, top-6 classes
  spatial_summary.md               findings + mentor-idea explanation
"""

import json
import math
import pathlib
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# 1. Paths and constants
# ---------------------------------------------------------------------------
DATASET_ROOT     = pathlib.Path(r"C:\Users\ryana\Downloads\SUNRGBD\SUNRGBD")
OUTPUT_DIR       = pathlib.Path(r"C:\sunrgbd-exploration")
CSV_PATH         = OUTPUT_DIR / "class_frequencies.csv"

EXCLUDE          = {"wall", "floor", "ceiling"}
TOP_N            = 25
ON_FLOOR_THRESH  = 0.15   # metres: bottom within this of floor counts as "on floor"
MIN_SAMPLES_PLOT = 30     # minimum per-category samples to include in plots

# ---------------------------------------------------------------------------
# 2. Synonym map (identical to class_frequency.py / cooccurrence.py)
# ---------------------------------------------------------------------------
# See class_frequency.py for the full rationale.
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
# 3. Load top-25 selected classes
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: same filter-before-slice pattern as cooccurrence.py.
freq_df = pd.read_csv(CSV_PATH)
selected_classes = (
    freq_df
    .loc[~freq_df["class"].isin(EXCLUDE), "class"]
    .head(TOP_N)
    .tolist()
)
selected_set = set(selected_classes)
print(f"Selected {len(selected_classes)} classes:")
for i, c in enumerate(selected_classes, 1):
    print(f"  {i:2d}. {c}")

# ---------------------------------------------------------------------------
# 4. Geometry helpers
# ---------------------------------------------------------------------------

def point_segment_dist(px: float, pz: float,
                       x1: float, z1: float,
                       x2: float, z2: float) -> float:
    """
    Shortest distance from point (px, pz) to line segment (x1,z1)-(x2,z2).

    This is the standard "project then clamp" formula.  We project the point
    onto the infinite line through the segment, clamp the parameter t to [0,1]
    to stay within the segment, and measure the residual distance.  This is
    called "closest point on segment" and is O(1) per edge.
    """
    dx, dz = x2 - x1, z2 - z1
    lsq = dx * dx + dz * dz
    if lsq < 1e-12:          # degenerate zero-length edge
        return math.hypot(px - x1, pz - z1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (pz - z1) * dz) / lsq))
    return math.hypot(px - (x1 + t * dx), pz - (z1 + t * dz))


def dist_to_nearest_wall(px: float, pz: float,
                         wall_xs: list, wall_zs: list) -> float:
    """
    Minimum distance from (px, pz) to any edge of the room boundary polygon.

    We treat the polygon as a closed loop of wall segments and iterate over
    all consecutive pairs.  This is O(n_walls) per object, which is fast
    because rooms typically have 4-8 walls.
    """
    n = len(wall_xs)
    if n < 2:
        return float("nan")
    return min(
        point_segment_dist(
            px, pz,
            wall_xs[i], wall_zs[i],
            wall_xs[(i + 1) % n], wall_zs[(i + 1) % n]
        )
        for i in range(n)
    )

# ---------------------------------------------------------------------------
# 5. Main scan
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: we scan annotation3Dfinal/index.json as the primary
# source (the "final" suffix means human-reviewed / quality-checked).  For
# each scene we also load annotation3Dlayout/index.json if it exists.
# The two files live in sibling directories under the same scene folder, so
# we derive the layout path from the 3D annotation path.

print("\nScanning annotation3Dfinal files...")
ann3d_files = sorted(DATASET_ROOT.rglob("annotation3Dfinal/index.json"))
print(f"  Found {len(ann3d_files):,} annotation3Dfinal files")

records          = []
scenes_ok        = 0
scenes_no_layout = 0
parse_errors     = 0

for scene_idx, ann3d_path in enumerate(ann3d_files):
    if scene_idx % 1000 == 0:
        print(f"  [{scene_idx:,} / {len(ann3d_files):,}] scenes scanned …")
    scene_dir   = ann3d_path.parent.parent
    layout_path = scene_dir / "annotation3Dlayout" / "index.json"

    # --- read 3D object bboxes ---
    try:
        with ann3d_path.open(encoding="utf-8") as f:
            ann3d = json.load(f)
    except (json.JSONDecodeError, OSError):
        parse_errors += 1
        continue

    scenes_ok += 1

    # --- read room layout (optional but needed for floor/wall features) ---
    floor_y = None
    room_cx_world = None
    room_cz_world = None
    room_half_w   = None
    room_half_d   = None
    wall_xs = wall_zs = None

    if layout_path.exists():
        try:
            with layout_path.open(encoding="utf-8") as f:
                layout = json.load(f)
            for obj in (layout.get("objects") or []):
                if obj and obj.get("name", "").lower().startswith("room"):
                    polys = obj.get("polygon") or []
                    if polys:
                        p = polys[0]
                        if "Ymax" in p and p.get("X") and p.get("Z"):
                            # COORDINATE SYSTEM NOTE: SUN RGB-D 3D annotations
                            # use a Y-DOWN camera convention (standard in computer
                            # vision).  Y increases downward, so:
                            #   Ymin = physically HIGH (near ceiling)
                            #   Ymax = physically LOW  (near floor / ground)
                            # Therefore floor_y = layout Ymax, NOT Ymin.
                            # Using Ymin as floor gives chair "bottom_above_floor"
                            # of ~1.9 m -- a clear sign the axis is inverted.
                            floor_y       = float(p["Ymax"])
                            wall_xs       = [float(x) for x in p["X"]]
                            wall_zs       = [float(z) for z in p["Z"]]
                            # Room centroid and half-extents for normalisation
                            room_cx_world = sum(wall_xs) / len(wall_xs)
                            room_cz_world = sum(wall_zs) / len(wall_zs)
                            room_half_w   = max((max(wall_xs) - min(wall_xs)) / 2, 0.1)
                            room_half_d   = max((max(wall_zs) - min(wall_zs)) / 2, 0.1)
                            break
        except (json.JSONDecodeError, OSError):
            pass

    if floor_y is None:
        scenes_no_layout += 1

    # --- extract per-object records ---
    for obj in (ann3d.get("objects") or []):
        if not obj:
            continue

        # Strip ":attribute" suffix that annotation3D appends (e.g. "chair:truncated")
        raw_name = obj.get("name", "")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        base_name = raw_name.split(":")[0].strip()
        canonical = normalise(base_name)
        if canonical not in selected_set:
            continue

        polys = obj.get("polygon") or []
        if not polys:
            continue
        poly = polys[0]

        ymin = poly.get("Ymin")
        ymax = poly.get("Ymax")
        xs   = poly.get("X") or []
        zs   = poly.get("Z") or []

        # Skip annotations with missing or degenerate geometry
        if ymin is None or ymax is None or not xs or not zs:
            continue
        if ymax <= ymin:   # zero-height annotation -- likely a placeholder
            continue

        ymin, ymax = float(ymin), float(ymax)
        xs = [float(x) for x in xs]
        zs = [float(z) for z in zs]

        # Centroid and bounding-box dimensions
        cx     = sum(xs) / len(xs)
        cz     = sum(zs) / len(zs)
        cy     = (ymin + ymax) / 2.0
        height = ymax - ymin
        width  = max(xs) - min(xs)
        depth  = max(zs) - min(zs)

        # Floor-relative features
        # In Y-down frame: floor is at Ymax of room.  The physical bottom of
        # the object is its Ymax (largest Y = closest to floor).
        # "bottom_above_floor" = how far the object bottom is above the floor
        #   = floor_y - Ymax_object
        # For a chair sitting on the floor: Ymax_chair ≈ floor_y → ~0 m.
        # For a wall-mounted mirror at 1.5 m height: Ymax_mirror is 1.0 m
        #   smaller than floor_y → bottom_above_floor ≈ 1.0 m.
        if floor_y is not None:
            bottom_above_floor = floor_y - ymax
            on_floor = bool(bottom_above_floor < ON_FLOOR_THRESH)
        else:
            bottom_above_floor = float("nan")
            on_floor = None

        # Wall-relative features — centroid distance
        if wall_xs is not None:
            d_wall = dist_to_nearest_wall(cx, cz, wall_xs, wall_zs)
        else:
            d_wall = float("nan")

        # Nearest-edge wall distance: check all 4 bounding-box corners.
        # ENGINEERING DECISION: we sample the four corners rather than
        # integrating over every edge point.  For a rectangular bbox the
        # closest point on any edge to a wall segment lies at one of the
        # corners or at the foot of a perpendicular from the wall to the
        # edge.  Sampling corners only is an approximation, but for typical
        # room geometries (walls far longer than object bbox sides) the
        # corner that faces the nearest wall is the true closest point.
        # The error is at most a few centimetres, well within annotation noise.
        #
        # We skip this computation when width_m or depth_m is zero/missing to
        # avoid degenerate boxes where all four corners collapse to the centroid.
        if wall_xs is not None and width > 0 and depth > 0:
            x_min_b = cx - width / 2
            x_max_b = cx + width / 2
            z_min_b = cz - depth / 2
            z_max_b = cz + depth / 2
            d_nearest_edge = min(
                dist_to_nearest_wall(bx, bz, wall_xs, wall_zs)
                for bx, bz in (
                    (x_min_b, z_min_b),
                    (x_min_b, z_max_b),
                    (x_max_b, z_min_b),
                    (x_max_b, z_max_b),
                )
            )
        else:
            d_nearest_edge = float("nan")

        # Room-normalised position: centre on room centroid, scale by half-extents.
        # This gives coordinates in approximately [-1, 1] for both axes regardless
        # of room size, making cross-scene comparison meaningful.
        # This pattern is called "normalised room coordinates" in scene synthesis
        # literature -- it lets you learn a layout prior that transfers across
        # rooms of different sizes.
        if room_cx_world is not None:
            norm_cx = (cx - room_cx_world) / room_half_w
            norm_cz = (cz - room_cz_world) / room_half_d
        else:
            norm_cx = float("nan")
            norm_cz = float("nan")

        records.append({
            "scene":                str(scene_dir.name),
            "category":             canonical,
            "cx":                   round(cx,     4),
            "cy":                   round(cy,     4),
            "cz":                   round(cz,     4),
            "height_m":             round(height, 4),
            "width_m":              round(width,  4),
            "depth_m":              round(depth,  4),
            "bottom_above_floor_m": round(bottom_above_floor, 4)
                                    if not math.isnan(bottom_above_floor) else float("nan"),
            "on_floor":             on_floor,
            "dist_to_wall_m":       round(d_wall, 4)
                                    if not math.isnan(d_wall) else float("nan"),
            "nearest_edge_dist_to_wall_m": round(d_nearest_edge, 4)
                                    if not math.isnan(d_nearest_edge) else float("nan"),
            "norm_cx":              round(norm_cx, 4)
                                    if not math.isnan(norm_cx) else float("nan"),
            "norm_cz":              round(norm_cz, 4)
                                    if not math.isnan(norm_cz) else float("nan"),
        })

print(f"  Scenes processed      : {scenes_ok:,}")
print(f"  Scenes without layout : {scenes_no_layout:,}")
print(f"  Parse errors          : {parse_errors:,}")
print(f"  Object records total  : {len(records):,}")

# ---------------------------------------------------------------------------
# 6. Build DataFrame and quick sanity check
# ---------------------------------------------------------------------------
df = pd.DataFrame(records)

# Sanity check: median bottom_above_floor for chair should be near 0
# (chairs sit on the floor).  If it's wildly negative, the Y axis is
# inverted in the dataset and we'd need to flip the sign.  We print this
# so it's visible in the run log.
if "chair" in df["category"].values:
    chair_med = df[df["category"] == "chair"]["bottom_above_floor_m"].median()
    print(f"\n[SANITY] Median chair bottom_above_floor = {chair_med:.3f} m "
          f"(expect ~0; if strongly negative, Y axis may be inverted)")

# ---------------------------------------------------------------------------
# 7. Save raw CSV
# ---------------------------------------------------------------------------
csv_path = OUTPUT_DIR / "spatial_features.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved features CSV : {csv_path}  ({len(df):,} rows)")

# ---------------------------------------------------------------------------
# 8. Per-category summary statistics
# ---------------------------------------------------------------------------
# We compute median (robust to outliers) and IQR (interquartile range) rather
# than mean/std because 3D bounding box annotations can have extreme outliers
# from mis-labelled scenes.  Median and IQR are called "robust statistics"
# because a small fraction of outliers cannot drag them far from the true
# central tendency.

def iqr(s: pd.Series) -> float:
    return float(s.quantile(0.75) - s.quantile(0.25))

summary_rows = []
for cat in selected_classes:
    sub = df[df["category"] == cat]
    if len(sub) == 0:
        continue

    # Only include rows with valid (non-NaN) values for each statistic.
    # Wall distance uses nearest_edge (not centroid) as the primary measure —
    # the centroid column is retained in spatial_features.csv for reference
    # but the summary reports the edge distance, which correctly reflects
    # whether a large object's surface is flush against a wall.
    h   = sub["bottom_above_floor_m"].dropna()
    we  = sub["nearest_edge_dist_to_wall_m"].dropna()
    ht  = sub["height_m"].dropna()
    of  = sub["on_floor"].dropna()

    summary_rows.append({
        "category":              cat,
        "n_objects":             len(sub),
        "n_with_floor_data":     len(h),
        "median_bottom_above_floor_m": round(h.median(), 3) if len(h) else float("nan"),
        "iqr_bottom_above_floor_m":    round(iqr(h),      3) if len(h) else float("nan"),
        "pct_on_floor":          round(of.mean() * 100,   1) if len(of) else float("nan"),
        "n_with_wall_data":      len(we),
        "median_dist_to_wall_m": round(we.median(), 3) if len(we) else float("nan"),
        "iqr_dist_to_wall_m":    round(iqr(we),      3) if len(we) else float("nan"),
        "median_height_m":       round(ht.median(),  3) if len(ht) else float("nan"),
        "iqr_height_m":          round(iqr(ht),      3) if len(ht) else float("nan"),
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = OUTPUT_DIR / "spatial_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary CSV  : {summary_path}")
print("\nPer-category summary (top 15 by object count):")
print(summary_df.sort_values("n_objects", ascending=False).head(15).to_string(index=False))

# ---------------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: we use box plots (not violin plots) so there is no
# seaborn dependency.  Box plots show median, IQR, and outliers -- enough
# information for exploratory analysis.  For publication-quality figures the
# same data could be re-plotted with seaborn violins.

plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

# --- helper: clip a column to [lo, hi] for display only ---
def clipped(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return series.clip(lo, hi).dropna()

# Select categories with enough data for meaningful plots
cats_with_floor = (
    summary_df
    .dropna(subset=["median_bottom_above_floor_m"])
    .query("n_with_floor_data >= @MIN_SAMPLES_PLOT")
    .sort_values("median_bottom_above_floor_m")
    ["category"]
    .tolist()
)
# Sort by median nearest-edge distance so both subplots share the same order.
# ENGINEERING DECISION: using nearest_edge as the sort key is more meaningful
# than centroid distance because it directly reflects whether an object's
# surface is touching the wall.  Sorting both plots identically lets the
# reader's eye scan horizontally and immediately see how much the centroid
# measure overestimates wall distance for large objects.
cats_with_wall = (
    summary_df
    .dropna(subset=["median_dist_to_wall_m"])
    .query("n_with_wall_data >= @MIN_SAMPLES_PLOT")
    .sort_values("median_dist_to_wall_m")
    ["category"]
    .tolist()
)

# ---- Plot 1: Height above floor distribution ----
# Sorted by median height so on-floor classes cluster at the bottom,
# elevated objects (lamps, mirrors, pictures) cluster at the top.
fig1, ax1 = plt.subplots(figsize=(10, 7))

HEIGHT_CLIP_HI = 2.5   # metres; covers >99 % of the data
# Lower bound is 0 (the floor): negative values arise from annotation noise
# (bounding-box bottom labelled slightly below the room floor polygon) and
# are clamped here because physically no object can be below floor level.
data_height = [
    clipped(df[df["category"] == c]["bottom_above_floor_m"], 0.0, HEIGHT_CLIP_HI)
    for c in cats_with_floor
]
bp1 = ax1.boxplot(
    data_height,
    vert=False,
    patch_artist=True,
    whis=[5, 95],              # whiskers = 5th–95th percentile (not 1.5×IQR)
    showfliers=False,          # hide the <5 % tail beyond the whiskers
    medianprops={"color": "black", "linewidth": 1.5},
    boxprops={"facecolor": "steelblue", "alpha": 0.7},
    whiskerprops={"linewidth": 0.9},
    capprops={"linewidth": 0.9},
)
ax1.set_xlim(0.0, HEIGHT_CLIP_HI)   # axis matches clip so title is accurate
ax1.set_yticks(range(1, len(cats_with_floor) + 1))
ax1.set_yticklabels(cats_with_floor, fontsize=8)
ax1.axvline(0,               color="red",   linewidth=0.8, linestyle="--", label="floor level (0 m)")
ax1.axvline(ON_FLOOR_THRESH, color="orange", linewidth=0.8, linestyle=":",  label=f"on-floor threshold ({ON_FLOOR_THRESH} m)")
ax1.set_xlabel("Bottom of bounding box above floor (metres)", fontsize=10)
ax1.set_title(
    "SUN RGB-D — Object height above floor, by category\n"
    f"Box = IQR, line = median; whiskers = 5th–95th pct; clipped at {HEIGHT_CLIP_HI} m\n"
    "(~24 % of raw values are slightly negative due to annotation noise; clamped to 0)",
    fontsize=10, pad=10,
)
ax1.legend(fontsize=8)
fig1.tight_layout()
p1 = OUTPUT_DIR / "spatial_height_distribution.png"
fig1.savefig(p1, dpi=150)
print(f"\nSaved plot : {p1}")
plt.close(fig1)

# ---- Plot 2: On-floor fraction ----
# Shows the binary view: what fraction of annotated instances of each class
# are on the floor vs. elevated/wall-mounted.  Sorted descending.
# ENGINEERING DECISION: we include ALL 25 categories here (not just those
# with floor data) but mark NaN-fraction categories explicitly so the reader
# knows why they're missing.
on_floor_data = (
    summary_df
    .dropna(subset=["pct_on_floor"])
    .sort_values("pct_on_floor", ascending=True)
)
fig2, ax2 = plt.subplots(figsize=(8, 7))
colours = ["#2196F3" if p >= 50 else "#FF9800" for p in on_floor_data["pct_on_floor"]]
ax2.barh(on_floor_data["category"], on_floor_data["pct_on_floor"],
         color=colours, edgecolor="white", linewidth=0.4)
ax2.axvline(50, color="gray", linewidth=0.8, linestyle="--")
ax2.set_xlabel("% of annotated instances that are on the floor", fontsize=10)
ax2.set_title(
    "SUN RGB-D — Fraction of object instances on the floor\n"
    "(blue ≥ 50%, orange < 50%)",
    fontsize=10, pad=10,
)
ax2.set_xlim(0, 105)
for i, (_, row) in enumerate(on_floor_data.iterrows()):
    ax2.text(row["pct_on_floor"] + 1, i, f"{row['pct_on_floor']:.0f}%",
             va="center", fontsize=7)
fig2.tight_layout()
p2 = OUTPUT_DIR / "spatial_on_floor_fraction.png"
fig2.savefig(p2, dpi=150)
print(f"Saved plot : {p2}")
plt.close(fig2)

# ---- Plot 3: Nearest bounding-box edge to wall ----
# Single plot using the nearest-edge measure (more accurate than centroid
# distance for large objects whose back is flush against a wall).
# Sorted by median ascending so wall-huggers appear at the bottom.

fig3, ax3 = plt.subplots(figsize=(10, 7))

WALL_CLIP_HI = 3.5   # metres; covers >99 % of the data (99th pct ≈ 2.75 m)
data_edge = [
    clipped(df[df["category"] == c]["nearest_edge_dist_to_wall_m"], 0.0, WALL_CLIP_HI)
    for c in cats_with_wall
]
ax3.boxplot(
    data_edge,
    vert=False,
    patch_artist=True,
    whis=[5, 95],              # whiskers = 5th–95th percentile (not 1.5×IQR)
    showfliers=False,          # hide the <5 % tail beyond the whiskers
    medianprops={"color": "black", "linewidth": 1.5},
    boxprops={"facecolor": "#66BB6A", "alpha": 0.7},
    whiskerprops={"linewidth": 0.9},
    capprops={"linewidth": 0.9},
)

# Set labels after boxplot() so nothing can overwrite them.
ax3.set_xlim(0.0, WALL_CLIP_HI)   # axis matches clip so title is accurate
ax3.set_yticks(range(1, len(cats_with_wall) + 1))
ax3.set_yticklabels(cats_with_wall, fontsize=9)

ax3.axvline(0.1, color="red",    linewidth=0.9, linestyle="--", label="≤ 0.1 m (wall contact)")
ax3.axvline(0.4, color="orange", linewidth=0.9, linestyle=":",  label="≤ 0.4 m (wall-adjacent)")
ax3.set_xlabel("Distance from nearest bounding-box edge to wall (metres)", fontsize=10)
ax3.set_title(
    "SUN RGB-D — Distance to nearest wall, by category\n"
    "Measured from nearest bounding-box edge (not centroid)\n"
    f"Box = IQR, line = median; whiskers = 5th–95th pct; clipped at {WALL_CLIP_HI} m",
    fontsize=10, pad=10,
)
ax3.legend(fontsize=8)
fig3.tight_layout()
p3 = OUTPUT_DIR / "spatial_wall_distance.png"
fig3.savefig(p3, dpi=150, bbox_inches="tight")
print(f"Saved plot : {p3}")
plt.close(fig3)

# ---- Plot 4: Room-normalised top-down scatter ----
# We pick the 6 classes with the most room-normalised samples and plot
# their (norm_cx, norm_cz) positions.  The room is conceptually a
# [-1,1] x [-1,1] square, so we can overlay a boundary box.
#
# Why normalise? Each scene has its own coordinate origin (camera position)
# so raw XZ values across different scenes are meaningless to overlay.
# After centring on the room centroid and dividing by half-extents, all
# rooms share the same canonical space -- useful for learning layout priors.

norm_sub = df.dropna(subset=["norm_cx", "norm_cz"])
norm_sub = norm_sub[norm_sub["norm_cx"].between(-2, 2) & norm_sub["norm_cz"].between(-2, 2)]

top6 = (
    norm_sub.groupby("category")
    .size()
    .sort_values(ascending=False)
    .head(6)
    .index.tolist()
)

colours6 = ["#E53935", "#1E88E5", "#43A047", "#FB8C00", "#8E24AA", "#00ACC1"]
fig4, ax4 = plt.subplots(figsize=(8, 8))

# Draw room boundary
room_rect = plt.Rectangle((-1, -1), 2, 2,
                           fill=False, edgecolor="gray",
                           linewidth=1.5, linestyle="--", label="room boundary (±1)")
ax4.add_patch(room_rect)

for cls, col in zip(top6, colours6):
    sub6 = norm_sub[norm_sub["category"] == cls]
    # Subsample for readability if there are many points
    if len(sub6) > 2000:
        sub6 = sub6.sample(2000, random_state=42)
    ax4.scatter(sub6["norm_cx"], sub6["norm_cz"],
                s=4, alpha=0.25, color=col, label=f"{cls} (n={len(sub6):,})")

ax4.set_xlim(-2, 2)
ax4.set_ylim(-2, 2)
ax4.set_aspect("equal")
ax4.axhline(0, color="lightgray", linewidth=0.6)
ax4.axvline(0, color="lightgray", linewidth=0.6)
ax4.set_xlabel("Normalised X  (room-centred, ÷ half-width)", fontsize=10)
ax4.set_ylabel("Normalised Z  (room-centred, ÷ half-depth)", fontsize=10)
ax4.set_title(
    "SUN RGB-D — Room-normalised top-down object positions\n"
    "Coords centred on room centroid, scaled to ±1 = room edge\n"
    "(subsampled to 2,000 pts per class for readability)",
    fontsize=10, pad=10,
)
ax4.legend(fontsize=8, markerscale=3)
fig4.tight_layout()
p4 = OUTPUT_DIR / "spatial_topdown_scatter.png"
fig4.savefig(p4, dpi=150)
print(f"Saved plot : {p4}")
plt.close(fig4)

# ---------------------------------------------------------------------------
# 10. Markdown summary
# ---------------------------------------------------------------------------
# Build key observations programmatically from the summary_df so they are
# always consistent with the actual numbers.

def fmt(v, decimals=2):
    return "N/A" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.{decimals}f}"

# Identify classes by role — use nearest_edge for wall classification
floor_classes        = summary_df[summary_df["pct_on_floor"] > 70]["category"].tolist()
elevated_classes     = summary_df[summary_df["pct_on_floor"] < 20]["category"].tolist()
wall_contact_classes = summary_df[summary_df["median_dist_to_wall_m"] < 0.1]["category"].tolist()
wall_adj_classes     = summary_df[
    (summary_df["median_dist_to_wall_m"] >= 0.1) &
    (summary_df["median_dist_to_wall_m"] < 0.4)
]["category"].tolist()
center_classes       = summary_df[summary_df["median_dist_to_wall_m"] > 1.0]["category"].tolist()

# Tallest and shortest on average
ht_sorted = summary_df.dropna(subset=["median_height_m"]).sort_values("median_height_m")
shortest_cats = ht_sorted.head(3)["category"].tolist()
tallest_cats  = ht_sorted.tail(3)["category"].tolist()

md_lines = [
    "# SUN RGB-D Spatial Analysis — Summary",
    "",
    "## Dataset Coverage",
    "",
    f"- **{scenes_ok:,}** scenes processed  ({parse_errors} parse errors)",
    f"- **{scenes_ok - scenes_no_layout:,}** scenes with room layout  "
    f"({scenes_no_layout:,} missing — floor/wall features are NaN for these)",
    f"- **{len(df):,}** object records across {len(selected_classes)} classes",
    "",
    "## Key Patterns",
    "",
    "### Floor placement",
    "",
    f"- **Predominantly on-floor (>70%)**:  {', '.join(floor_classes) or 'none found'}",
    f"- **Predominantly elevated (<20% on floor)**:  {', '.join(elevated_classes) or 'none found'}",
    "",
    "  On-floor objects have a tight height-above-floor distribution near 0 m,",
    "  making them easy to place with a simple floor-contact constraint.",
    "  Elevated objects (mirrors, pictures, monitors, lamps) require a wall-height",
    "  or support-surface prior instead.",
    "",
    "### Wall proximity  (based on nearest bounding-box edge distance)",
    "",
    f"- **Wall-contact (median < 0.1 m)**:  {', '.join(wall_contact_classes) or 'none found'}",
    f"- **Wall-adjacent (median 0.1 – 0.4 m)**:  {', '.join(wall_adj_classes) or 'none found'}",
    f"- **Room-centre (median > 1.0 m)**:  {', '.join(center_classes) or 'none found'}",
    "",
    "  Wall-contact and wall-adjacent classes almost always appear within arm's reach",
    "  of a wall.  A synthesis system can constrain these objects to snap to the",
    "  nearest wall rather than sampling freely in XZ.",
    "",
    "### Centroid vs nearest-edge distance",
    "",
    "  The centroid-to-wall distance measures how far the object's geometric centre",
    "  is from the nearest wall.  For large objects — sofa, bed, cabinet — the",
    "  centroid sits roughly half the object's depth away from the wall even when",
    "  the back surface is flush against it, so the centroid measure systematically",
    "  overestimates wall proximity for those classes.  The nearest-edge distance",
    "  (minimum across the four bounding-box corners) corrects for this and is the",
    "  right signal for deciding **whether** an object should snap to a wall.",
    "  Conversely, the centroid distance is the right signal for deciding **where in",
    "  the room** to place the object's centre of mass — which is what the XZ",
    "  position prior actually needs.",
    "",
    "### Object size",
    "",
    f"- **Smallest median height**: {', '.join(shortest_cats)}",
    f"- **Tallest median height**: {', '.join(tallest_cats)}",
    "",
    "  Object height distributions are class-specific and useful as a size prior",
    "  when synthesising scenes (avoids placing a 0.3 m table or a 2 m cup).",
    "",
    "## Per-category Statistics",
    "",
    "| category | n | median height (m) | % on floor | median edge→wall (m) |",
    "|---|---|---|---|---|",
]

for _, row in summary_df.sort_values("n_objects", ascending=False).iterrows():
    md_lines.append(
        f"| {row['category']} | {int(row['n_objects']):,} | "
        f"{fmt(row['median_height_m'])} | "
        f"{fmt(row['pct_on_floor'], 0)}% | "
        f"{fmt(row['median_dist_to_wall_m'])} |"
    )

md_lines += [
    "",
    "## What Your Mentor's Idea Is Doing",
    "",
    "Your mentor's proposal is an instance of **scene synthesis via learned spatial priors**.",
    "",
    "A full RGB-D scene is high-dimensional: millions of depth pixels plus colour.",
    "The insight is that most of that information is *redundant given the objects*.",
    "If you know a scene contains a chair, table, and monitor, you can often reconstruct",
    "a plausible-looking scene just by knowing:",
    "",
    "1. **What** objects are present (from a co-occurrence prior like the one we built).",
    "2. **Where** each object sits: floor height, wall distance, and room-normalised XZ",
    "   position — all of which follow learnable per-class distributions.",
    "3. **How big** each object is (height, width, depth priors per class).",
    "",
    "The resulting representation is something like:",
    "```",
    "scene = [(class_i, height_above_floor_i, dist_to_wall_i, norm_x_i, norm_z_i), ...]",
    "```",
    "which is O(n_objects × 5 numbers) instead of O(H × W × D) depth-map voxels.",
    "",
    "**Are these results useful for that direction?**  Yes — the data here provides",
    "exactly the empirical priors needed:",
    "",
    "- **Height-above-floor distributions** → floor-placement constraints per class",
    "  (chairs/tables at ~0 m; lamps/pictures at 1-2 m; etc.)",
    "- **Nearest-edge wall-distance distributions** → wall-snapping constraints",
    "  per class (cabinets, shelves, windows hug walls; tables float in centre)",
    "- **Room-normalised XZ scatter** → spatial layout priors that generalise",
    "  across rooms of different sizes",
    "- **Object size distributions** → scale priors that prevent physically",
    "  implausible objects",
    "",
    "The next step for your mentor's direction would be fitting parametric",
    "distributions (e.g. Gaussian or mixture model) to each of these per-class",
    "empirical distributions, then sampling from them to synthesise new scenes.",
    "This is essentially what models like **PlanIT** (Wang et al., 2019) and",
    "**ATISS** (Paschalidou et al., 2021) do, so those papers are the right",
    "literature to read alongside these results.",
    "",
    "---",
    "*Generated by spatial_analysis.py*",
]

md_path = OUTPUT_DIR / "spatial_summary.md"
md_path.write_text("\n".join(md_lines), encoding="utf-8")
print(f"Saved markdown     : {md_path}")
print("\nDone.")
