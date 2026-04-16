"""
class_frequency.py
------------------
Scans every scene in the SUN RGB-D dataset and counts how many scenes each
object class appears in ("scene frequency").  A class is counted once per
scene even if it has multiple instances in that scene, so the number tells
you "in how many scenes will I encounter this class?" -- the right metric
for choosing which classes are worth modelling.

Outputs:
  class_frequencies.csv   -- full table, sorted by scene_frequency desc
  class_frequencies.png   -- bar chart of the top 50 classes
"""

import json
import pathlib
import collections

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# 1. Locate every annotation2Dfinal/index.json file in the dataset
# ---------------------------------------------------------------------------
# Using rglob("annotation2Dfinal/index.json") finds all annotation files
# regardless of subdataset nesting depth (handles the extra level in sun3ddata).

DATASET_ROOT = pathlib.Path(r"C:\Users\ryana\Downloads\SUNRGBD\SUNRGBD")
OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")

# ---------------------------------------------------------------------------
# Synonym map: variant name -> canonical name
# ---------------------------------------------------------------------------
# SUN RGB-D annotations are crowd-sourced and have no enforced vocabulary, so
# the same object gets many different labels.  Without merging, the frequency
# table fragments one real class into many low-count variants, causing two
# problems:
#   1. The true scene-frequency of a concept (e.g. "sofa") is underestimated
#      because scenes labelled "couch" are counted separately.
#   2. The co-occurrence matrix misses relationships: a "couch"+"chair" scene
#      and a "sofa"+"chair" scene should both count toward sofa-chair affinity.
#
# Design choices:
#   - We map variants to a single lowercase canonical name.
#   - We only merge names that refer to the same physical object class.
#     "bookshelf" -> "shelf" is safe; "notebook" -> "book" is NOT (a notebook
#     is a laptop/writing pad, not a bound book).
#   - Numbered suffixes (chair1, chair2) are instance annotations: annotators
#     label individual instances per scene.  Because we deduplicate to a set
#     per scene, chair1 and chair2 both map to "chair" and contribute only once.
#   - Plurals (curtains, pillows) are unambiguously the same class.
#
# This is called a "label normalisation" or "class consolidation" step and is
# standard pre-processing for any dataset with free-text annotations.

import re as _re

# Build the synonym lookup.  Keys are raw annotation names (after strip/lower);
# values are the canonical name to count under.
_SYNONYMS: dict[str, str] = {}

def _add(canonical: str, *variants: str) -> None:
    """Register all variants (and the canonical itself) in _SYNONYMS."""
    _SYNONYMS[canonical] = canonical
    for v in variants:
        _SYNONYMS[v] = canonical

# Seating
_add("sofa",    "couch", "sofachair", "sofas", "couches")
_add("chair",   "armchair", "chairs")

# Surfaces
_add("shelf",   "shelves", "bookshelf", "bookshelves", "shelve", "bookshelve")
_add("counter", "countertop", "countertops")
_add("desk",    "desktable")

# Storage
_add("cabinet", "cupboard", "cupboards", "cabinets", "filingcabinet",
     "filecabinet", "storagecabinet")

# Display / computing
_add("monitor", "computermonitor", "pcmonitor", "monitors")

# Soft furnishings
_add("pillow",  "pillows", "cushion", "cushions", "throwpillow", "throwpillows")
_add("curtain", "curtains", "windowcurtain")

# Books (singular/plural only — "notebook" is NOT merged: it means laptop/pad)
_add("book",    "books")

# Lighting
_add("lamp",    "tablelamp", "desklamp", "floorlamp", "nightlamp", "walllamp")

# Windows / blinds
_add("window",  "windows")
_add("blinds",  "blind", "windowblinds", "windowblind")

# Misc plurals / spelling variants
_add("bottle",  "bottles")
_add("bag",     "bags")
_add("mirror",  "mirrors")
_add("door",    "doors")
_add("table",   "tables")
_add("bed",     "beds")
_add("box",     "boxes")

# Numbered instance suffixes: chair1, chair2, sofa3, window1, …
# These are handled dynamically in the normalisation function below.
_STRIP_SUFFIX = _re.compile(r"^(.+?)\d+$")

def normalise(raw: str) -> str:
    """
    Apply strip/lower then synonym mapping to a raw annotation name.

    Pattern used:
      1. strip + lower  (already done at call site, but safe to repeat)
      2. look up in _SYNONYMS  (explicit variant -> canonical)
      3. strip trailing digits (chair1 -> chair, then re-check _SYNONYMS)
    """
    name = raw.strip().lower()
    if name in _SYNONYMS:
        return _SYNONYMS[name]
    # Strip trailing digit suffix and try again (handles chair1, sofa2, …)
    m = _STRIP_SUFFIX.match(name)
    if m:
        base = m.group(1)
        if base in _SYNONYMS:
            return _SYNONYMS[base]
        return base   # e.g. "lamp3" -> "lamp" even if not in synonym map
    return name

print("Scanning for annotation files…")
annotation_files = sorted(DATASET_ROOT.rglob("annotation2Dfinal/index.json"))
print(f"  Found {len(annotation_files):,} scenes")

# ---------------------------------------------------------------------------
# 2. Read each annotation file and record which classes appear in that scene
# ---------------------------------------------------------------------------
# The JSON structure is:
#   { "objects": [{"name": "chair"}, {"name": "table"}, ...], ... }
#
# We build a set of class names per scene (deduplicating multiple instances),
# then increment a counter for each class in that set.  This gives
# scene-frequency rather than instance-frequency.

scene_frequency: collections.Counter = collections.Counter()
scenes_with_no_objects = 0
parse_errors = 0

for i, ann_path in enumerate(annotation_files):
    if i % 1000 == 0:
        print(f"  Processing scene {i:,} / {len(annotation_files):,} …")

    try:
        with ann_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        parse_errors += 1
        continue

    # Extract the unique set of canonical class names in this scene.
    # normalise() applies strip/lower AND the synonym map, so "couch" and
    # "sofa" both become "sofa", and "chair1"/"chair2" both become "chair".
    # Because we build a set, duplicates after normalisation are collapsed
    # automatically -- a scene with both "chair1" and "chair2" still counts
    # "chair" only once.
    objects = data.get("objects") or []
    classes_in_scene = {
        normalise(obj["name"])
        for obj in objects
        if obj and isinstance(obj.get("name"), str) and obj["name"].strip()
    }

    if not classes_in_scene:
        scenes_with_no_objects += 1
        continue

    scene_frequency.update(classes_in_scene)

print(f"\nDone.")
print(f"  Scenes parsed successfully : {len(annotation_files) - parse_errors:,}")
print(f"  Scenes with no objects     : {scenes_with_no_objects:,}")
print(f"  Parse errors               : {parse_errors:,}")
print(f"  Unique classes found       : {len(scene_frequency):,}")

# ---------------------------------------------------------------------------
# 3. Build a sorted DataFrame
# ---------------------------------------------------------------------------
# Convert the Counter to a DataFrame and sort descending by scene_frequency.
# We also add a "rank" column and a "pct_of_scenes" column so you can see
# at a glance what fraction of the dataset contains each class.

total_scenes = len(annotation_files) - parse_errors

df = pd.DataFrame(
    scene_frequency.most_common(),   # returns list of (class, count) sorted desc
    columns=["class", "scene_frequency"],
)
df.index = df.index + 1              # rank starts at 1
df.index.name = "rank"
df["pct_of_scenes"] = (df["scene_frequency"] / total_scenes * 100).round(2)

print(f"\nTop 20 classes:")
print(df.head(20).to_string())

# ---------------------------------------------------------------------------
# 4. Save the full table as CSV
# ---------------------------------------------------------------------------
csv_path = OUTPUT_DIR / "class_frequencies.csv"
df.to_csv(csv_path)
print(f"\nSaved full frequency table: {csv_path}")

# ---------------------------------------------------------------------------
# 5. Plot a bar chart of the top 50 classes
# ---------------------------------------------------------------------------
TOP_N = 50
top = df.head(TOP_N)

fig, ax = plt.subplots(figsize=(16, 7))

bars = ax.bar(
    range(len(top)),
    top["scene_frequency"],
    color="steelblue",
    edgecolor="white",
    linewidth=0.4,
)

# X-axis: class names rotated so they don't overlap
ax.set_xticks(range(len(top)))
ax.set_xticklabels(top["class"], rotation=55, ha="right", fontsize=8.5)

# Y-axis: raw scene count on left, percentage on right
ax.set_ylabel("Scene frequency (# scenes containing class)", fontsize=11)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

# Secondary y-axis showing percentage of total scenes
ax2 = ax.twinx()
ax2.set_ylim(
    ax.get_ylim()[0] / total_scenes * 100,
    ax.get_ylim()[1] / total_scenes * 100,
)
ax2.set_ylabel("% of all scenes", fontsize=11)
ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))

ax.set_title(
    f"SUN RGB-D — Top {TOP_N} object classes by scene frequency\n"
    f"({total_scenes:,} scenes total, {len(scene_frequency):,} unique classes)",
    fontsize=13,
    pad=14,
)
ax.set_xlim(-0.6, len(top) - 0.4)

# Annotate the count above each bar (only the first 20 to avoid clutter)
for idx, (count, pct) in enumerate(
    zip(top["scene_frequency"], top["pct_of_scenes"])
):
    if idx < 20:
        ax.text(
            idx, count + total_scenes * 0.003,
            f"{count:,}",
            ha="center", va="bottom", fontsize=6.5, color="dimgray",
        )

fig.tight_layout()
chart_path = OUTPUT_DIR / "class_frequencies.png"
fig.savefig(chart_path, dpi=150)
print(f"Saved bar chart: {chart_path}")
plt.show()
