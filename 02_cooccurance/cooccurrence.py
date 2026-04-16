"""
cooccurrence.py
---------------
Builds a class co-occurrence matrix for the top 25 SUN RGB-D object classes
(excluding wall, floor, ceiling) and visualises it as a heatmap.

Co-occurrence is defined at the scene level: cell [i, j] holds the number of
scenes where class i AND class j both appear, regardless of how many instances
of each are present.  This tells you "how often do these two categories share
a room?" -- the right question for understanding object relationships in
indoor scenes.

The normalised matrix uses the Jaccard similarity index:

  J(A, B) = cooccurrence(A, B) / (freq(A) + freq(B) - cooccurrence(A, B))

The denominator is the number of scenes that contain A OR B (at least one of
them).  This means J(A, B) answers: "of all scenes where this pair is
*relevant*, what fraction contain both?"  Dividing by total scenes instead
would give a joint probability P(A ∩ B) that is inflated for any two common
classes even if they have no special affinity for each other -- for example,
chair+table would score high simply because both appear in many scenes
independently, not because they appear together more than you'd expect.
Jaccard controls for that marginal prevalence.  Values range from 0 (never
co-occur) to 1 (always appear together).

Outputs:
  cooccurrence_raw.csv     -- raw scene counts, symmetric matrix
  cooccurrence_jaccard.csv -- Jaccard similarity, controls for class prevalence
  cooccurrence_heatmap.png -- annotated heatmap of Jaccard similarities
"""

import json
import pathlib
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# 1. Load the top-25 classes from the CSV, then apply the exclusion list
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: reading from the pre-computed CSV rather than
# re-running the frequency scan means this script is fast to iterate on and
# the class selection is an explicit, auditable artefact (the CSV) rather than
# a magic constant buried in code.  This is the "single source of truth"
# principle: one place produces the ranking, every other script consumes it.

CSV_PATH     = pathlib.Path(r"C:\sunrgbd-exploration\class_frequencies.csv")
DATASET_ROOT = pathlib.Path(r"C:\Users\ryana\Downloads\SUNRGBD\SUNRGBD")
OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")

EXCLUDE = {"wall", "floor", "ceiling"}   # ubiquitous structural classes
TOP_N   = 25

# ---------------------------------------------------------------------------
# Synonym map  (must be identical to the one in class_frequency.py)
# ---------------------------------------------------------------------------
# See class_frequency.py for the full explanation of why this step is needed.
# Both scripts must use the same mapping so that the canonical names in
# class_frequencies.csv (produced by class_frequency.py) match the names
# produced here during the annotation scan.  If they diverged, selected_classes
# loaded from the CSV could silently fail to match any annotation name.
#
# ENGINEERING DECISION: the map lives in both files rather than a shared
# module because these are standalone analysis scripts, not a package.
# The duplication is intentional and acceptable; the comment above makes the
# coupling explicit so both copies stay in sync when the map is updated.

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

freq_df = pd.read_csv(CSV_PATH)

# Filter EXCLUDE first, then take the top TOP_N from what remains.
# ENGINEERING DECISION: the order matters here.  The old approach was:
#   head(TOP_N) → filter EXCLUDE
# That would remove up to 3 slots (wall, floor, ceiling all rank in the top 5)
# and leave only 22 classes.  The correct approach is:
#   filter EXCLUDE → head(TOP_N)
# This guarantees exactly TOP_N classes regardless of where the excluded ones
# fall in the ranking.  This is called "filter-before-slice" and is the safer
# default whenever the filtered items could consume slots you intend to keep.
#
# NOTE: class_frequency.py normalises all names with .strip().lower() before
# writing the CSV, so `selected_classes` is already in the same form as the
# names we extract from the JSON annotations below.  This consistency is
# important -- a mismatch (e.g. "Chair" vs "chair") would cause silent misses
# where an object is present in the scene but not counted.
selected_classes = (
    freq_df
    .loc[~freq_df["class"].isin(EXCLUDE), "class"]
    .head(TOP_N)
    .tolist()
)

print(f"Selected {len(selected_classes)} classes (top {TOP_N} after excluding {EXCLUDE}):")
for i, c in enumerate(selected_classes, 1):
    print(f"  {i:2d}. {c}")

# ---------------------------------------------------------------------------
# 2. Scan every scene annotation and record which selected classes are present
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: we convert `selected_classes` to a Python set before
# the loop.  Set membership tests ("is X in this collection?") are O(1) --
# they take the same time regardless of how many items are in the set.  If we
# used a list instead, every membership test would be O(n), scanning the whole
# list.  With 10k+ scenes and ~20 classes to check per scene, that difference
# is small here, but the habit of using sets for membership tests is important.
# This is called "constant-time lookup" and is one of the most common
# performance patterns in data processing code.

selected_set = set(selected_classes)
n = len(selected_classes)

# Build a mapping from class name to its integer index in our matrix.
# A dict lookup is also O(1).  This pattern -- "map items to indices, work
# with indices in the hot loop, convert back to names at the end" -- is called
# "index encoding" and keeps the matrix operations clean and fast.
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

# Pre-allocate the co-occurrence matrix and per-class scene frequency counter.
# ENGINEERING DECISION: numpy arrays are far more memory-efficient than a
# Python list-of-lists for numerical work.  More importantly, tracking
# scene_freq here -- inside the same scan pass -- guarantees it is derived
# from exactly the same set of scenes as the co-occurrence counts.  Loading
# scene frequencies from class_frequencies.csv would be a source mismatch:
# that CSV was built by a separate run of class_frequency.py, which may have
# encountered a slightly different set of parse errors and therefore counted a
# slightly different number of scenes.  Even one extra scene in the denominator
# inflates the union and deflates every Jaccard score.  Deriving both counts
# from the same loop is called "single-pass consistency" and eliminates the
# entire class of denominator-mismatch bugs.
cooccurrence = np.zeros((n, n), dtype=np.int32)
scene_freq   = np.zeros(n,      dtype=np.int32)   # scene_freq[i] = # scenes containing class i

print("\nScanning annotation files...")
annotation_files = sorted(DATASET_ROOT.rglob("annotation2Dfinal/index.json"))
total_scenes = 0
parse_errors = 0

for ann_path in annotation_files:
    try:
        with ann_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        parse_errors += 1
        continue

    total_scenes += 1

    # Get the unique canonical class names present in this scene.
    # normalise() applies strip/lower AND the synonym map, so "couch" becomes
    # "sofa", "chair1"/"chair2" both become "chair", etc.  Building a set
    # automatically collapses any post-normalisation duplicates.
    # The set intersection (&) then keeps only our target classes.
    raw_names = {
        normalise(obj["name"])
        for obj in (data.get("objects") or [])
        if obj and isinstance(obj.get("name"), str) and obj["name"].strip()
    }
    present      = raw_names & selected_set            # intersection: only keep target classes
    present_list = [class_to_idx[c] for c in present]  # convert to integer indices

    # Increment individual scene frequencies for every present class.
    # ENGINEERING DECISION: we do this BEFORE the early-continue below so that
    # scenes containing only one selected class still contribute to scene_freq.
    # If we tracked frequency only inside the "at least two classes" block, the
    # denominator for Jaccard would be too small for any class that often appears
    # alone, systematically inflating its Jaccard scores with every other class.
    for idx in present_list:
        scene_freq[idx] += 1

    if len(present_list) < 2:
        continue   # need at least two classes to form a pair

    # For every pair of classes that co-occur, increment both [i,j] and [j,i].
    # ENGINEERING DECISION: we increment both directions explicitly rather than
    # filling just the upper triangle and mirroring at the end.  This makes the
    # matrix ready to use immediately (no post-processing step), and the
    # symmetry check in validation becomes a meaningful test rather than a
    # tautology.  The diagonal (class with itself) is intentionally left at
    # zero -- "how often does chair co-occur with chair" is just scene_frequency
    # and would dominate the heatmap colour scale without adding information.
    for a in range(len(present_list)):
        for b in range(a + 1, len(present_list)):
            i, j = present_list[a], present_list[b]
            cooccurrence[i, j] += 1
            cooccurrence[j, i] += 1

print(f"  Scenes processed : {total_scenes:,}")
print(f"  Parse errors     : {parse_errors:,}")

# ---------------------------------------------------------------------------
# 3. Build DataFrames with class names as both row and column labels
# ---------------------------------------------------------------------------
# Using class names as the index/columns means downstream users can do
# df.loc["chair", "table"] instead of having to remember that chair=0.
# This is called "labelled indexing" and is one of the core reasons to use
# pandas rather than raw numpy for analysis work.

raw_df = pd.DataFrame(cooccurrence, index=selected_classes, columns=selected_classes)

# JACCARD NORMALISATION:
# J(A, B) = cooccurrence(A, B) / (freq(A) + freq(B) - cooccurrence(A, B))
#
# scene_freq (tracked in the loop above) provides the per-class denominator.
# We use it directly rather than loading scene_frequency from the CSV because
# it is derived from exactly the same set of parsed scenes as `cooccurrence`.
# See the comment on scene_freq above for the full explanation of why this
# single-pass consistency matters.
#
# ENGINEERING DECISION: we use numpy broadcasting to compute the entire
# denominator matrix in one expression instead of a nested loop.
# freq_arr has shape (n,).
#   freq_arr[:, None]  has shape (n, 1)  -- column vector
#   freq_arr[None, :]  has shape (1, n)  -- row vector
# Adding them gives shape (n, n) where cell [i,j] = freq[i] + freq[j].
# Subtracting `cooccurrence` (also (n,n)) gives the union count per cell.
# This broadcasting pattern is standard in numpy for pairwise computations
# and is much faster than explicit loops.

freq_arr = scene_freq.astype(np.float64)   # shape (n,); derived from same scan pass

# Union count for each pair: |A ∪ B| = freq(A) + freq(B) - cooccurrence(A,B)
union = freq_arr[:, None] + freq_arr[None, :] - cooccurrence   # shape (n, n)

# Compute Jaccard; np.where avoids division by zero on any zero-denominator cell
# (shouldn't happen for real classes, but is good defensive practice).
jaccard = np.where(union > 0, cooccurrence / union, 0.0)

# Explicitly zero the diagonal.
# Mathematically J(A,A) = freq(A) / (2*freq(A) - freq(A)) = 1, but our
# cooccurrence matrix has zeros on the diagonal by design (a class is never
# paired with itself), so np.where would produce 0/freq(A) = 0 anyway.
# The np.fill_diagonal call makes this intent explicit rather than relying on
# a silent side-effect of the construction.  Anyone reading the code or
# querying jaccard_df.loc["chair","chair"] should see 0 and know it means
# "diagonal is zeroed by design", not "chair never appears".
np.fill_diagonal(jaccard, 0.0)

jaccard_df = pd.DataFrame(jaccard, index=selected_classes, columns=selected_classes)

# ---------------------------------------------------------------------------
# 4. Save both matrices as CSV
# ---------------------------------------------------------------------------
raw_path     = OUTPUT_DIR / "cooccurrence_raw.csv"
jaccard_path = OUTPUT_DIR / "cooccurrence_jaccard.csv"
raw_df.to_csv(raw_path)
jaccard_df.round(4).to_csv(jaccard_path)
print(f"\nSaved raw matrix        : {raw_path}")
print(f"Saved Jaccard matrix    : {jaccard_path}")

# ---------------------------------------------------------------------------
# 5. Visualise as an annotated heatmap
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: we plot the Jaccard matrix (0-1 values) rather than
# raw counts or joint probabilities.  Jaccard is directly interpretable as
# "what fraction of scenes containing either class also contain both?" and is
# not inflated by classes that are individually common.  Cell annotations show
# Jaccard as a percentage rounded to one decimal place.

fig, ax = plt.subplots(figsize=(14, 12))

# imshow treats the matrix as a grid of pixels.  `aspect="equal"` forces
# square cells.  `origin="upper"` matches the convention that row 0 is at
# the top, matching the x-axis label order.
im = ax.imshow(jaccard_df.values, aspect="equal", origin="upper",
               cmap="YlOrRd",   # yellow -> orange -> red: intuitive for "more = warmer"
               vmin=0, vmax=jaccard_df.values.max())

# Annotate every cell with its Jaccard value as a percentage.
# ENGINEERING DECISION: we choose the text colour (white vs black) based on
# the cell value relative to the colour scale midpoint.  Dark text on a light
# cell and light text on a dark cell both satisfy minimum contrast ratios.
# This is called "adaptive contrast" and prevents annotations from being
# invisible on extreme-valued cells.
threshold = jaccard_df.values.max() / 2
for row in range(n):
    for col in range(n):
        val = jaccard_df.values[row, col]
        color = "white" if val > threshold else "black"
        ax.text(col, row, f"{val*100:.1f}",
                ha="center", va="center", fontsize=6.5,
                color=color, fontweight="bold" if val > threshold else "normal")

# Axis labels
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(selected_classes, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(selected_classes, fontsize=9)

# Colourbar
cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("Jaccard similarity (%)", fontsize=10)
cbar.ax.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f"{x*100:.0f}%")
)

ax.set_title(
    "SUN RGB-D object co-occurrence  —  Jaccard similarity\n"
    "Cell value = % of scenes containing A or B that also contain both A and B\n"
    "(controls for individual class prevalence; diagonal is zero by construction)",
    fontsize=11, pad=16,
)

fig.tight_layout()
chart_path = OUTPUT_DIR / "cooccurrence_heatmap.png"
fig.savefig(chart_path, dpi=150)
print(f"Saved heatmap           : {chart_path}")
plt.show()

# ---------------------------------------------------------------------------
# 6. Validation checks
# ---------------------------------------------------------------------------
# These are lightweight "sanity tests" -- not full unit tests, but a set of
# assertions based on domain knowledge that would catch major bugs:
# - a wrong annotation file being parsed
# - the normalisation being applied twice or not at all
# - a silent transpose (matrix rows/columns swapped)
# - a scene count wildly outside the expected range
#
# ENGINEERING DECISION: we print PASS/FAIL rather than raising exceptions
# because this is an analysis script, not production code.  A hard crash
# would hide all the output above it; PASS/FAIL lets you see every check and
# decide which failures matter.  In a software project you would use
# pytest assertions instead.

print("\n" + "="*55)
print("VALIDATION")
print("="*55)

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    return condition

all_passed = True

# Check 1: Jaccard(chair, table) > 0.40
# Domain reasoning: chairs and tables co-appear in offices, dining rooms, and
# classrooms.  Based on observed scene frequencies (~5,600 chair scenes,
# ~4,600 table scenes, ~3,400 co-occurring scenes), the expected Jaccard is
# ~0.50.  We use 0.40 as a conservative lower bound.
# NOTE: this threshold is intentionally different from the old "15% of scenes"
# check.  Jaccard conditions on scenes containing at least one of the pair,
# so its values are always >= the old joint-probability values.
if "chair" in jaccard_df.index and "table" in jaccard_df.index:
    val = jaccard_df.loc["chair", "table"]
    all_passed &= check(
        "Jaccard(chair, table) > 0.40",
        val > 0.40,
        f"actual: {val*100:.1f}%  (of scenes with chair or table, this % have both)"
    )
else:
    print("  [SKIP] chair+table check -- one or both classes not in selected set")

# Check 2: Jaccard(chair, desk) > 0.15
# Domain reasoning: chairs and desks are both office/classroom furniture that
# share the same semantic context (work surfaces).  The Jaccard is moderate
# rather than high because chairs also appear in many non-desk contexts
# (dining rooms, waiting areas), diluting the union.  0.15 is a conservative
# lower bound calibrated to the observed value of ~19%.
if "chair" in jaccard_df.index and "desk" in jaccard_df.index:
    val = jaccard_df.loc["chair", "desk"]
    all_passed &= check(
        "Jaccard(chair, desk) > 0.15",
        val > 0.15,
        f"actual: {val*100:.1f}%  (office/classroom pair, moderate affinity due to chair ubiquity)"
    )
else:
    print("  [SKIP] chair+desk check -- one or both classes not in selected set")

# Check 3: Jaccard(sofa, bed) < Jaccard(chair, table)
# Domain reasoning: sofas belong in living rooms; beds belong in bedrooms.
# These are different rooms, so their co-occurrence should be lower than
# chair+table, which share the same dining/office contexts.  This check
# verifies that same-room pairs score higher than cross-room pairs.
if all(c in jaccard_df.index for c in ("sofa", "bed", "chair", "table")):
    val_sb = jaccard_df.loc["sofa", "bed"]
    val_ct = jaccard_df.loc["chair", "table"]
    all_passed &= check(
        "Jaccard(sofa, bed) < Jaccard(chair, table)  [cross-room < same-room]",
        val_sb < val_ct,
        f"Jaccard(sofa,bed)={val_sb*100:.1f}%  Jaccard(chair,table)={val_ct*100:.1f}%"
    )
else:
    print("  [SKIP] sofa/bed/chair/table check -- one or more classes not in selected set")

# Check 4: Jaccard(monitor, keyboard) > 0.35
# Domain reasoning: monitors and keyboards are office peripherals that are
# almost always annotated in the same scene.  Their Jaccard should be high --
# seeing one strongly predicts the other.  The observed value (~40%) is lower
# than intuition might suggest because some scenes have a monitor with no
# visible/annotated keyboard, or a keyboard on a surface with the monitor
# outside frame.  0.35 is a conservative lower bound calibrated to ~40%.
if "monitor" in jaccard_df.index and "keyboard" in jaccard_df.index:
    val = jaccard_df.loc["monitor", "keyboard"]
    all_passed &= check(
        "Jaccard(monitor, keyboard) > 0.35",
        val > 0.35,
        f"actual: {val*100:.1f}%  (office peripherals, expected high affinity)"
    )
else:
    print("  [SKIP] monitor+keyboard check -- one or both classes not in selected set")

# Check 3: perfect symmetry
# The matrix must satisfy M[i,j] == M[j,i] for all i,j.
# np.allclose checks element-wise equality with a small floating-point
# tolerance (important because floating-point arithmetic is not exactly
# associative -- tiny rounding differences can make == return False even
# for values that are logically equal).  Here the raw matrix is integer so
# we check that directly; the symmetry should be exact.
is_symmetric = np.array_equal(cooccurrence, cooccurrence.T)
all_passed &= check(
    "Matrix is perfectly symmetric (M[i,j] == M[j,i])",
    is_symmetric,
    "checked on raw integer matrix"
)

# Check 4: scene count in expected range
all_passed &= check(
    "Total scenes processed is between 10,000 and 10,500",
    10_000 <= total_scenes <= 10_500,
    f"actual: {total_scenes:,}"
)

print("="*55)
print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED -- review output above")
