# -*- coding: utf-8 -*-
"""
incremental_splits.py
---------------------
Evaluates four continual-learning data splits across two strategies and two
granularities over the top-24 SUN RGB-D object classes.

STRATEGIES
  Semantic   — classes grouped by room type (bedroom / office / kitchen).
               Groupings are motivated by Jaccard co-occurrence values.
               Stage order follows semantic progression, not frequency.
               NOTE: under the strict filter this produces low survival for
               early stages because high-frequency classes (window 28%,
               door 25%) land in the last stage and become forbidden
               for all earlier stages.

  Survival   — classes ordered strictly by descending scene frequency.
               Most frequent classes go first so they are never "future
               forbidden" for any later stage.  Maximises scene counts at
               the cost of semantic coherence within each stage.

GRANULARITIES
  A — 3 stages × 8 classes each
  B — 6 stages × 4 classes each (finer-grained version of A)

WHY 24 NOT 25:
  All prior analysis scripts used TOP_N=25.  The 25th class is 'sign'
  (535 scenes, 5.2%).  It is dropped here as a practical equal-stage
  design choice: 24 is divisible by both 8 (for 3-stage splits) and 4
  (for 6-stage splits); 25 is not.  This is arithmetical convenience,
  not a finding from the analysis.

Top-24 classes (by scene frequency, wall/floor/ceiling excluded):
  chair, table, window, door, cabinet, shelf, desk, box, picture, book,
  paper, lamp, sofa, pillow, bed, monitor, curtain, bottle, bag, mirror,
  counter, keyboard, light, cup

STRICT FILTER (unchanged throughout):
  A scene survives Stage N if and only if:
    (a) at least one Stage N class is present       [same as SDCoT]
    (b) zero classes from any Stage N+1, N+2, …    [stricter than SDCoT]
  Condition (b) is what creates the large survival losses for early stages.

Outputs (three CSVs):
  incremental_splits_semantic.csv   — surviving scenes for Sem-A and Sem-B
  incremental_splits_survival.csv   — surviving scenes for Surv-A and Surv-B
  incremental_splits_all.csv        — all four splits combined
"""

import json
import pathlib
import re
import sys

import pandas as pd

# Force UTF-8 output on Windows (cp1252 console chokes on em-dashes, ×, etc.)
# ENGINEERING DECISION: reconfigure at startup rather than escaping every
# print — one line of setup vs. dozens of substitutions, and the stage names
# in the data structures remain readable as-written.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CSV_PATH     = pathlib.Path(r"C:\sunrgbd-exploration\class_frequencies.csv")
DATASET_ROOT = pathlib.Path(r"C:\Users\ryana\Downloads\SUNRGBD\SUNRGBD")
OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")

EXCLUDE = {"wall", "floor", "ceiling"}
TOP_N   = 24
# All prior scripts used TOP_N=25.  'sign' (rank 25, 535 scenes, 5.2%) is
# dropped here solely because 24 divides evenly into 8 (3-stage) and 4
# (6-stage), while 25 does not.  This is a practical equal-stage design
# choice — it is not derived from any prior analysis result.

UNVIABLE_THRESHOLD = 50   # flag stages with fewer surviving scenes than this

# ---------------------------------------------------------------------------
# Synonym map — MUST stay identical to cooccurrence.py
# PATTERN: "canonical form + variants" table.  Both scripts share the same
# normalisation contract so that class names match across runs.
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
    """Apply strip/lower, synonym map, and numeric-suffix stripping."""
    name = raw.strip().lower()
    if name in _SYNONYMS:
        return _SYNONYMS[name]
    m = _STRIP_SUFFIX.match(name)
    if m:
        base = m.group(1)
        return _SYNONYMS.get(base, base)
    return name

# ---------------------------------------------------------------------------
# Step 2 — Load classes and define splits
# ---------------------------------------------------------------------------

freq_df = pd.read_csv(CSV_PATH)

# PATTERN: filter-before-slice.  Exclude first, then head(N) so we always
# get exactly TOP_N classes regardless of where excluded items ranked.
selected_classes: list[str] = (
    freq_df
    .loc[~freq_df["class"].isin(EXCLUDE), "class"]
    .head(TOP_N)
    .tolist()
)

print("=" * 70)
print(f"TOP {TOP_N} CLASSES  (wall/floor/ceiling excluded; 'sign' rank-25 dropped for arithmetic divisibility)")
print("=" * 70)
for i, c in enumerate(selected_classes, 1):
    row = freq_df.loc[freq_df["class"] == c].iloc[0]
    print(f"  {i:2d}. {c:<12s}  {row['scene_frequency']:,} scenes  ({row['pct_of_scenes']:.1f}%)")

# ── SPLIT A: 3 stages × 8 classes ──────────────────────────────────────────
#
# Grouping rationale (supported by Jaccard values from cooccurrence_jaccard.csv):
#
# Stage 1 — Bedroom / Lounge  (sleep + comfort furnishings)
#   Classes: bed, pillow, lamp, curtain, mirror, picture, sofa, light
#
#   These 8 classes share the home-comfort semantic context.  Key Jaccard pairs:
#     bed  ↔ pillow  : 0.369  — highest pair in entire matrix
#     pillow ↔ lamp  : 0.295  — bedside lamp + pillow
#     bed  ↔ lamp    : 0.288
#     sofa ↔ pillow  : 0.228  — living-room overlap (decorative pillows on sofa)
#     picture ↔ lamp : 0.207  — bedroom/lounge wall art near light source
#     picture ↔ pillow: 0.184 — soft-furnishing + wall art
#     mirror ↔ bed   : 0.146  — dressing area
#     curtain ↔ pillow: 0.125 — curtained bedroom window
#   None of these classes have strong affinity for desks, monitors, or counters,
#   making them a coherent "Stage 1" that is semantically distant from Stage 2/3.
#
# Stage 2 — Office / Study  (work surfaces + knowledge objects)
#   Classes: desk, monitor, keyboard, chair, table, shelf, book, paper
#
#   Key Jaccard pairs:
#     monitor ↔ keyboard : 0.451  — strongest pair in the matrix; almost always
#                                    annotated together in office scenes
#     desk ↔ monitor    : 0.203
#     desk ↔ keyboard   : 0.176
#     chair ↔ desk      : 0.193
#     chair ↔ table     : 0.519  — second-highest pair; dining + classroom scenes
#     shelf ↔ book      : 0.216
#     paper ↔ book      : 0.203
#   These 8 classes collectively cover every SUN RGB-D "work or study" scene.
#
# Stage 3 — Kitchen / Utility / Circulation  (functional + transitional)
#   Classes: counter, cabinet, bottle, cup, box, bag, window, door
#
#   Key Jaccard pairs:
#     cabinet ↔ counter : 0.240  — kitchen surfaces
#     bottle ↔ counter  : 0.168
#     cup    ↔ counter  : 0.111
#     bottle ↔ cabinet  : 0.146
#   window (27.8%) and door (24.9%) are placed here on semantic grounds:
#   they are architectural elements with no strong affinity to bedroom or office
#   objects (Jaccard with any Stage 1/2 class < 0.12).
#   IMPORTANT — strict-filter consequence: because they land in Stage 3, they
#   are "future classes" forbidden in Stage 1 and Stage 2.  Placing them LAST
#   MINIMISES Stage 1/2 survival — the opposite of what the previous version of
#   this comment said.  Putting high-frequency classes in the LAST stage
#   maximises the contamination barrier for all earlier stages.
#   This is the semantic split's honest cost: good groupings, low survival.
#   See the survival-oriented splits (SPLIT_A_SURV / SPLIT_B_SURV) below for
#   the contrast.

# PATTERN: OrderedDict-like structure via plain dict (Python 3.7+ preserves
# insertion order).  Stage ordering is critical: index position determines
# which stages are "future" for each stage N.
SPLIT_A: dict[str, list[str]] = {
    "Stage 1 – Bedroom/Lounge":  ["bed", "pillow", "lamp", "curtain",
                                   "mirror", "picture", "sofa", "light"],
    "Stage 2 – Office/Study":    ["desk", "monitor", "keyboard", "chair",
                                   "table", "shelf", "book", "paper"],
    "Stage 3 – Kitchen/Utility": ["counter", "cabinet", "bottle", "cup",
                                   "box", "bag", "window", "door"],
}

# ── SPLIT B: 6 stages × 4 classes ──────────────────────────────────────────
#
# Each Split A stage is divided into two finer-grained stages.  The
# within-stage ordering preserves the highest-Jaccard pairs together.
#
# Stage 1 — Bedroom core         bed, pillow, lamp, curtain
#   The four strongest-linked bedroom objects (top-3 Jaccard pairs all live
#   here).  A model trained on Stage 1 sees the canonical "bedroom look".
#
# Stage 2 — Bedroom accessories  mirror, picture, sofa, light
#   Softer bedroom/lounge signals.  sofa + pillow Jaccard (0.228) motivated
#   putting sofa in the bedroom half; light and mirror are accessories.
#   After Stage 1, the model already knows bedroom context, so Stage 2 adds
#   supplementary objects without changing the scene type.
#
# Stage 3 — Office core          monitor, keyboard, desk, chair
#   The highest-Jaccard cluster in the dataset: monitor+keyboard=0.451.
#   desk+monitor=0.203, chair+desk=0.193.  These 4 define "office scene".
#
# Stage 4 — Study / library      table, shelf, book, paper
#   Shared between offices and classrooms/libraries.  chair+table=0.519 means
#   chair (Stage 3) appears heavily in Stage 4 scenes, which the strict filter
#   *does not penalise* (only future stages are forbidden).  shelf+book=0.216,
#   paper+book=0.203.
#
# Stage 5 — Kitchen              counter, cabinet, bottle, cup
#   cabinet+counter=0.240, bottle+counter=0.168.  Coherent kitchen cluster.
#
# Stage 6 — Storage / circulation  box, bag, window, door
#   Transitional and storage objects placed last on semantic grounds.
#   Strict-filter consequence (corrected from prior version): placing window
#   and door in Stage 6 makes them forbidden in Stages 1-5.  Window (27.8%)
#   and door (24.9%) appearing in Stage 6 HURTS earlier-stage survival.
#   This is the semantic split's cost, not a benefit.

SPLIT_B: dict[str, list[str]] = {
    "Stage 1 – Bedroom core":        ["bed", "pillow", "lamp", "curtain"],
    "Stage 2 – Bedroom accessories": ["mirror", "picture", "sofa", "light"],
    "Stage 3 – Office core":         ["monitor", "keyboard", "desk", "chair"],
    "Stage 4 – Study/library":       ["table", "shelf", "book", "paper"],
    "Stage 5 – Kitchen":             ["counter", "cabinet", "bottle", "cup"],
    "Stage 6 – Storage/circulation": ["box", "bag", "window", "door"],
}

# ── SURVIVAL-ORIENTED SPLIT A: 3 stages × 8 classes ────────────────────────
#
# Principle: order classes by DESCENDING scene frequency so that the most
# ubiquitous classes land in Stage 1 and are therefore never "future forbidden"
# classes for any later stage.  This minimises the contamination barrier.
#
# Under the strict filter, a class in Stage N+1 is forbidden in Stage N scenes.
# Placing chair (57%) and window (28%) in Stage 3 would eliminate the majority
# of Stage 1 and Stage 2 scenes.  Placing them in Stage 1 means they are never
# forbidden anywhere.
#
# Frequency-descending order (from class_frequencies.csv):
#   chair 57%, table 46%, window 28%, door 25%, cabinet 19%, shelf 16%,
#   desk 15%, box 15%, picture 14%, book 14%, paper 13%, lamp 13%,
#   sofa 13%, pillow 12%, bed 11%, monitor 9%, curtain 8%, bottle 8%,
#   bag 8%, mirror 7%, counter 7%, keyboard 7%, light 6%, cup 6%
#
# Stage 1 gets the 8 most frequent classes → they are never forbidden.
# Stage 2's forbidden classes are Stage 3's 8 least-frequent classes (max 8%).
# Stage 3 has no forbidden classes (last stage → 100% survival by definition).
#
# Trade-off: Stage 1 mixes structural (window, door), seating (chair, table),
# and storage (cabinet, shelf, box) objects — semantically heterogeneous.

SPLIT_A_SURV: dict[str, list[str]] = {
    "Stage 1 – High-freq (structural/furniture)": [
        "chair", "table", "window", "door", "cabinet", "shelf", "desk", "box"],
    "Stage 2 – Mid-freq (decor/study objects)": [
        "picture", "book", "paper", "lamp", "sofa", "pillow", "bed", "monitor"],
    "Stage 3 – Low-freq (kitchen/accessories)": [
        "curtain", "bottle", "bag", "mirror", "counter", "keyboard", "light", "cup"],
}

# ── SURVIVAL-ORIENTED SPLIT B: 6 stages × 4 classes ────────────────────────
#
# Same principle applied at finer granularity: each block of 4 is sorted
# by frequency within the same descending order.  Stage 1 = top-4 by freq,
# Stage 6 = bottom-4.

SPLIT_B_SURV: dict[str, list[str]] = {
    "Stage 1 – Top-freq 1-4":    ["chair", "table", "window", "door"],
    "Stage 2 – Top-freq 5-8":    ["cabinet", "shelf", "desk", "box"],
    "Stage 3 – Mid-freq 9-12":   ["picture", "book", "paper", "lamp"],
    "Stage 4 – Mid-freq 13-16":  ["sofa", "pillow", "bed", "monitor"],
    "Stage 5 – Low-freq 17-20":  ["curtain", "bottle", "bag", "mirror"],
    "Stage 6 – Low-freq 21-24":  ["counter", "keyboard", "light", "cup"],
}

# Print all four proposed groupings
print()
print("=" * 70)
print("SEMANTIC SPLIT A — 3 stages x 8 classes  (bedroom / office / kitchen)")
print("=" * 70)
for stage, classes in SPLIT_A.items():
    print(f"  {stage}: {', '.join(classes)}")

print()
print("=" * 70)
print("SEMANTIC SPLIT B — 6 stages x 4 classes  (finer-grained semantic)")
print("=" * 70)
for stage, classes in SPLIT_B.items():
    print(f"  {stage}: {', '.join(classes)}")

print()
print("=" * 70)
print("SURVIVAL SPLIT A — 3 stages x 8 classes  (frequency-descending)")
print("=" * 70)
for stage, classes in SPLIT_A_SURV.items():
    print(f"  {stage}: {', '.join(classes)}")

print()
print("=" * 70)
print("SURVIVAL SPLIT B — 6 stages x 4 classes  (frequency-descending)")
print("=" * 70)
for stage, classes in SPLIT_B_SURV.items():
    print(f"  {stage}: {', '.join(classes)}")

# ---------------------------------------------------------------------------
# Step 3 — Build scene-level class presence matrix
# ---------------------------------------------------------------------------
# PATTERN: collect rows as a list of dicts, then call pd.DataFrame(records)
# once at the end.  Growing a DataFrame inside a loop via pd.concat is O(n²)
# in total time; deferred materialisation is O(n).
# "scene" is the path to the scene root (parent of annotation2Dfinal/).

selected_set = set(selected_classes)

print()
print("=" * 70)
print("STEP 3 — Scanning annotation files...")
print("=" * 70)
annotation_files = sorted(DATASET_ROOT.rglob("annotation2Dfinal/index.json"))
total_scenes = 0
parse_errors = 0
records = []

for ann_path in annotation_files:
    try:
        with ann_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        parse_errors += 1
        continue

    total_scenes += 1

    # Set intersection: only keep target classes; normalise handles synonyms
    raw_names = {
        normalise(obj["name"])
        for obj in (data.get("objects") or [])
        if obj and isinstance(obj.get("name"), str) and obj["name"].strip()
    }
    present = raw_names & selected_set

    # Scene root = two levels up from annotation2Dfinal/index.json
    scene_root = str(ann_path.parent.parent)

    row: dict = {"scene": scene_root}
    for cls in selected_classes:
        row[cls] = 1 if cls in present else 0
    records.append(row)

print(f"  Scenes processed : {total_scenes:,}")
print(f"  Parse errors     : {parse_errors:,}")

# PATTERN: labelled DataFrame with binary columns.  Downstream code can do
# presence_df["chair"].sum() instead of tracking indices manually.
presence_df = pd.DataFrame(records)
print(f"  Matrix shape     : {presence_df.shape}  "
      f"(rows=scenes, cols=scene + {TOP_N} class flags)")

# Quick sanity check: scene frequencies should roughly match class_frequencies.csv
print()
print("  Scene-frequency sanity check (from presence matrix):")
for cls in selected_classes[:6]:  # spot-check top 6
    n = presence_df[cls].sum()
    print(f"    {cls:<12s}: {n:,} scenes")

# ---------------------------------------------------------------------------
# Step 4 & 5 — Apply strict filter and report
# ---------------------------------------------------------------------------
# For stage N:
#   has_stage  = scenes with ≥1 Stage N class
#   no_future  = scenes with 0 classes from stages N+1, N+2, …
#   surviving  = has_stage AND no_future
#
# PATTERN: compute the "future classes" set once per stage outside any inner
# loop (precomputing loop-invariant data).  This is O(n_stages) set-union
# operations rather than O(n_stages × n_scenes).

SPLITS: dict[str, dict[str, list[str]]] = {
    # NOTE: Sem-A and Sem-B are the original (current) splits from the first
    # version of this script.  The groupings are unchanged; only the comments
    # were corrected to accurately describe the strict-filter consequences.
    "Sem-A":  SPLIT_A,       # semantic, 3 stages x 8
    "Sem-B":  SPLIT_B,       # semantic, 6 stages x 4
    "Surv-A": SPLIT_A_SURV,  # survival-oriented, 3 stages x 8
    "Surv-B": SPLIT_B_SURV,  # survival-oriented, 6 stages x 4
}

# PATTERN: collect (split, stage, total, survived, pct_lost) as we go so the
# comparison table does not need to re-run the filter a second time.
# Re-running would be redundant and could silently diverge if SPLITS changed.
csv_rows: list[dict] = []
stage_stats: list[dict] = []   # for the summary table at the end

print()
print("=" * 70)
print("STEPS 4 & 5 — STRICT FILTER + RESULTS")
print("=" * 70)

unviable_stages: list[tuple[str, str, int]] = []

for split_name, split_def in SPLITS.items():
    stage_list        = list(split_def.items())   # [(label, [classes]), …]
    n_stages          = len(stage_list)
    stage_class_lists = [cls_list for _, cls_list in stage_list]

    print()
    print(f"  SPLIT {split_name}  ({'x'.join(str(len(c)) for _,c in stage_list)} "
          f"= {sum(len(c) for _,c in stage_list)} classes across {n_stages} stages)")
    print("  " + "-" * 66)

    for stage_idx, (stage_label, stage_cls) in enumerate(stage_list):
        stage_set = set(stage_cls)

        # Future classes: all classes belonging to stages N+1, N+2, …
        # PRECOMPUTED OUTSIDE THE SCENE LOOP — O(n_stages) not O(n_stages × n_scenes)
        future_classes: set[str] = set()
        for j in range(stage_idx + 1, n_stages):
            future_classes |= set(stage_class_lists[j])

        # ── Condition (a): at least one Stage N class present ────────────
        has_stage = presence_df[list(stage_set)].sum(axis=1) > 0

        # ── Condition (b): no future-stage class present ─────────────────
        # This is the strict part.  SDCoT only enforces (a).
        if future_classes:
            no_future = presence_df[sorted(future_classes)].sum(axis=1) == 0
        else:
            no_future = pd.Series(True, index=presence_df.index)

        total_with_stage = int(has_stage.sum())
        surviving_mask   = has_stage & no_future
        surviving        = int(surviving_mask.sum())
        pct_lost = (
            100.0 * (1 - surviving / total_with_stage)
            if total_with_stage > 0 else float("nan")
        )

        flag = "  *** UNVIABLE ***" if surviving < UNVIABLE_THRESHOLD else ""
        if surviving < UNVIABLE_THRESHOLD:
            unviable_stages.append((split_name, stage_label, surviving))

        # Accumulate stats for the summary table
        stage_stats.append({
            "split":    split_name,
            "stage":    stage_label,
            "classes":  ", ".join(stage_cls),
            "total":    total_with_stage,
            "survived": surviving,
            "pct_lost": pct_lost,
        })

        print(f"\n  {stage_label}")
        print(f"    Classes           : {', '.join(stage_cls)}")
        if future_classes:
            print(f"    Forbidden (future): {', '.join(sorted(future_classes))}")
        else:
            print(f"    Forbidden (future): (none — last stage)")
        print(f"    Total scenes with >=1 stage class : {total_with_stage:>6,}")
        print(f"    Surviving strict filter            : {surviving:>6,}{flag}")
        print(f"    Percentage lost                    : {pct_lost:>6.1f}%")

        # Collect CSV rows for surviving scenes
        for _, scene_row in presence_df[surviving_mask].iterrows():
            present_here = [c for c in selected_classes if scene_row[c] == 1]
            csv_rows.append({
                "split":           split_name,
                "stage":           stage_label,
                "scene":           scene_row["scene"],
                "classes_present": "|".join(present_here),
            })

# ---------------------------------------------------------------------------
# Step 5 (cont.) — Save CSVs
# ---------------------------------------------------------------------------
# Save three CSVs: per strategy family + combined.
# "Per family" means the full set of surviving scenes for each strategy,
# which is what a downstream training script would consume.

all_df  = pd.DataFrame(csv_rows, columns=["split", "stage", "scene", "classes_present"])
sem_df  = all_df[all_df["split"].str.startswith("Sem")]
surv_df = all_df[all_df["split"].str.startswith("Surv")]

path_all  = OUTPUT_DIR / "incremental_splits_all.csv"
path_sem  = OUTPUT_DIR / "incremental_splits_semantic.csv"
path_surv = OUTPUT_DIR / "incremental_splits_survival.csv"

all_df.to_csv(path_all,   index=False)
sem_df.to_csv(path_sem,   index=False)
surv_df.to_csv(path_surv, index=False)

print()
print("=" * 70)
print("SAVED CSVs")
print("=" * 70)
print(f"  {path_sem}   ({len(sem_df):,} rows)")
print(f"  {path_surv}  ({len(surv_df):,} rows)")
print(f"  {path_all}   ({len(all_df):,} rows, all four splits)")

# ---------------------------------------------------------------------------
# Step 6 — Viability check
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("STEP 6 — VIABILITY CHECK  (threshold: >= 50 scenes per stage)")
print("=" * 70)
if not unviable_stages:
    print("  All stages meet the 50-scene minimum.")
else:
    print(f"  {len(unviable_stages)} stage(s) below threshold:")
    for spl, stage, count in unviable_stages:
        print(f"    {spl} | {stage} : {count} scenes")

# ---------------------------------------------------------------------------
# Final summary table — suitable for sharing with mentor
# ---------------------------------------------------------------------------
# PATTERN: build from stage_stats collected during the filter loop above,
# not from a second pass over presence_df.  Single-pass consistency means
# the table is guaranteed to match the per-stage printout above.

print()
print("=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)
print()

strategies = [
    ("SEMANTIC SPLITS (room-type grouping, Jaccard-motivated)",   ["Sem-A",  "Sem-B"]),
    ("SURVIVAL SPLITS (frequency-descending, maximise scene counts)", ["Surv-A", "Surv-B"]),
]

for strategy_label, split_keys in strategies:
    print(f"  {strategy_label}")
    print(f"  {'Split':<8} {'Stage':<36} {'Total':>6} {'Survive':>8} {'Lost%':>7}")
    print("  " + "-" * 62)
    for row in stage_stats:
        if row["split"] not in split_keys:
            continue
        flag = " (*)" if row["survived"] < UNVIABLE_THRESHOLD else ""
        print(f"  {row['split']:<8} {row['stage']:<36} "
              f"{row['total']:>6,} {row['survived']:>8,} {row['pct_lost']:>6.1f}%{flag}")
    print()

print("  (*) = fewer than 50 scenes; stage is unviable for training")
print()
print("  INTERPRETATION")
print("  " + "-" * 62)
print("  Under the strict filter, a scene is accepted for Stage N only if")
print("  it contains >=1 Stage N class AND zero classes from any later stage.")
print("  Semantic splits: early stages lose 90%+ of scenes because high-")
print("  frequency classes (window 28%, door 25%, chair 57%) land in later")
print("  stages and become forbidden for all earlier stages.")
print("  Survival splits: losses are more balanced because the most frequent")
print("  classes go first and are never forbidden for any later stage.")
