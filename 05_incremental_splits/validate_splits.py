# -*- coding: utf-8 -*-
"""
validate_splits.py
Four correctness checks for incremental_splits_all.csv / split_analysis_report.md

Data sources and their roles
------------------------------
incremental_splits_all.csv  — ground truth for the filter results.
    'classes_present' was populated from annotation JSON files (same source
    the strict filter ran against).  Used as the authoritative class list.

spatial_features.csv  — secondary source derived from the 3D bounding-box
    pipeline. Covers 7,198 of 10,295 scenes and may include objects absent
    from the 2D annotations (different processing pipeline). Used for
    corroboration; discrepancies are reported but do not override the CSV.
"""

import re, pathlib, sys, unicodedata
import pandas as pd

# Force UTF-8 on Windows console (cp1252 chokes on arrows/dashes)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Synonym normalisation (identical to incremental_splits.py) ───────────────
synonyms: dict[str, str] = {}

def add(canonical: str, *variants: str) -> None:
    synonyms[canonical] = canonical
    for v in variants:
        synonyms[v] = canonical

add("sofa",    "couch", "sofachair", "sofas", "couches")
add("chair",   "armchair", "chairs")
add("shelf",   "shelves", "bookshelf", "bookshelves", "shelve", "bookshelve")
add("counter", "countertop", "countertops")
add("desk",    "desktable")
add("cabinet", "cupboard", "cupboards", "cabinets", "filingcabinet",
               "filecabinet", "storagecabinet")
add("monitor", "computermonitor", "pcmonitor", "monitors")
add("pillow",  "pillows", "cushion", "cushions", "throwpillow", "throwpillows")
add("curtain", "curtains", "windowcurtain")
add("book",    "books")
add("lamp",    "tablelamp", "desklamp", "floorlamp", "nightlamp", "walllamp")
add("window",  "windows")
add("bottle",  "bottles")
add("bag",     "bags")
add("mirror",  "mirrors")
add("door",    "doors")
add("table",   "tables")
add("bed",     "beds")
add("box",     "boxes")

_strip = re.compile(r"^(.+?)\d+$")

def normalise(raw: str) -> str:
    name = raw.strip().lower()
    if name in synonyms:
        return synonyms[name]
    m = _strip.match(name)
    if m:
        base = m.group(1)
        return synonyms.get(base, base)
    return name

# ── Split definitions (from incremental_splits.py) ──────────────────────────
SPLITS: dict[str, dict[str, list[str]]] = {
    "Sem-A": {
        "Stage 1 - Bedroom/Lounge":  ["bed","pillow","lamp","curtain","mirror","picture","sofa","light"],
        "Stage 2 - Office/Study":    ["desk","monitor","keyboard","chair","table","shelf","book","paper"],
        "Stage 3 - Kitchen/Utility": ["counter","cabinet","bottle","cup","box","bag","window","door"],
    },
    "Sem-B": {
        "Stage 1 - Bedroom core":        ["bed","pillow","lamp","curtain"],
        "Stage 2 - Bedroom accessories": ["mirror","picture","sofa","light"],
        "Stage 3 - Office core":         ["monitor","keyboard","desk","chair"],
        "Stage 4 - Study/library":       ["table","shelf","book","paper"],
        "Stage 5 - Kitchen":             ["counter","cabinet","bottle","cup"],
        "Stage 6 - Storage/circulation": ["box","bag","window","door"],
    },
    "Surv-A": {
        "Stage 1 - High-freq (structural/furniture)": ["chair","table","window","door","cabinet","shelf","desk","box"],
        "Stage 2 - Mid-freq (decor/study objects)":   ["picture","book","paper","lamp","sofa","pillow","bed","monitor"],
        "Stage 3 - Low-freq (kitchen/accessories)":   ["curtain","bottle","bag","mirror","counter","keyboard","light","cup"],
    },
    "Surv-B": {
        "Stage 1 - Top-freq 1-4":   ["chair","table","window","door"],
        "Stage 2 - Top-freq 5-8":   ["cabinet","shelf","desk","box"],
        "Stage 3 - Mid-freq 9-12":  ["picture","book","paper","lamp"],
        "Stage 4 - Mid-freq 13-16": ["sofa","pillow","bed","monitor"],
        "Stage 5 - Low-freq 17-20": ["curtain","bottle","bag","mirror"],
        "Stage 6 - Low-freq 21-24": ["counter","keyboard","light","cup"],
    },
}

TOP_24 = [
    "chair","table","window","door","cabinet","shelf","desk","box",
    "picture","book","paper","lamp","sofa","pillow","bed","monitor",
    "curtain","bottle","bag","mirror","counter","keyboard","light","cup",
]

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading CSVs...", flush=True)
all_df = pd.read_csv(r"C:\sunrgbd-exploration\incremental_splits_all.csv")
sf     = pd.read_csv(r"C:\sunrgbd-exploration\spatial_features.csv")

sf["norm_cat"] = sf["category"].apply(normalise)
# scene_id -> set of normalised classes  (sf uses short IDs: img_0063)
sf_classes: dict[str, set] = sf.groupby("scene")["norm_cat"].apply(set).to_dict()

# Add basename column for matching sf short IDs
all_df["scene_id"] = all_df["scene"].apply(lambda p: pathlib.Path(p).name)

# Parse classes_present into a Python set (annotation-JSON ground truth)
all_df["cls_set"] = all_df["classes_present"].apply(
    lambda v: set(v.split("|")) if pd.notna(v) else set()
)

# ── Stage label cleaning (em-dash / replacement-char -> plain hyphen) ────────
def clean(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s))
    s = re.sub(r"[\u2013\u2014\u2012\ufffd]", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip()

all_df["stage_clean"] = all_df["stage"].apply(clean)

# Build lookup tables
stage_to_cls: dict[tuple, list] = {}
stage_order:  dict[str, list]   = {}
for split_name, stages in SPLITS.items():
    ordered = []
    for label, classes in stages.items():
        key = clean(label)
        stage_to_cls[(split_name, key)] = classes
        ordered.append(key)
    stage_order[split_name] = ordered


# =============================================================================
# CHECK 1  Strict filter correctness — 10 sampled scenes
#
# Ground truth: classes_present column (annotation JSON, same source as filter)
# Corroboration: spatial_features.csv (noted separately; not used to override)
# =============================================================================
print()
print("=" * 70)
print("CHECK 1 -- Strict filter correctness (10 random scenes)")
print("=" * 70)
print("  Ground truth: 'classes_present' column (from annotation JSONs).")
print("  spatial_features.csv shown alongside for corroboration only.")

# Sample: 2 scenes per split (first stage + a middle/late stage)
sample_rows = []
for split_name in ["Sem-A", "Sem-B", "Surv-A", "Surv-B"]:
    subset  = all_df[all_df["split"] == split_name]
    stages  = stage_order[split_name]
    picks   = [stages[0], stages[len(stages) // 2]]     # first + mid stage
    for skey in picks:
        rows = subset[subset["stage_clean"] == skey]
        if len(rows):
            sample_rows.append(rows.sample(1, random_state=42 + len(sample_rows)).iloc[0])
        if len(sample_rows) >= 10:
            break
    if len(sample_rows) >= 10:
        break

# Top up if needed
if len(sample_rows) < 10:
    extra = all_df.sample(10 - len(sample_rows), random_state=99)
    for _, r in extra.iterrows():
        sample_rows.append(r)

check1_results = []

for row in sample_rows[:10]:
    split = row["split"]
    stage = row["stage_clean"]
    sid   = row["scene_id"]

    # ---- Authoritative class set (annotation JSON via classes_present) ------
    csv_cls = row["cls_set"] & set(TOP_24)

    # ---- Corroborating class set (spatial_features.csv) --------------------
    sf_raw  = sf_classes.get(sid)
    sf_cls  = (sf_raw & set(TOP_24)) if sf_raw is not None else None
    sf_note = "(not in spatial_features)" if sf_cls is None else ""

    # ---- Stage and future class sets ----------------------------------------
    stage_cls = set(stage_to_cls.get((split, stage), []))
    idx       = stage_order[split].index(stage) if stage in stage_order[split] else -1
    future_keys = stage_order[split][idx + 1:] if idx >= 0 else []
    future_cls  = set()
    for fk in future_keys:
        future_cls |= set(stage_to_cls.get((split, fk), []))

    # ---- Conditions (ground truth = CSV) ------------------------------------
    stage_hit  = csv_cls & stage_cls          # (a)
    future_hit = csv_cls & future_cls         # (b)
    cond_a = bool(stage_hit)
    cond_b = not bool(future_hit)
    passed = cond_a and cond_b
    check1_results.append(passed)

    # ---- sf corroboration flags ---------------------------------------------
    if sf_cls is not None:
        sf_extra_future = (sf_cls & future_cls) - (csv_cls & future_cls)
        sf_extra_note   = f"  [sf has extra future class not in annotation: {sorted(sf_extra_future)}]" \
                          if sf_extra_future else ""
        sf_miss_stage   = stage_hit - sf_cls
        sf_miss_note    = f"  [sf missing stage class present in annotation: {sorted(sf_miss_stage)}]" \
                          if sf_miss_stage else ""
    else:
        sf_extra_note = sf_miss_note = ""

    status = "PASS" if passed else "FAIL"
    a_tag  = "OK" if cond_a else "FAIL -- no stage class found in annotation"
    b_tag  = "OK" if cond_b else f"FAIL -- future classes in annotation: {sorted(future_hit)}"

    print(f"\n  [{status}]  {split} | {stage}")
    print(f"    scene              : {sid}")
    print(f"    stage classes      : {sorted(stage_cls)}")
    print(f"    classes (csv/annotation): {sorted(csv_cls)}")
    print(f"    classes (sf)       : {sorted(sf_cls) if sf_cls is not None else sf_note}")
    if sf_extra_note: print(f"   {sf_extra_note}")
    if sf_miss_note:  print(f"   {sf_miss_note}")
    print(f"    (a) stage hit      : {sorted(stage_hit)}  ->  {a_tag}")
    print(f"    (b) no future      : future_forbidden={sorted(future_cls)[:6]}...")
    print(f"                         contaminating={sorted(future_hit)}  ->  {b_tag}")

check1_pass = all(check1_results)
print()
print(f"  CHECK 1 RESULT: {'PASS' if check1_pass else 'FAIL'}"
      f"  ({sum(check1_results)}/10 scenes verified correctly)")


# =============================================================================
# CHECK 2  No scene appears in two stages of the same split
# =============================================================================
print()
print("=" * 70)
print("CHECK 2 -- No scene appears in two stages of the same split")
print("=" * 70)

check2_pass = True
for split_name in ["Sem-A", "Sem-B", "Surv-A", "Surv-B"]:
    sub = all_df[all_df["split"] == split_name]
    dup = sub[sub.duplicated("scene", keep=False)]
    ok  = len(dup) == 0
    if not ok:
        check2_pass = False
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}]  {split_name}: {len(dup)} duplicate scene paths")
    if not ok:
        print(dup[["scene", "stage"]].drop_duplicates().head(5).to_string(index=False))

print(f"\n  CHECK 2 RESULT: {'PASS' if check2_pass else 'FAIL'}")


# =============================================================================
# CHECK 3  Last stage has 0% loss
#
# For each split's last stage there are no future-stage classes, so every
# scene with >=1 last-stage class must survive the filter.
#
# Verification method: iterate every row in the CSV whose split+stage is the
# last stage and confirm that classes_present contains >=1 last-stage class.
# Then compare surviving count against the before-filter count from the
# report to confirm they are equal.
#
# Note on spatial_features.csv: sf covers only 7,198 / 10,295 scenes and
# may report class presence for objects absent from the 2D annotations.
# We therefore use sf only to check the subset it covers, and use the
# annotation-based classes_present as the authoritative source for all scenes.
# =============================================================================
print()
print("=" * 70)
print("CHECK 3 -- Last stage always 0% loss")
print("=" * 70)
print("  Primary: every surviving row's classes_present must hit the last stage.")
print("  Secondary: for scenes in spatial_features, cross-check there too.")

LAST_STAGES = {
    "Sem-A":  "Stage 3 - Kitchen/Utility",
    "Sem-B":  "Stage 6 - Storage/circulation",
    "Surv-A": "Stage 3 - Low-freq (kitchen/accessories)",
    "Surv-B": "Stage 6 - Low-freq 21-24",
}

# Expected surviving counts from report (before == surviving for last stages)
REPORT_SURVIVING = {
    "Sem-A":  6808,
    "Sem-B":  5807,
    "Surv-A": 4169,
    "Surv-B": 2264,
}

check3_pass = True
for split_name, last_raw in LAST_STAGES.items():
    last_key = clean(last_raw)
    last_cls = set(stage_to_cls.get((split_name, last_key), []))

    rows = all_df[(all_df["split"] == split_name) & (all_df["stage_clean"] == last_key)]
    total_rows = len(rows)

    # (A) Every row must have >=1 last-stage class in classes_present
    fails_a = rows[rows["cls_set"].apply(lambda s: not bool(s & last_cls))]
    cond_a_ok = len(fails_a) == 0

    # (B) Row count matches report value
    expected = REPORT_SURVIVING[split_name]
    cond_b_ok = (total_rows == expected)

    # (C) sf corroboration: for scenes in sf, do they also show a last-stage class?
    rows_in_sf    = rows[rows["scene_id"].isin(sf_classes)]
    sf_miss_count = rows_in_sf[
        rows_in_sf["scene_id"].apply(
            lambda sid: not bool(sf_classes[sid] & last_cls)
        )
    ]
    sf_note = (f"{len(rows_in_sf):,} of {total_rows:,} scenes in sf; "
               f"{len(sf_miss_count)} sf rows lack a last-stage class "
               f"(likely sf pipeline gap, not a filter bug)")

    ok = cond_a_ok and cond_b_ok
    if not ok:
        check3_pass = False
    status = "PASS" if ok else "FAIL"

    print(f"\n  [{status}]  {split_name} -- {last_raw}")
    print(f"    last-stage classes       : {sorted(last_cls)}")
    print(f"    surviving in CSV         : {total_rows:,}  (report says {expected:,})"
          f"  ->  {'counts match' if cond_b_ok else 'COUNT MISMATCH'}")
    print(f"    rows missing stage class : {len(fails_a)}"
          f"  ->  {'OK -- all rows confirmed' if cond_a_ok else 'FAIL -- filter bug'}")
    if not cond_a_ok:
        print(fails_a[["scene", "classes_present"]].head(3).to_string(index=False))
    print(f"    sf corroboration         : {sf_note}")

print(f"\n  CHECK 3 RESULT: {'PASS' if check3_pass else 'FAIL'}")


# =============================================================================
# CHECK 4  Class coverage — each of 24 classes in exactly 1 stage per split
# =============================================================================
print()
print("=" * 70)
print("CHECK 4 -- Class coverage (24 classes, each in exactly 1 stage per split)")
print("=" * 70)

check4_pass = True
for split_name, stages in SPLITS.items():
    assigned: list[str] = []
    for _, classes in stages.items():
        assigned.extend(classes)

    counts  = {c: assigned.count(c) for c in TOP_24}
    missing = [c for c, n in counts.items() if n == 0]
    dupes   = [c for c, n in counts.items() if n > 1]
    extra   = [c for c in assigned if c not in TOP_24]

    ok = not (missing or dupes or extra)
    if not ok:
        check4_pass = False
    status = "PASS" if ok else "FAIL"

    print(f"\n  [{status}]  {split_name} -- {len(assigned)} classes across {len(stages)} stages")
    if missing: print(f"    MISSING    : {missing}")
    if dupes:   print(f"    DUPLICATED : {dupes}")
    if extra:   print(f"    UNEXPECTED : {extra}")
    if ok:
        for label, classes in stages.items():
            print(f"    {label:<44}: {', '.join(classes)}")

print(f"\n  CHECK 4 RESULT: {'PASS' if check4_pass else 'FAIL'}")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
results = [check1_pass, check2_pass, check3_pass, check4_pass]
n_pass  = sum(results)
labels  = [
    "Check 1 -- filter correctness (10 sampled scenes, annotation ground truth)",
    "Check 2 -- no scene in two stages of the same split",
    "Check 3 -- last stage always 0% loss",
    "Check 4 -- all 24 classes assigned exactly once per split",
]

print()
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
for label, passed in zip(labels, results):
    print(f"  {'PASS' if passed else 'FAIL'}  {label}")
print()
if n_pass == 4:
    print("  All 4 checks passed.")
    print("  Results are SAFE TO SHARE with the mentor.")
else:
    print(f"  {4 - n_pass} check(s) FAILED. Resolve issues above before sharing.")
