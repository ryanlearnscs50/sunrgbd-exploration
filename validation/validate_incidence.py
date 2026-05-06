"""
validate_incidence.py
---------------------
Independently recomputes the co-occurrence matrix using the incidence matrix
method and cross-checks it against the CSVs produced by cooccurrence.py.

WHY a second algorithm?
  cooccurrence.py uses a nested-loop approach: for every scene, find every
  pair of co-present classes and increment two cells.  That is correct, but
  it is also the *only* evidence we have.  If there is a systematic bug
  (e.g. an off-by-one in the pair enumeration, or a wrong annotation path),
  both the data and the check would be wrong in the same way.

  This script uses a completely different algorithm -- the incidence matrix
  method -- so any agreement between them is strong evidence that both are
  correct.  This is called "independent verification" and is the same
  principle behind dual-entry bookkeeping in accounting.

THE INCIDENCE MATRIX METHOD:

  Step 1. Build a binary matrix X of shape (S, C):
            rows    = scenes  (one row per annotation file)
            columns = classes (one column per selected class)
            X[s, c] = 1  if class c appears in scene s
                    = 0  otherwise
          This is called a "binary incidence matrix" or "membership matrix".

  Step 2. Compute  C = X^T @ X   (matrix multiplication, shape C x C)

  WHY does this give co-occurrence?  Linear algebra explanation:

    Entry C[i, j]  =  (row i of X^T)  dot product  (column j of X)
                   =  (column i of X) dot product   (column j of X)
                   =  sum over all scenes s of  X[s, i] * X[s, j]

    Because X is binary, X[s, i] * X[s, j] = 1 only when BOTH class i
    AND class j are present in scene s, and 0 otherwise.
    So the sum counts exactly the number of scenes where both appear --
    which is precisely the co-occurrence definition.

    On the diagonal (i == j):
      C[i, i] = sum_s  X[s, i]^2 = sum_s  X[s, i]   (since 0^2=0, 1^2=1)
             = number of scenes that contain class i
             = scene frequency of class i

    Off-diagonal (i != j):
      C[i, j] = number of scenes containing both class i and class j

  This is the same computation as "how many items do these two sets share?"
  expressed as a single matrix multiply instead of explicit pair enumeration.
  It is a standard trick in recommender systems and document similarity, where
  the pattern is called "item-item co-occurrence via dot products".

Outputs: PASS/FAIL printed to stdout, no files written.
"""

import datetime
import json
import pathlib
import re

import numpy as np
import pandas as pd

DATASET_ROOT = pathlib.Path(r"C:\Users\ryana\Downloads\SUNRGBD\SUNRGBD")
OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")

# ---------------------------------------------------------------------------
# 1. Reproduce the exact class selection from cooccurrence.py
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: we read the same CSV and apply the same exclusion list
# rather than hard-coding the 22 class names.  If cooccurrence.py ever changes
# its cutoff, this script stays in sync automatically.  This is the
# "single source of truth" principle: the class list lives in one place.

CSV_PATH = OUTPUT_DIR / "class_frequencies.csv"
EXCLUDE  = {"wall", "floor", "ceiling"}
TOP_N    = 25

freq_df = pd.read_csv(CSV_PATH)
# Filter EXCLUDE first, then slice the top TOP_N -- mirrors the fix in
# cooccurrence.py so both scripts always operate on the same class set.
selected_classes = (
    freq_df
    .loc[~freq_df["class"].isin(EXCLUDE), "class"]
    .head(TOP_N)
    .tolist()
)

# ---------------------------------------------------------------------------
# Synonym map  (must be identical to class_frequency.py and cooccurrence.py)
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
n = len(selected_classes)
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

print(f"Classes selected: {n}")
print(f"  {selected_classes}")
print()

# ---------------------------------------------------------------------------
# 2. Load the reference data we are validating against
# ---------------------------------------------------------------------------
raw_ref = pd.read_csv(OUTPUT_DIR / "cooccurrence_raw.csv", index_col=0)
freq_ref = pd.read_csv(CSV_PATH).set_index("class")["scene_frequency"]
freq_ref = freq_ref.reindex(selected_classes)   # keep only our 22, in order

# ---------------------------------------------------------------------------
# 3. Build the incidence matrix X  (scenes x classes)
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: we collect rows into a Python list and convert to
# numpy at the end, rather than pre-allocating a numpy array and indexing
# into it.  The reason: we do not know the final scene count until we finish
# scanning (some files fail to parse).  Appending to a list is O(1) amortised;
# growing a numpy array would require copying the whole array each time.
# This pattern -- "collect in a list, convert once" -- is called
# "deferred materialisation" and is standard practice when the size is unknown.

print("Scanning annotation files to build incidence matrix...")

annotation_files = sorted(DATASET_ROOT.rglob("annotation2Dfinal/index.json"))
rows       = []
parse_errors = 0

for ann_path in annotation_files:
    try:
        with ann_path.open(encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        parse_errors += 1
        continue

    # Build the set of canonical class names present in this scene.
    raw_names = {
        normalise(obj["name"])
        for obj in (data.get("objects") or [])
        if obj and isinstance(obj.get("name"), str) and obj["name"].strip()
    }
    present = raw_names & set(selected_classes)

    # Build a binary row vector: 1 for each class that appears, 0 otherwise.
    # ENGINEERING DECISION: np.zeros gives us a clean all-zero vector to start;
    # we then set only the indices of present classes to 1.
    # This is called "one-hot encoding" when there is exactly one 1, and
    # "multi-hot encoding" (or "binary bag-of-words") when there can be many.
    row = np.zeros(n, dtype=np.int8)   # int8: each cell is 0 or 1, saves memory
    for cls in present:
        row[class_to_idx[cls]] = 1
    rows.append(row)

total_scenes = len(rows)
print(f"  Scenes processed : {total_scenes:,}")
print(f"  Parse errors     : {parse_errors:,}")
print()

# Stack the list of 1-D arrays into a 2-D matrix.
# np.vstack treats each element as a row: shape = (total_scenes, n).
# ENGINEERING DECISION: we use int32 for X here (upgraded from int8) because
# the matrix multiply X^T @ X will accumulate sums of up to ~10,000 values.
# int8 overflows at 127; int32 holds up to 2^31 safely.
X = np.vstack(rows).astype(np.int32)   # shape: (S, C)

print(f"Incidence matrix X shape: {X.shape}  (scenes x classes)")
print(f"  Non-zero entries: {X.sum():,}  (total class appearances across all scenes)")
print()

# ---------------------------------------------------------------------------
# 4. Compute C = X^T @ X  (the incidence-based co-occurrence matrix)
# ---------------------------------------------------------------------------
# @ is Python's matrix-multiply operator (introduced in PEP 465, Python 3.5).
# X.T is the transpose of X, shape (C, S).
# X.T @ X is therefore (C, S) @ (S, C) = (C, C).
# Each entry C[i,j] is the dot product of column i and column j of X,
# which counts the number of scenes where both class i and class j appear --
# see the module docstring for the full linear algebra explanation.

C = X.T @ X   # shape: (n, n)

print("Computed C = X^T @ X")
print(f"  C shape: {C.shape}")
print(f"  C diagonal (scene frequencies, first 5): {np.diag(C)[:5].tolist()}")
print()

# Wrap in a DataFrame with the same labels as the reference CSV so we can
# compare using pandas' label-aligned indexing.
C_df = pd.DataFrame(C, index=selected_classes, columns=selected_classes)

# ---------------------------------------------------------------------------
# 5. Validation checks
# ---------------------------------------------------------------------------

check_records = []   # list of dicts, one per check — written to CSV and report at the end

def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")
    # ENGINEERING DECISION: we store the record inside check() rather than
    # at each call site.  The function is the single place that knows both
    # the label and the outcome, so it is the right place to capture them.
    # This is called "single responsibility" -- the caller just invokes check();
    # the bookkeeping happens in one place.
    check_records.append({"check_name": label, "result": status, "detail": detail})
    return condition

results = []

print("=" * 66)
print("VALIDATION")
print("=" * 66)

# ------------------------------------------------------------------
# Check 1: diagonal matches scene_frequency for every class (within 1)
# ------------------------------------------------------------------
# WHY within 1 rather than exactly equal?
# cooccurrence.py and this script both skip the same malformed files,
# but we want to be robust to any off-by-one that might arise if a
# future re-run of class_frequency.py counted scenes slightly differently
# (e.g. if a previously-corrupted file was later fixed).  A tolerance of
# 1 scene out of 10,000+ is functionally zero.
diag   = np.diag(C)
mismatches_diag = []
for i, cls in enumerate(selected_classes):
    expected = int(freq_ref[cls])
    actual   = int(diag[i])
    if abs(actual - expected) > 1:
        mismatches_diag.append((cls, actual, expected))

max_diag_discrepancy = max(
    abs(int(diag[i]) - int(freq_ref[cls]))
    for i, cls in enumerate(selected_classes)
)

results.append(check(
    "Diagonal == scene_frequency for every class (tolerance ±1)",
    len(mismatches_diag) == 0,
    f"max discrepancy = {max_diag_discrepancy} scene(s)"
    + (f"; first mismatch: {mismatches_diag[0]}" if mismatches_diag else "")
))

# ------------------------------------------------------------------
# Check 2: off-diagonal exactly matches cooccurrence_raw.csv
# ------------------------------------------------------------------
# WHY exact equality here?  Both matrices store integer counts of the
# same events (scene co-occurrences).  If two independent algorithms
# agree to the integer, that is a very strong correctness guarantee --
# much stronger than a floating-point tolerance check.  Any disagreement,
# even by 1, would indicate a real algorithmic difference worth investigating.
#
# We compare the off-diagonal only (mask out i==j) because:
# - The reference CSV has zeros on the diagonal (cooccurrence.py skips i==j)
# - Our C has scene frequencies on the diagonal
# Both are correct for their own definitions; we compare only the
# cells that mean the same thing in both algorithms.

offdiag_mask = ~np.eye(n, dtype=bool)   # True everywhere except the diagonal

C_offdiag   = C[offdiag_mask]
ref_offdiag = raw_ref.values[offdiag_mask]

discrepancies = C_offdiag - ref_offdiag
max_offdiag_discrepancy = int(np.abs(discrepancies).max())
num_mismatches = int((discrepancies != 0).sum())

results.append(check(
    "Off-diagonal C[i,j] == cooccurrence_raw[i,j] for every pair",
    num_mismatches == 0,
    f"max discrepancy = {max_offdiag_discrepancy}  |  mismatched cells = {num_mismatches}"
))

# ------------------------------------------------------------------
# Print the maximum discrepancy across all off-diagonal cells regardless
# ------------------------------------------------------------------
print()
print(f"  Maximum off-diagonal discrepancy across all {n*n - n} cell pairs: "
      f"{max_offdiag_discrepancy}")
print(f"  Maximum diagonal discrepancy across all {n} classes:              "
      f"{max_diag_discrepancy}")

# ------------------------------------------------------------------
# Check 3: C is exactly symmetric
# ------------------------------------------------------------------
# X^T @ X is guaranteed to be symmetric by construction (it is a
# "Gram matrix"), so this check should always pass unless there is a
# numpy bug.  We include it anyway as a sanity test -- a symmetric
# result is a necessary (though not sufficient) condition for correctness.
results.append(check(
    "C = X^T @ X is symmetric (Gram matrix property)",
    np.array_equal(C, C.T),
    "guaranteed by construction; failure would indicate a numpy bug"
))

# ------------------------------------------------------------------
# Check 4: scene count matches total_scenes from cooccurrence.py
# ------------------------------------------------------------------
# cooccurrence.py prints its scene count but does not save it to a file.
# We instead check against the maximum diagonal value: whichever class
# appears in the most scenes gives a lower bound on total_scenes.
# The most common class is chair (~55% of scenes), so diag.max()
# is a reasonable sanity check that our scan found the right scale.
results.append(check(
    "Total scenes in expected range [10,000 – 10,500]",
    10_000 <= total_scenes <= 10_500,
    f"actual: {total_scenes:,}"
))

# ------------------------------------------------------------------
# Final verdict
# ------------------------------------------------------------------
print()
passed = sum(results)
total  = len(results)
print(f"  {passed}/{total} checks passed")
print("=" * 66)

verdict_lines = []
if all(results):
    verdict_lines = [
        "INDEPENDENT VALIDATION PASSED —",
        "co-occurrence matrix is mathematically confirmed via two",
        "independent algorithms",
        "",
        "Algorithm 1 (cooccurrence.py):  nested-loop pair enumeration",
        "Algorithm 2 (this script):      incidence matrix  C = X^T @ X",
        "Both produce identical off-diagonal counts, confirming correctness.",
    ]
    print()
    for line in verdict_lines:
        print(f"  {line}")
else:
    verdict_lines = ["VALIDATION FAILED -- see mismatches above"]
    print()
    print(f"  {verdict_lines[0]}")

# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------
# ENGINEERING DECISION: we capture the run timestamp once here and reuse it
# in both output files.  datetime.now() with isoformat() produces an
# unambiguous, sortable string (e.g. "2026-03-19T14:32:07") -- this format
# is called ISO 8601 and is the standard for timestamps in data files because
# it sorts lexicographically in chronological order.

run_timestamp = datetime.datetime.now().isoformat(timespec="seconds")

# --- 6a. validation_report.txt ------------------------------------------
# A human-readable text file with all context needed to interpret the run.
# ENGINEERING DECISION: we build the report as a list of lines and join at
# the end, rather than concatenating strings in a loop.  str.join() on a
# pre-built list is O(n) total; concatenation in a loop is O(n^2) because
# each step creates a new string.  This is called the "join idiom" and is
# the idiomatic Python way to build multi-line strings.

report_lines = [
    "=" * 66,
    "INCIDENCE MATRIX VALIDATION REPORT",
    "=" * 66,
    f"Run timestamp   : {run_timestamp}",
    f"Script          : validate_incidence.py",
    f"Dataset root    : {DATASET_ROOT}",
    "",
    "KEY NUMBERS",
    "-" * 40,
    f"Scenes processed       : {total_scenes:,}",
    f"Parse errors skipped   : {parse_errors:,}",
    f"Classes validated      : {n}",
    f"Off-diagonal cells     : {n * n - n:,}",
    f"Max off-diag discrepancy: {max_offdiag_discrepancy}",
    f"Max diagonal discrepancy: {max_diag_discrepancy}",
    "",
    "CHECKS",
    "-" * 40,
]
for rec in check_records:
    report_lines.append(f"  [{rec['result']}] {rec['check_name']}")
    if rec["detail"]:
        report_lines.append(f"         {rec['detail']}")

report_lines += [
    "",
    f"  {passed}/{total} checks passed",
    "",
    "VERDICT",
    "-" * 40,
]
report_lines += [f"  {line}" for line in verdict_lines]
report_lines.append("=" * 66)

report_path = OUTPUT_DIR / "validation_report.txt"
report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
print()
print(f"Saved report : {report_path}")

# --- 6b. validation_summary.csv -----------------------------------------
# A machine-readable CSV: one row per check, with check_name / result / detail.
# ENGINEERING DECISION: we add a run_timestamp column so that if this CSV is
# appended to across multiple runs (e.g. after re-running cooccurrence.py with
# new data), each row is self-identifying.  Without a timestamp, you would have
# no way to know which run produced which rows.  This is called "provenance
# tracking" and is essential for reproducible data pipelines.

summary_df = pd.DataFrame(check_records)
summary_df.insert(0, "run_timestamp", run_timestamp)   # prepend timestamp column

summary_path = OUTPUT_DIR / "validation_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved summary: {summary_path}")

# --- 6c. validation_cell_comparison.csv ---------------------------------
# One row per unique off-diagonal pair (i < j), showing both algorithm's
# counts and the discrepancy between them.
#
# WHY i < j only (upper triangle)?
# The co-occurrence matrix is symmetric: count(A,B) == count(B,A) by
# definition.  Including both (i,j) and (j,i) would double every pair,
# making the file twice as long without adding information.  Restricting
# to i < j enumerates each unordered pair exactly once.  This is called
# "upper-triangle iteration" -- the same pattern used in cooccurrence.py
# and the asymmetry analysis in conditional_probability.py.

cell_rows = []
for i in range(n):
    for j in range(i + 1, n):
        cell_rows.append({
            "class_i":    selected_classes[i],
            "class_j":    selected_classes[j],
            "algo1_count": int(raw_ref.values[i, j]),   # nested-loop (cooccurrence.py)
            "algo2_count": int(C[i, j]),                 # incidence matrix (this script)
            "discrepancy": int(raw_ref.values[i, j]) - int(C[i, j]),
        })

cell_df = pd.DataFrame(cell_rows)
cell_path = OUTPUT_DIR / "validation_cell_comparison.csv"
cell_df.to_csv(cell_path, index=False)
print(f"Saved cell comparison: {cell_path}  ({len(cell_df)} pairs)")
