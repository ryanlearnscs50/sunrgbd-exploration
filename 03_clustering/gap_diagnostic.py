"""
gap_diagnostic.py
-----------------
For every focal class, prints and saves the full Gap statistic breakdown.

NOTATION NOTE:
  W_k^(b) uses (b) as a LABEL (index), not a power.
  W_k^(b) = the within-cluster inertia from running k-means on null dataset #b.
  With B=20, you get 20 separate W_k^(1), W_k^(2), ..., W_k^(20) per k.
  mean_b[log(W_k^(b))] = average of those 20 log-inertia values.
  This is NOT b * log(W_k).

For each focal class and each k the output shows:
  - log(W_k)              : log-inertia on the REAL data
  - mean_b[log(W_k^(b))] : average log-inertia over B=20 null datasets
  - Gap(k)                : mean_b[log(W_k^(b))] - log(W_k)
  - sk                    : std(log W_k^(b)) * sqrt(1 + 1/B)
  - 1-SE rule step-by-step walkthrough for k selection
  - Decisive comparison: the exact values that made the rule fire

Output saved to: gap_diagnostic_full.txt
"""

import pathlib
import sys
import numpy as np
import pandas as pd
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────
JACCARD_THRESHOLD = 0.10
MIN_SCENES        = 80
MIN_FEATURES      = 2
K_RANGE           = range(2, 11)
B                 = 20
RANDOM_STATE      = 42
EXCLUDE           = {"wall", "floor", "ceiling"}
TOP_N             = 25

OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")
JACCARD_CSV  = OUTPUT_DIR / "02_cooccurance" / "cooccurrence_jaccard.csv"
FREQ_CSV     = OUTPUT_DIR / "01_class_frequencies" / "class_frequencies.csv"
FEATURES_CSV = OUTPUT_DIR / "04_spatial_analysis" / "spatial_features.csv"
OUT_TXT      = OUTPUT_DIR / "03_clustering" / "gap_diagnostic_full.txt"

# ── Load data ─────────────────────────────────────────────────────────────────
jac     = pd.read_csv(JACCARD_CSV, index_col=0)
sf      = pd.read_csv(FEATURES_CSV)
freq_df = pd.read_csv(FREQ_CSV)

selected_classes = (
    freq_df.loc[~freq_df["class"].isin(EXCLUDE), "class"]
    .head(TOP_N).tolist()
)

scene_class = (
    sf[sf["category"].isin(selected_classes)]
    .groupby(["scene", "category"])
    .size()
    .unstack(fill_value=0)
    .clip(upper=1)
)
for cls in selected_classes:
    if cls not in scene_class.columns:
        scene_class[cls] = 0
scene_class = scene_class[selected_classes]


# ── Gap statistic for one focal class ────────────────────────────────────────
def run_gap(X, k_range, B, rng):
    feature_probs = X.values.mean(axis=0)
    feature_probs = np.clip(feature_probs, 1e-6, 1 - 1e-6)
    records = {}
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        log_W_real = np.log(km.inertia_ + 1e-10)

        null_log_Ws = []
        for _ in range(B):
            X_ref = rng.binomial(1, feature_probs, size=X.shape).astype(float)
            km_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_ref.fit(X_ref)
            null_log_Ws.append(np.log(km_ref.inertia_ + 1e-10))

        null_log_Ws = np.array(null_log_Ws)
        records[k] = {
            "log_W_real": float(log_W_real),
            "mean_null":  float(null_log_Ws.mean()),
            "gap":        float(null_log_Ws.mean() - log_W_real),
            "sk":         float(null_log_Ws.std() * np.sqrt(1.0 + 1.0 / B)),
        }
    return records


def pick_k(records):
    ks = sorted(records)
    for i, k in enumerate(ks[:-1]):
        k_next = ks[i + 1]
        if records[k]["gap"] >= records[k_next]["gap"] - records[k_next]["sk"]:
            return k
    return max(ks, key=lambda k: records[k]["gap"])


# ── Output helpers ────────────────────────────────────────────────────────────
lines = []

def emit(text=""):
    print(text)
    lines.append(text)


# ── Header ────────────────────────────────────────────────────────────────────
WIDE = "=" * 90

emit(WIDE)
emit("GAP STATISTIC DIAGNOSTIC — FULL BREAKDOWN")
emit("SUN RGB-D co-occurrence clustering  |  Jaccard threshold=0.10  |  B=20 null datasets")
emit()
emit("NOTATION GUIDE:")
emit("  W_k           = within-cluster inertia when k-means is fit on REAL data")
emit("  W_k^(b)       = inertia when k-means is fit on NULL dataset number b")
emit("                  (b) is an INDEX label, not a power  —  W_k^(b) != W_k^b")
emit("  log(W_k^(b))  = log of that single null-dataset inertia")
emit("  mean_b[...]   = average of the 20 values  log(W_k^(1)), ..., log(W_k^(20))")
emit("  Gap(k)        = mean_b[log(W_k^(b))] - log(W_k)")
emit("  sk            = std(log W_k^(b)) * sqrt(1 + 1/B)  [uncertainty correction]")
emit(WIDE)

rng = np.random.default_rng(RANDOM_STATE)
summary_rows = []

# ── Per-focal-class loop ──────────────────────────────────────────────────────
for focal in selected_classes:
    focal_scenes = scene_class[scene_class[focal] == 1]
    n_scenes = len(focal_scenes)

    if n_scenes < MIN_SCENES:
        emit()
        emit("─" * 90)
        emit(f"FOCAL CLASS: {focal.upper()}   [SKIPPED — only {n_scenes} scenes, need >= {MIN_SCENES}]")
        continue

    others  = [c for c in selected_classes if c != focal]
    jac_row = jac.loc[focal, others]
    keep    = jac_row[jac_row >= JACCARD_THRESHOLD].index.tolist()

    if len(keep) < MIN_FEATURES:
        emit()
        emit("─" * 90)
        emit(f"FOCAL CLASS: {focal.upper()}   [SKIPPED — only {len(keep)} features after threshold]")
        continue

    X = focal_scenes[keep].copy()
    records = run_gap(X, K_RANGE, B, rng)
    chosen = pick_k(records)

    emit()
    emit("─" * 90)
    emit(f"FOCAL CLASS : {focal.upper()}")
    emit(f"Scenes      : {n_scenes}  |  Features ({len(keep)}): {', '.join(keep)}")
    emit(f"Chosen k    : {chosen}  (by Tibshirani 1-SE rule)")
    emit()

    # ── Per-k table ──────────────────────────────────────────────────────────
    col_w = 24
    hdr = (f"  {'k':>2}  {'log(W_k)  [real]':>18}  "
           f"{'mean_b[log(W_k^(b))]  [null avg]':>{col_w}}  "
           f"{'Gap(k)':>8}  {'sk':>8}  note")
    emit(hdr)
    emit("  " + "·" * (len(hdr) - 2))

    max_gap_k = max(records, key=lambda k: records[k]["gap"])
    for k, r in sorted(records.items()):
        tags = []
        if k == chosen:
            tags.append("CHOSEN k")
        if k == max_gap_k and k != chosen:
            tags.append("argmax Gap")
        tag_str = "  <- " + ", ".join(tags) if tags else ""
        emit(
            f"  {k:>2}  {r['log_W_real']:>18.6f}  "
            f"{r['mean_null']:>{col_w}.6f}  "
            f"{r['gap']:>+8.4f}  {r['sk']:>8.4f}"
            f"{tag_str}"
        )

    # ── 1-SE rule step-by-step ────────────────────────────────────────────────
    emit()
    emit("  1-SE rule: pick SMALLEST k where Gap(k) >= Gap(k+1) - sk(k+1)")
    emit("  " + "·" * 74)
    ks = sorted(records)
    rule_fired = False
    for i, k in enumerate(ks[:-1]):
        k_next = ks[i + 1]
        lhs = records[k]["gap"]
        rhs = records[k_next]["gap"] - records[k_next]["sk"]
        fires = lhs >= rhs
        tag = ""
        if fires and not rule_fired:
            tag = "  <- CHOSEN HERE"
            rule_fired = True
        elif fires:
            tag = "  (also True, ignored — already chosen above)"
        emit(
            f"  k={k}: Gap({k})={lhs:+.4f}  >=  "
            f"Gap({k_next})-sk({k_next})={rhs:+.4f}  =>  "
            f"{'True ' if fires else 'False'}{tag}"
        )

    if not rule_fired:
        emit(f"  No step fired -> fallback: argmax Gap -> k={chosen}")

    # ── Decisive comparison zoomed in ────────────────────────────────────────
    k1 = chosen
    k2 = chosen + 1
    if k1 in records and k2 in records:
        emit()
        emit(f"  DECISIVE COMPARISON  (k={k1} vs k={k2}):")
        emit(f"    mean_b[ log(W_{k1}^(b)) ]  [null avg at k={k1}]  =  {records[k1]['mean_null']:.6f}")
        emit(f"    mean_b[ log(W_{k2}^(b)) ]  [null avg at k={k2}]  =  {records[k2]['mean_null']:.6f}")
        emit(f"    log(W_{k1})  [real data]                          =  {records[k1]['log_W_real']:.6f}")
        emit(f"    log(W_{k2})  [real data]                          =  {records[k2]['log_W_real']:.6f}")
        emit(f"    Gap({k1}) = {records[k1]['mean_null']:.6f} - {records[k1]['log_W_real']:.6f} = {records[k1]['gap']:+.6f}")
        emit(f"    Gap({k2}) = {records[k2]['mean_null']:.6f} - {records[k2]['log_W_real']:.6f} = {records[k2]['gap']:+.6f}")
        emit(f"    sk({k2})                                          =  {records[k2]['sk']:.6f}")
        emit(f"    Gap({k2}) - sk({k2})                              =  {records[k2]['gap'] - records[k2]['sk']:+.6f}")
        emit(f"    Gap({k1}) >= Gap({k2}) - sk({k2})?               =  {records[k1]['gap'] >= records[k2]['gap'] - records[k2]['sk']}")

    summary_rows.append({
        "focal_class":        focal,
        "n_scenes":           n_scenes,
        "n_features":         len(keep),
        "chosen_k":           chosen,
        "gap_at_chosen_k":    round(records[chosen]["gap"], 4),
        "argmax_gap_k":       max_gap_k,
        "rule_fired_early":   chosen != max_gap_k,
    })

# ── Cross-class summary ───────────────────────────────────────────────────────
emit()
emit(WIDE)
emit("SUMMARY ACROSS ALL FOCAL CLASSES")
emit(WIDE)
emit(f"  {'Focal class':<14}  {'Scenes':>7}  {'Features':>9}  {'Chosen k':>9}  "
     f"{'Gap at chosen k':>16}  {'Argmax k':>9}  {'Rule fired early?':>18}")
emit("  " + "─" * 83)
for row in summary_rows:
    early = "Yes (Occams razor)" if row["rule_fired_early"] else "No (used argmax)"
    emit(
        f"  {row['focal_class']:<14}  {row['n_scenes']:>7}  {row['n_features']:>9}  "
        f"{row['chosen_k']:>9}  {row['gap_at_chosen_k']:>+16.4f}  "
        f"{row['argmax_gap_k']:>9}  {early:>18}"
    )

emit()
emit(f"Saved to: {OUT_TXT}")

# ── Write file ────────────────────────────────────────────────────────────────
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
