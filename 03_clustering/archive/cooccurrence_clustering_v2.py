"""
cooccurrence_clustering_v2.py
-----------------------------
Identical to cooccurrence_clustering.py except:
  - K_RANGE widened from range(2, 6)  →  range(2, 11)   (k = 2 .. 10)
  - Every (focal_class, k, silhouette_score) pair is recorded and saved to
    silhouette_scores_by_k.csv — one row per class per k tried.
  - Output CSVs are named cluster_assignments_t010_v2.csv and
    cluster_summary_t010_v2.csv so the originals are preserved.
  - A comparison table is printed at the end: old best-k (from k=2..5, read
    from cluster_summary_t010.csv) vs new best-k (from k=2..10).

WHY widen k?
  K-Means with a capped range of k=2..5 will always select from a small set.
  Some focal classes may have natural cluster counts between 6 and 10 — if so,
  the silhouette score improves and the cluster assignments change.  The
  comparison table makes it explicit which classes benefited.

All other choices (Jaccard threshold = 0.10, random_state = 42, n_init = 10,
binary co-occurrence features, silhouette as selection criterion) are unchanged.
"""

import warnings
import pathlib
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# 1. Paths and constants
# ---------------------------------------------------------------------------
OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")
JACCARD_CSV  = OUTPUT_DIR / "cooccurrence_jaccard.csv"
FEATURES_CSV = OUTPUT_DIR / "spatial_features.csv"
FREQ_CSV     = OUTPUT_DIR / "class_frequencies.csv"
OLD_SUMMARY  = OUTPUT_DIR / "cluster_summary_t010.csv"   # for comparison table

EXCLUDE = {"wall", "floor", "ceiling"}
TOP_N   = 25

JACCARD_THRESHOLD = 0.10
MIN_SCENES        = 80
MIN_FEATURES      = 2

# CHANGE: widened from range(2, 6) to range(2, 11)
K_RANGE     = range(2, 11)    # k = 2, 3, 4, 5, 6, 7, 8, 9, 10
K_RANGE_OLD = range(2, 6)     # k = 2, 3, 4, 5   (used in threshold sensitivity to match v1)

SENSITIVITY_CLASS = "chair"
THRESHOLDS        = np.round(np.arange(0.00, 0.32, 0.02), 2)

# ---------------------------------------------------------------------------
# 2. Load inputs
# ---------------------------------------------------------------------------
jac     = pd.read_csv(JACCARD_CSV, index_col=0)
sf      = pd.read_csv(FEATURES_CSV)
freq_df = pd.read_csv(FREQ_CSV)

selected_classes = (
    freq_df.loc[~freq_df["class"].isin(EXCLUDE), "class"]
    .head(TOP_N).tolist()
)

print(f"Top-{TOP_N} classes: {selected_classes}")

# ---------------------------------------------------------------------------
# 3. Build scene × class binary matrix
# ---------------------------------------------------------------------------
# PATTERN: "binary bag-of-classes" — each scene is a 0/1 vector over the
# top-25 classes.  Multiple instances of the same class in one scene collapse
# to 1 (clip upper=1) because context, not count, drives clustering.
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

print(f"\nScene x class matrix: {scene_class.shape[0]:,} scenes x {scene_class.shape[1]} classes")

# ---------------------------------------------------------------------------
# 4. Cluster each focal class  (k = 2 .. 10)
# ---------------------------------------------------------------------------
# For every focal class, every k in K_RANGE is attempted and its silhouette
# score is recorded — not just the winner.  This produces the full
# silhouette_scores_by_k.csv table.
#
# ENGINEERING DECISION: record silhouette scores even for non-winning k values.
# Without this, the "wider search benefited X classes" comparison would require
# re-running v1; reading from the saved CSV is sufficient for the winners only,
# but the full silhouette curve is needed to understand the shape of the
# optimisation landscape.

results  = {}   # focal -> {labels, k, silhouette, features, n_scenes, X}
sil_rows = []   # every (focal, k, silhouette_score) attempt

print("\nClustering per focal class (k=2..10):")

for focal in selected_classes:
    focal_scenes = scene_class[scene_class[focal] == 1]
    n_scenes = len(focal_scenes)

    if n_scenes < MIN_SCENES:
        print(f"  {focal:12s}: skipped — too few scenes ({n_scenes})")
        continue

    others  = [c for c in selected_classes if c != focal]
    jac_row = jac.loc[focal, others]
    keep    = jac_row[jac_row >= JACCARD_THRESHOLD].index.tolist()

    if len(keep) < MIN_FEATURES:
        print(f"  {focal:12s}: skipped — only {len(keep)} features after threshold")
        continue

    X = focal_scenes[keep].copy()

    best_k, best_sil, best_labels = 2, -1.0, None

    for k in K_RANGE:
        if k >= n_scenes:
            break
        km  = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbl = km.fit_predict(X)
        if len(np.unique(lbl)) < 2:
            continue
        sil = silhouette_score(X, lbl, metric="euclidean")

        # Record every score for silhouette_scores_by_k.csv
        sil_rows.append({
            "focal_class":      focal,
            "k":                k,
            "silhouette_score": round(float(sil), 4),
        })

        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, lbl

    if best_labels is None:
        print(f"  {focal:12s}: no valid clustering")
        continue

    results[focal] = {
        "labels":     pd.Series(best_labels, index=focal_scenes.index),
        "k":          best_k,
        "silhouette": best_sil,
        "features":   keep,
        "n_scenes":   n_scenes,
        "X":          X,
    }
    print(f"  {focal:12s}: k={best_k}, sil={best_sil:.3f}, "
          f"scenes={n_scenes:4d}, features={len(keep)}")

# ---------------------------------------------------------------------------
# 5. Cluster profile heatmaps  (saved as _v2 to avoid overwriting originals)
# ---------------------------------------------------------------------------
display_classes = [c for c in selected_classes if c in results]
n_disp = len(display_classes)
COLS   = 2
ROWS   = (n_disp + COLS - 1) // COLS

fig_p, axes = plt.subplots(ROWS, COLS, figsize=(15, 4.5 * ROWS))
axes_flat   = np.array(axes).flatten()

for idx, focal in enumerate(display_classes):
    ax  = axes_flat[idx]
    res = results[focal]
    labels   = res["labels"]
    features = res["features"]
    X        = res["X"]

    cluster_ids  = sorted(np.unique(labels), key=lambda c: -(labels == c).sum())
    profile = pd.DataFrame(
        {cid: X.loc[labels == cid, features].mean() for cid in cluster_ids}
    ).T
    cluster_sizes = [(labels == cid).sum() for cid in cluster_ids]

    col_var = profile.var(axis=0)
    profile = profile[col_var.sort_values(ascending=False).index]

    im = ax.imshow(profile.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(profile.columns)))
    ax.set_xticklabels(profile.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cluster_ids)))
    ax.set_yticklabels(
        [f"C{i+1}  n={cluster_sizes[i]}" for i in range(len(cluster_ids))],
        fontsize=8,
    )
    ax.set_title(
        f"{focal}   k={res['k']}   sil={res['silhouette']:.2f}",
        fontsize=9, fontweight="bold",
    )
    plt.colorbar(im, ax=ax, shrink=0.75, label="mean presence")

for idx in range(n_disp, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig_p.suptitle(
    "SUN RGB-D - Co-occurrence cluster profiles  (k=2..10)\n"
    "Each row = one cluster of scenes containing the focal class\n"
    "Each column = a co-occurring class (sorted by discriminativeness)\n"
    "Colour = fraction of scenes in that cluster where the co-class is present\n"
    f"Jaccard threshold = {JACCARD_THRESHOLD}",
    fontsize=10, y=1.005,
)
fig_p.tight_layout()
p_prof = OUTPUT_DIR / "cluster_profiles_t010_v2.png"
fig_p.savefig(p_prof, dpi=130, bbox_inches="tight")
print(f"\nSaved: {p_prof}")
plt.close(fig_p)

# ---------------------------------------------------------------------------
# 6. Threshold sensitivity (unchanged logic; k swept over K_RANGE_OLD = 2..5
#    to keep this sensitivity plot comparable with v1)
# ---------------------------------------------------------------------------
print(f"\nThreshold sensitivity sweep for '{SENSITIVITY_CLASS}' (k=2..5 for comparability with v1):")

sens_rows = []
if SENSITIVITY_CLASS in scene_class.columns:
    fs      = scene_class[scene_class[SENSITIVITY_CLASS] == 1]
    others  = [c for c in selected_classes if c != SENSITIVITY_CLASS]
    X_full  = fs[others]
    jac_row = jac.loc[SENSITIVITY_CLASS, others]

    for thresh in THRESHOLDS:
        keep   = jac_row[jac_row >= thresh].index.tolist()
        n_feat = len(keep)
        if n_feat < MIN_FEATURES:
            break
        X_t = X_full[keep]
        best_k_t, best_sil_t = 2, -1.0
        for k in K_RANGE_OLD:     # compare with v1 on same k range
            if k >= len(X_t): break
            km  = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X_t)
            if len(np.unique(lbl)) < 2: continue
            sil = silhouette_score(X_t, lbl, metric="euclidean")
            if sil > best_sil_t:
                best_k_t, best_sil_t = k, sil
        sens_rows.append({
            "threshold":       float(thresh),
            "n_features":      n_feat,
            "best_k":          best_k_t,
            "best_silhouette": round(best_sil_t, 4),
        })
        print(f"  thresh={thresh:.2f}  features={n_feat:2d}  k={best_k_t}  sil={best_sil_t:.4f}")

    sens_df = pd.DataFrame(sens_rows)
    fig_s, (ax_feat, ax_sil) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_feat.bar(sens_df["threshold"], sens_df["n_features"],
                width=0.014, color="steelblue", alpha=0.85)
    ax_feat.axvline(JACCARD_THRESHOLD, color="red", linestyle="--",
                    linewidth=1.2, label=f"default threshold ({JACCARD_THRESHOLD})")
    ax_feat.set_ylabel("Co-class features retained", fontsize=10)
    ax_feat.set_title(
        f"Threshold sensitivity - '{SENSITIVITY_CLASS}'\n"
        "How trimming low-Jaccard co-classes affects clustering quality",
        fontsize=10,
    )
    ax_feat.legend(fontsize=8)
    ax_feat.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax_sil.plot(sens_df["threshold"], sens_df["best_silhouette"],
                marker="o", color="darkorange", linewidth=1.8, markersize=5)
    ax_sil.axvline(JACCARD_THRESHOLD, color="red", linestyle="--", linewidth=1.2)
    ax_sil.axhline(0.10, color="gray", linestyle=":",
                   linewidth=0.9, label="0.10 - weak-structure boundary")
    ax_sil.set_xlabel("Jaccard threshold", fontsize=10)
    ax_sil.set_ylabel("Best silhouette score", fontsize=10)
    ax_sil.legend(fontsize=8)
    fig_s.tight_layout()

    p_sens = OUTPUT_DIR / "cluster_threshold_sensitivity_v2.png"
    fig_s.savefig(p_sens, dpi=130)
    print(f"\nSaved: {p_sens}")
    plt.close(fig_s)

# ---------------------------------------------------------------------------
# 7. Save CSVs
# ---------------------------------------------------------------------------
assign_rows  = []
summary_rows = []

for focal, res in results.items():
    labels   = res["labels"]
    X        = res["X"]
    features = res["features"]

    for scene, cid in labels.items():
        assign_rows.append({"focal_class": focal, "scene": scene, "cluster": int(cid)})

    for cid in sorted(np.unique(labels)):
        mask  = labels == cid
        n     = int(mask.sum())
        means = X.loc[mask, features].mean().sort_values(ascending=False)
        top3  = ", ".join(f"{c}={v:.2f}" for c, v in means.head(3).items())
        summary_rows.append({
            "focal_class":    focal,
            "cluster":        int(cid) + 1,
            "n_scenes":       n,
            "pct_of_class":   round(n / res["n_scenes"] * 100, 1),
            "top_co_classes": top3,
            "silhouette":     round(res["silhouette"], 3),
            "k":              res["k"],
        })

# silhouette_scores_by_k.csv — full silhouette curve per focal class
sil_df = pd.DataFrame(sil_rows, columns=["focal_class", "k", "silhouette_score"])
sil_df.to_csv(OUTPUT_DIR / "silhouette_scores_by_k.csv", index=False)

pd.DataFrame(assign_rows).to_csv(
    OUTPUT_DIR / "cluster_assignments_t010_v2.csv", index=False
)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "cluster_summary_t010_v2.csv", index=False)

print("\nSaved:")
print(f"  silhouette_scores_by_k.csv      ({len(sil_df)} rows)")
print(f"  cluster_assignments_t010_v2.csv ({len(assign_rows)} rows)")
print(f"  cluster_summary_t010_v2.csv     ({len(summary_rows)} rows)")

# ---------------------------------------------------------------------------
# 8. Comparison table: old best-k (2..5) vs new best-k (2..10)
# ---------------------------------------------------------------------------
# WHY read old results from the saved CSV rather than re-running with k=2..5?
# Re-running would be redundant — KMeans with random_state=42 is deterministic,
# so the saved CSV is authoritative.  Reading it avoids doubling compute time.

print()
print("=" * 80)
print("COMPARISON: old best-k (k=2..5)  vs  new best-k (k=2..10)")
print("=" * 80)

old_sum = pd.read_csv(OLD_SUMMARY)
# One row per focal_class (silhouette and k are repeated for every cluster row)
old_best = (
    old_sum.groupby("focal_class")[["k", "silhouette"]]
    .first()
    .rename(columns={"k": "old_k", "silhouette": "old_sil"})
)

new_best = (
    summary_df.groupby("focal_class")[["k", "silhouette"]]
    .first()
    .rename(columns={"k": "new_k", "silhouette": "new_sil"})
)

comp = old_best.join(new_best, how="outer").reset_index()
comp["k_changed"] = comp["old_k"] != comp["new_k"]
comp["sil_delta"]  = (comp["new_sil"] - comp["old_sil"]).round(3)

# Sort: classes where k changed first, then alphabetically
comp_sorted = comp.sort_values(
    ["k_changed", "focal_class"], ascending=[False, True]
).reset_index(drop=True)

print(f"\n  {'Focal class':<14} {'Old k':>5} {'Old sil':>8} {'New k':>6} {'New sil':>8} {'Delta sil':>10} {'k changed?':>11}")
print("  " + "-" * 72)

n_changed    = 0
n_improved   = 0
n_worsened   = 0

for _, row in comp_sorted.iterrows():
    changed_flag = "YES  <--" if row["k_changed"] else "no"
    if row["k_changed"]:
        n_changed += 1
    delta = row["sil_delta"]
    if delta > 0:
        n_improved += 1
    elif delta < 0:
        n_worsened += 1

    print(f"  {row['focal_class']:<14} "
          f"{int(row['old_k']):>5} "
          f"{row['old_sil']:>8.3f} "
          f"{int(row['new_k']):>6} "
          f"{row['new_sil']:>8.3f} "
          f"{delta:>+10.3f} "
          f"{changed_flag:>11}")

print()
print(f"  Classes where best k changed  : {n_changed} / {len(comp)}")
print(f"  Classes with improved sil     : {n_improved}")
print(f"  Classes with worsened sil     : {n_worsened}  "
      f"(negative delta = k=2..5 winner was already globally optimal)")
print()
print("  NOTE: a negative delta is not a bug — it means the k=2..5 run happened")
print("  to find the same best-k (since k<=5 is still in the new range) and")
print("  KMeans is deterministic, so the result is identical.  Delta = 0.000 for")
print("  unchanged classes.  A positive delta means a k in 6..10 beat all k<=5.")

print("\nDone.")
