"""
cooccurrence_clustering.py
--------------------------
For each of the top-25 SUN RGB-D classes, every scene containing that class
is represented as a binary feature vector: which OTHER top-25 classes also
appear in that scene?  K-Means then clusters those scenes, revealing distinct
"room contexts" for each object class.

Key idea: a chair in an office scene (co-occurs with desk, monitor, keyboard)
is in a fundamentally different context from a chair in a bedroom (co-occurs
with bed, pillow, lamp).  Co-occurrence vectors encode that context compactly.

Two configurable choices are exposed and analysed:
  1. Jaccard threshold -- drop co-class features whose Jaccard similarity with
     the focal class is below this value.  Low-Jaccard features are nearly
     always absent, so they contribute mostly noise to the clustering.
  2. k (number of clusters) -- chosen automatically per class by the silhouette
     method: try k=2..5 and keep the k with the highest silhouette score.

Outputs
-------
  cluster_profiles.png              heatmap grid: cluster × co-class presence
  cluster_threshold_sensitivity.png silhouette score and feature count vs threshold
  cluster_assignments.csv           per-scene cluster label for each focal class
  cluster_summary.csv               cluster sizes + top distinguishing co-classes
"""

import warnings
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 1. Paths and constants
# ---------------------------------------------------------------------------
OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")
JACCARD_CSV  = OUTPUT_DIR / "cooccurrence_jaccard.csv"
FEATURES_CSV = OUTPUT_DIR / "spatial_features.csv"
FREQ_CSV     = OUTPUT_DIR / "class_frequencies.csv"

EXCLUDE = {"wall", "floor", "ceiling"}
TOP_N   = 25

# Clustering hyperparameters
JACCARD_THRESHOLD = 0.10   # co-classes below this Jaccard are dropped as features
MIN_SCENES        = 80     # skip clustering for classes appearing in < 80 scenes
MIN_FEATURES      = 2      # need at least 2 features to form meaningful clusters
K_RANGE           = range(2, 6)   # try k = 2, 3, 4, 5

# Threshold sensitivity sweep (section 6)
SENSITIVITY_CLASS  = "chair"
THRESHOLDS         = np.round(np.arange(0.00, 0.32, 0.02), 2)

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
# ENGINEERING DECISION: we collapse every instance of a class in a scene into
# a single 1/0 indicator.  Multiple chairs in the same room all share the same
# context; counting them separately would just add identical duplicate rows
# without adding information.  The co-occurrence metric was also computed at
# this scene-level granularity, so the feature space and the Jaccard prior are
# consistent with each other.
#
# Pattern: "binary bag-of-classes" -- analogous to the bag-of-words model in
# NLP, but for object categories instead of words.

scene_class = (
    sf[sf["category"].isin(selected_classes)]
    .groupby(["scene", "category"])
    .size()
    .unstack(fill_value=0)
    .clip(upper=1)          # binarise: 1 = present, 0 = absent
)
# Ensure every top-25 column exists even if no scene contains that class
for cls in selected_classes:
    if cls not in scene_class.columns:
        scene_class[cls] = 0
scene_class = scene_class[selected_classes]   # canonical column order

print(f"\nScene × class matrix: {scene_class.shape[0]:,} scenes × {scene_class.shape[1]} classes")

# ---------------------------------------------------------------------------
# 4. Cluster each focal class
# ---------------------------------------------------------------------------
# For focal class X:
#   (a) Keep only scenes where X is present.
#   (b) Drop column X (always 1 → zero variance → useless feature).
#   (c) Jaccard-based feature selection: retain co-class Y only if
#       Jaccard(X, Y) >= JACCARD_THRESHOLD.
#       Low-Jaccard co-classes appear near-randomly alongside X (low base rate
#       AND low affinity), so they add near-zero-variance dimensions that
#       dilute cluster signal.  Dropping them is an instance of
#       "variance-based feature selection" guided by a domain prior (Jaccard).
#   (d) K-Means silhouette sweep: try k = 2..5, keep k with highest
#       silhouette score.  The silhouette score for a point measures how
#       similar it is to its own cluster compared to the nearest other cluster;
#       averaged over all points it summarises overall cluster quality.
#       Range: [-1, 1]; values > 0.1 indicate at least weak structure.
#   (e) Store labels indexed by scene name.

results = {}

print("\nClustering per focal class:")
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
# 5. Cluster profile heatmaps
# ---------------------------------------------------------------------------
# For each focal class, one panel: a heatmap whose
#   rows   = clusters (labelled with size), sorted largest → smallest
#   columns = surviving co-class features, sorted by variance across cluster
#             means — the most "discriminating" features appear on the left
#
# Colour = mean presence rate (0–1) of that co-class in that cluster's scenes.
#
# Reading the plot: clusters with high values (dark orange/red) for specific
# co-classes define distinct room types.  E.g. a chair cluster with high
# "monitor" and "keyboard" but low "bed" and "pillow" is an office context.
#
# Sorting columns by variance is a pattern called "discriminant ordering":
# place the features that differ most between groups at the most salient
# position (left), so the eye naturally sees the meaningful differences.

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

    # Build cluster profile matrix
    cluster_ids = sorted(np.unique(labels),
                         key=lambda c: -(labels == c).sum())   # largest first
    profile = pd.DataFrame(
        {cid: X.loc[labels == cid, features].mean() for cid in cluster_ids}
    ).T
    cluster_sizes = [(labels == cid).sum() for cid in cluster_ids]

    # Sort columns by inter-cluster variance (most discriminating leftmost)
    col_var = profile.var(axis=0)
    profile = profile[col_var.sort_values(ascending=False).index]

    im = ax.imshow(profile.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(profile.columns)))
    ax.set_xticklabels(profile.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(cluster_ids)))
    row_labels = [f"C{i+1}  n={cluster_sizes[i]}" for i in range(len(cluster_ids))]
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(
        f"{focal}   k={res['k']}   sil={res['silhouette']:.2f}",
        fontsize=9, fontweight="bold"
    )
    plt.colorbar(im, ax=ax, shrink=0.75, label="mean presence")

for idx in range(n_disp, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig_p.suptitle(
    "SUN RGB-D — Co-occurrence-based cluster profiles\n"
    "Each row = one cluster of scenes containing the focal class\n"
    "Each column = a co-occurring class (sorted by discriminativeness)\n"
    "Colour = fraction of scenes in that cluster where the co-class is present\n"
    f"Jaccard threshold = {JACCARD_THRESHOLD}",
    fontsize=10, y=1.005
)
fig_p.tight_layout()
p_prof = OUTPUT_DIR / "cluster_profiles_t010.png"
fig_p.savefig(p_prof, dpi=130, bbox_inches="tight")
print(f"\nSaved: {p_prof}")
plt.close(fig_p)

# ---------------------------------------------------------------------------
# 6. Threshold sensitivity analysis
# ---------------------------------------------------------------------------
# We sweep JACCARD_THRESHOLD from 0.00 to 0.30 for the focal class "chair".
# At each threshold we record:
#   - n_features : how many co-class features survive
#   - best_silhouette : silhouette score of the optimal-k clustering
#   - best_k : which k was chosen
#
# The goal: find the "sweet spot" where trimming noise features improves
# cluster quality without discarding genuine signal.
#
# Conceptually this is a "regularisation path" — the same idea as sweeping
# the regularisation strength λ in Lasso regression, watching how model
# complexity and performance change together.

print(f"\nThreshold sensitivity sweep for '{SENSITIVITY_CLASS}':")

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
        for k in K_RANGE:
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
        print(f"  thresh={thresh:.2f}  features={n_feat:2d}  "
              f"k={best_k_t}  sil={best_sil_t:.4f}")

    sens_df = pd.DataFrame(sens_rows)

    fig_s, (ax_feat, ax_sil) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_feat.bar(sens_df["threshold"], sens_df["n_features"],
                width=0.014, color="steelblue", alpha=0.85)
    ax_feat.axvline(JACCARD_THRESHOLD, color="red", linestyle="--",
                    linewidth=1.2, label=f"default threshold ({JACCARD_THRESHOLD})")
    ax_feat.set_ylabel("Co-class features retained", fontsize=10)
    ax_feat.set_title(
        f"Threshold sensitivity — '{SENSITIVITY_CLASS}'\n"
        "How trimming low-Jaccard co-classes affects clustering quality",
        fontsize=10
    )
    ax_feat.legend(fontsize=8)
    ax_feat.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

    ax_sil.plot(sens_df["threshold"], sens_df["best_silhouette"],
                marker="o", color="darkorange", linewidth=1.8, markersize=5)
    ax_sil.axvline(JACCARD_THRESHOLD, color="red", linestyle="--", linewidth=1.2)
    ax_sil.axhline(0.10, color="gray", linestyle=":", linewidth=0.9,
                   label="0.10 — weak-structure boundary")
    ax_sil.set_xlabel("Jaccard threshold", fontsize=10)
    ax_sil.set_ylabel("Best silhouette score", fontsize=10)
    ax_sil.legend(fontsize=8)

    fig_s.tight_layout()
    p_sens = OUTPUT_DIR / "cluster_threshold_sensitivity.png"
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

pd.DataFrame(assign_rows).to_csv(OUTPUT_DIR / "cluster_assignments_t010.csv", index=False)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(OUTPUT_DIR / "cluster_summary_t010.csv", index=False)
print("\nSaved: cluster_assignments_t010.csv, cluster_summary_t010.csv")

print("\nCluster summary (sorted by focal class, then cluster):")
print(summary_df.sort_values(["focal_class", "cluster"]).to_string(index=False))

print("\nDone.")
