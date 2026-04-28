"""
cooccurrence_clustering_v3.py
-----------------------------
Fixes the k-selection bias from v2 by replacing silhouette-based k selection
with the Gap Statistic (Tibshirani, Walther & Hastie, 2001).

══════════════════════════════════════════════════════════════════════════════
 PART 1 — FROM THE PAPER  (Tibshirani, Walther & Hastie, J. R. Stat. Soc. B,
                            2001, "Estimating the number of clusters in a data
                            set via the gap statistic")
══════════════════════════════════════════════════════════════════════════════

WHY NOT SILHOUETTE FOR K SELECTION
  Silhouette score measures how tightly packed each cluster is relative to its
  nearest neighbour cluster.  As k grows, each cluster gets smaller and tighter
  by construction — you are splitting the same data into more pieces.  So the
  silhouette score almost always rises with k, meaning argmax(silhouette) will
  always favour the largest k you allow.  This is not a bug in silhouette; it
  simply measures the wrong thing when the goal is to find the *natural* number
  of clusters.

THE GAP STATISTIC  [PAPER]
  The gap statistic asks a different question:
    "How much better does k-means do on the *real* data than it would on
     completely structureless (null) data with the same feature statistics?"

  Concretely, for each k:
    1. Fit k-means on the real data X.  Record log(W_k), where W_k is the
       within-cluster sum of squared distances (= inertia).
    2. Generate B random reference datasets from a null distribution
       (no cluster structure).  Fit k-means on each; record log(W_k^(b)).
    3. Gap(k) = mean_b[ log(W_k^(b)) ] − log(W_k)
       A large Gap means the real data clusters FAR better than noise would.

  This is the core formula from the paper (equation 3).  All of the above is
  taken directly from Tibshirani et al. without modification.

THE 1-SE RULE  [PAPER]
  The paper proposes picking the *smallest* k such that:
    Gap(k) ≥ Gap(k+1) − s(k+1)
  where s(k) = std(log W_k^(b)) × sqrt(1 + 1/B)  [the sqrt(1+1/B) factor
  corrects for the fact that we are comparing to an estimated, not true, mean].

  This is Occam's razor expressed statistically: prefer the simplest model
  (fewest clusters) that is not significantly beaten by the next one.
  Without the 1-SE rule, argmax(Gap) can still creep upward.

  Implemented verbatim in best_k_from_gap() below.

THE NULL DISTRIBUTION IN THE PAPER
  Tibshirani et al. recommend drawing reference data uniformly over the
  axis-aligned bounding box of X.  This is the default in the original paper.

══════════════════════════════════════════════════════════════════════════════
 PART 2 — ADAPTATIONS SPECIFIC TO THIS DATASET
         (SUN RGB-D co-occurrence binary matrices)
══════════════════════════════════════════════════════════════════════════════

ADAPTED NULL DISTRIBUTION  [THIS PROJECT]
  The paper's uniform bounding-box reference breaks down for binary (0/1) data:
  uniform samples over [0, 1]^d would mostly produce values near 0.5, which
  are not valid binary observations and create artificially low inertia in the
  reference fits.

  Instead, we draw each reference feature independently from
  Bernoulli(p_j), where p_j = mean(X[:, j]) is the observed marginal
  frequency of class j in the real data.  This:
    • Preserves each feature's real-world prevalence (e.g. "chair" appears in
      ~70% of scenes, so reference columns for "chair" also average ~70%).
    • Destroys all between-feature correlation — which is the cluster signal
      we are trying to measure.
  This adaptation is not in the original paper; it is a standard practice for
  binary/count data (see also: cluster stability literature, e.g. Fang & Wang).

FOCAL-CLASS SUBMATRIX  [THIS PROJECT]
  The paper applies the gap statistic to a full data matrix.  Here we run it
  once per focal class: we first filter to scenes that contain the focal class,
  then build a feature matrix of *other* co-occurring classes filtered by a
  Jaccard threshold.  This means we are asking "how do scenes containing X
  sub-cluster by what else they contain?" rather than "how do all scenes cluster?"

  This scope restriction is a domain decision — we care about the internal
  diversity of contexts for a given object class — not a change to the statistic.

JACCARD THRESHOLD = 0.10  [THIS PROJECT]
  Features (co-occurring classes) are only included if their Jaccard index with
  the focal class is ≥ 0.10.  Jaccard < 0.10 classes appear so rarely alongside
  the focal class that they contribute mainly noise to the feature matrix.

SCENE × CLASS BINARY ENCODING  [THIS PROJECT]
  Each row of the feature matrix X is one scene; each column is a class.
  Cell = 1 if that class appears at least once in that scene, 0 otherwise.
  We clip raw instance counts to 1 (presence/absence), because k-means on
  raw counts would be dominated by classes with unusually high instance counts
  rather than co-occurrence patterns.

══════════════════════════════════════════════════════════════════════════════
 CHANGES FROM V2
══════════════════════════════════════════════════════════════════════════════
  - Two helper functions added: compute_gap_and_fits(), best_k_from_gap()
  - K selection in the main loop now uses gap statistic, not silhouette argmax.
  - Silhouette scores are still computed and saved for cross-check / curiosity.
  - New output: gap_stats_by_k_v3.csv  (gap, sk, log_W per focal class per k)
  - Comparison table now shows: v2 best-k (silhouette) vs v3 best-k (gap).
  - Output files suffixed _v3.
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
OUTPUT_DIR    = pathlib.Path(r"C:\sunrgbd-exploration")
JACCARD_CSV   = OUTPUT_DIR / "cooccurrence_jaccard.csv"
FEATURES_CSV  = OUTPUT_DIR / "spatial_features.csv"
FREQ_CSV      = OUTPUT_DIR / "class_frequencies.csv"
V2_SUMMARY    = OUTPUT_DIR / "cluster_summary_t010_v2.csv"  # for comparison

EXCLUDE = {"wall", "floor", "ceiling"}
TOP_N   = 25

JACCARD_THRESHOLD = 0.10
MIN_SCENES        = 80
MIN_FEATURES      = 2

K_RANGE     = range(2, 11)   # k = 2 .. 10  (same search space as v2)
K_RANGE_OLD = range(2, 6)    # for threshold-sensitivity plot (keep v1 comparable)

# GAP_B: number of reference bootstrap datasets per k per focal class.
# Higher B → more stable Gap estimate but proportionally more compute.
# 50 is the standard for publications; 20 is fast enough for exploration.
GAP_B = 20

SENSITIVITY_CLASS = "chair"
THRESHOLDS        = np.round(np.arange(0.00, 0.32, 0.02), 2)


# ---------------------------------------------------------------------------
# 2. Gap statistic helpers
# ---------------------------------------------------------------------------

def compute_gap_and_fits(X, k_range, B=GAP_B, random_state=42):
    """
    For every k in k_range, fit KMeans on X and on B null-reference datasets,
    then compute the Gap statistic.

    PATTERN: the function returns two dicts so the caller can use both the
    chosen k's cluster labels AND the full Gap curve without running KMeans twice.

    Returns
    -------
    gap_stats : dict  {k: {'gap': float, 'sk': float, 'log_W': float}}
        gap  = mean_b[log W_b] - log W_real
        sk   = std(log W_b) * sqrt(1 + 1/B)   [Tibshirani correction factor]
        log_W= log(inertia of real KMeans fit)

    fits : dict  {k: (labels_array, silhouette_score)}
        Cached so the caller can retrieve labels for the chosen k without refitting.
    """
    rng = np.random.default_rng(random_state)

    # Marginal Bernoulli probabilities — one per feature column.
    # Drawing from these creates null data that has the same class-frequency
    # profile but zero co-occurrence structure.
    feature_probs = X.values.mean(axis=0)
    feature_probs = np.clip(feature_probs, 1e-6, 1 - 1e-6)  # avoid degenerate Bernoulli

    gap_stats = {}
    fits = {}

    for k in k_range:
        if k >= len(X):
            break

        # Real fit
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)

        if len(np.unique(labels)) < 2:
            continue

        sil = silhouette_score(X, labels, metric="euclidean")
        # +1e-10 guard: inertia should always be > 0 on real data, but just in case
        log_W = np.log(km.inertia_ + 1e-10)

        fits[k] = (labels, sil)

        # Reference fits: B independent null datasets
        ref_log_Ws = []
        for _ in range(B):
            # This is "independent Bernoulli sampling" — the null distribution.
            # It is equivalent to shuffling each column independently, which
            # destroys between-feature correlation while keeping marginals fixed.
            X_ref = rng.binomial(1, feature_probs, size=X.shape).astype(float)
            km_ref = KMeans(n_clusters=k, random_state=42, n_init=10)
            km_ref.fit(X_ref)
            ref_log_Ws.append(np.log(km_ref.inertia_ + 1e-10))

        ref_log_Ws = np.array(ref_log_Ws)
        gap_stats[k] = {
            "gap":   float(ref_log_Ws.mean() - log_W),
            "sk":    float(ref_log_Ws.std() * np.sqrt(1.0 + 1.0 / B)),
            "log_W": float(log_W),
        }

    return gap_stats, fits


def best_k_from_gap(gap_stats):
    """
    Tibshirani 1-SE rule:
      Choose the *smallest* k such that Gap(k) >= Gap(k+1) - s(k+1).

    WHY smallest, not largest?
      We want the simplest model that is statistically indistinguishable from
      the best model.  Once Gap(k) is within one SE of the next k's Gap, the
      extra clusters are not meaningfully earning their complexity.

    Fallback: if no such k is found (Gap monotonically increases beyond 1 SE
    at every step), return the k with the highest Gap value.
    """
    ks = sorted(gap_stats)
    for i, k in enumerate(ks[:-1]):
        k_next = ks[i + 1]
        if gap_stats[k]["gap"] >= gap_stats[k_next]["gap"] - gap_stats[k_next]["sk"]:
            return k
    # No early stop found — fall back to argmax Gap
    return max(ks, key=lambda k: gap_stats[k]["gap"])


# ---------------------------------------------------------------------------
# 3. Load inputs
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
# 4. Build scene × class binary matrix  (unchanged from v2)
# ---------------------------------------------------------------------------
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
# 5. Cluster each focal class using Gap statistic k selection
# ---------------------------------------------------------------------------
results  = {}   # focal -> {labels, k, silhouette, features, n_scenes, X}
sil_rows = []   # every (focal, k, silhouette) attempt — for cross-check
gap_rows = []   # every (focal, k, gap, sk, log_W)

print(f"\nClustering per focal class (k=2..10, Gap statistic, B={GAP_B} bootstraps):")

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

    # --- GAP STATISTIC (replaces the old argmax-silhouette loop) ---
    # compute_gap_and_fits runs KMeans on the real data AND on B reference
    # datasets for every k, caching results so we only fit each k once.
    gap_stats, fits = compute_gap_and_fits(X, K_RANGE)

    if not fits:
        print(f"  {focal:12s}: no valid clustering found")
        continue

    chosen_k = best_k_from_gap(gap_stats)
    best_labels_arr, best_sil = fits[chosen_k]
    best_labels = pd.Series(best_labels_arr, index=focal_scenes.index)

    results[focal] = {
        "labels":     best_labels,
        "k":          chosen_k,
        "silhouette": best_sil,
        "features":   keep,
        "n_scenes":   n_scenes,
        "X":          X,
    }

    # Record silhouette curve (all k)
    for k, (_, sil_k) in fits.items():
        sil_rows.append({
            "focal_class":      focal,
            "k":                k,
            "silhouette_score": round(float(sil_k), 4),
        })

    # Record gap curve (all k)
    for k, gs in gap_stats.items():
        gap_rows.append({
            "focal_class": focal,
            "k":           k,
            "gap":         round(gs["gap"], 4),
            "sk":          round(gs["sk"], 4),
            "log_W":       round(gs["log_W"], 4),
        })

    chosen_gap = gap_stats[chosen_k]["gap"]
    print(f"  {focal:12s}: k={chosen_k} (gap={chosen_gap:+.3f}, sil={best_sil:.3f}), "
          f"scenes={n_scenes:4d}, features={len(keep)}")

# ---------------------------------------------------------------------------
# 6. Cluster profile heatmaps
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
    "SUN RGB-D — Co-occurrence cluster profiles  (k selected by Gap Statistic, v3)\n"
    "Each row = one cluster of scenes containing the focal class\n"
    "Each column = a co-occurring class (sorted by discriminativeness)\n"
    "Colour = fraction of scenes in that cluster where the co-class is present\n"
    f"Jaccard threshold = {JACCARD_THRESHOLD}  |  Gap bootstrap B={GAP_B}",
    fontsize=10, y=1.005,
)
fig_p.tight_layout()
p_prof = OUTPUT_DIR / "cluster_profiles_t010_v3.png"
fig_p.savefig(p_prof, dpi=130, bbox_inches="tight")
print(f"\nSaved: {p_prof}")
plt.close(fig_p)

# ---------------------------------------------------------------------------
# 7. Gap curve plots — one subplot per focal class
#    Shows Gap(k) ± 1 SE, with the chosen k marked.
#    This lets you *see* whether the 1-SE rule fired early or went to argmax.
# ---------------------------------------------------------------------------
gap_df = pd.DataFrame(gap_rows)

fig_g, g_axes = plt.subplots(ROWS, COLS, figsize=(14, 4 * ROWS), sharex=False)
g_axes_flat   = np.array(g_axes).flatten()

for idx, focal in enumerate(display_classes):
    ax  = g_axes_flat[idx]
    sub = gap_df[gap_df["focal_class"] == focal].sort_values("k")

    ax.errorbar(sub["k"], sub["gap"], yerr=sub["sk"],
                marker="o", color="steelblue", linewidth=1.6,
                markersize=5, capsize=3, label="Gap ± 1 SE")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")

    chosen_k = results[focal]["k"]
    chosen_gap_val = sub.loc[sub["k"] == chosen_k, "gap"].values[0]
    ax.axvline(chosen_k, color="red", linestyle="--", linewidth=1.2,
               label=f"chosen k={chosen_k}")
    ax.scatter([chosen_k], [chosen_gap_val], color="red", zorder=5, s=60)

    ax.set_title(f"{focal}", fontsize=9, fontweight="bold")
    ax.set_xlabel("k", fontsize=8)
    ax.set_ylabel("Gap statistic", fontsize=8)
    ax.legend(fontsize=7)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

for idx in range(n_disp, len(g_axes_flat)):
    g_axes_flat[idx].set_visible(False)

fig_g.suptitle(
    "Gap statistic curves per focal class\n"
    "Red dashed line = k chosen by Tibshirani 1-SE rule\n"
    "A gap above 0 means real data clusters better than null (Bernoulli) data",
    fontsize=10, y=1.005,
)
fig_g.tight_layout()
p_gap = OUTPUT_DIR / "gap_curves_v3.png"
fig_g.savefig(p_gap, dpi=130, bbox_inches="tight")
print(f"Saved: {p_gap}")
plt.close(fig_g)

# ---------------------------------------------------------------------------
# 8. Threshold sensitivity (unchanged logic, k swept over K_RANGE_OLD = 2..5)
# ---------------------------------------------------------------------------
print(f"\nThreshold sensitivity sweep for '{SENSITIVITY_CLASS}' (k=2..5, for v1 comparability):")

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
        for k in K_RANGE_OLD:
            if k >= len(X_t): break
            km  = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X_t)
            if len(np.unique(lbl)) < 2: continue
            sil = silhouette_score(X_t, lbl, metric="euclidean")
            if sil > best_sil_t:
                best_k_t, best_sil_t = k, sil
        sens_rows.append({
            "threshold": float(thresh),
            "n_features": n_feat,
            "best_k": best_k_t,
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

    p_sens = OUTPUT_DIR / "cluster_threshold_sensitivity_v3.png"
    fig_s.savefig(p_sens, dpi=130)
    print(f"\nSaved: {p_sens}")
    plt.close(fig_s)

# ---------------------------------------------------------------------------
# 9. Save CSVs
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

summary_df = pd.DataFrame(summary_rows)
sil_df     = pd.DataFrame(sil_rows)
gap_out_df = pd.DataFrame(gap_rows)

summary_df.to_csv(OUTPUT_DIR / "cluster_summary_t010_v3.csv", index=False)
pd.DataFrame(assign_rows).to_csv(OUTPUT_DIR / "cluster_assignments_t010_v3.csv", index=False)
sil_df.to_csv(OUTPUT_DIR / "silhouette_scores_by_k_v3.csv", index=False)
gap_out_df.to_csv(OUTPUT_DIR / "gap_stats_by_k_v3.csv", index=False)

print("\nSaved:")
print(f"  cluster_summary_t010_v3.csv      ({len(summary_rows)} rows)")
print(f"  cluster_assignments_t010_v3.csv  ({len(assign_rows)} rows)")
print(f"  silhouette_scores_by_k_v3.csv    ({len(sil_df)} rows)")
print(f"  gap_stats_by_k_v3.csv            ({len(gap_out_df)} rows)")

# ---------------------------------------------------------------------------
# 10. Comparison table: v2 best-k (silhouette) vs v3 best-k (gap statistic)
# ---------------------------------------------------------------------------
print()
print("=" * 90)
print("COMPARISON: v2 best-k (argmax silhouette)  vs  v3 best-k (Gap statistic 1-SE rule)")
print("=" * 90)

v2_sum = pd.read_csv(V2_SUMMARY)
v2_best = (
    v2_sum.groupby("focal_class")[["k", "silhouette"]]
    .first()
    .rename(columns={"k": "v2_k", "silhouette": "v2_sil"})
)

v3_best = (
    summary_df.groupby("focal_class")[["k", "silhouette"]]
    .first()
    .rename(columns={"k": "v3_k", "silhouette": "v3_sil"})
)

comp = v2_best.join(v3_best, how="outer").reset_index()
comp["k_changed"] = comp["v2_k"] != comp["v3_k"]
comp["k_delta"]   = (comp["v3_k"] - comp["v2_k"]).astype("Int64")

comp_sorted = comp.sort_values(["k_changed", "focal_class"], ascending=[False, True]).reset_index(drop=True)

print(f"\n  {'Focal class':<14} {'v2 k':>5} {'v2 sil':>8} {'v3 k':>6} {'v3 sil':>8} {'Δk':>5} {'Changed?':>10}")
print("  " + "-" * 62)

n_changed = 0
n_lower   = 0
n_higher  = 0

for _, row in comp_sorted.iterrows():
    changed_flag = "YES  <--" if row["k_changed"] else "no"
    if row["k_changed"]:
        n_changed += 1
        if row["k_delta"] < 0:
            n_lower += 1
        else:
            n_higher += 1

    print(f"  {row['focal_class']:<14} "
          f"{int(row['v2_k']):>5} "
          f"{row['v2_sil']:>8.3f} "
          f"{int(row['v3_k']):>6} "
          f"{row['v3_sil']:>8.3f} "
          f"{int(row['k_delta']):>+5} "
          f"{changed_flag:>10}")

print()
print(f"  Classes where k changed   : {n_changed} / {len(comp)}")
print(f"    of which k decreased    : {n_lower}   (gap rule prefers simpler model)")
print(f"    of which k increased    : {n_higher}   (gap curve peaked later)")
print()
print("  INTERPRETATION:")
print("  A decrease in k means v2 was over-clustering — silhouette was rewarding")
print("  tighter-but-artificial splits.  The gap statistic only accepts a higher k")
print("  when real data clusters *meaningfully better* than null (Bernoulli) data.")
print("  A lower v3_sil for the same class is expected and correct: the chosen k")
print("  is simpler, so per-cluster cohesion is naturally lower — but the cluster")
print("  boundaries correspond to real structure, not arbitrary splitting.")

print("\nDone.")
