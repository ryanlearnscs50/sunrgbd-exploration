"""
threshold_sweep.py
------------------
Runs the full K-Means clustering pipeline across all top-25 classes at a
range of Jaccard thresholds above 0.10 and reports per-class silhouette scores
plus aggregate stats.  No PNGs are written — this is a fast comparison tool.

We deliberately skip very high thresholds (>0.20) because at that point most
classes are left with <3 features, making silhouette=1.0 a trivial artefact
(clusters collapse to "has feature X or doesn't") rather than genuine structure.
"""

import warnings
import pathlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

OUTPUT_DIR   = pathlib.Path(r"C:\sunrgbd-exploration")
JACCARD_CSV  = OUTPUT_DIR / "cooccurrence_jaccard.csv"
FEATURES_CSV = OUTPUT_DIR / "spatial_features.csv"
FREQ_CSV     = OUTPUT_DIR / "class_frequencies.csv"

EXCLUDE    = {"wall", "floor", "ceiling"}
TOP_N      = 25
MIN_SCENES = 80
K_RANGE    = range(2, 6)

# Thresholds to test — we already know 0.05 and 0.10, so focus on the gap above
THRESHOLDS = [0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]

# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
jac     = pd.read_csv(JACCARD_CSV, index_col=0)
sf      = pd.read_csv(FEATURES_CSV)
freq_df = pd.read_csv(FREQ_CSV)

selected_classes = (
    freq_df.loc[~freq_df["class"].isin(EXCLUDE), "class"]
    .head(TOP_N).tolist()
)

scene_class = (
    sf[sf["category"].isin(selected_classes)]
    .groupby(["scene", "category"]).size()
    .unstack(fill_value=0).clip(upper=1)
)
for cls in selected_classes:
    if cls not in scene_class.columns:
        scene_class[cls] = 0
scene_class = scene_class[selected_classes]

# ---------------------------------------------------------------------------
# Cluster one focal class at a given threshold; return (sil, k, n_features)
# or None if skipped.
# ---------------------------------------------------------------------------
def cluster_focal(focal, threshold):
    focal_scenes = scene_class[scene_class[focal] == 1]
    if len(focal_scenes) < MIN_SCENES:
        return None
    others  = [c for c in selected_classes if c != focal]
    jac_row = jac.loc[focal, others]
    keep    = jac_row[jac_row >= threshold].index.tolist()
    if len(keep) < 2:
        return None
    X = focal_scenes[keep]
    best_k, best_sil = 2, -1.0
    for k in K_RANGE:
        if k >= len(X): break
        lbl = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        if len(np.unique(lbl)) < 2: continue
        sil = silhouette_score(X, lbl, metric="euclidean")
        if sil > best_sil:
            best_k, best_sil = k, sil
    return (round(best_sil, 4), best_k, len(keep))

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
rows = []
print(f"Sweeping {len(THRESHOLDS)} thresholds across all classes ...\n")

for thresh in THRESHOLDS:
    sils, n_feat_list, valid = [], [], 0
    for focal in selected_classes:
        res = cluster_focal(focal, thresh)
        if res is None:
            continue
        sil, k, nf = res
        sils.append(sil)
        n_feat_list.append(nf)
        valid += 1
    rows.append({
        "threshold":     thresh,
        "n_classes":     valid,
        "mean_sil":      round(np.mean(sils), 4) if sils else float("nan"),
        "median_sil":    round(np.median(sils), 4) if sils else float("nan"),
        "min_sil":       round(np.min(sils), 4) if sils else float("nan"),
        "mean_features": round(np.mean(n_feat_list), 1) if n_feat_list else float("nan"),
    })
    print(f"  thresh={thresh:.2f}  classes={valid:2d}  "
          f"mean_sil={rows[-1]['mean_sil']:.4f}  "
          f"median_sil={rows[-1]['median_sil']:.4f}  "
          f"mean_features={rows[-1]['mean_features']:.1f}")

# ---------------------------------------------------------------------------
# Per-class breakdown at each threshold
# ---------------------------------------------------------------------------
print("\n\nPer-class silhouette at each threshold:")
print(f"{'class':<12}" + "".join(f"  t={t:.2f}" for t in THRESHOLDS))
print("-" * (12 + 9 * len(THRESHOLDS)))

for focal in selected_classes:
    results_per_thresh = []
    for thresh in THRESHOLDS:
        res = cluster_focal(focal, thresh)
        if res is None:
            results_per_thresh.append("  skip")
        else:
            results_per_thresh.append(f"  {res[0]:.3f}")
    line = f"{focal:<12}" + "".join(results_per_thresh)
    print(line)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
sweep_df = pd.DataFrame(rows)
print("\n\nAggregate summary:")
print(sweep_df.to_string(index=False))

best_row = sweep_df.loc[sweep_df["mean_sil"].idxmax()]
print(f"\nBest mean silhouette: threshold={best_row['threshold']:.2f}  "
      f"mean_sil={best_row['mean_sil']:.4f}  "
      f"n_classes={int(best_row['n_classes'])}")
