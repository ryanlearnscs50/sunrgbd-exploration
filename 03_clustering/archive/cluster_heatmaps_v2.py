"""
cluster_heatmaps_v2.py
----------------------
Produces standardised 10×10 heatmap panels for every clustered focal class,
using the v2 clustering results (k=2..10, Jaccard threshold=0.10).

Design rules
------------
- Fixed 10×10 grid per panel.
- Columns = top-10 co-class features ranked by inter-cluster variance.
  Extra columns (when a class has < 10 surviving Jaccard features) are
  padded on the right with solid grey (RGB 0.75, 0.75, 0.75).
- Rows = clusters sorted largest → smallest.
  Extra rows (when best-k < 10) are padded at the bottom with solid grey.
- Grey cells carry no text annotation.
- Non-grey cells: numeric value (2 d.p.); white text if value > 0.5,
  dark-grey text otherwise.
- Panel title: "class  k=N  sil=X.XX"   (fontsize 11pt)
- Axis tick labels: 9pt   |   Cell annotations: 7pt
- Colorbar label: "Fraction of scenes where co-class is present"
- 3-column grid layout for the combined figure.

WHY rebuild profiles from the raw data rather than from cluster_summary?
  cluster_summary_t010_v2.csv stores only the top-3 co-classes per cluster.
  The heatmap needs the full set of surviving features for every cluster so
  that inter-cluster variance can be computed correctly and columns can be
  sorted by discriminativeness.  The raw spatial features + cluster
  assignments give us that full matrix without re-running KMeans.

Outputs
-------
  cluster_profiles_v2.png          — combined 3-column grid, dpi=150
  cluster_plots/cluster_profile_*.png — one figure per focal class, dpi=150
"""

import copy
import pathlib
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = pathlib.Path(r"C:\sunrgbd-exploration")
SUBDIR     = OUTPUT_DIR / "cluster_plots"
SUBDIR.mkdir(exist_ok=True)

JACCARD_CSV  = OUTPUT_DIR / "cooccurrence_jaccard.csv"
FEATURES_CSV = OUTPUT_DIR / "spatial_features.csv"
FREQ_CSV     = OUTPUT_DIR / "class_frequencies.csv"
ASSIGN_CSV   = OUTPUT_DIR / "cluster_assignments_t010_v2.csv"
SUMMARY_CSV  = OUTPUT_DIR / "cluster_summary_t010_v2.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
JACCARD_THRESHOLD = 0.10
EXCLUDE           = {"wall", "floor", "ceiling"}
TOP_N             = 25
GRID              = 10                     # fixed panel dimensions
GREY_RGB          = (0.75, 0.75, 0.75)    # padding colour

# ---------------------------------------------------------------------------
# Load all inputs
# ---------------------------------------------------------------------------
print("Loading data...", flush=True)
jac        = pd.read_csv(JACCARD_CSV, index_col=0)
sf         = pd.read_csv(FEATURES_CSV)
freq_df    = pd.read_csv(FREQ_CSV)
assign_df  = pd.read_csv(ASSIGN_CSV)
summary_df = pd.read_csv(SUMMARY_CSV)

selected_classes: list[str] = (
    freq_df.loc[~freq_df["class"].isin(EXCLUDE), "class"]
    .head(TOP_N).tolist()
)

# ---------------------------------------------------------------------------
# Build scene × class binary matrix
# ---------------------------------------------------------------------------
# PATTERN: "binary bag-of-classes" — identical to the clustering script so
# that feature values are consistent with what KMeans was trained on.
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

print(f"Scene x class matrix: {scene_class.shape[0]:,} x {scene_class.shape[1]}")

# Best-k and silhouette per focal class (one entry per class, from summary)
best_params: dict[str, dict] = (
    summary_df.groupby("focal_class")[["k", "silhouette"]]
    .first()
    .to_dict("index")
)

# ---------------------------------------------------------------------------
# Profile builder
# ---------------------------------------------------------------------------
def build_profile(focal: str):
    """
    Rebuild the full cluster profile matrix for a focal class from the raw
    scene×feature matrix and the saved cluster assignments.

    Returns
    -------
    profile : DataFrame  shape (n_real_clusters, n_top_features)
        Rows = clusters sorted by size (largest first).
        Columns = up to 10 features sorted by inter-cluster variance (desc).
        Columns are already the final top-10 slice.
    cluster_sizes : list[int]
        Scene count for each cluster row, in the same order.
    n_real_features : int
        How many features survived the Jaccard filter (before the top-10 cap).
        Used only for diagnostic printing.
    """
    # Jaccard-based feature selection (same rule as clustering script)
    others  = [c for c in selected_classes if c != focal]
    jac_row = jac.loc[focal, others]
    keep    = jac_row[jac_row >= JACCARD_THRESHOLD].index.tolist()

    # Scene × feature sub-matrix (scenes that contain the focal class)
    focal_scenes = scene_class[scene_class[focal] == 1]
    X = focal_scenes[keep]

    # Cluster assignments (0-based KMeans labels from the v2 CSV)
    fa = (
        assign_df[assign_df["focal_class"] == focal]
        .set_index("scene")["cluster"]
    )

    # Align: restrict to scenes present in BOTH X and the assignments table.
    # ENGINEERING DECISION: spatial_features.csv covers a subset of scenes
    # (~7k of ~10k).  KMeans was run on that subset, so assignments only
    # exist for scenes in spatial_features.  The intersection is therefore
    # identical to the set KMeans saw — no information is lost.
    common = X.index.intersection(fa.index)
    X      = X.loc[common]
    labels = fa.loc[common]

    # Sort clusters by size (largest first)
    cluster_ids   = sorted(labels.unique(), key=lambda c: -(labels == c).sum())
    cluster_sizes = [int((labels == c).sum()) for c in cluster_ids]

    # Profile matrix: rows = clusters, columns = features
    profile = pd.DataFrame(
        {cid: X.loc[labels == cid].mean() for cid in cluster_ids}
    ).T   # shape: (n_clusters, n_features)

    # Sort columns by inter-cluster variance (most discriminating leftmost)
    # PATTERN: "discriminant ordering" — the feature that differs most across
    # clusters goes to column 0 so the eye naturally sees the key separators.
    col_var     = profile.var(axis=0)
    sorted_cols = col_var.sort_values(ascending=False).index.tolist()
    n_real_feat = len(sorted_cols)

    top_cols = sorted_cols[:GRID]
    profile  = profile[top_cols]

    return profile, cluster_sizes, n_real_feat


# ---------------------------------------------------------------------------
# Panel renderer
# ---------------------------------------------------------------------------
def render_panel(ax, focal: str, profile: pd.DataFrame,
                 cluster_sizes: list[int], best_k: int, best_sil: float,
                 cmap_with_bad):
    """
    Draw one standardised 10×10 heatmap panel onto ax.

    Real cells get the YlOrRd colour + numeric annotation.
    Padding cells (grey) get no annotation.

    Returns the AxesImage (needed for attaching a colorbar).
    """
    n_r = min(len(cluster_sizes), GRID)   # actual cluster rows
    n_c = min(len(profile.columns), GRID) # actual feature columns

    # Build 10×10 data array; NaN marks padded cells
    data_grid = np.full((GRID, GRID), np.nan)
    data_grid[:n_r, :n_c] = profile.values[:n_r, :n_c]

    # PATTERN: masked array + cmap.set_bad() → NaN renders as solid grey
    # without affecting the colormap scale for real data.
    masked = np.ma.masked_invalid(data_grid)
    im = ax.imshow(masked, aspect="auto", cmap=cmap_with_bad, vmin=0, vmax=1)

    # Numeric annotations for real cells only
    for r in range(n_r):
        for c in range(n_c):
            v = data_grid[r, c]
            if not np.isnan(v):
                txt_color = "white" if v > 0.5 else "#444444"
                ax.text(c, r, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=7, color=txt_color)

    # X-axis: feature names for real columns, empty string for padding
    x_labels = list(profile.columns[:n_c]) + [""] * (GRID - n_c)
    ax.set_xticks(range(GRID))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)

    # Y-axis: "C1  n=XXXX" for real rows, empty string for padding
    y_labels = (
        [f"C{i + 1}  n={cluster_sizes[i]:,}" for i in range(n_r)]
        + [""] * (GRID - n_r)
    )
    ax.set_yticks(range(GRID))
    ax.set_yticklabels(y_labels, fontsize=9)

    ax.set_title(f"{focal}  k={best_k}  sil={best_sil:.2f}",
                 fontsize=11, fontweight="bold")

    return im


# ---------------------------------------------------------------------------
# Prepare colormap (copy so we don't mutate the global instance)
# ---------------------------------------------------------------------------
cmap_with_bad = copy.copy(plt.cm.YlOrRd)
cmap_with_bad.set_bad(color=GREY_RGB)

# Classes that have cluster assignments
display_classes = [
    c for c in selected_classes
    if c in best_params and not assign_df[assign_df["focal_class"] == c].empty
]
print(f"Focal classes to plot: {len(display_classes)}")

# Pre-build all profiles (reuse for both grid and individual figures)
profiles: dict[str, tuple] = {}
for focal in display_classes:
    prof, sizes, n_feat = build_profile(focal)
    profiles[focal] = (prof, sizes, n_feat)
    params = best_params[focal]
    print(f"  {focal:<12}: k={params['k']:>2}, sil={params['silhouette']:.3f}, "
          f"real_features={n_feat}, real_clusters={len(sizes)}")

# ---------------------------------------------------------------------------
# Combined grid figure  (3-column layout)
# ---------------------------------------------------------------------------
COLS       = 3
N_CLASSES  = len(display_classes)
GRID_ROWS  = (N_CLASSES + COLS - 1) // COLS

# Panel sizing:
#   - 5.0 in wide per panel (10 cols + y-tick labels)
#   - 5.8 in tall per panel (10 rows + rotated x-tick labels + title)
#   - 1.2 in extra width on right for colorbar
PANEL_W = 5.0
PANEL_H = 5.8
FIG_W   = COLS * PANEL_W + 1.6   # ~16.6 in
FIG_H   = GRID_ROWS * PANEL_H + 1.5   # ~48.9 in for 8 rows

print(f"\nBuilding grid figure ({COLS} cols x {GRID_ROWS} rows)  "
      f"figsize=({FIG_W:.1f}, {FIG_H:.1f})  dpi=150 ...")

fig, axes = plt.subplots(
    GRID_ROWS, COLS,
    figsize=(FIG_W, FIG_H),
)
axes_flat = np.array(axes).flatten()

last_im = None
for idx, focal in enumerate(display_classes):
    ax     = axes_flat[idx]
    prof, sizes, _ = profiles[focal]
    params = best_params[focal]
    im = render_panel(ax, focal, prof, sizes,
                      params["k"], params["silhouette"], cmap_with_bad)
    # Add per-panel colorbar (same pattern as original script)
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("Fraction of scenes where\nco-class is present",
                   fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    last_im = im

# Hide unused panels
for idx in range(N_CLASSES, len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle(
    "SUN RGB-D  —  Co-occurrence cluster profiles  (k=2..10, Jaccard threshold = 0.10)\n"
    "Rows = clusters sorted by size (largest first)   "
    "Columns = top-10 co-class features (sorted by inter-cluster discriminativeness)\n"
    "Cell value = fraction of cluster scenes where that co-class is present   "
    "Grey cells = grid padding",
    fontsize=11, y=1.002,
)

fig.tight_layout(h_pad=1.8, w_pad=0.8)

out_grid = OUTPUT_DIR / "cluster_profiles_v2.png"
fig.savefig(out_grid, dpi=150, bbox_inches="tight")
print(f"Saved: {out_grid}")
plt.close(fig)

# ---------------------------------------------------------------------------
# Individual per-class figures  →  cluster_plots/cluster_profile_{cls}.png
# ---------------------------------------------------------------------------
print(f"\nSaving {N_CLASSES} individual figures to {SUBDIR} ...")

for focal in display_classes:
    prof, sizes, _ = profiles[focal]
    params = best_params[focal]

    fig_s, ax_s = plt.subplots(figsize=(7.5, 7.5))
    im_s = render_panel(ax_s, focal, prof, sizes,
                        params["k"], params["silhouette"], cmap_with_bad)

    cbar_s = fig_s.colorbar(im_s, ax=ax_s, shrink=0.88, pad=0.02)
    cbar_s.set_label("Fraction of scenes where co-class is present", fontsize=9)
    cbar_s.ax.tick_params(labelsize=8)

    fig_s.tight_layout()
    out_ind = SUBDIR / f"cluster_profile_{focal}.png"
    fig_s.savefig(out_ind, dpi=150, bbox_inches="tight")
    plt.close(fig_s)
    print(f"  Saved: cluster_profile_{focal}.png")

print(f"\nAll done.")
print(f"  Grid figure : {out_grid}")
print(f"  Individual  : {SUBDIR}  ({N_CLASSES} files)")
