"""
SUN RGB-D Preprocessing Pipeline Visualizer
=============================================
Illustrates the full path from raw sensor data → 3D point cloud:

    RGB image
    Raw depth (uint16 PNG, holes present)
    Filled depth (depth_bfx, bilateral-filter inpainting)
    Metric conversion  (uint16 / 10000 → metres)
    Back-projection   (2-D pixel + depth → 3-D XYZ via camera intrinsics)

Produces three output figures:
  panel_2d.png          — side-by-side 2-D pipeline overview
  panel_depth_stats.png — depth coverage and distribution
  panel_3d.png          — matplotlib 3-D scatter of the coloured point cloud
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from PIL import Image
from pathlib import Path
import open3d as o3d

# ─── 0. Paths ────────────────────────────────────────────────────────────────
SCENE_DIR = Path("C:/Users/ryana/Downloads/SUNRGBD/SUNRGBD/kv1/NYUdata/NYU0001")
OUT_DIR   = Path("C:/sunrgbd-exploration/06_preprocessing")

RGB_PATH       = SCENE_DIR / "image"       / "NYU0001.jpg"
DEPTH_PATH     = SCENE_DIR / "depth"       / "NYU0001.png"
DEPTH_BFX_PATH = SCENE_DIR / "depth_bfx"  / "NYU0001.png"
INTRINSICS_TXT = SCENE_DIR / "intrinsics.txt"

# ─── 1. Load data ────────────────────────────────────────────────────────────
rgb_img   = np.array(Image.open(RGB_PATH))                  # (H, W, 3) uint8
depth_raw = np.array(Image.open(DEPTH_PATH),  dtype=np.uint16)   # (H, W) uint16
depth_bfx = np.array(Image.open(DEPTH_BFX_PATH), dtype=np.uint16)

H, W = depth_raw.shape

# ─── 2. Parse camera intrinsics ──────────────────────────────────────────────
# The intrinsics file stores the 3×3 camera matrix K as a flat row:
#   fx  0  cx  0  fy  cy  0  0  1
# K projects a 3-D camera-frame point (X, Y, Z) to image pixel (u, v):
#   u = fx * X/Z + cx
#   v = fy * Y/Z + cy
# We invert this to go from pixel (u,v) + depth Z back to 3-D.
vals = list(map(float, INTRINSICS_TXT.read_text().split()))
fx, cx = vals[0], vals[2]
fy, cy = vals[4], vals[5]

# ─── 3. Metric depth conversion ──────────────────────────────────────────────
# SUN RGB-D stores depth as uint16 in units of 0.1 mm (i.e. divide by 10 000
# to get metres).  Raw zeros mean "no reading" — the sensor had no return.
DEPTH_SCALE = 10_000.0

depth_m_raw = depth_raw.astype(np.float32) / DEPTH_SCALE    # metres, 0 = hole
depth_m_bfx = depth_bfx.astype(np.float32) / DEPTH_SCALE   # metres, filled

# ─── 4. Hole mask ────────────────────────────────────────────────────────────
# "Holes" are pixels where the raw sensor returned 0 (no valid depth).
# depth_bfx fills them using a bilateral filter — a smoothing filter that
# respects edges by weighting neighbours by both spatial distance and
# intensity difference.
hole_mask   = (depth_raw == 0)          # True where raw has no reading
filled_mask = hole_mask & (depth_bfx > 0)   # holes that were successfully filled

n_pixels   = H * W
n_holes    = hole_mask.sum()
n_filled   = filled_mask.sum()
coverage_raw = 1.0 - n_holes / n_pixels
coverage_bfx = (depth_bfx > 0).sum() / n_pixels

# ─── 5. Back-projection to 3-D point cloud ───────────────────────────────────
# Back-projection is the inverse of perspective projection.
# For every valid pixel (u, v) with depth Z:
#   X = (u - cx) * Z / fx
#   Y = (v - cy) * Z / fy
#   Z = depth_m[v, u]
# This gives a point in the camera coordinate frame (metres).
# The operation is called "un-projecting" or "lifting" from 2-D to 3-D.

valid = depth_m_bfx > 0                                 # boolean mask (H, W)
v_idx, u_idx = np.where(valid)                          # pixel row/col indices

Z = depth_m_bfx[v_idx, u_idx]
X = (u_idx - cx) * Z / fx
Y = (v_idx - cy) * Z / fy

xyz    = np.stack([X, Y, Z], axis=1)                    # (N, 3) point cloud
colors = rgb_img[v_idx, u_idx] / 255.0                  # (N, 3) normalised RGB

# Sub-sample for plotting — plotting all ~240k points in matplotlib is slow
rng  = np.random.default_rng(42)
step = max(1, len(xyz) // 10_000)
idx  = np.arange(0, len(xyz), step)
xyz_sub    = xyz[idx]
colors_sub = colors[idx]

# ─── 6. Figure A: 2-D pipeline overview ──────────────────────────────────────
fig_a, axes = plt.subplots(2, 3, figsize=(16, 9))
fig_a.suptitle("SUN RGB-D Preprocessing Pipeline — NYU0001 (Kinect v1)",
               fontsize=14, fontweight="bold")

# Panel 1: RGB
axes[0, 0].imshow(rgb_img)
axes[0, 0].set_title("(1) RGB image  [561×427]")
axes[0, 0].axis("off")

# Panel 2: Raw depth — colourmap; zeros are shown as black
depth_display = np.ma.masked_where(depth_raw == 0, depth_m_raw)
im2 = axes[0, 1].imshow(depth_display, cmap="plasma", vmin=0.5, vmax=4.0)
axes[0, 1].set_title(f"(2) Raw depth  [uint16 → metres]\n"
                     f"Coverage: {coverage_raw*100:.1f}%  |  Holes: {n_holes:,} px")
axes[0, 1].axis("off")
plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, label="depth (m)")

# Panel 3: Filled depth (depth_bfx)
im3 = axes[0, 2].imshow(depth_m_bfx, cmap="plasma", vmin=0.5, vmax=4.0)
axes[0, 2].set_title(f"(3) Bilateral-filtered depth  [depth_bfx]\n"
                     f"Coverage: {coverage_bfx*100:.1f}%  |  Filled: {n_filled:,} px")
axes[0, 2].axis("off")
plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, label="depth (m)")

# Panel 4: Hole visualisation — where are the raw sensor holes?
hole_vis = np.zeros((*hole_mask.shape, 3), dtype=np.uint8)
hole_vis[~hole_mask]  = [180, 220, 180]   # valid   → light green
hole_vis[filled_mask] = [255, 165,   0]   # filled  → orange
hole_vis[hole_mask & ~filled_mask] = [220, 50, 50]   # unfilled → red

axes[1, 0].imshow(hole_vis)
axes[1, 0].set_title("(4) Hole map\n"
                     "  green=valid  orange=filled  red=unfilled")
axes[1, 0].axis("off")

# Panel 5: Depth difference (filled − raw) — magnitude of inpainting
diff = depth_m_bfx - depth_m_raw
diff[~hole_mask] = 0.0                          # only show filled regions
im5 = axes[1, 1].imshow(diff, cmap="hot", vmin=0, vmax=diff.max())
axes[1, 1].set_title("(5) Inpainted depth change\n(depth_bfx − raw, holes only)")
axes[1, 1].axis("off")
plt.colorbar(im5, ax=axes[1, 1], fraction=0.046, label="Δ depth (m)")

# Panel 6: RGB overlaid with depth contours (fusion preview)
axes[1, 2].imshow(rgb_img, alpha=0.75)
contour_depth = np.where(depth_m_bfx > 0, depth_m_bfx, np.nan)
cset = axes[1, 2].contour(contour_depth, levels=10, cmap="coolwarm", alpha=0.7)
axes[1, 2].set_title("(6) RGB + depth iso-contours\n(preview of 2-D/3-D fusion)")
axes[1, 2].axis("off")

plt.tight_layout()
out_a = OUT_DIR / "panel_2d.png"
fig_a.savefig(out_a, dpi=150, bbox_inches="tight")
plt.close(fig_a)
print(f"Saved: {out_a}")

# ─── 7. Figure B: Depth statistics ───────────────────────────────────────────
fig_b, axes_b = plt.subplots(1, 2, figsize=(12, 4))
fig_b.suptitle("Depth Statistics — NYU0001", fontsize=13, fontweight="bold")

# Left: histogram comparing raw vs filled distributions
valid_raw = depth_m_raw[depth_m_raw > 0].ravel()
valid_bfx = depth_m_bfx[depth_m_bfx > 0].ravel()

axes_b[0].hist(valid_raw, bins=80, color="steelblue", alpha=0.6, label=f"Raw  ({len(valid_raw):,} px)")
axes_b[0].hist(valid_bfx, bins=80, color="darkorange", alpha=0.5, label=f"Filled  ({len(valid_bfx):,} px)")
axes_b[0].set_xlabel("Depth (metres)")
axes_b[0].set_ylabel("Pixel count")
axes_b[0].set_title("Depth distribution before/after hole-filling")
axes_b[0].legend()
axes_b[0].grid(True, alpha=0.3)

# Right: coverage bar — what fraction of the frame has valid depth?
labels  = ["Raw", "Filled (bfx)"]
covers  = [coverage_raw * 100, coverage_bfx * 100]
colours = ["steelblue", "darkorange"]
bars = axes_b[1].bar(labels, covers, color=colours, width=0.4)
axes_b[1].set_ylim(0, 105)
axes_b[1].set_ylabel("Valid pixels (%)")
axes_b[1].set_title("Depth coverage before/after hole-filling")
axes_b[1].grid(True, alpha=0.3, axis="y")
for bar, val in zip(bars, covers):
    axes_b[1].text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                   f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
out_b = OUT_DIR / "panel_depth_stats.png"
fig_b.savefig(out_b, dpi=150, bbox_inches="tight")
plt.close(fig_b)
print(f"Saved: {out_b}")

# ─── 8. Figure C: 3-D point cloud (matplotlib) ───────────────────────────────
# The axes convention follows the camera frame:
#   +X  right,  +Y  down,  +Z  forward (into the scene)
# We flip Y here (Y → -Y) so "up" is visually up in the plot.

# For depth-colored views we map Z (forward distance) to a plasma colormap.
# This renders more clearly than per-point RGB in matplotlib's software renderer.
Z_sub    = xyz_sub[:, 2]
depth_norm = (Z_sub - Z_sub.min()) / (Z_sub.max() - Z_sub.min() + 1e-6)
depth_colors = plt.cm.plasma(depth_norm)[:, :3]   # (N, 3) — drop alpha

fig_c = plt.figure(figsize=(18, 10))
fig_c.suptitle("3-D Point Cloud — NYU0001 (back-projected from filled depth)",
               fontsize=13, fontweight="bold")

# We show four views: two coloured by depth, two coloured by RGB.
# Using larger point size (s=2) makes individual points visible at 150 dpi.
view_configs = [
    dict(elev=20,  azim=-60,  title="(A) Perspective — depth-coloured",  c=depth_colors),
    dict(elev=90,  azim=-90,  title="(B) Top-down — depth-coloured",     c=depth_colors),
    dict(elev=20,  azim=-60,  title="(C) Perspective — RGB-coloured",    c=colors_sub),
    dict(elev=10,  azim=30,   title="(D) Side view — RGB-coloured",      c=colors_sub),
]

for i, cfg in enumerate(view_configs):
    ax = fig_c.add_subplot(2, 2, i + 1, projection="3d")
    ax.scatter(xyz_sub[:, 0],        # X  (right)
               xyz_sub[:, 2],        # Z  (depth, forward)
               -xyz_sub[:, 1],       # -Y (up in plot)
               c=cfg["c"],
               s=2,
               linewidths=0,
               depthshade=False)     # depthshade=False preserves per-point colour
    ax.set_xlabel("X (m)", fontsize=8)
    ax.set_ylabel("Z — depth (m)", fontsize=8)
    ax.set_zlabel("Height (m)", fontsize=8)
    ax.set_title(cfg["title"], fontsize=10)
    ax.view_init(elev=cfg["elev"], azim=cfg["azim"])
    ax.tick_params(labelsize=7)

plt.tight_layout()
out_c = OUT_DIR / "panel_3d.png"
fig_c.savefig(out_c, dpi=150, bbox_inches="tight")
plt.close(fig_c)
print(f"Saved: {out_c}")

# ─── 9. Open3D interactive viewer (optional — comment out if headless) ────────
# open3d gives a much richer interactive 3-D experience. It is left here as
# a separate step so the script can run headless above and you can opt in.
def view_open3d():
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.stack([xyz[:, 0], -xyz[:, 1], xyz[:, 2]], axis=1)   # flip Y → up
    )
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="SUN RGB-D Point Cloud — NYU0001",
        width=1280, height=720,
    )

# Uncomment to launch the interactive viewer:
# view_open3d()

print("\nAll figures saved to:", OUT_DIR)
print(f"  panel_2d.png         — 2-D preprocessing pipeline")
print(f"  panel_depth_stats.png — depth coverage & histogram")
print(f"  panel_3d.png          — 3-D point cloud scatter")
