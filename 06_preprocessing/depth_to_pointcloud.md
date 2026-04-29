# From 2D Pixels to 3D Point Cloud — SUN RGB-D Preprocessing Pipeline

**Dataset:** SUN RGB-D (Song et al., 2015)
**Sensors covered:** Kinect v1, Kinect v2, Asus Xtion, Intel RealSense

This document describes every step that transforms a raw RGB image and a raw
depth map into a coloured 3D point cloud. Steps 1–3 are preprocessing of the
raw sensor data. Steps 4–5 are the core geometric transformation. Steps 6–8 are
post-processing to clean and compact the result.

---

## Background: The Pinhole Camera Model

A pinhole camera maps a 3D world point onto a 2D image plane by projecting
through a single point — the **camera centre** (also called the optical centre).
The distance from the camera centre to the image plane is the **focal length** f.

### Deriving the Projection Formula with Similar Triangles

Consider a 3D point **P** at coordinates (X, Y, Z) in the **camera frame**
(Z is the depth, pointing forward; X points right; Y points downward).

Look at the scene from the side (the XZ plane):

```
 Camera centre
       O
       |\
       | \
       |  \         (the ray from O to P)
       |   \
  -----+----x_img--  <-- image plane (at distance f from O)
       |       \
       |        \
       |         \
       |          P = (X, Z)
       |
       +-------> Z (depth, into scene)
```

The image plane sits at distance `f` from O. The point P is at depth Z.
Two right triangles share the same angle at O:

| Triangle | base | height |
|---|---|---|
| Small (image plane) | f | x_img |
| Large (scene depth) | Z | X |

By similar triangles:

```
x_img     X
----- = -----
  f       Z

            f · X
x_img  =  -------
              Z
```

The same reasoning applies in the vertical direction:

```
            f · Y
y_img  =  -------
              Z
```

### Converting to Pixel Coordinates

`x_img` and `y_img` are distances in physical units (millimetres). To convert to
pixel indices (u, v), divide by the pixel size and add the **principal point**
(cx, cy) — the pixel at which the optical axis pierces the image plane:

```
u = fx · (X/Z) + cx        where fx = f / pixel_width_x
v = fy · (Y/Z) + cy        where fy = f / pixel_width_y
```

`fx` and `fy` are called the **focal lengths in pixel units**. They absorb the
physical focal length and the sensor pixel size into a single number. For most
cameras `fx ≈ fy`; they differ slightly due to non-square pixels.

In matrix form this is the standard **intrinsic matrix K**:

```
    ⎡ fx   0   cx ⎤
K = ⎢  0  fy   cy ⎥
    ⎣  0   0    1 ⎦
```

### The Back-Projection (Inverse) Formula

We have the pixel (u, v) and the depth value Z = d. We want (X, Y, Z).
Rearranging the projection equations:

```
Z = d
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
```

This is called **back-projection** or **unprojection** — lifting a 2D pixel
back into 3D space. It is the core of the entire pipeline.

---

## Step 1 — Depth Map Encoding in SUN RGB-D

The depth sensor returns a 16-bit greyscale PNG (`depth/`). Each pixel stores
depth in units of **0.1 mm** (i.e. tenths of a millimetre).

To convert to **metres**:

```python
depth_metres = depth_uint16.astype(float) / 10_000.0
```

A raw pixel value of 0 means **no valid reading** — a hole in the depth map.

---

## Step 2 — Why Depth Maps Have Holes

Structured-light and time-of-flight depth sensors fail to return valid readings
in several situations:

| Cause | What happens | Common objects affected |
|---|---|---|
| **Specular reflection** | IR light bounces away from sensor instead of returning | Metal, glass, glossy plastic |
| **Dark surface absorption** | Dark material absorbs IR light, returns nothing | Black fabric, matte-black objects |
| **Edge / occlusion shadow** | The IR projector and sensor camera have a small physical gap (baseline). Objects at depth edges cast a "shadow" that the projector illuminates but the camera cannot see | Object boundaries, corners |
| **Transparent surfaces** | IR passes through glass rather than reflecting | Windows, glass tables |
| **Range limits** | Objects too close (<0.5 m) or too far (>5–8 m) for the sensor | Very close foreground |

The result is a depth image with irregular black holes, particularly along object
edges and on reflective or dark surfaces.

---

## Step 3 — Hole Filling: Cross-Bilateral Filter

SUN RGB-D provides a pre-filled depth image in the `depth_bfx/` folder for
every scene. The filling uses a **cross-bilateral filter** guided by the RGB
image.

**What a bilateral filter does:**

A regular Gaussian blur fills holes by averaging neighbouring pixels, weighted
only by spatial distance — which smears depth across object edges.

A bilateral filter adds a second weight based on **value similarity**:

```
weight(p, q) = exp( −‖p−q‖²/(2σ_s²) )   ← spatial proximity
             × exp( −‖I_p−I_q‖²/(2σ_r²) ) ← intensity (range) similarity
```

A **cross-bilateral** filter uses the RGB image's intensities for the range
weight while filling the depth map. This means:
- Pixels on the same side of a colour edge get high weight → smooth fill
- Pixels across a colour edge (where a real depth discontinuity likely exists)
  get low weight → edge is preserved

The SUN RGB-D toolbox applies this iteratively at multiple scales (coarse-to-fine
inpainting) so that large holes are also filled.

> **Note:** The bilateral filter is applied by the dataset curators.
> `depth_bfx` is included in the raw download. You do not need to rerun it.

---

## Step 4 — Depth–RGB Alignment

The depth sensor and the RGB camera are separate physical devices mounted a few
centimetres apart. Their images are not naturally aligned — a pixel at (u, v)
in the RGB image does not correspond to pixel (u, v) in the raw depth image.

Alignment (also called **registration** or **rectification**) involves:

1. **Intrinsic calibration:** Each camera has its own fx, fy, cx, cy.
2. **Extrinsic calibration:** A 3×3 rotation R and 3D translation t describing
   the rigid-body transform from the depth camera frame to the RGB camera frame.
3. **Reprojection:** Each depth pixel is back-projected to 3D using the depth
   camera intrinsics, transformed by (R, t), then re-projected to 2D using the
   RGB camera intrinsics.

SUN RGB-D performs this alignment during dataset creation. The
`intrinsics.txt` file in each scene folder stores the **RGB camera intrinsics**,
and `depth_bfx` is already warped into the RGB camera's image space. You can
therefore use a single set of intrinsics (from `intrinsics.txt`) for both
the depth and RGB images.

---

## Step 5 — Back-Projection to 3D (Camera Frame)

With holes filled and images aligned, apply the inverse formula to every valid
depth pixel:

```python
# u_idx, v_idx: pixel column and row indices of valid (non-zero) pixels
Z = depth_metres[v_idx, u_idx]
X = (u_idx − cx) * Z / fx
Y = (v_idx − cy) * Z / fy
```

Each (X, Y, Z) triple is a point in the **camera coordinate frame**:

```
+X  →  right
+Y  ↓  down
+Z  →  forward (into the scene, away from camera)
```

This is a right-handed coordinate system centred at the camera's optical centre.

---

## Step 6 — Attaching RGB Colours

Each point (X, Y, Z) maps back to the pixel (u, v) it came from. The RGB
values at that pixel are attached directly:

```python
rgb = rgb_image[v_idx, u_idx]          # shape (N, 3), dtype uint8 (0–255)
xyzrgb = np.hstack([xyz, rgb])         # shape (N, 6)
```

The result is a **coloured point cloud**: each point carries 6 values
(X, Y, Z, R, G, B). This is the standard format used by Open3D and CloudCompare.

---

## Step 7 — Coordinate System Conversion (Camera → World Frame)

The camera frame has +Y pointing **downward** (row indices increase going down
the image). For downstream tasks — visualisation, scene understanding,
consistency with gravity — we convert to a world frame where +Y points **up**.

The transformation is a simple **axis flip**:

```python
X_world =  X          # right stays right
Y_world = −Y          # flip: camera-down becomes world-down-negated = up
Z_world =  Z          # depth forward stays forward
```

In SUN RGB-D's own toolbox the convention is:
- +X: right
- +Y: up (after flipping)
- +Z: forward

This is also what the Open3D visualiser in `preprocessing_viz.py` applies
(`-xyz[:, 1]` for the vertical axis in 3D plots).

> For more complex sensor rigs (cameras tilted or mounted at angles), the full
> rotation from camera to world uses the **extrinsic matrix** — a 4×4
> homogeneous transform. For axis-aligned Kinect captures (camera roughly
> horizontal), the Y-flip is sufficient.

---

## Step 8 — Outlier Removal

Back-projected depth maps always contain stray points — "noise" that appears
as isolated points floating in mid-air, caused by sensor noise at hole
boundaries, mixed pixels at object edges, or small depth errors.

**Statistical outlier removal** (Open3D `remove_statistical_outlier`):

For each point, compute the mean distance to its k nearest neighbours.
Points where this mean distance is more than `std_ratio` standard deviations
above the global mean are removed.

```python
pcd_clean, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
```

Typical parameters for indoor RGBD scenes: `nb_neighbors=20`, `std_ratio=2.0`.

---

## Step 9 — Voxel Downsampling (Optional)

A single SUN RGB-D frame contains ~200,000–300,000 points at full resolution.
For training or downstream processing this is often more than needed.

**Voxel downsampling** divides 3D space into a uniform grid of cubic voxels of
side length `voxel_size`. All points within each voxel are replaced by their
centroid (average position and colour).

```python
pcd_down = pcd_clean.voxel_down_sample(voxel_size=0.02)  # 2 cm voxels
```

Effect: reduces point count by ~10–50× while preserving the shape of surfaces.
The smaller the voxel size, the more detail is kept and the larger the output.

---

## Pipeline Summary

```
Raw RGB image  +  Raw depth PNG (uint16, 0.1 mm units)
        │
        ▼  Step 1 — Convert uint16 → metres  (÷ 10,000)
        │
        ▼  Step 2 — Identify holes  (pixels where depth == 0)
        │           Causes: specular reflection, dark surfaces,
        │           edge shadows, transparency, range limits
        │
        ▼  Step 3 — Cross-bilateral hole filling  [depth_bfx]
        │           RGB-guided: fills holes without blurring real edges
        │
        ▼  Step 4 — Depth–RGB alignment  [pre-done by SUN RGB-D]
        │           Single intrinsics.txt covers aligned image space
        │
        ▼  Step 5 — Back-projection  (per valid pixel)
        │           Z = d
        │           X = (u − cx) · Z / fx
        │           Y = (v − cy) · Z / fy
        │           → 3D point cloud in camera frame
        │
        ▼  Step 6 — Attach RGB  →  XYZRGB (6D) point cloud
        │
        ▼  Step 7 — Flip Y axis  (camera-down → world-up)
        │           X_w = X,  Y_w = −Y,  Z_w = Z
        │
        ▼  Step 8 — Statistical outlier removal
        │           Remove stray points (sensor noise, edge artefacts)
        │
        ▼  Step 9 — Voxel downsampling  (optional, e.g. 2 cm voxels)
        │
        ▼
 Final coloured 3D point cloud
```

---

## Key Parameters in SUN RGB-D

| Parameter | Description | Source |
|---|---|---|
| `fx`, `fy` | Focal lengths in pixel units | `intrinsics.txt` (row 0 col 0, row 1 col 1) |
| `cx`, `cy` | Principal point (optical axis pixel) | `intrinsics.txt` (row 0 col 2, row 1 col 2) |
| Depth scale | 10,000 raw units = 1 metre | Dataset convention (0.1 mm per unit) |
| Depth range | ~0.5 m – 7 m (Kinect v1/v2) | Sensor hardware limit |
| `depth_bfx` | Bilateral-filtered depth, pre-aligned to RGB | Provided in dataset |

---

## References

- Song, S., Lichtenberg, S., & Xiao, J. (2015). **SUN RGB-D: A RGB-D Scene
  Understanding Benchmark Suite.** CVPR 2015.
- Kopf, J. et al. (2007). **Joint Bilateral Upsampling.** ACM SIGGRAPH 2007.
  *(Cross-bilateral filtering principle used for depth hole filling.)*
- Hartley, R. & Zisserman, A. (2004). **Multiple View Geometry in Computer
  Vision** (2nd ed.). Cambridge University Press.
  *(Pinhole camera model, intrinsic matrix, back-projection.)*
