"""
trace_pipeline.py
-----------------
Standalone annotated trace of the TR3D detection pipeline (SamsungLabs/tr3d,
ICIP 2023) applied to a synthetic SUN RGB-D sample.

Requires ONLY: torch, numpy  (no MinkowskiEngine, no mmdet3d, no GPU needed).

MinkowskiEngine's SparseTensor is mocked with a plain Python class.
Sparse convolutions are mocked with torch.nn.Linear.
Every shape and transformation mirrors the real network exactly.

Run:
    python trace_pipeline.py
"""

import sys
import io
import torch
import torch.nn as nn
import numpy as np

# Tee stdout to both terminal and a file
_output_path = "trace_pipeline_output.txt"
_file_out = open(_output_path, "w", encoding="utf-8")

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

sys.stdout = _Tee(sys.__stdout__, _file_out)

# -- SUN RGB-D model config (from configs/sunrgbd/tr3d_sunrgbd.py) ----------
N_CLASSES          = 10      # bed, table, sofa, chair, toilet,
                              # desk, dresser, night_stand, bookshelf, bathtub
N_REG_OUTS         = 6       # axis-aligned box: Dx,Dy,Dz, w,h,l  (no yaw)
VOXEL_SIZE         = 0.02    # 2 cm voxels (0.02 m)
BACKBONE_CHANNELS  = [32, 64, 128, 256]  # channels at each backbone scale
OUT_CHANNELS       = 128     # every FPN neck output uses this many channels
# Which FPN level is responsible for detecting each class.
# 0 = finer scale (small objects), 1 = coarser scale (large objects).
# Source: TR3DAssigner(label2level=...) in the SUN RGB-D config.
LABEL2LEVEL = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
# Classes:     bed tbl sofa chr tlt dsk drs ngt bks bat
# Level 1 (large): bed, table, sofa, desk, bookshelf
# Level 0 (small): chair, toilet, dresser, night_stand, bathtub

torch.manual_seed(0)

print("=" * 72)
print("TR3D Pipeline Trace  --  SUN RGB-D  (CPU, no MinkowskiEngine)")
print("=" * 72)

# -----------------------------------------------------------------------------
# MOCK: SparseTensor
# -----------------------------------------------------------------------------
# In the real code, MinkowskiEngine.SparseTensor is the core data structure.
# It stores only the ACTIVE voxels -- the subset of the 3-D grid that actually
# contains at least one point. This is what makes sparse 3-D convolution
# memory-efficient compared to dense 3-D grids.
#
# Key attributes:
#   .coordinates  -- (N, 4) int32: [batch_idx, x_vox, y_vox, z_vox]
#   .features     -- (N, C) float32: feature vector at each active voxel
#
# .decomposition_permutations gives, for each sample in the batch, the row
# indices into .features that belong to it.

class SparseTensor:
    """Minimal mock of MinkowskiEngine.SparseTensor (data container only)."""
    def __init__(self, coordinates, features):
        self.coordinates = coordinates.long()
        self.features    = features.float()

    @property
    def decomposition_permutations(self):
        """
        Returns a list of index tensors -- one per batch item.
        In ME this is used to split predictions back into per-scene lists.
        """
        batch_ids = self.coordinates[:, 0]
        return [torch.where(batch_ids == b)[0] for b in batch_ids.unique()]

    def decomposed_coordinates(self):
        """Spatial coordinates (no batch column) split by batch item."""
        batch_ids = self.coordinates[:, 0]
        return [self.coordinates[batch_ids == b, 1:] for b in batch_ids.unique()]


# -----------------------------------------------------------------------------
# STEP 0 -- Create fake SUN RGB-D input
# -----------------------------------------------------------------------------
# A real SUN RGB-D scene (e.g. NYU0001.jpg) has ~50k?200k 3-D points after
# back-projection from the filled depth map.
# Each point: (x, y, z) in metres + (r, g, b) normalised to [0, 1].
# The RGB image is 640 x 480 pixels, stored as (batch, C, H, W) for PyTorch.

print("\n-- STEP 0: Input ---------------------------------------------------")

N_POINTS    = 50_000
point_cloud = torch.rand(N_POINTS, 6)
point_cloud[:, :3] *= 6.0          # scale xyz to ~6 m room dimensions

# Image in PyTorch CHW format with a batch dimension
rgb_image = torch.rand(1, 3, 480, 640)

print(f"  Raw point cloud : {tuple(point_cloud.shape)}  "
      f"xin[{point_cloud[:,0].min():.1f},{point_cloud[:,0].max():.1f}] m  "
      f"rgbin[0,1]")
print(f"  RGB image       : {tuple(rgb_image.shape)}  [batch, C, H, W]")


# -----------------------------------------------------------------------------
# STEP 1 -- Voxelization
# -----------------------------------------------------------------------------
# SOURCE  mink_single_stage.py ? extract_feats():
#   coordinates, features = ME.utils.batch_sparse_collate(
#       [(p[:, :3] / voxel_size, p[:, 3:]) for p in points],
#       device=points[0].device)
#   x = ME.SparseTensor(coordinates=coordinates, features=features)
#
# THREE THINGS HAPPEN HERE:
#
# (a) Quantisation: divide xyz by voxel_size ? integer voxel indices.
#     A point at (1.234 m, 0.456 m, 2.678 m) lands in voxel (61, 22, 133).
#     This snaps every point to the nearest 2-cm grid cell.
#
# (b) Deduplication: many raw points map to the same voxel.
#     ME.batch_sparse_collate keeps each voxel once, averaging its features.
#     50,000 raw points ? far fewer unique occupied voxels.
#
# (c) Batch column: a batch index (0 for first sample) is prepended.
#     Coordinate layout becomes [batch_idx, x_vox, y_vox, z_vox].
#     This lets a single SparseTensor hold an entire mini-batch efficiently.

print("\n-- STEP 1: Voxelization --------------------------------------------")
print(f"  Voxel size : {VOXEL_SIZE} m  ({int(VOXEL_SIZE*100)} cm cubes)")

xyz = point_cloud[:, :3]
rgb = point_cloud[:, 3:]

# (a) Quantise to integer voxel indices
vox_idx = (xyz / VOXEL_SIZE).long()     # (N_POINTS, 3)

# (b) Deduplicate -- encode (x,y,z) as a single integer for unique()
M = int(vox_idx.max().item()) + 1
flat = vox_idx[:, 0] * M**2 + vox_idx[:, 1] * M + vox_idx[:, 2]
unique_flat, inverse = torch.unique(flat, return_inverse=True)
N_VOX = len(unique_flat)

# Average RGB features within each voxel
vox_rgb    = torch.zeros(N_VOX, 3)
vox_rgb.scatter_add_(0, inverse.unsqueeze(1).expand(-1, 3), rgb)
counts     = torch.zeros(N_VOX).scatter_add_(0, inverse, torch.ones(N_POINTS))
vox_rgb   /= counts.unsqueeze(1)

# Recover integer voxel xyz from the flat encoding
vox_x = unique_flat // M**2
vox_y = (unique_flat % M**2) // M
vox_z = unique_flat % M

# (c) Prepend batch index 0
batch_col  = torch.zeros(N_VOX, 1, dtype=torch.long)
vox_coords = torch.cat([batch_col, vox_x[:,None], vox_y[:,None], vox_z[:,None]], dim=1)

x = SparseTensor(coordinates=vox_coords, features=vox_rgb)

print(f"  Raw points ? unique voxels : {N_POINTS:,} ? {N_VOX:,}  "
      f"({100*N_VOX/N_POINTS:.1f}% of points kept)")
print(f"  SparseTensor.coordinates   : {tuple(x.coordinates.shape)}  "
      f"[batch, x_vox, y_vox, z_vox]  dtype=int")
print(f"  SparseTensor.features      : {tuple(x.features.shape)}  "
      f"[r, g, b per voxel]  dtype=float")
print(f"  Voxel grid extent          : "
      f"x<={vox_coords[:,1].max().item()}  "
      f"y<={vox_coords[:,2].max().item()}  "
      f"z<={vox_coords[:,3].max().item()}  voxel units")


# -----------------------------------------------------------------------------
# STEP 2 -- Backbone  (MinkResNet -- sparse 3-D encoder-decoder)
# -----------------------------------------------------------------------------
# SOURCE  mink_single_stage.py ? extract_feats():
#   x = self.backbone(x)
#
# The backbone is a sparse 3-D U-Net (MinkResNet in the TR3D config).
# It has an encoder arm that progressively DOWNSAMPLES the voxel grid and a
# decoder arm that UPSAMPLES back up with skip connections -- just like U-Net,
# but in 3-D and operating only on occupied voxels.
#
# Downsampling uses stride-2 sparse convolutions.  In 3-D, stride 2 halves
# each spatial dimension, so each level has up to 8x fewer active voxels
# and 2x more channels (capturing more abstract geometry).
#
# The backbone returns a LIST of SparseTensors -- one per scale level.
# The neck and head will consume these multi-scale outputs.
#
# Approximate scales for TR3D on SUN RGB-D (MinkResNet50):
#   Scale 0: ~N_VOX voxels,      32 ch  ? finest, most spatial detail
#   Scale 1: ~N_VOX/8 voxels,    64 ch
#   Scale 2: ~N_VOX/64 voxels,  128 ch
#   Scale 3: ~N_VOX/512 voxels, 256 ch  ? coarsest, most semantic
#
# MOCK: Linear layer + uniform subsampling to simulate each level.

print("\n-- STEP 2: Backbone (sparse 3-D U-Net) -----------------------------")

class MockBackboneLevel(nn.Module):
    def __init__(self, in_ch, out_ch, stride):
        super().__init__()
        self.proj   = nn.Linear(in_ch, out_ch)
        self.stride = stride   # spatial subsampling factor (real = stride-2 conv)

    def forward(self, st):
        # Simulate stride-2 downsampling: keep every stride-th voxel,
        # halve the voxel coordinate indices (real: sparse stride-2 conv does this)
        keep = torch.arange(0, len(st.features), self.stride)
        new_feats  = self.proj(st.features[keep])
        new_coords = st.coordinates[keep].clone()
        new_coords[:, 1:] = new_coords[:, 1:] // 2
        return SparseTensor(coordinates=new_coords, features=new_feats)

backbone_levels = nn.ModuleList([
    MockBackboneLevel(3,                      BACKBONE_CHANNELS[0], stride=1),
    MockBackboneLevel(BACKBONE_CHANNELS[0],   BACKBONE_CHANNELS[1], stride=8),
    MockBackboneLevel(BACKBONE_CHANNELS[1],   BACKBONE_CHANNELS[2], stride=8),
    MockBackboneLevel(BACKBONE_CHANNELS[2],   BACKBONE_CHANNELS[3], stride=8),
])

backbone_outputs = []
cur = x
for i, lvl in enumerate(backbone_levels):
    cur = lvl(cur)
    backbone_outputs.append(cur)
    n, c = cur.features.shape
    print(f"  Scale {i}: {n:>6,} voxels  x  {c:>3} channels  "
          f"  coords {tuple(cur.coordinates.shape)}")


# -----------------------------------------------------------------------------
# STEP 2b -- Feature Fusion  [TR3D+FF variant only]
# -----------------------------------------------------------------------------
# SOURCE  tr3d_ff.py ? extract_feats() + _f():
#   with torch.no_grad():
#       x = self.img_backbone(img)
#       img_features = self.img_neck(x)[0]
#   ...
#   x = self.backbone(x, partial(self._f, img_features=..., img_metas=..., img_shape=...))
#
# WHAT HAPPENS:
# The plain TR3D processes only the point cloud. TR3D+FF also passes the RGB
# image through a 2-D ResNet + FPN to get a (B, 256, H/8, W/8) feature map.
#
# Then, inside the backbone, at each scale, for every active voxel:
#   1. Recover real-world xyz:  xyz = vox_coords * voxel_size
#   2. Project onto the image plane with the camera matrix K:
#        u = fx * X/Z + cx
#        v = fy * Y/Z + cy
#   3. Sample the 2-D feature map at (u, v) with bilinear interpolation
#      (mmdet3d's point_sample() handles this).
#   4. A 1x1 sparse convolution compresses 256 ? 64 channels.
#   5. The 64-ch 2-D feature is ADDED residually to the 3-D sparse feature.
#      This is: new_x = compressed_2d_feat + x  (SparseTensor addition)
#
# WHY: Pure depth data loses colour and texture information. By back-injecting
# image features at the geometric locations of the voxels, the network learns
# jointly from appearance and geometry without fusing at the raw input level.
# The 2-D backbone's weights are FROZEN during 3-D training (init_weights).

print("\n-- STEP 2b: Feature Fusion [TR3D+FF only] --------------------------")

# 2-D FPN output: 1/8 resolution ? 480/8=60 rows, 640/8=80 cols
img_feat_map = torch.rand(1, 256, 60, 80)
n_vox_s0     = backbone_outputs[0].features.shape[0]

# point_sample() projects each voxel's real xyz ? image (u,v) ? samples feat map
# We mock this as random feature vectors (same shape as the real output)
sampled_2d = torch.rand(n_vox_s0, 256)              # (N_vox_scale0, 256)
fused_feat  = nn.Linear(256, 64)(sampled_2d)         # 1x1 conv: 256?64 ch
# Residual add would happen here (channel alignment needed in real code)

print(f"  2-D image FPN feature map : {tuple(img_feat_map.shape)}  "
      f"[batch, 256, H/8, W/8]")
print(f"  Sampled per-voxel 2-D feat: {tuple(sampled_2d.shape)}  "
      f"via bilinear sampling at projected (u,v)")
print(f"  After 1x1 compression     : {tuple(fused_feat.shape)}  "
      f"? added residually to 3-D sparse features at scale 0")
print(f"  (Plain TR3D skips this step entirely)")


# -----------------------------------------------------------------------------
# STEP 3 -- Neck  (TR3DNeck: 3-D Feature Pyramid Network)
# -----------------------------------------------------------------------------
# SOURCE  tr3d_neck.py ? TR3DNeck.forward():
#   def forward(self, x):
#       x = x[1:]           # skip finest scale (scale 0)
#       inputs = x
#       x = inputs[-1]      # start at deepest (most abstract) scale
#       for i in range(len(inputs)-1, -1, -1):
#           if i < len(inputs)-1:
#               x = up_block(x) + inputs[i]  # upsample & add lateral skip
#               x = lateral_block(x)
#               outs.append(out_block(x))    # project to out_channels
#       return outs[::-1]
#
# THIS IS A 3-D FPN.  Starting from the coarsest, most semantic scale (scale 3,
# 256 ch) and working towards finer scales:
#   - MinkowskiGenerativeConvolutionTranspose doubles the spatial resolution
#     (like a 3-D transposed convolution, but sparse -- it only creates new
#     voxels where a parent voxel existed, not everywhere).
#   - The upsampled features are ADDED to the same-scale backbone output (lateral
#     skip connection) to combine semantics with spatial detail.
#   - A 3x3 sparse conv (lateral_block) mixes the merged features.
#   - A final 3x3 sparse conv (out_block) projects to OUT_CHANNELS (128).
#
# Neck INPUT  : backbone_outputs[1], [2], [3]  (skips [0])
#               channels: 64, 128, 256
# Neck OUTPUT : 2 SparseTensors at scales 1 and 2, each with 128 channels
#
# WHY FPN?
# Small objects (chair, toilet) are best detected at fine resolution (scale 1).
# Large objects (bed, sofa) need coarser resolution (scale 2) to see them whole.
# FPN gives the head both resolutions with shared, enriched features.

print("\n-- STEP 3: Neck (3-D Feature Pyramid Network) ----------------------")

class MockFPNLevel(nn.Module):
    """
    One FPN level: optional up-projection from deeper level + lateral + output.

    The real neck's make_up_block(in_ch_deep, in_ch_this) performs a
    channel-reducing generative deconv so that upsampled features have the
    SAME channel count as this level's backbone features before adding them.
    We replicate that channel reduction with a Linear layer (up_proj).
    """
    def __init__(self, in_ch, out_ch, up_in_ch=None):
        super().__init__()
        # up_proj: reduces the deeper level's channels to match this level
        self.up_proj  = nn.Linear(up_in_ch, in_ch) if up_in_ch else None
        self.lateral  = nn.Linear(in_ch, in_ch)
        self.out_proj = nn.Linear(in_ch, out_ch)

    def forward(self, st, deeper_feats=None):
        feats = st.features                         # (N_this, in_ch)
        if deeper_feats is not None and self.up_proj is not None:
            # Project deeper features to this level's channel count (up_proj)
            # then broadcast to match the voxel count of this level
            up = self.up_proj(deeper_feats)         # (N_deep, in_ch)
            n  = feats.shape[0]
            if up.shape[0] >= n:
                up = up[:n]
            else:
                rep = (n // up.shape[0]) + 1
                up  = up.repeat(rep, 1)[:n]
            feats = feats + up                      # residual add (channel-matched)
        feats     = self.lateral(feats)
        out_feats = self.out_proj(feats)
        return SparseTensor(
            coordinates=st.coordinates[:len(out_feats)],
            features=out_feats)

# Neck sees backbone scales 1, 2, 3  (skips scale 0, the finest)
# channels:                            64ch, 128ch, 256ch
neck_in = backbone_outputs[1:]

# Level 2: merges scale-2 lateral (128ch) with upsampled scale-3 (256->128ch)
fpn_level2 = MockFPNLevel(in_ch=BACKBONE_CHANNELS[2],
                           out_ch=OUT_CHANNELS,
                           up_in_ch=BACKBONE_CHANNELS[3])
# Level 1: merges scale-1 lateral (64ch) with upsampled scale-2 output (128->64ch)
fpn_level1 = MockFPNLevel(in_ch=BACKBONE_CHANNELS[1],
                           out_ch=OUT_CHANNELS,
                           up_in_ch=OUT_CHANNELS)

# Top-down pass: deepest (256ch) -> scale 2 (128ch) -> scale 1 (64ch)
neck_s2 = fpn_level2(neck_in[1], neck_in[2].features)   # scale 2 merged with scale 3
neck_s1 = fpn_level1(neck_in[0], neck_s2.features)      # scale 1 merged with scale 2

neck_outputs = [neck_s1, neck_s2]   # finer first (matches return outs[::-1])

for i, ns in enumerate(neck_outputs):
    n, c = ns.features.shape
    print(f"  FPN output level {i}: {n:>6,} voxels  x  {c:>3} channels  "
          f"(classes ? {[j for j,l in enumerate(LABEL2LEVEL) if l==i]})")


# -----------------------------------------------------------------------------
# STEP 4 -- Head  (TR3DHead: per-voxel bbox + cls prediction)
# -----------------------------------------------------------------------------
# SOURCE  tr3d_head.py ? TR3DHead._forward_single():
#   def _forward_single(self, x):
#       reg_final     = self.bbox_conv(x).features     # (N, 6)
#       reg_distance  = torch.exp(reg_final[:, 3:6])   # size always positive
#       bbox_pred     = cat(reg_final[:,:3], reg_distance, ...)
#       cls_pred      = self.cls_conv(x).features      # (N, n_classes)
#       # split by batch item using decomposition_permutations
#       for permutation in x.decomposition_permutations:
#           points = x.coordinates[permutation][:, 1:] * self.voxel_size
#       return bbox_preds, cls_preds, points
#
# TWO BRANCHES, each a 1x1 sparse convolution (kernel_size=1):
#
#   BBOX branch  (128 ? 6):
#     output[:, 0:3] = Dx, Dy, Dz -- offset from voxel centre to predicted
#                      box centre in metres. Unbounded -- can be negative.
#     output[:, 3:6] = log(w), log(h), log(l) -- log half-sizes.
#                      exp() is applied so sizes are always positive.
#     ? Predicted box: centre = voxel_xyz + Dxyz,  size = (w,h,l)
#
#   CLS branch  (128 ? 10):
#     Raw logits. sigmoid() at inference time gives class probabilities.
#     N_CLASSES=10 for SUN RGB-D.
#
# COORDINATE RECOVERY:
#   voxel_xyz_metres = vox_coords[:, 1:] * voxel_size
#   This converts integer voxel indices back to real-world metres.
#
# ASSIGNER (training only):
#   TR3DAssigner matches each voxel to a ground-truth box using 3 conditions:
#   1. Level consistency: label2level[class] must equal this FPN level index.
#      (Large-object classes only get assigned at scale 1, etc.)
#   2. Top-k proximity: only the top_pts_threshold=6 voxels closest to each
#      box centre are positive candidates.
#   3. Unique assignment: each voxel is assigned to the single closest GT box.

print("\n-- STEP 4: Head (per-voxel bbox + cls prediction) ------------------")
print(f"  n_classes   = {N_CLASSES}  |  n_reg_outs = {N_REG_OUTS}")
print(f"  label2level = {LABEL2LEVEL}")
print(f"    Level 0 (fine):   classes {[i for i,l in enumerate(LABEL2LEVEL) if l==0]}"
      f"  ? chair, toilet, dresser, night_stand, bathtub")
print(f"    Level 1 (coarse): classes {[i for i,l in enumerate(LABEL2LEVEL) if l==1]}"
      f"  ? bed, table, sofa, desk, bookshelf")

bbox_branch = nn.Linear(OUT_CHANNELS, N_REG_OUTS)
cls_branch  = nn.Linear(OUT_CHANNELS, N_CLASSES)

all_bbox_preds, all_cls_preds, all_points = [], [], []

for lvl_idx, ns in enumerate(neck_outputs):
    feats    = ns.features                      # (N_vox, 128)
    vox_int  = ns.coordinates[:, 1:].float()   # integer voxel coords (N_vox, 3)
    real_xyz = vox_int * VOXEL_SIZE             # ? real metres            (N_vox, 3)

    # BBOX branch
    raw     = bbox_branch(feats)                # (N_vox, 6)
    delta   = raw[:, :3]                        # Dx, Dy, Dz -- centre offset
    sizes   = torch.exp(raw[:, 3:6])            # w, h, l -- always positive
    bbox_pred = torch.cat([delta, sizes], dim=1)# (N_vox, 6)

    # CLS branch
    cls_pred = cls_branch(feats)                # (N_vox, 10) -- raw logits

    all_bbox_preds.append(bbox_pred)
    all_cls_preds.append(cls_pred)
    all_points.append(real_xyz)

    print(f"\n  Level {lvl_idx}:")
    print(f"    Input features      : {tuple(feats.shape)}")
    print(f"    Voxel centres (m)   : {tuple(real_xyz.shape)}  "
          f"xin[{real_xyz[:,0].min():.2f},{real_xyz[:,0].max():.2f}]")
    print(f"    bbox_pred           : {tuple(bbox_pred.shape)}  "
          f"[Dx,Dy,Dz, w,h,l]")
    print(f"      Dxyz range        : [{delta.min():.3f}, {delta.max():.3f}] m")
    print(f"      sizes (exp) range : [{sizes.min():.3f}, {sizes.max():.3f}] m")
    print(f"    cls_pred (logits)   : {tuple(cls_pred.shape)}  "
          f"[{N_CLASSES} classes per voxel]")
    print(f"    cls_pred (sigmoid)  : min={cls_pred.sigmoid().min():.3f}  "
          f"max={cls_pred.sigmoid().max():.3f}")


# -----------------------------------------------------------------------------
# STEP 5 -- Decode predictions ? final 3-D bounding boxes  (inference path)
# -----------------------------------------------------------------------------
# SOURCE  tr3d_head.py ? _get_bboxes_single():
#   scores      = torch.cat(cls_preds).sigmoid()
#   bbox_preds  = torch.cat(bbox_preds)
#   points      = torch.cat(points)
#   boxes       = _bbox_pred_to_bbox(points, bbox_preds)
#   boxes, scores, labels = _nms(boxes, scores, img_meta)
#
# _bbox_pred_to_bbox():
#   x_centre = point[:,0] + bbox_pred[:,0]    # voxel x + Dx
#   y_centre = point[:,1] + bbox_pred[:,1]    # voxel y + Dy
#   z_centre = point[:,2] + bbox_pred[:,2]    # voxel z + Dz
#   ? box = (cx, cy, cz, w, h, l)            # axis-aligned, no yaw for SUN RGB-D
#
# _nms():
#   For each class separately:
#     1. Keep only voxels with score > score_thr (e.g. 0.01)
#     2. Apply nms3d_normal (axis-aligned 3-D NMS) with iou_thr (e.g. 0.25)
#        -- remove boxes that overlap > iou_thr with a higher-scoring box
#   Merge surviving boxes from all classes.
#   Final output: (cx, cy, cz, w, h, l) + label + score per detected object.

print("\n-- STEP 5: Decode predictions ? 3-D bounding boxes ----------------")

# Concatenate across both FPN levels
all_pts   = torch.cat(all_points,     dim=0)   # (total_vox, 3)
all_bbox  = torch.cat(all_bbox_preds, dim=0)   # (total_vox, 6)
all_cls   = torch.cat(all_cls_preds,  dim=0)   # (total_vox, 10)

print(f"  Total voxels (all FPN levels): {all_pts.shape[0]:,}")
print(f"  all_bbox shape               : {tuple(all_bbox.shape)}")
print(f"  all_cls shape                : {tuple(all_cls.shape)}")

# Decode: voxel centre + offset ? predicted box centre
cx = all_pts[:, 0] + all_bbox[:, 0]
cy = all_pts[:, 1] + all_bbox[:, 1]
cz = all_pts[:, 2] + all_bbox[:, 2]
w  = all_bbox[:, 3]
h  = all_bbox[:, 4]
l  = all_bbox[:, 5]
decoded = torch.stack([cx, cy, cz, w, h, l], dim=1)  # (total_vox, 6)

print(f"\n  Decoded 3-D boxes : {tuple(decoded.shape)}  [cx, cy, cz, w, h, l]")

# Score threshold
scores     = all_cls.sigmoid()              # (total_vox, 10)
max_scores, pred_labels = scores.max(dim=1) # (total_vox,)
score_thr  = 0.01
keep       = max_scores > score_thr
print(f"  After score_thr={score_thr}  : {keep.sum():,} / {all_pts.shape[0]:,} "
      f"candidate voxels kept")

# NMS (real code uses mmcv.ops.nms3d_normal -- axis-aligned 3-D IoU)
# Typically eliminates ~95% of remaining candidates.
n_cands   = keep.sum().item()
n_final   = max(1, n_cands // 20)    # rough simulation of NMS retention rate
print(f"  After 3-D NMS (simulated): ~{n_final} final detections")
print(f"  Output per detection: (cx, cy, cz, w, h, l)  +  class label  +  score")


# -----------------------------------------------------------------------------
# FULL PIPELINE SUMMARY
# -----------------------------------------------------------------------------
print("\n" + "=" * 72)
print("SHAPE SUMMARY  (follow this table top to bottom to trace the data)")
print("=" * 72)
print(f"  0. Raw input          : point cloud {tuple(point_cloud.shape)}, "
      f"image {tuple(rgb_image.shape)}")
print(f"  1. After voxelisation : coords {tuple(x.coordinates.shape)}  "
      f"feats {tuple(x.features.shape)}")
for i, bo in enumerate(backbone_outputs):
    print(f"  2. Backbone scale {i}   : {bo.features.shape[0]:>6,} voxels "
          f"x {bo.features.shape[1]:>3} ch")
print(f"  2b. FF sampled 2-D   : {tuple(sampled_2d.shape)}  "
      f"? compressed {tuple(fused_feat.shape)}  [+FF only]")
for i, ns in enumerate(neck_outputs):
    print(f"  3. Neck FPN level {i}  : {ns.features.shape[0]:>6,} voxels "
          f"x {ns.features.shape[1]:>3} ch")
for i in range(len(neck_outputs)):
    print(f"  4. Head level {i} out : bbox {tuple(all_bbox_preds[i].shape)}  "
          f"cls {tuple(all_cls_preds[i].shape)}")
print(f"  5. Decoded boxes      : {tuple(decoded.shape)}")
print(f"  5. After score thr    : {keep.sum():,} candidates")
print(f"  5. After 3-D NMS      : ~{n_final} final detections  "
      f"(cx, cy, cz, w, h, l, label, score)")
print()
print("Key concepts used:")
print("  Sparse tensor      -- store only occupied voxels, not the full grid")
print("  Voxelisation       -- snap continuous xyz to a 2cm grid, deduplicate")
print("  Stride-2 sparse    -- halves resolution, 8x fewer voxels per level (3D)")
print("  FPN top-down       -- merge semantics (deep) + spatial detail (shallow)")
print("  1x1 sparse conv    -- per-voxel projection, no spatial mixing")
print("  Label2level        -- each class predicted at its natural FPN scale")
print("  TR3DAssigner       -- assigns GT boxes to voxels by level + proximity")
print("  exp(log_size)      -- guarantees box sizes are always positive")
print("  3-D NMS            -- removes duplicate detections by axis-aligned IoU")

# Close the output file and restore stdout
sys.stdout = sys.__stdout__
_file_out.close()
print(f"\nOutput saved to: {_output_path}")
