# Reference methods: two detectors, two memory units

The maintained codebase exposes four method paths. Use these named methods
for reference-aligned work. Separate custom policies remain available for
controlled research variants.

| Detector | Scene memory: LDMR | Object memory: TR3D_OBJ-style extension |
|---|---|---|
| TR3D | `configs/memory/scannet_s5_ldmr.py` | `configs/memory/scannet_s5_tr3d_obj.py` |
| UniDet3D | `adapters/tools/run_sequence.py --memory-mode scene --recipe ldmr` | `adapters/tools/run_sequence.py --memory-mode object --recipe tr3d_obj` |

TR3D additionally has the original SUN40 S5 paths:
`configs/memory/sunrgbd_s5_ldmr.py` and `configs/memory/sunrgbd_s5_tr3d_obj.py`.
UniDet3D currently supports ScanNet35 S5, not a validated SUN port.

## What follows the references

**LDMR scene method:** current-class natural scenes plus saved scene visibility,
frozen-teacher pseudo labels, Design-2 scene selection, original LD-drop reviewing,
120 scene budget for ScanNet. The TR3D configs inherit original LDMR directly.

**Object method:** pseudo supervision plus per-class cropped objects, 100 crops
per class, minimum 20 points, the reference's absolute 1 cm crop-boundary
tolerance, up to three attempted pastes with probability .7,
class-uniform then object-uniform draws **with replacement**, IoU collision
threshold .3, floor placement, jitter 1m and 10 placement attempts. SUN enables
random yaw; ScanNet remains axis-aligned. Confidence filtering is .5 at every
increment, as in the TR3D_OBJ configuration. Object Design-2 and source-seat
reviewing are separate experiments; the named reference object method does not
silently substitute either.

`tr3d_obj_replace` preserves the reference bank's unconditional replacement of
a random existing entry after capacity. It is **not unbiased reservoir sampling**;
that naming error in the reference must not be mistaken for a different algorithm.
The implementation keeps bounded authoritative crop storage and versioned state.
The host reads bank, insertion and pseudo settings from the same ScanNet object
profile used by TR3D; only transform argument names and host-specific geometry
are adapted. Scene selectors and reviewing reuse the original LDMR functions.

## Training exposure and optimization

| Setting | TR3D ScanNet scene | TR3D SUN object | UniDet3D named recipes |
|---|---|---|---|
| Stage schedule | 12 epochs × 15 repeats | 5/2/2/1/1 epochs × 15 repeats | 12 nominal epochs × 15 = 180 single-pass epochs per stage |
| Batch | 16 | 16 | 8 by default; explicit overrides are recorded |
| Optimizer | Original AdamW .001 / weight decay .0001 | Same | Native host AdamW .0002 / weight decay .05 |
| LR decay | Original segmented runner; see below | Step at max(1, stage epochs−1) | Native polynomial decay over the full 180 passes |
| Scene reviews | Original repeat-unit segments | Not used in object reference | At passes 36, 72, 108, 144; capture also at 0 and 180 |

The original LDMR scene trainer executes the ScanNet exposure as **five resumed
runner epochs, each repeating the dataset 36 times**. Its optimizer state resumes
between segments. The inherited step milestones `[8, 11]` are not reached by that
five-epoch runner, so this reference scene path keeps LR .001 throughout the
stage. The maintained TR3D path preserves this behavior. A config-level listing
of 12 epochs and milestones 8/11 alone does not describe the executed schedule.
Ordinary validation in this scene path occurs at the five segment ends.

`--epochs` for a named host recipe means **nominal** epochs; the sequence records
both that value and its effective pass count. Checkpoints are named by effective
host epoch (for example `epoch_180.pth`). Sampling receipts and LD captures also
use effective-pass units. Validation and ordinary checkpoints occur every 15
passes (12 times per stage), avoiding 180 full validation sweeps. LD captures at
36/72/108/144 are independent of that interval. Custom recipes keep their default
interval of one pass; `--validation-interval` can change it and must divide the
stage length.
Changing batch size changes optimization and is not an equivalent reproduction.

The ScanNet transfer of TR3D_OBJ uses LDMR's ScanNet class split and exposure;
the reference TR3D_OBJ reported results are SUN40, not ScanNet35. UniDet3D retains
its detector, matcher and optimizer; it uses explicit box/distance targets so
teacher boxes survive into its loss. Matching exposure does not establish an
optimal host schedule or transfer the original paper's accuracy claim.

## Commands

Run legacy commands from the repository root in the legacy environment. Configure
data locations using the existing installation/data guide and native config overrides.
Top-level `--cfg-options data.samples_per_gpu=... data.train.dataset.ann_file=...`
now reach the actual stage configuration. `base_config.data.*` remains an alias;
do not provide both spellings for the same field. Stage definitions still own
stage epochs and learning rates.

```bash
/path/to/legacy/bin/python tools/train_incremental.py \
  configs/memory/scannet_s5_ldmr.py --seed 201 --work-dir /path/to/new/tr3d_scene

/path/to/legacy/bin/python tools/train_incremental.py \
  configs/memory/sunrgbd_s5_tr3d_obj.py --seed 200 --work-dir /path/to/new/tr3d_object
```

For either host method, use its own Python environment and prepared ScanNet35
metadata with genuine superpoints:

```bash
/path/to/host/bin/python /path/to/source/adapters/tools/run_sequence.py \
  --host-python /path/to/host/bin/python --ldmr-python /path/to/legacy/bin/python \
  --upstream-root /path/to/UniDet3D \
  --train-info /path/to/full/train/infos.pkl --val-info /path/to/full/val/infos.pkl \
  --work-dir /path/to/new/host_scene --memory-mode scene --recipe ldmr \
  --epochs 12 --batch-size 8 --workers 2 --seed 201
```

For the host object method change both `--memory-mode object` and
`--recipe tr3d_obj`, and use a new work directory. Recipes own their pseudo,
selection and reviewing settings; omit manual `--ld-epochs`/`--reviewing` switches.
Both recipes enable pseudo supervision and box targets. Do not continue a
historical run under a changed recipe; continuation checks reject that mismatch.

## Explicit implementation differences

Reference behavior is the anchor, not a claim of byte-identical training:

- Object population uses per-class RNG streams; the teammate bank interleaves
  a single RNG across classes. The replacement rule matches, but selected IDs
  need not match a multi-class reference run with the same seed.
- Geometry retains explicit gravity/bottom-center conversion. Placement jitter
  is clipped to valid scene bounds; yaw-aware collision bounds are preserved.
  These avoid ambiguous/invalid geometry rather than reproducing reference bugs.
- Two original SUN training annotations have zero height. They remain in scene
  learning-dynamics GT counts, matching LDMR's false-negative denominator. The
  object bank excludes them because they cannot supply positive-volume targets.
- ScanNet has no arbitrary pasted yaw. UniDet3D additionally maintains point,
  instance and superpoint alignment; these are host requirements.
- Loss-logit masking is not enabled. The reference `objbank_pseudo` condition
  supplies the unmasked comparison; `objbank_masking` is a separate ablation.

The maintained crop frame is box-local. SUN random yaw is applied once by the
final source-plus-random angle, keeping pasted points aligned with the target
box. The reference replacement population keeps at most the class quota plus
one incoming crop in memory. On all ScanNet stage-1 training data, its 700-crop
bank and rich host payload are byte-identical to the earlier all-candidate export.

That byte-identity result concerns the earlier streaming-storage change. The
subsequent crop-boundary correction intentionally changes payloads: the corrected
ScanNet stage-1 bank retains the same 700 IDs, but 685 crops gain context points
(185,350 additional points in total). All 700 crops and their rich mask rows
match the teammate extraction oracle. Checkpoints record `crop_tolerance` and
reject continuation under a different tolerance; historical files without that
field retain the old 1e-6 default. Re-export reference banks before using the
corrected named recipe. Crop membership uses float64 intermediates, as in TR3D_OBJ.

Reference revisions: LDMR `ab67f3d7a3e6e98c6f66182832bb37e8656cceaa`,
TR3D_OBJ `cd667a180c3cbf4eae29cf779ea907592400266f`,
UniDet3D `940a730a09711b0bf266fd972504da29a83b91f6`.

## Validation status

The [current validation report](unidet3d_validation.md) records reference-source
comparisons, complete real-data bank population, and native CUDA loss/backward.
Full-sequence accuracy for the named 180-pass recipes remains unmeasured.
Matching method components and training exposure does not establish equal
accuracy across detectors with different architectures and optimizers.
