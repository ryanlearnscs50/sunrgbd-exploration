# LDMR — SUN RGB-D reproduction, Week 1 findings

**Date:** 2026-07-23 · **Hardware:** 2x RTX 3090, Ubuntu 20.04.6
**Task:** clone the code, set up the environment, evaluate the released
checkpoints on SUN RGB-D using the 40-class metadata.

---

## Headline

**The released SUN RGB-D checkpoints reproduce their reported numbers.**

| Protocol | Final ckpt | Ours mAP@0.25 | Reported | Delta |
|---|---|---|---|---|
| 3-stage (20+10+10) | `stage_03.pth` | **0.2935** | 0.2912 | +0.0023 |
| 5-stage (8x5) | `stage_05.pth` | **0.2503** | 0.2510 | −0.0007 |
| 10-stage (4x10) | `stage_10.pth` | **0.1938** | 0.1938 | −0.0000 |

All three deltas are well inside run-to-run evaluation noise. The
3-stage run also gives mAP@0.50 = 0.1760 and mAR@0.25 = 0.7661.

Every intermediate stage of all three protocols was also evaluated (18
checkpoints total) in an overnight sweep — full table in
**`logs/SWEEP_REPORT.txt`**, machine-readable in `logs/results.json`.

### Full per-stage sweep (mAP@0.25, over classes seen so far)

| Protocol | Per-stage mAP@0.25 → final |
|---|---|
| 3-stage  | 0.1978 → 0.2487 → **0.2935** |
| 5-stage  | 0.1070 → 0.1580 → 0.2019 → 0.2367 → **0.2503** |
| 10-stage | 0.0553 → 0.1024 → 0.1214 → 0.1484 → 0.1658 → 0.1674 → 0.1823 → 0.1916 → 0.1950 → **0.1938** |

**Reading the intermediate stages correctly.** In `SWEEP_REPORT.txt` the
early-stage "reported" deltas look alarmingly large (e.g. 5-stage stage 1:
ours 0.1070 vs reported 0.5361). This is **not** a reproduction failure — it
is an evaluation-scope mismatch:

- Our sweep always evaluates over **all 40 classes**.
- The manifest's per-stage "reported" numbers are computed over only the
  classes **seen so far** (8 classes at 5-stage stage 1).

So at stage 1 our number averages 32 never-trained classes (scoring ≈ 0)
into the mean, while the reported number averages only the 8 trained ones.
The two definitions converge exactly at the **final stage**, where all 40
classes have been seen — and there every protocol matches to within noise
(the [MATCH] rows). The final-stage numbers are the meaningful comparison;
the intermediate rows are a monotone learning curve, not a like-for-like
check.

---

## The one real gotcha: the 40-class metadata

Your reminder to "use the 40-class meta data files" turned out to be the
single thing standing between a working eval and a silently wrong one.

This server already had SUN RGB-D metadata at `/data3/sunrgbd/sunrgbd_40/`
with filenames matching *exactly* what the configs expect
(`sunrgbd_infos_{train,val}_40class.pkl`). Those files are **not** the LDMR
ones. Loading them directly:

- **48 distinct labels spanning 0–51**, not 40 spanning 0–39
- `annos['name']` disagrees with `annos['class']` in 11,597 instances —
  e.g. `name='bed'` carries `label=10`, which under the 40-class vocabulary
  is `garbage_bin`

Your `SUNRGBDDataset` label-space contract check caught this immediately and
printed exactly the right diagnosis, including the fix. Worth calling out:
without that check this would have produced plausible-looking but meaningless
mAP, and it would have been very hard to trace. It's a good piece of defensive
engineering.

**Resolution:** downloaded the real PKLs from
`huggingface.co/datasets/Peisheng/LDMR-data` into `meta_data/sunrgbd/` —
verified strictly 0–39, exactly 40 labels, names and indices agreeing, same
5050 val scenes — and pointed `repo/data/sunrgbd/*.pkl` at those. Point
clouds still symlink to `/data3/sunrgbd` (they're fine and shared); only the
metadata label space differed. Nothing under `/data3/sunrgbd` was modified.

---

## Environment setup: four blockers and their fixes

For anyone repeating this on a similar box. Full detail in `MEMORY.md`.

1. **MinkowskiEngine — the bundled wheel doesn't work here.**
   `MinkowskiEngine-0.5.4-cp39-cp39-linux_x86_64.whl` installs but
   `import MinkowskiEngine` fails with `GLIBC_2.34 not found`. The wheel was
   built on Ubuntu 22.04 (glibc 2.35); this box is 20.04.6 (glibc 2.31).
   Built from source instead (commit `405b39c`, CUDA 11.3, `TORCH_CUDA_ARCH_LIST=8.6`).
   Two sub-blockers: no `python3.9-dev` headers and no sudo — obtained via
   `apt download libpython3.9-dev` + `dpkg-deb -x` and pointed `CPATH` at both
   the `python3.9` include dir *and* its parent (`pyconfig.h` includes a path
   relative to the parent). System gcc 9.4.0 works as-is — the installation
   guide's local-GCC-9 procedure is unnecessary on this box.

2. **`pip install -e .` fails** with `AssertionError: assert req_to_install.is_direct`.
   Cause: `install_requires` pulls `nuscenes-devkit`, which unconditionally
   hard-requires the full `jupyter` metapackage, which breaks pip's legacy
   resolver. Fix: `pip install --no-build-isolation --no-deps -e .`.

3. **`ModuleNotFoundError: nuscenes`** on `import mmdet3d` — the dataset
   registry eagerly imports `nuscenes_mono_dataset` even for SUN RGB-D-only
   work. Fix: `pip install --no-deps nuscenes-devkit==1.1.9`.

4. **`ImportError: cannot import name 'gcd' from 'fractions'`** via
   trimesh→networkx. `requirements` resolved to networkx 2.2, which uses an
   API removed in Python 3.9. Fix: `networkx==2.8.8`. Also needed
   `mmsegmentation==0.24.1` (`mmdet3d/__init__.py` imports and version-asserts
   `mmseg`, but nothing declares it).

**Suggestion:** items 2–4 are all things a fresh cloner will hit. A short
"known install issues" block in the README, or pinning `networkx` and adding
`mmsegmentation` to requirements, would save the next person a couple of hours.

---

## Notes on the results

Per-class AP@0.25 at 3-stage / stage 3 shows the expected long-tailed shape:

- **Strong:** `toilet` .873, `bed` .863, `chair` .799, `night_stand` .725,
  `sofa` .666, `sofa_chair` .646
- **Near-dead:** `paper` .0001, `book` .003, `door` .011, `picture` .014,
  `keyboard` .035, `box` .035

The weak classes are the small/thin/rare ones — consistent with SUN RGB-D's
class imbalance rather than anything specific to the incremental protocol.
Recall stays high where AP is low in several cases (`cabinet` AP .056 but
AR .777, `drawer` AP .053 / AR .706), i.e. the detector is finding these
objects but scoring/localising them poorly, not missing them outright.

Per-stage mAP is computed over the classes seen *so far*, so the stage
sequence is not a like-for-like comparison — later stages average over a
larger and harder class set. Keep that in mind when reading the curve in
`SWEEP_REPORT.txt` as a "forgetting" measure.

---

## Open items

- **The uploaded files.** You mentioned uploading checkpoints and metadata,
  but nothing was found under `/data3/ryan` or `~`. Everything here came from
  the public HuggingFace repos instead. If you did upload them somewhere,
  point me at the path and I'll diff against what I fetched.
- **ScanNetV2** is the next scoped dataset. Not started. It needs extracted
  ScanNet point clouds on this server, which hasn't been verified yet.
- **Paper PDF** — `paper/` is still empty; `scp` it there if you want me to
  read it.
- **Config↔protocol mapping**: the README documents the 10-stage eval command
  verbatim. The 3- and 5-stage mappings I inferred from the training commands
  plus each manifest's `source_run` string, and the reproduced numbers confirm
  the inference was right. Might be worth stating all three explicitly in the
  README.
