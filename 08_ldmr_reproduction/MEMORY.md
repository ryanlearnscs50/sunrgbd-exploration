# LDMR environment and reproduction notes

The released SUN RGB-D checkpoint evaluation is complete. Later object replay
experiments are in `../09_object_level_memory/`; pseudo supervision, dose and LR
experiments are in `../10_object_memory_pseudo/`.

## Environment: FINAL WORKING STATE (2026-07-23)

Everything below is done and verified — do not redo it.

| Component | Status |
|---|---|
| Repo clone, dataset symlinks, checkpoint downloads (18 ckpts, SHA256-verified) | ✅ |
| venv at `/home/ryan/.venvs/ldmr-venv`, symlinked as `repo/venv` | ✅ |
| PyTorch 1.12.1+cu113 / torchvision 0.13.1+cu113 | ✅ `torch.cuda.is_available()==True` |
| mmcv-full 1.6.0, mmdet 2.24.1 | ✅ |
| **MinkowskiEngine 0.5.4 (built from source)** | ✅ GPU smoke test passes (SparseTensor + MinkowskiConvolution) |
| mmsegmentation 0.24.1 (+ mmcls 0.25.0 pulled in) | ✅ required — `mmdet3d/__init__.py` imports `mmseg` |
| networkx 2.8.8 (upgraded from 2.2) | ✅ required — 2.2 does `from fractions import gcd`, removed in Py3.9 |
| nuscenes-devkit 1.1.9 installed `--no-deps` | ✅ required — see below |
| `pip install -e .` (mmdet3d 1.0.0rc3) | ✅ **must use `--no-deps`** — see below |
| `import mmdet3d` | ✅ prints 1.0.0rc3 |

### Four install gotchas, all now solved (don't rediscover)

1. **`pip install -e .` fails** with `AssertionError: assert
   req_to_install.is_direct` — setup.py's `install_requires` pulls
   `nuscenes-devkit`, which hard-requires the `jupyter` metapackage, which
   breaks pip's legacy resolver. **Fix: `pip install --no-cache-dir
   --no-build-isolation --no-deps -e .`** (all real deps installed by hand
   already). Note setup.py passes **no `ext_modules`** — this fork relies on
   mmcv's CUDA ops, so there is nothing to compile here. That's expected.
2. **`ModuleNotFoundError: nuscenes`** on `import mmdet3d` — the dataset
   registry eagerly imports `nuscenes_mono_dataset`. This risk was predicted
   last session and did happen. **Fix: `pip install --no-cache-dir --no-deps
   "nuscenes-devkit==1.1.9"`** (skips the jupyter chain).
3. **`ModuleNotFoundError: mmseg`** — **Fix: `pip install
   "mmsegmentation==0.24.1"`** (mmdet3d asserts `0.20.0 <= mmseg <= 1.0.0`).
4. **`ImportError: cannot import name 'gcd' from 'fractions'`** via
   trimesh→networkx — **Fix: `pip install "networkx==2.8.8"`**.

### MinkowskiEngine build recipe (already done; kept for reproducibility)

The repo's **prebuilt wheel does not work here** — it's built on Ubuntu 22.04
(glibc 2.35); this server is Ubuntu 20.04.6 (glibc 2.31), so
`import MinkowskiEngine` fails with `GLIBC_2.34 not found`. Built from source
instead (source at `/home/ryan/.venvs/ldmr-build/MinkowskiEngine`, commit
`405b39c`, the same one tr3d used):

```bash
cd /home/ryan/.venvs/ldmr-build/MinkowskiEngine
PYINC=/home/ryan/.venvs/ldmr-build/python3.9-dev-root/usr/include/python3.9
PYINC_PARENT=/home/ryan/.venvs/ldmr-build/python3.9-dev-root/usr/include
nohup env CUDA_HOME=/usr/local/cuda-11.3 PATH=/usr/local/cuda-11.3/bin:$PATH \
  FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=8 \
  CPATH="$PYINC:$PYINC_PARENT" \
  /data3/ryan/ldmr_exploration/repo/venv/bin/python setup.py install \
  --blas=openblas --force_cuda > me_build.log 2>&1 &
```

Two blockers baked into that command: (a) no `python3.9-dev` headers and no
sudo — got them via `apt download libpython3.9-dev` + `dpkg-deb -x` into
`~/.venvs/ldmr-build/python3.9-dev-root`, then pointed `CPATH` at **both**
`.../include/python3.9` (for `Python.h`) and its parent (because `pyconfig.h`
includes `<x86_64-linux-gnu/python3.9/pyconfig.h>` relative to the parent).
(b) System CUDA 11.3 at `/usr/local/cuda-11.3` and system gcc 9.4.0 both work
as-is — the LDMR guide's local-GCC-9 dance is unnecessary on this box.

## CRITICAL: the 40-class metadata — `/data3/sunrgbd/sunrgbd_40/` is WRONG for LDMR

The filenames match the expected metadata names, but their label spaces differ.

`/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_{train,val}_40class.pkl` (the
preexisting files owned by `peisheng`) are **NOT** LDMR-compatible. LDMR's
`SUNRGBDDataset` has a built-in label-space contract check that rejects them:

```
ValueError: SUNRGBDDataset: SUNRGBD label-space contract violation detected.
out_of_range_labels(count): 40:170, 41:126, ... 50:631, 51:546
name_index_mismatches=11597
mismatch_example: scene_id=1, name=bed, label=10, mapped_name=garbage_bin
```

Verified directly: those PKLs carry **48 distinct labels spanning 0–51** (a
different, larger label space), and `annos['name']` disagrees with
`annos['class']` under the 40-class vocabulary.

**The correct files** are the released ones from
`https://huggingface.co/datasets/Peisheng/LDMR-data` (`sunrgbd/` subfolder).
Verified: **strictly 0–39, exactly 40 labels, names and indices agree**
(`bed`→5, `night_stand`→14, `dresser`→23, `lamp`→8). Same 5050 val scenes.

Downloaded to `/data3/ryan/ldmr_exploration/meta_data/sunrgbd/` and the
symlinks `repo/data/sunrgbd/sunrgbd_infos_{train,val}_40class.pkl` now point
**there**, not at `/data3/sunrgbd/sunrgbd_40/`. Everything else under
`repo/data/sunrgbd` (`points/`, `sunrgbd_trainval/`, `OFFICIAL_SUNRGBD/`)
still symlinks to `/data3/sunrgbd` and is fine — the point clouds are shared;
only the *metadata* label space differed.

## Running an evaluation

```bash
cd /data3/ryan/ldmr_exploration/repo
OMP_NUM_THREADS=8 nohup venv/bin/python tools/eval_incremental.py \
  <config> <checkpoint> --eval mAP > ../logs/<name>.log 2>&1 &
```

Config ↔ protocol ↔ expected final mAP@0.25 (from README + manifests):

| Protocol | Config (`configs/incremental/sunrgbd/`) | Final ckpt | Reported |
|---|---|---|---|
| 3-stage (20+10+10) | `tr3d_dynamic_head_20x10x10_pseudo_memory_ld_design2_reviewing_521.py` | `stage_03.pth` | 0.2912 |
| 5-stage (8x5) | `tr3d_dynamic_head_8x5_pseudo_memory_ld_design2_reviewing_52211.py` | `stage_05.pth` | 0.2510 |
| 10-stage (4x10) | `tr3d_dynamic_head_4x10_pseudo_memory_ld_design2_reviewing_6111111111.py` | `stage_10.pth` | 0.1938 |

The 10-stage config↔protocol pairing is confirmed verbatim by the README's
Evaluation section; the other two are inferred from the training commands plus
each manifest's `source_run` string (e.g. 3-stage's starts
`sunrgbd_s3_ld2revpse...` = design2 + reviewing + pseudo → the `..._ld_design2_
reviewing_521.py` config). Per-stage reported mAPs live in
`checkpoints/sunrgbd_*stage/manifest.json` and in each ckpt's
`['meta']['ldmr']`.

Eval speed: ~7 task/s on one 3090 over 5050 val scenes ≈ 12 min per run.
Evaluations use detached processes (`nohup ... &`).

## RESULTS (2026-07-23)

Final-stage reproduction on the released checkpoints, 40-class HF metadata,
5050 val scenes, single 3090 each:

| Protocol | Ours mAP@0.25 | Reported | Delta | Verdict |
|---|---|---|---|---|
| 3-stage `stage_03` | **0.2935** | 0.2912 | +0.0023 | ✅ reproduces |
| 5-stage `stage_05` | **0.2503** | 0.2510 | −0.0007 | ✅ reproduces |
| 10-stage `stage_10` | **0.1938** | 0.1938 | −0.0000 | ✅ reproduces |

Deltas are well within run-to-run eval noise, so **LDMR's released SUN RGB-D
checkpoints reproduce their reported numbers.** 3-stage also gives
mAP@0.50 = 0.1760 and mAR@0.25 = 0.7661.

Per-class shape at 3-stage/stage-3 (sanity — matches the expected long-tail):
strong on `toilet` .873, `bed` .863, `chair` .799, `night_stand` .725;
near-dead on `paper` .0001, `book` .003, `door` .011, `picture` .014.

**Full 18-checkpoint sweep COMPLETE (finished 2026-07-23 02:17, all rc=0).**
Every intermediate stage of all three protocols evaluated. Results in
`logs/SWEEP_REPORT.txt` + `logs/results.json`; marker `logs/SWEEP.DONE`.
All three final-stage numbers MATCH reported (see table above).

**IMPORTANT — how to read intermediate-stage deltas (not a bug):** the sweep
always evaluates over all 40 classes, but the manifest's per-stage "reported"
numbers are computed over classes-seen-*so-far* (e.g. 8 classes at 5-stage
stage 1). So early-stage deltas look huge (5-stage s1: ours 0.107 vs reported
0.536) purely because ours averages in ~32 never-trained classes scoring ≈ 0.
The two definitions converge only at the final stage, where all 40 are seen —
and there every protocol matches. Final-stage rows are the meaningful check.
