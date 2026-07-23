# LDMR Exploration — Local Memory

Scope: this file only covers `/data3/ryan/ldmr_exploration`. It is separate from
the assistant's global cross-project memory — nothing here leaks to other
folders/projects, and nothing from outside should be assumed relevant unless
copied in here explicitly.

## Project

- Reproducing & evaluating **LDMR** (Learning-Dynamics-driven Memory and
  Review) — incremental 3D object detection, ECCV 2026.
- Paper: Peisheng Qian, Jie Xu, Xulei Yang, Na Zhao. arXiv:2607.14560
  (not yet uploaded locally — see "Paper PDF" below).
- Repo: https://github.com/qianpeisheng/LDMR — cloned to `./repo`.
- Backbone: TR3D. Datasets: SUN RGB-D (40 classes), ScanNetV2 (35 classes).
- Checkpoints (public): https://huggingface.co/Peisheng/LDMR
- Dataset metadata (public): https://huggingface.co/datasets/Peisheng/LDMR-data
- Protocols released for SUN RGB-D: 3-stage (20+10+10, seed 200, reported
  29.12 mAP@0.25), 5-stage (8x5, seed 200, 25.10), 10-stage (4x10, seed 200,
  19.38). Every stage's checkpoint is released, not just final.

## Server environment

- 2x RTX 3090 (24GB), driver 560.35.05, system CUDA toolkit 12.6 (nvcc).
  No sudo access for `ryan`.
- LDMR requires: Python 3.9, PyTorch 1.12.1+cu113, MinkowskiEngine 0.5.4,
  mmcv-full 1.6.0, mmdet 2.24.1. `/usr/bin/python3.9` is available on this
  server.
- Existing venvs under `~/.venvs` (`tr3d-venv`, `tr3d-build`) belong to the
  unrelated `tr3d` project (Python 3.8) — do not reuse them for LDMR.
- The repo ships a **prebuilt MinkowskiEngine 0.5.4 wheel** for
  cp39-linux_x86_64 at `repo/MinkowskiEngine/MinkowskiEngine-0.5.4-cp39-cp39-linux_x86_64.whl`,
  built on Ubuntu 22.04 + CUDA 11.3 + RTX 3090 (arch 8.6) per
  `repo/MinkowskiEngine/MinkowskiEngine_Installation_Guide.md`. Our GPUs match
  (RTX 3090) — try installing this wheel directly before attempting a from-source
  build (which needs a local CUDA 11.3 toolkit + GCC-9, neither present yet).
- `repo/activate_server_env.sh` defaults to `$HOME/local/cuda-11.3`,
  `$HOME/local/openblas`, `$HOME/opt/gcc-9` — this was `peisheng`'s original
  machine layout, none of it exists under ryan's `$HOME`. Override or skip
  this script; only needed if we end up building ME from source.

## Dataset (SUN RGB-D)

- A fully prepared SUN RGB-D dataset already exists on this server at
  `/data3/sunrgbd` (owned by user `peisheng`, group/other-readable):
  `OFFICIAL_SUNRGBD/`, `sunrgbd_trainval/`, `points/` (12G), `matlab/`,
  `meta_data/`, and a `sunrgbd_40/` subfolder containing
  `sunrgbd_infos_train_40class.pkl` and `sunrgbd_infos_val_40class.pkl`.
- Those filenames match exactly what the LDMR configs expect at
  `data/sunrgbd/sunrgbd_infos_{train,val}_40class.pkl` — likely means the
  MATLAB extraction + `create_data.py --use-40-classes` step is already done
  and we can skip it.
- Plan: symlink `repo/data/sunrgbd` contents to `/data3/sunrgbd` rather than
  copying (12G+35G+7.9G) or re-running MATLAB extraction. Confirmed with Ryan
  before doing this — see [[TASKS]].
- Hard rule: never write to, modify, or delete anything under `/data3/sunrgbd`
  — it's outside this project folder and owned by another user.

## Checkpoints

- As of 2026-07-23, searched `/data3/ryan` and `~` and found **no** uploaded
  LDMR checkpoint or metadata files despite Ryan saying they were uploaded.
  The 40-class SUN RGB-D `.pkl` metadata that already exists at
  `/data3/sunrgbd/sunrgbd_40/` may be what he meant, or he may mean something
  not yet located.
- Checkpoints are also just publicly downloadable from
  https://huggingface.co/Peisheng/LDMR (dirs `sunrgbd_3stage`, `sunrgbd_5stage`,
  `sunrgbd_10stage`, plus scannet equivalents). Each checkpoint carries
  provenance in `torch.load(...)['meta']['ldmr']` (protocol/stage/mAP).
- Status: **blocked** — waiting on Ryan for location or go-ahead to pull from
  HF directly. See [[TASKS]] open questions.

## Decisions (confirmed by Ryan, 2026-07-23)

- Pull checkpoints from HuggingFace directly rather than waiting for upload.
- Evaluate all three SUN RGB-D protocols (3/5/10-stage), not just one.
- Symlink `repo/data/sunrgbd` to `/data3/sunrgbd` rather than copying.

## Gotcha: /data3 is mounted noexec — venv cannot live inside this project folder

- `/data3` is mounted `ext4 rw,nosuid,nodev,noexec,relatime` (confirmed via
  `findmnt /data3`). Any `.so`/binary under `/data3` fails to `dlopen`/`mmap`
  with `PROT_EXEC` — this is what looked like a `Permission denied` on
  `venv/bin/pip` and later `failed to map segment from shared object` when
  importing `torch`. It is a real OS-level mount restriction, not a Claude
  Code sandbox artifact.
- The sibling `tr3d` project (`/data3/ryan/tr3d`) hit and documented this
  exact issue in `tr3d/CLAUDE.md` ("Environment notes" section): its venv
  lives at `/home/ryan/.venvs/tr3d-venv` (4.8G), with `tr3d/venv` being a
  symlink to it. Going *through* the symlink from `/data3` works fine — the
  kernel resolves exec permission from the file's actual mount, not the path
  used to reach it.
- **LDMR mirrors this pattern**: venv lives at `/home/ryan/.venvs/ldmr-venv`,
  with `repo/venv` symlinked to it. Ryan confirmed this 2026-07-23, with one
  constraint: **be economical with `/home` disk space** — the root filesystem
  (`/`, which `/home` lives on) has only ~71GB free out of 879GB (92% used).
  tr3d's venv is ~4.8G as a size reference for what to expect; avoid
  unnecessary bloat (pip cache, build artifacts) and don't create additional
  large dirs under `/home` beyond `~/.venvs/ldmr-venv`.
- Useful reusable facts pulled from `tr3d/CLAUDE.md` for this exact
  Python3.9/PyTorch1.12.1+cu113/MinkowskiEngine0.5.4/mmcv1.6.0/mmdet2.24.1
  stack on this server:
  - System CUDA 11.3 toolkit already installed at `/usr/local/cuda-11.3`
    (`nvcc` release 11.3.109) — no need to build/install our own.
  - mmcv-full 1.6.0 wheel for this stack comes from
    `download.openmmlab.com/mmcv/dist/cu113/torch1.12.0/` (torch1.12.0 index
    also serves 1.12.1 users — mmcv wheels are keyed by minor version).
  - `requirements/runtime.txt`'s numpy 1.21.5 / numba 0.53.0 pins are "overly
    conservative" per tr3d's notes and conflict with PyTorch 1.12.1 + OpenMMLab;
    tr3d relaxed to numpy 1.23.5 / numba 0.56.4. Worth trying the same if LDMR's
    pins cause resolution failures.
  - GPUs are 2x RTX 3090 (sm_86), which PyTorch+cu113 supports natively — no
    `TORCH_CUDA_ARCH_LIST` override needed at runtime (only for building ME
    from source, where tr3d used `TORCH_CUDA_ARCH_LIST="8.6"`).
- Workaround for the earlier `venv/bin/pip` exec issue specifically (now moot
  since the venv is no longer under `/data3`, but noting for completeness):
  prefer `venv/bin/python -m pip ...` over calling wrapper scripts directly
  when execution does have to happen from a noexec-mounted path.

## Status log

- 2026-07-23: Folder created (`MEMORY.md`, `TASKS.md`, `repo/`, `checkpoints/`,
  `paper/`). Repo cloned. Server/GPU/dataset recon done.
  `repo/data/sunrgbd` symlinked to `/data3/sunrgbd` (OFFICIAL_SUNRGBD,
  sunrgbd_trainval, points, + the two 40-class `.pkl` info files) — verified
  the paths match what `configs/tr3d/tr3d_sunrgbd-3d-40class.py` expects.
  Downloaded and SHA256-verified all 18 SUN RGB-D checkpoints (3-stage,
  5-stage, 10-stage, every stage) from huggingface.co/Peisheng/LDMR into
  `checkpoints/`. Python 3.9 venv created at `repo/venv`. PyTorch install
  in progress.

- 2026-07-23 (later session): **Environment build COMPLETE and verified.**
  MinkowskiEngine from-source build finished successfully and passes the GPU
  smoke test. mmdet3d editable-installed. First eval launched. See
  "Environment: FINAL WORKING STATE" below.

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

This is what Ryan meant by "remember to use the 40-class meta data files", and
it is a real trap because the filenames match exactly.

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

(Note for the sibling tr3d project: this is a *different* issue from the
CAGroup3D-points problem recorded in global memory, but it rhymes — borrowed
SUN RGB-D artifacts on this server are repeatedly not what they claim.)

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
Run detached (`nohup ... &`) per Ryan's standing instruction.

## RESULTS (2026-07-23)

Final-stage reproduction on the released checkpoints, 40-class HF metadata,
5050 val scenes, single 3090 each:

| Protocol | Ours mAP@0.25 | Reported | Delta | Verdict |
|---|---|---|---|---|
| 3-stage `stage_03` | **0.2935** | 0.2912 | +0.0023 | ✅ reproduces |
| 5-stage `stage_05` | **0.2503** | 0.2510 | −0.0007 | ✅ reproduces |
| 10-stage `stage_10` | pending (in sweep) | 0.1938 | — | — |

Deltas are well within run-to-run eval noise, so **LDMR's released SUN RGB-D
checkpoints reproduce their reported numbers.** 3-stage also gives
mAP@0.50 = 0.1760 and mAR@0.25 = 0.7661.

Per-class shape at 3-stage/stage-3 (sanity — matches the expected long-tail):
strong on `toilet` .873, `bed` .863, `chair` .799, `night_stand` .725;
near-dead on `paper` .0001, `book` .003, `door` .011, `picture` .014.

Full 18-checkpoint sweep launched 2026-07-23 01:28 (detached, both GPUs) —
see `run_full_sweep.sh` / `collect_results.py`; outputs land in
`logs/SWEEP_REPORT.txt` + `logs/results.json`, marker `logs/SWEEP.DONE`.

## NEXT SESSION: START HERE (written 2026-07-23 ~01:35, session ending)

The environment is **done**. Do not rebuild anything. The only thing in
flight is the 18-checkpoint eval sweep.

### 1. Check whether the overnight sweep finished

```bash
cd /data3/ryan/ldmr_exploration
ls logs/SWEEP.DONE            # exists => finished
cat logs/SWEEP_REPORT.txt     # the results table
cat logs/sweep_master.log     # per-job START/END + rc, in case something failed
```

The sweep was launched detached at 2026-07-23 01:28 (`nohup bash
run_full_sweep.sh &`, master PID was 920818, PPID=1, own process group) so it
is independent of the Claude Code session that started it. Expected runtime
~50 min for the 16 remaining jobs across both GPUs, i.e. it should have
finished around 02:20.

**If `SWEEP.DONE` is missing** (box rebooted, job killed, whatever): just
re-run it. It is idempotent — `run_job()` skips any checkpoint whose log
already contains "Overall", so completed work is never redone:

```bash
cd /data3/ryan/ldmr_exploration
nohup bash run_full_sweep.sh > logs/sweep_master.log 2>&1 &
```

**If the sweep finished but you want the report regenerated** (e.g. after
adding runs by hand):

```bash
repo/venv/bin/python collect_results.py > logs/SWEEP_REPORT.txt
```

### 2. Then finish Week 1

- Fold the sweep numbers into `FINDINGS.md` (the write-up for Ryan — already
  drafted, has a placeholder section for the per-stage curve).
- Tick the remaining boxes in [[TASKS]].

### 3. Two open threads to raise with Ryan

- He said he "uploaded the checkpoints and meta data files", but nothing was
  ever found under `/data3/ryan` or `~`. Everything used so far came from the
  public HuggingFace repos instead (with his go-ahead). If he did upload
  something, diff it against `checkpoints/` and `meta_data/sunrgbd/`.
- ScanNetV2 is the next scoped item. **Not started, and not verified as
  feasible** — it needs extracted ScanNet point clouds on this server, which
  nobody has checked for yet. Check that *before* promising a timeline.

### Files in this folder

| Path | What |
|---|---|
| `MEMORY.md` | this file — env recipe, gotchas, results |
| `TASKS.md` | weekly tasking checklist |
| `FINDINGS.md` | the write-up for Ryan |
| `run_full_sweep.sh` | evaluates all 18 ckpts over 2 GPUs, idempotent |
| `collect_results.py` | parses all `logs/eval_*.log` -> report + `results.json` |
| `checkpoints/sunrgbd_{3,5,10}stage/` | 18 released ckpts + manifests |
| `meta_data/sunrgbd/` | **the correct 40-class PKLs** (from HF) |
| `logs/` | per-eval logs, `SWEEP_REPORT.txt`, `results.json`, `SWEEP.DONE` |
| `repo/` | the LDMR clone; `repo/venv` -> `/home/ryan/.venvs/ldmr-venv` |
| `paper/` | empty — Ryan hasn't dropped the PDF here yet |
