# LDMR Exploration — Task Tracker

See [[MEMORY]] for background/context. This file tracks weekly tasking only.

## Week 1 (assigned 2026-07-23)

Goal: reproduce the LDMR setup and evaluate the provided checkpoints on
SUN RGB-D, using the 40-class metadata.

- [x] Create `ldmr_exploration` folder with local memory + task tracking
- [x] Clone https://github.com/qianpeisheng/LDMR to `./repo`
- [x] Recon server env (GPU, CUDA, Python versions) and existing SUN RGB-D data
- [x] Decisions confirmed with Ryan: pull checkpoints from HuggingFace, run
      all three protocols (3/5/10-stage), symlink the dataset
- [x] Download + SHA256-verify all 18 SUN RGB-D checkpoints from
      huggingface.co/Peisheng/LDMR into `checkpoints/`
- [x] Create Python 3.9 venv at `/home/ryan/.venvs/ldmr-venv`
      (symlinked as `repo/venv`; `/data3` is noexec, mirrors tr3d's setup)
- [x] PyTorch 1.12.1+cu113, mmcv-full 1.6.0, mmdet 2.24.1, runtime deps
- [x] **MinkowskiEngine 0.5.4 built from source** (prebuilt wheel is glibc-2.35,
      this box is 2.31) — GPU smoke test passes
- [x] mmsegmentation 0.24.1 + networkx 2.8.8 + nuscenes-devkit (`--no-deps`)
- [x] `pip install -e .` with `--no-deps`; `import mmdet3d` OK (1.0.0rc3)
- [x] **Found and fixed the 40-class metadata trap**: the preexisting
      `/data3/sunrgbd/sunrgbd_40/*.pkl` are NOT LDMR-compatible (labels 0-51,
      name/index mismatches) despite identical filenames. Downloaded the real
      ones from huggingface.co/datasets/Peisheng/LDMR-data into
      `meta_data/sunrgbd/` and repointed the symlinks. See MEMORY.md.
- [~] Eval final-stage checkpoint of each protocol, record mAP@0.25
      - [x] 3-stage `stage_03.pth`: **0.2935** vs 0.2912 reported (+0.0023) ✅
      - [x] 5-stage `stage_05.pth`: **0.2503** vs 0.2510 reported (-0.0007) ✅
      - [ ] 10-stage `stage_10.pth` (expect 0.1938) — in the sweep below
- [~] Compare against README-reported numbers — 2/3 done, both MATCH
- [~] **Full 18-checkpoint sweep RUNNING** (launched 2026-07-23 01:28, detached,
      both GPUs, ~50 min). `bash run_full_sweep.sh`, master log
      `logs/sweep_master.log`. On completion it writes `logs/SWEEP.DONE`,
      `logs/SWEEP_REPORT.txt` and `logs/results.json` via `collect_results.py`.
      Re-runnable: it skips any checkpoint whose log already has results.
- [ ] Write up findings for Ryan

Skipped for now: `pytest -q` (65 tests) — the real end-to-end check is the
eval run itself, which exercises the same import + model + data path.

## How to send me the paper PDF

There's no chat-based file upload in this environment — I only see what's on
the server's filesystem. Easiest options:
- `scp`/`rsync` the PDF from your machine to
  `/data3/ryan/ldmr_exploration/paper/` on this server, or
- if you're already on the server, just `cp` it there yourself.
Once it's in that folder I'll pick it up automatically.

## Backlog / later weeks

- ScanNetV2 evaluation
- Training reproduction (if scoped for a later week)
