# 08 — LDMR reproduction (SUN RGB-D)

Week-1 reproduction of **LDMR** (Learning-Dynamics-driven Memory and Review,
ECCV 2026, Qian et al.) — incremental 3D object detection on a TR3D backbone.
Repo: https://github.com/qianpeisheng/LDMR · Checkpoints/data:
https://huggingface.co/Peisheng/LDMR

**Goal:** stand up the environment and confirm the released SUN RGB-D
checkpoints reproduce their reported mAP@0.25, across all three incremental
protocols, using the 40-class metadata.

## Result — all three protocols reproduce

| Protocol | Final ckpt | Ours mAP@.25 | Reported | Delta |
|---|---|---|---|---|
| 3-stage (20+10+10) | stage_03 | **0.2935** | 0.2912 | +0.0023 |
| 5-stage (8×5)      | stage_05 | **0.2503** | 0.2510 | −0.0007 |
| 10-stage (4×10)    | stage_10 | **0.1938** | 0.1938 | −0.0000 |

All 18 checkpoints (every intermediate stage) were evaluated over the 5050
SUN RGB-D val scenes on 2× RTX 3090.

## Files

| File | What |
|---|---|
| `FINDINGS.md` | The write-up: headline, the 40-class metadata trap, the four env blockers, per-class notes, how to read the intermediate stages |
| `logs/SWEEP_REPORT.txt` | Full 18-checkpoint results table + forgetting curve |
| `logs/results.json` | Machine-readable results |
| `logs/sweep_master.log` | Per-job START/END/rc for the overnight sweep |
| `run_full_sweep.sh` | Idempotent 2-GPU sweep driver (skips checkpoints already done) |
| `collect_results.py` | Parses per-eval logs → report + JSON |
| `MEMORY.md` | Full environment recipe + gotchas (reproducibility reference) |
| `TASKS.md` | Week-1 task checklist |

## The one thing that mattered: the 40-class metadata

This server already had SUN RGB-D metadata with the *exact* filenames the
configs expect — but a different (0–51) label space, with names disagreeing
with class indices in 11,597 instances. LDMR's `SUNRGBDDataset` contract check
caught it. The correct PKLs come from
https://huggingface.co/datasets/Peisheng/LDMR-data (strictly 0–39, 40 labels,
names/indices agree). See `FINDINGS.md` for detail.
