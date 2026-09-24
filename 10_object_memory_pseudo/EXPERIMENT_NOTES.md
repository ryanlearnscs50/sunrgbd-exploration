# Experiment notes — September 23–24, 2026

## Protocol

- SUN RGB-D 40 classes, frequency-order S5 (8×5), stages 2–5 continued from
  the released stage-1 checkpoint. Continuation seeds: 201, 202, 203.
- Starting checkpoint SHA256:
  `8ff54ab599834f58477e3511d10af0252d206a9b42d8c12e4aa796b07b55ed96`.
- Random object bank: 20 crops/class, source-relative height, at most one paste.
  Dose means scene-level attempt probability (0, .25, .50, 1), not accepted dose.
- Pseudo confidence .50; NMS .30; pseudo-versus-GT threshold .25. No reviewing.
- Continuation epochs: 2/2/1/1. Original step milestones [8,11] leave LR constant
  at .001. Cosine uses per-iteration decay toward .0001 without warmup.
- Final metrics cover all 40 classes. Earlier stages cover seen classes only;
  old/new cohorts are recomputed at each stage.

## Reproduction layout

The source overlay in `../09_object_level_memory/ldmr_overlay/` contains complete
modified files relative to upstream LDMR `ab67f3d`; it is not a standalone package.
Apply it to a full upstream checkout, using the file paths below the overlay root.
Set up a separate experiment workspace as follows:

```
experiment_workspace/
  repo/                    full LDMR checkout with overlay applied
    venv/                  working Python environment (or symlink)
  checkpoints/sunrgbd_5stage/stage_01.pth
  launch_week3_experiments.py
  run_week3_experiment.sh
  collect_week3_status.py
  summarize_week3_results.py
  summarize_week3_replication.py
  run_week3_overnight.py
  monitor_week3_cutoff.py
  monitor_week3_cosine_replication.py
```

Copy the Python/shell files from this directory into that workspace. The archived
shell runner has a server-specific `ROOT=/data3/ryan/ldmr_exploration`; change it
to the new workspace before running. Python scripts derive ROOT from their own
location. Use a fresh `week3_runs/` for new training: the published folder is an
archive, and the launcher deliberately refuses existing run identities.

After installing the environment, dataset and checkpoint, inspect commands with:

```bash
repo/venv/bin/python launch_week3_experiments.py --dry-run \
  --modes pseudo_only dose25 --seed 201
```

Omitting `--dry-run` starts detached jobs on the two GPUs. Corrected cosine
identifiers are `pseudo_cosine_v2` and `dose25_cosine_v2`. Archived cutoff/queue
scripts encode the original experiment campaign, including its historical cutoff
and source hashes; they require a new plan/state for another campaign and are not
general-purpose launch commands.

Summaries consume local `incremental_logs/`, which is not included. Published
JSON and report tables can be inspected without that directory. Original absolute
paths and PIDs in manifests identify the archived runs; they are not portable
paths or evidence that the processes are still active. Source hashes record the
code at launch time; report-wording edits after completion can differ. One existing
trainer docstring was also reworded and trailing whitespace normalized for
publication, without changing execution.

## Runtime and data

The tested environment is Python 3.9, PyTorch 1.12.1+cu113, MinkowskiEngine 0.5.4,
mmcv-full 1.6.0 and mmdet 2.24.1, on two RTX 3090 GPUs. Use the released
[40-class metadata](https://huggingface.co/datasets/Peisheng/LDMR-data), not similarly
named metadata with a larger label space. Checkpoints are available from
[Peisheng/LDMR](https://huggingface.co/Peisheng/LDMR).

## Verification and limitations

Publication checks: 13 focused scheduler, insertion and pipeline tests passed
from the LDMR checkout root. Command:

```bash
venv/bin/python -m pytest -q tests/test_incremental_stage_lr_config.py \
  tests/test_exemplar_insertion.py tests/test_object_memory_pipeline_wiring.py
```

The scheduler tests resolve configs relative to the checkout root. Python syntax,
JSON parsing, shell syntax, and publication diff whitespace were also checked.

All twelve valid runs have zero exits, completion markers, correctly scoped
stage-2–5 metrics and nonempty final checkpoints. Every replay run has a final
bank. All four valid cosine runs show actual stage LR decay. Seed-202 initial
replay banks match completely between schedules, including point arrays.

The two original cosine attempts are excluded because stage config preparation
ignored the incremental scheduler override. The fix propagates an explicit
`lr_config` by deep copy while retaining the base-config fallback. Regression
coverage includes stage preparation, hook behavior, copy isolation and fallback.

The seeds vary continuation training and bank randomness, not stage-1 training.
Candidate selection on seed 201, evolving teachers, stochastic inference and
augmentation RNG limit causal and statistical interpretation. All experiments
finished on September 24; no additional work is queued.
