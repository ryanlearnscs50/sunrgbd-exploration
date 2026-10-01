# Week 4 protocol and reproduction notes

The completed sweep contains six full-training runs: budgets of 100, 50 and
20 objects per class, each at seeds 200 and 201. Stage 1 is trained from
scratch separately in every run. All six completed ten stages. Four further
stage-2 continuations compare random and largest-point-count selection;
these have separate checkpoint lineage and are excluded from the full sweep.

## Fixed training settings

- SUN RGB-D, 40 frequency-ordered classes, ten stages of four new classes.
- Six initial epochs, then one epoch per stage; RepeatDataset 15, batch 16.
- AdamW LR 0.001, weight decay 0.0001; initial epoch 6 uses LR 0.0001.
- Later stages use LR 0.001 throughout their single epoch.
- Minimum crop support 20 points; random within-class bank selection.
- Insertion probability 0.7, up to three candidate crops, ten placement attempts.
- Collision threshold 0.3, XY jitter ±1 m, scene-minimum floor placement.
- Pseudo confidence 0.5, NMS 0.3, pseudo/GT IoU 0.25, maximum 100 per scene.
- No scene replay, reviewing, extra crop yaw, cosine schedule or loss masking.

The capacity caps are the only configuration changes across the main sweep.
Configured replay dose is fixed; accepted paste counts were not logged.
Runtime includes evaluation, pseudo generation and bank construction.

## Source and environment

Use upstream LDMR revision `ab67f3d` and apply the repository's
[implementation overlay](../09_object_level_memory/ldmr_overlay/) at its root.
The overlay contains the complete accumulated object-memory changes and the
four `week4_s10_object*_pseudo.py` configurations. Reference comparisons are
pinned in [the protocol audit](WEEK_4_PROTOCOL_AUDIT.md).

The training host uses Python 3.9, PyTorch 1.12.1+cu113, MMCV/MMDetection3D
and a CUDA-enabled MinkowskiEngine build. Existing data and metadata paths
must be configured for the new host; the dataset is not included. See the
[environment notes](../08_ldmr_reproduction/) for the established setup.

From the configured LDMR checkout, a full-run command is:

```bash
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 python tools/train_incremental_scene.py \
  configs/incremental/sunrgbd/week4_s10_object100_pseudo.py \
  --work-dir /path/to/results/week4_object100_s200 \
  --start-stage 1 --end-stage 10 --seed 200
```

Repeat with the 50/20 configurations and seeds 200/201. These are reproduction
instructions; no further training is scheduled by this publication.

The largest-point-count path was repaired after the original full runs failed.
Its population wrapper now scans all eligible crops and keeps a bounded top-K
heap, with stable ties. The random branch preserves its existing seeded order.
This changes candidate coverage for the alternative selector while limiting
retained crop storage. For background, see the standard library `heapq`
documentation on bounded priority queues.

## Evidence and verification

`week4_runs/analysis.json` contains full-run summaries and class/resource
details. Each full run also includes ten original stage metric JSON files,
its manifest and terminal markers. Original manifests retain training-host
paths as provenance; they are not portable path configuration.

`week4_runs/selection_recovery/audit.json` records the four stage-2 runs,
checkpoint hashes and paired deltas. `pseudo_content_audit.json` records
exact scene-payload comparisons: pseudo labels differ beyond timestamps.
The next selector comparison should reuse shared cached pseudo labels.

Run `python3 verify_week4_results.py` for a standard-library-only check of
published stage metrics, paired deltas, bank counts and summary figures.
The included `week4_*.py` host analyzers additionally require the original
local run directories, model artifacts and environment. They preserve the
analysis implementation; they are not substitutes for the portable check.

Before publication, the host re-audited all six full runs and four diagnostic
runs. Checks covered terminal exits, metric scope/means, finite losses, LR,
bank capacity and selector, nonempty checkpoints and pseudo artifacts, and
diagnostic checkpoint lineage. The 26 focused bank, insertion, population,
pipeline and LR regression tests passed. The independent offline selector
audit verified all 40 classes against exhaustive ranking.

Model payloads were not deserialized by the completion audits. A separate
content check read trusted local pseudo-label pickles. The public numerical
check does not read pickle files or validate the omitted model weights.
