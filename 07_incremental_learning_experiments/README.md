# 07 — Incremental-Learning Experiments (TR3D on SUN RGB-D)

Class-incremental 3D object detection: take [TR3D](https://github.com/SamsungLabs/tr3d) (a sparse-voxel
MinkowskiEngine detector) and teach it new object classes **in stages**, where each stage only ever sees
labels for its *own new* classes — never the old ones. The central question is **catastrophic forgetting**:
how much does the model forget earlier classes as it learns later ones, and which technique best prevents it?

The structural reference is **SDCoT** ("Static-Dynamic Co-Teaching", AAAI 2022), a VoteNet-based incremental
detector. We port its ideas onto TR3D's very different dense-voxel architecture.

> This folder is a **reference snapshot of the code, configs, design docs, and result tables** — not a
> runnable repo. The multi-GB inputs (per-stage `.pkl` label files, `points/*.bin` point clouds) and the
> trained checkpoints (`*.pth`) are **not** included; they live on the working server. Paths in the scripts
> point at those server locations.

---

## The task (three steps, as assigned)

- **Step A — data foundation.** Copy the dataset plumbing and swap in *our* object-detection label files
  for the reference codebase's. (Done earlier; the relevant label source is the 40-class SUN RGB-D mmdet3d
  pkl filtered to our split classes.)
- **Step B — incremental setup + class orderings.** Adapt TR3D into a multi-stage incremental pipeline, and
  decide the **order classes are introduced** across stages — trying more than one. We build **two
  independent schedules**: a **3-stage** and a **6-stage** ordering. (See `data_prep/` + `split_outputs/`.)
- **Step C — training variants.** Implement and compare **(1) naive fine-tuning** (baseline, no forgetting
  mitigation) and **(2) pseudo-labeling** (the priority). We additionally built an optional **(3)
  distillation** variant as the closest faithful port of full SDCoT. (See `step_c_tools/` + `model_code/`.)

Results for all three variants on both schedules are in [`results/`](results/RESULTS.md).

---

## How the two schedules are built (Step B, in brief)

The earlier split work (`../05_incremental_splits/`) assigned classes to stages using **2D scene-level
class presence**. That turned out to be the wrong signal: a class can be *drawn* in a room type yet have
almost **no 3D bounding boxes** there (e.g. "door" appears in 157 classroom 2D-annotations but has a 3D box
in only 1). Training a 3D detector on such a class gives it nothing to learn.

So the **v4** builder here re-derives everything from **actual 3D-box counts**: each class is assigned to the
room type where it has the most 3D boxes, a support **floor** (≥20 training scenes) drops classes too thin to
learn, and the stage **order** is chosen by a per-class maximin sweep (maximize the smallest class's scene
count first). Result: two schedules whose every trained class actually has boxes to learn from.

- **3-stage** (30 classes): `classroom → office → bedroom`
- **6-stage** (32 classes): `classroom → bathroom → bedroom → office → kitchen → living_room`

---

## Folder map

```
data_prep/            Step A/B: build the splits, carve per-stage pkls, generate TR3D configs
  incremental_splits_v4_3dbox.py   the v4 split builder (3D-box driven, maximin ordering)
  carve_v4.py                      carve per-stage train (novel-only) + cumulative val pkls
  gen_configs_v4.py                emit the per-stage TR3D configs
  STEP_B_DESIGN.md                 design doc: mapping SDCoT's pipeline onto TR3D
  archive/                         superseded precursors (v3 builder, v2-era carve/config helpers)

split_outputs/        the v4 split products (small, human-readable)
  split_report_v4.md               summary of the resulting schedules
  stage_def_*.json, stage_manifest_*.json, *_assignments_*.csv, *_lift_*.csv, *_box_counts_*.csv

step_c_tools/         Step C: the training + evaluation scripts
  expand_head.py                   grow the classifier head between stages, copying old weights
  make_pseudo_labels.py            offline pseudo-label generator (the priority variant)
  run_naive_v4.sh                  variant 1: naive fine-tuning
  run_pseudo_v4.sh                 variant 2: pseudo-labeling
  run_distill_v4.sh                variant 3: pseudo-labeling + logit distillation
  analyze_incremental.py           parse eval logs -> per-stage old/new mAP + forgetting
  verify_pseudo_quality.py         proxy precision/recall/F1 of teacher pseudo-labels vs threshold
  run_pseudo_v4_thr045.sh          pseudo retrain at the tuned 0.45 confidence threshold
  run_joint_v4.sh                  fully-supervised (joint) upper-bound run
  epoch_study/                     optimal-epoch study: epoch-budget sweep + per-epoch scout traces
  archive/                         earlier harness skeletons

model_code/           modified TR3D / mmdet3d source (see model_code/README.md for the diffs)
  mink_single_stage_inc.py         new detector subclass adding the distillation teacher
  tr3d_head.py                     head with the empty-batch crash fix
  detectors__init__.py             registers the new detector

configs/              generated per-stage configs (3stage/, 6stage/, generated_configs.json)
                      + joint.py (the fully-supervised upper-bound config) per schedule

results/              RESULTS.md (3-way comparison) + INCREMENTAL_RESULTS.md (comprehensive report)
  eval_logs/                       extracted per-stage AP tables (naive/pseudo/distill + joint + thr045)
  epoch_study/                     optimal-epoch report + sweep curve
```

The `data_prep/build_joint_pkl.py` script builds the joint (all-labels) train pkls for the upper-bound run.

---

## Reproduce (on the working server, in order)

```bash
# Step B: build splits -> carve pkls -> generate configs
python data_prep/incremental_splits_v4_3dbox.py
python data_prep/carve_v4.py
python data_prep/gen_configs_v4.py

# Step C: run each variant for each schedule (GPU id as 2nd arg)
bash step_c_tools/run_naive_v4.sh   3stage 0
bash step_c_tools/run_pseudo_v4.sh  3stage 0
bash step_c_tools/run_distill_v4.sh 3stage 0   # optional 3rd variant
# ...repeat with 6stage

# Analyze
python step_c_tools/analyze_incremental.py --root work_dirs/incremental_v4_pseudo --sched 3stage --metric 0.25

# Follow-up experiments
python data_prep/build_joint_pkl.py                          # build joint (all-labels) train pkls
bash   step_c_tools/run_joint_v4.sh         3stage 0         # fully-supervised upper bound
bash   step_c_tools/run_pseudo_v4_thr045.sh 3stage 0         # pseudo at the tuned 0.45 threshold
bash   step_c_tools/epoch_study/run_epoch_study_all.sh       # optimal-epoch sweep (3stage)
```

---

## Headline result

Pseudo-labeling is the workhorse — on the clean 3-stage schedule it roughly **halves forgetting** and lifts
old-class mAP ~**15×** over naive fine-tuning. Distillation adds only a small extra margin, and only on the
harder 6-stage tail where it was designed to help. Full numbers in [`results/RESULTS.md`](results/RESULTS.md),
with the comprehensive report (joint upper bound, threshold tuning, epoch study) in
[`results/INCREMENTAL_RESULTS.md`](results/INCREMENTAL_RESULTS.md).

**Follow-up findings.** The 3-stage incremental result reaches **~66%** of its fully-supervised ceiling
(0.1366 vs joint 0.2062 mAP@0.25); 6-stage only ~15%, confirming its collapse is a schedule artefact, not a
data limit. Tuning the pseudo-label confidence threshold to **0.45** lifts old-class mAP +35% and cuts
forgetting ~20%. An optimal-epoch study shows the paper's **12 epochs is already right** for incremental
stages — epoch budget is not the lever for forgetting.
