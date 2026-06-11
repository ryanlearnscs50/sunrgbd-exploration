# Step C Results — three variants × two schedules

> **See also [`INCREMENTAL_RESULTS.md`](INCREMENTAL_RESULTS.md)** — the comprehensive,
> mentor-facing report: plain-language glossary, full per-stage *and* per-class tables for both
> schedules at mAP@0.25 **and** mAP@0.50, the **fully-supervised (joint) upper bound**, the
> **pseudo@0.45** threshold variant, and the **optimal-epoch study**. The summary below is the
> original 3-way Step-C comparison; the follow-up experiments added since are described at the
> bottom of this file under [Follow-up experiments](#follow-up-experiments-since-the-initial-snapshot).

All numbers are **mAP@0.25** (a detection counts as correct at ≥25% 3D-box overlap; the score is the
mean Average Precision over classes, 0–1). Produced by `step_c_tools/analyze_incremental.py` from the
per-stage cumulative-validation eval logs in [`eval_logs/`](eval_logs/).

**Columns.** `overall` = mean AP over *all* classes known at the final stage. `old` = mean AP over only the
classes from *earlier* stages (the retention metric — the heart of the problem). `forget` = mean
*forgetting* = average over old classes of `(AP when the class was introduced − AP at the final stage)`;
**higher = forgot more**.

| variant  | 3stage overall | 3stage old | 3stage forget | 6stage overall | 6stage old | 6stage forget |
|----------|:--------------:|:----------:|:-------------:|:--------------:|:----------:|:-------------:|
| naive    | 0.0994         | 0.0039     | +0.1465       | 0.0393         | 0.0000     | +0.2635       |
| pseudo   | **0.1280**     | **0.0587** | **+0.0931**   | 0.0423         | 0.0015     | +0.2612       |
| distill  | 0.1274         | 0.0547     | +0.0972       | **0.0451**     | **0.0089** | **+0.2575**   |

Per-stage overall mAP@0.25 progression (each stage evaluated on the cumulative val set):

| variant  | 3stage (s1→s3)            | 6stage (s1→s6)                                  |
|----------|---------------------------|-------------------------------------------------|
| naive    | .337 → .091 → .099        | .331 → .300 → .152 → .038 → .005 → .039         |
| pseudo   | .337 → .140 → .128        | .331 → .349 → .242 → .092 → .032 → .042         |
| distill  | .337 → .138 → .127        | .331 → .344 → .224 → .081 → .018 → .045         |

---

## The three variants

**1. Naive fine-tuning (baseline / floor).** Just keep training on each stage's new classes with no
protection for old ones. Between stages the classifier head is grown by `expand_head.py` (old class weights
copied verbatim, new columns randomly initialized) so old knowledge isn't wiped *at expansion time* — but
training then overwrites it anyway. Result is **textbook catastrophic forgetting**: by the final stage old
classes collapse toward 0 AP (6-stage: *every* old class → 0.0000). This is the bar the other variants beat.

**2. Pseudo-labeling (the priority).** Before training stage *t*, run the stage *t−1* model over stage *t*'s
training scenes, keep its confident old-class detections, and merge them in as if they were ground-truth
labels (offline, saved to `*_pseudo.pkl`; originals never overwritten). Now each stage is continually
reminded of the old classes. **Key tuning finding:** SDCoT's recommended confidence threshold (0.9–0.95) is
for VoteNet; TR3D's dense sigmoid-focal scores run far lower (max ~0.8, median ~0.04), so a 0.9 threshold
keeps **zero** boxes. Recalibrated to **0.5**, which yields sensible, correctly-distributed pseudo-labels.

**3. Distillation (optional; closest port of full SDCoT).** On top of pseudo-labels, keep the stage *t−1*
model as a **frozen teacher** and add a penalty that keeps the student's old-class output logits close to the
teacher's (centered-MSE on logits). The alignment that SDCoT works hard for is free here: teacher and student
voxelize the *same* points, so their sparse voxels line up 1-to-1 and logits can be compared directly. See
`../model_code/README.md`.

---

## Reading the numbers

- **3-stage: pseudo ≈ distill, both crush naive.** Old-class mAP jumps from 0.004 (naive) to ~0.058, and
  forgetting roughly halves (+0.147 → +0.093). Adding the distillation teacher on top of pseudo buys
  essentially nothing here — pseudo already does the work, so prefer it for simplicity.

- **6-stage: distillation is marginally best, exactly where intended.** It has the best final old-class mAP
  (0.0089 vs pseudo 0.0015 vs naive 0.0000) and the lowest forgetting (+0.2575). Per-stage, pseudo leads
  through stages 2–5 while distill edges ahead at the final stage (deepest old-class retention).

- **But all three collapse at 6-stage's stage 5.** That stage ("kitchen") introduces a single new class
  (fridge) in ~40 scenes — a degenerate one-class fine-tune that hammers every old class at once. The teacher
  by that point is itself several stages forgotten, and kitchen scenes contain few old objects to re-detect,
  so pseudo-labels can't supply enough defense. Distillation slows this tail bleed but doesn't stop it; it is
  the clearest structural weakness of the 6-stage schedule.

**Bottom line.** Pseudo-labeling is the workhorse mitigation. Distillation is worth it only on long schedules
with severe tail forgetting. The remaining open problem is the degenerate single-class stage.

---

## Follow-up experiments (since the initial snapshot)

Three follow-ups were run after the original three-variant comparison above. Full tables and prose are in
[`INCREMENTAL_RESULTS.md`](INCREMENTAL_RESULTS.md); the headlines:

### 1. Fully-supervised (joint) upper bound

One model trained from scratch on the **same scenes** as the incremental chain but with **all** class labels
present in every scene — the ceiling that any incremental method is chasing. Evaluated on the same final
cumulative val set. Tools: `data_prep/build_joint_pkl.py`, `configs/{3stage,6stage}/joint.py`,
`step_c_tools/run_joint_v4.sh`. Logs: `eval_logs/joint_{3stage,6stage}.txt`.

| schedule | joint mAP@0.25 | joint mAP@0.50 | best incremental @0.25 | % of ceiling |
|----------|:--------------:|:--------------:|:----------------------:|:------------:|
| 3-stage  | **0.2062**     | 0.1010         | pseudo@0.45 0.1366     | ~66%         |
| 6-stage  | **0.2803**     | 0.1542         | pseudo 0.0423          | ~15%         |

The 3-stage incremental result reaches two-thirds of its ceiling. The 6-stage ceiling is *higher* than the
3-stage one (joint does fine on the same scenes), so the 6-stage collapse is a property of the **incremental
schedule** (long chain + the degenerate fridge-only stage 5), not the data.

### 2. Pseudo-label threshold tuning (0.45)

A fine sweep of the global confidence cut-off (0.40–0.50, proxy precision/recall/F1 on the train-set teacher
via `step_c_tools/verify_pseudo_quality.py`) identified **0.45** as the balanced sweet spot, then a full
3-stage retrain confirmed it beats the 0.50 baseline on every old-class metric at a tiny new-class cost.
Tool: `step_c_tools/run_pseudo_v4_thr045.sh`. Logs: `eval_logs/pseudo_thr045_3stage_stage*.txt`.

| 3-stage final (AP@0.25) | old mAP | overall | new mAP | mean forgetting (17 old) |
|-------------------------|:-------:|:-------:|:-------:|:------------------------:|
| pseudo @0.50 (baseline) | 0.0587  | 0.1280  | 0.2187  | +0.0931                  |
| **pseudo @0.45**        | **0.0792** | **0.1366** | 0.2117 | **+0.0740**            |

Old-class mAP +35%, forgetting cut ~20% relative; biggest save is chair (drop +0.221 → +0.131). Only 3-stage
was retrained — the 6-stage teachers are too degraded for a threshold to help. **Adopt 0.45 for 3-stage pseudo.**

### 3. Optimal-epoch study (3-stage)

Tests the hypothesis that incremental stages train *too long* (the TR3D paper's 12 epochs) and overfit the
novel-only data at the expense of old-class retention. A full sweep over epoch budgets E∈{2,4,6,8,12} for
both naive and pseudo, plus per-stage per-epoch scout traces. Tools under `step_c_tools/epoch_study/`; report
+ curve in `results/epoch_study/`.

**Conclusion: epoch budget is the wrong lever — keep E=12.** Naive old-class retention is flat-near-zero and
noisy across all E (early-stop buys ~nothing). Pseudo old-class mAP rises *monotonically* with epochs
(E2 0.032 → E12 0.059), because pseudo injects teacher old-class labels into every stage, so each epoch
optimizes a joint old+new objective (rehearsal-by-distillation) rather than novel-only — there is no
forgetting pressure to stop early for.

> *Extension in progress (not in this snapshot):* because pseudo was still climbing at E=12, a sweep to
> E∈{16,24,36} is running to locate the plateau/overfit point. Results and the final
> `epoch_sweep_3stage.png` curve will land in a later update.

---

## Still-open follow-ups (not run)

- Distill **without** pseudo-labels — isolate distillation's own contribution.
- Per-stage (rather than global) pseudo-label thresholds.
- Directly address the 6-stage stage-5 single-class degeneracy (e.g. merge thin stages, or replay).
- A split where old classes **recur** across stages, to unlock pseudo-labeling's real ceiling.
