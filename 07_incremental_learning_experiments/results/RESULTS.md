# Step C Results — three variants × two schedules

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

## Optional follow-ups (not run)

- Distill **without** pseudo-labels — isolate distillation's own contribution.
- Sweep the pseudo-label confidence threshold (currently 0.5).
- Directly address the 6-stage stage-5 single-class degeneracy (e.g. merge thin stages, or replay).
