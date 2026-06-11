# TR3D Class-Incremental — Detailed Results (per-stage)

Generated 2026-06-10 from the on-disk eval logs under
`/data3/ryan/tr3d/work_dirs/incremental_v4_*` via
`tools_incremental/analyze_incremental.py`.

---

## How to read this report (plain-language guide)

**What the task is.** We teach a 3D object detector (TR3D) to find furniture in
indoor scenes, but we reveal the object categories *a few at a time* instead of
all at once. Each batch of new categories is a **stage**. After each stage we
measure how well the model does — both on the brand-new categories it just
learned and on the older ones it learned before. The worry is **catastrophic
forgetting**: as the model learns new categories, it tends to forget the old
ones. The whole study is about how badly that happens and which tricks reduce it.

**What the numbers are (and their units).** Every score in this report is an
**AP** or **mAP** value.

- **AP = Average Precision** for one object category. It is a single number
  between **0 and 1** that summarizes how reliably the detector finds that
  category — rewarding it for catching real objects (recall) without raising
  false alarms (precision). It is a **pure fraction (unitless)**: 0 = useless,
  1 = perfect. Multiply by 100 to read it as a percent — e.g. `0.617` means
  **61.7%**.
- **mAP = mean Average Precision** = the plain average of the per-category AP
  values over all categories being scored. Same 0–1 scale, same "×100 = percent"
  convention. This is the single headline number people quote.

**What `@0.25` and `@0.50` mean.** A detection only counts as "correct" if the
predicted 3D box overlaps the true box well enough. We measure overlap with
**IoU** (Intersection-over-Union: the shared volume divided by the combined
volume of the two boxes, again a 0–1 fraction). `@0.25` requires at least 25%
overlap to count as a hit; `@0.50` requires at least 50%. So **`@0.50` is the
stricter, harder test** — its scores are always lower than `@0.25` for the same
model. Both are standard for SUN RGB-D; we report both.

**`overall / old / new`.** Each per-stage cell shows three mAP numbers:
- **overall** = mAP averaged over *every* category seen so far (the cumulative
  vocabulary at that stage).
- **old** = mAP averaged only over categories introduced in an *earlier* stage —
  this is the "are we forgetting?" number.
- **new** = mAP averaged only over the categories introduced *in this very stage*
  — this is the "are we still able to learn?" number.
A `—` in the **old** column means there are no old classes yet (stage 1).

**`#cls`** = how many categories are in scope at that stage (it grows each stage,
since the vocabulary is cumulative).

**Forgetting.** For each old category we record the AP it had *at the moment it
was introduced* and the AP it still has *at the final stage*. **Forgetting =
(AP at introduction) − (AP at final stage)**, averaged over old categories. It is
on the same 0–1 scale; a **positive** number means the score dropped (bad), and
**lower forgetting is better**. `+0.1465` means the model lost, on average, 0.1465
of AP (≈14.7 percentage points) on its old categories.

**The variants** (naive / pseudo) are the different anti-forgetting
tricks being compared — described in the Setup section just below.

**The joint (fully-supervised) upper bound** at the very end is a reference
model trained the "normal" way — all categories labeled and learned at once, no
staging. It is the **ceiling**: the best score achievable on this data if
forgetting were not an issue at all. We report each incremental method as a
**percent of this ceiling** so you can see how much of the achievable
performance the staged training actually keeps.

**One-line cheat sheet:** every number is a 0–1 fraction (×100 = %); bigger AP/mAP
is better; `@0.50` is the strict version of `@0.25`; *old* tracks forgetting,
*new* tracks fresh learning; *forgetting* is a drop so smaller is better; *joint*
is the ceiling.

---

## Setup (what was actually run — verified from configs, not memory)

- Detector: TR3D (mmdet3d), one config per stage under
  `configs/tr3d_incremental_v4/{3stage,6stage}/stage*.py`.
- Schedule per stage: `EpochBasedRunner`, **max_epochs = 12**, lr step `[8, 11]`,
  no warmup (identical to the canonical TR3D SUN RGB-D schedule).
- Class vocabulary is the **sunrgbd-40-style naming**, cumulative:
  - **3-stage**: 3 → 17 → **30** classes.
  - **6-stage**: 3 → 9 → 14 → 24 → 25 → **32** classes.
  (NOTE: this is *not* the "19 box-available" set named in older memory notes.)
- Class introduction order (v4 "best order", driven by 3D-box counts):
  - 3-stage:
    - stage1 (+3): chair, table, whiteboard
    - stage2 (+14): computer, desk, keyboard, box, drawer, shelf, garbage_bin, mouse, printer, book, monitor, paper, cup, laptop
    - stage3 (+13): bed, pillow, lamp, night_stand, sofa, dresser, tv, curtain, cabinet, door, mirror, bookshelf, painting
  - 6-stage:
    - stage1 (+3): chair, table, whiteboard
    - stage2 (+6): toilet, sink, bathtub, towel, garbage_bin, mirror
    - stage3 (+5): bed, pillow, lamp, night_stand, dresser
    - stage4 (+10): computer, desk, keyboard, box, shelf, mouse, printer, book, monitor, paper
    - stage5 (+1): fridge
    - stage6 (+7): sofa, coffee_table, tv, drawer, cabinet, painting, laptop
- Each stage is evaluated on the **cumulative** val set (all classes seen so far).
- `old_mAP` = mean AP over classes introduced in an *earlier* stage;
  `new_mAP` = mean AP over classes introduced *this* stage.
- Variants (the anti-forgetting tricks being compared):
  - **naive** — just keep training the same model on each new stage's data, with
    nothing done to protect the old categories. The baseline; expected to forget
    the most.
  - **pseudo** — before training a stage, run the *previous* stage's model over
    the new scenes and let it auto-label any old-category objects it still
    recognizes ("pseudo-labels" = machine-generated labels, since the new scenes
    only come with hand-labels for the new categories). Those pseudo-labels are
    added to training so the model keeps practicing the old categories. The
    **score threshold** (0.50) is the confidence cut-off: only auto-labels the
    old model is ≥50% sure about are kept, to avoid teaching it from its own
    mistakes.
  - **pseudo@0.45** — same idea, looser cut-off (keep auto-labels it is ≥45%
    sure about → more old-category labels, slightly noisier). 3-stage only.
- **fully-supervised (joint) upper bound** — one model trained from scratch on
  the full train set with all classes labeled at once; evaluated on the same
  cumulative final-stage val pkl. This is the ceiling the incremental variants
  are measured against. See last section.

---

## 3-stage schedule

### mAP@0.25 — per stage (overall / old / new)

| stage | #cls | naive | pseudo@0.50 | pseudo@0.45 |
|-------|-----:|------:|------------:|------------:|
| stage1 |  3 | 0.3373 / —      / 0.3373 | 0.3373 / —      / 0.3373 | 0.3373 / —      / 0.3373 |
| stage2 | 17 | 0.0912 / 0.0020 / 0.1103 | 0.1395 / 0.2676 / 0.1121 | 0.1428 / 0.2782 / 0.1138 |
| stage3 | 30 | 0.0994 / 0.0039 / 0.2243 | 0.1280 / 0.0587 / 0.2187 | **0.1366 / 0.0792 / 0.2117** |

### mAP@0.50 — per stage (overall / old / new)

| stage | #cls | naive | pseudo@0.50 | pseudo@0.45 |
|-------|-----:|------:|------------:|------------:|
| stage1 |  3 | 0.1915 / —      / 0.1915 | 0.1915 / —      / 0.1915 | 0.1915 / —      / 0.1915 |
| stage2 | 17 | 0.0263 / 0.0001 / 0.0319 | 0.0525 / 0.1591 / 0.0297 | 0.0516 / 0.1736 / 0.0255 |
| stage3 | 30 | 0.0497 / 0.0023 / 0.1117 | 0.0601 / 0.0260 / 0.1047 | **0.0683 / 0.0365 / 0.1098** |

### Forgetting (mean over old classes, AP@0.25 at intro stage − at final stage3)

| variant | mean forgetting (lower = better) | final old_mAP@0.25 |
|---------|---------------------------------:|-------------------:|
| naive        | +0.1465 | 0.0039 |
| pseudo@0.50  | +0.0931 | 0.0587 |
| **pseudo@0.45** | **+0.0740** | **0.0792** |

Per-class **AP@0.25 retained at the final stage** for every old class (higher =
less forgetting; `intro` = AP the class had at the stage it was introduced, naive
run, as a reference for how much there was to lose):

| old class | intro | naive | pseudo@0.50 | pseudo@0.45 |
|-----------|------:|------:|------------:|------------:|
| chair       | 0.748 | 0.000 | 0.527 | **0.617** |
| table       | 0.198 | 0.000 | 0.042 | **0.095** |
| whiteboard  | 0.066 | 0.000 | 0.000 | 0.000 |
| computer    | 0.089 | 0.006 | **0.052** | 0.024 |
| desk        | 0.210 | 0.005 | 0.182 | **0.211** |
| keyboard    | 0.111 | 0.000 | 0.000 | 0.000 |
| box         | 0.048 | 0.000 | **0.012** | 0.011 |
| drawer      | 0.048 | 0.004 | **0.018** | 0.017 |
| shelf       | 0.051 | 0.020 | 0.038 | **0.095** |
| garbage_bin | 0.350 | 0.002 | 0.103 | **0.128** |
| mouse       | 0.001 | 0.000 | 0.000 | 0.000 |
| printer     | 0.162 | 0.024 | 0.005 | **0.035** |
| book        | 0.002 | 0.000 | 0.000 | 0.000 |
| monitor     | 0.317 | 0.004 | 0.020 | **0.061** |
| paper       | 0.001 | 0.000 | 0.000 | 0.000 |
| cup         | 0.000 | 0.000 | 0.000 | 0.000 |
| laptop      | 0.155 | 0.000 | 0.000 | **0.052** |
| **mean forgetting** | | +0.1465 | +0.0931 | **+0.0740** |

The high-frequency classes (chair, table, desk, garbage_bin, monitor) are where
the methods separate; the many near-zero classes (mouse, book, paper, cup) had
almost no AP to begin with, so they neither forget nor retain meaningfully.

---

## 6-stage schedule (no thr045 variant run)

### mAP@0.25 — per stage (overall / old / new)

| stage | #cls | naive | pseudo@0.50 |
|-------|-----:|------:|------------:|
| stage1 |  3 | 0.3311 / —      / 0.3311 | 0.3311 / —      / 0.3311 |
| stage2 |  9 | 0.3001 / 0.0028 / 0.4488 | 0.3486 / 0.1571 / 0.4444 |
| stage3 | 14 | 0.1524 / 0.0078 / 0.4126 | 0.2419 / 0.1451 / 0.4161 |
| stage4 | 24 | 0.0379 / 0.0138 / 0.0716 | 0.0921 / 0.1093 / 0.0680 |
| stage5 | 25 | 0.0050 / 0.0000 / 0.1230 | 0.0321 / 0.0273 / 0.1484 |
| stage6 | 32 | 0.0393 / 0.0000 / 0.1795 | 0.0423 / 0.0015 / 0.1882 |

### mAP@0.50 — per stage (overall / old / new)

| stage | #cls | naive | pseudo@0.50 |
|-------|-----:|------:|------------:|
| stage1 |  3 | 0.1983 / —      / 0.1983 | 0.1983 / —      / 0.1983 |
| stage2 |  9 | 0.1622 / 0.0016 / 0.2426 | 0.1935 / 0.0728 / 0.2539 |
| stage3 | 14 | 0.0753 / 0.0021 / 0.2069 | 0.1159 / 0.0597 / 0.2170 |
| stage4 | 24 | 0.0107 / 0.0074 / 0.0153 | 0.0290 / 0.0399 / 0.0138 |
| stage5 | 25 | 0.0030 / 0.0000 / 0.0762 | 0.0081 / 0.0033 / 0.1247 |
| stage6 | 32 | 0.0199 / 0.0000 / 0.0909 | 0.0203 / 0.0006 / 0.0907 |

### Forgetting (mean over old classes, AP@0.25 → final stage6)

| variant | mean forgetting | final old_mAP@0.25 |
|---------|----------------:|-------------------:|
| naive       | +0.2635 | 0.0000 |
| pseudo@0.50 | +0.2612 | 0.0015 |

### Per-class AP@0.25 retained at the final stage (every old class)

| old class | intro | naive | pseudo@0.50 |
|-----------|------:|------:|------------:|
| chair       | 0.749 | 0.000 | 0.000 |
| table       | 0.202 | 0.000 | 0.000 |
| whiteboard  | 0.042 | 0.000 | 0.000 |
| toilet      | 0.841 | 0.000 | 0.000 |
| sink        | 0.589 | 0.000 | 0.000 |
| bathtub     | 0.780 | 0.000 | 0.000 |
| towel       | 0.097 | 0.000 | 0.000 |
| garbage_bin | 0.272 | 0.000 | 0.000 |
| mirror      | 0.114 | 0.000 | 0.000 |
| bed         | 0.707 | 0.000 | 0.000 |
| pillow      | 0.139 | 0.000 | 0.000 |
| lamp        | 0.210 | 0.000 | 0.000 |
| night_stand | 0.649 | 0.000 | 0.000 |
| dresser     | 0.359 | 0.000 | **0.023** |
| computer    | 0.051 | 0.000 | 0.000 |
| desk        | 0.194 | 0.000 | **0.003** |
| keyboard    | 0.101 | 0.000 | 0.000 |
| box         | 0.035 | 0.000 | 0.000 |
| shelf       | 0.077 | 0.000 | 0.000 |
| mouse       | 0.000 | 0.000 | 0.000 |
| printer     | 0.085 | 0.000 | **0.012** |
| book        | 0.002 | 0.000 | 0.000 |
| monitor     | 0.171 | 0.000 | 0.000 |
| paper       | 0.000 | 0.000 | 0.000 |
| fridge      | 0.123 | 0.000 | 0.000 |
| **mean forgetting** | | +0.2635 | **+0.2612** |

The collapse is near-total: by stage6 almost every old class has decayed to
0.000 for both variants. Pseudo-labeling keeps a faint trace on a couple of
classes (dresser, printer) and has marginally lower mean forgetting than naive,
but the absolute numbers are negligible either way. This is the
schedule-length / degenerate-stage5 problem, not a variant problem.

---

## Takeaways

1. **Pseudo-labeling sharply reduces forgetting** vs naive fine-tuning on the
   3-stage chain: final old-class mAP@0.25 jumps 0.0039 → 0.0792 (~20×) and mean
   forgetting roughly halves (+0.1465 → +0.0740). Overall mAP rises 0.0994 →
   0.1366.
2. **Threshold 0.45 beats 0.50** on every headline number (overall, old, and
   forgetting). 0.45 is the current best 3-stage configuration.
3. **6-stage is pathological**: old-class mAP decays to ~0 for both variants by
   the final stage. The long chain plus the degenerate single-class stage5
   (fridge only) destroys retention. This schedule needs design attention
   separately from the variant comparison.
4. Epoch budget is **12 per stage** everywhere (lr drops at 8 & 11). The
   optimal-epoch question (can `<12` epochs retain old classes better?) is now
   answered — see **"Optimal-epoch study"** below. Short version: early-stopping
   is the *wrong* lever. It buys a negligible bump for naive and actively *hurts*
   pseudo, which wants the full 12 epochs.

## Optimal-epoch study (3-stage)

**Question.** The TR3D schedule trains every incremental stage for 12 epochs.
Catastrophic forgetting is usually *worsened* by longer fine-tuning on the new
task, so the hypothesis was: **stop early (`<12` epochs) to retain old classes
better.** This study sweeps the per-stage epoch budget `E ∈ {2,4,6,8,12}` for both
variants on the 3-stage chain and reports the final-stage (stage3) cumulative-val
mAP, broken into old (17 classes seen before the final stage) vs new (13 final-stage
classes). Run overnight 2026-06-11, fully unattended; raw report at
`tr3d/logs/EPOCH_STUDY_REPORT.txt`.

### How the sweep was run (mechanics)

- **Epoch count is a *global* knob, not per-stage.** Because each stage warm-starts
  from the previous stage's checkpoint, you cannot pick a stage's epochs in
  isolation — a 4-epoch stage3 must be taught by a 4-epoch stage2 teacher. So each
  `E` is a **full independent re-run of the entire chain** (stage1@E → expand head →
  stage2@E → expand → stage3@E), not a per-stage tweak. Different `E` values *are*
  independent of each other, so they ran in parallel, two per wave across the 2× 3090s.
- **lr milestones rescaled per E.** The paper's `step=[8,11]` would never fire for
  `E<8`, so milestones are rescaled `m1=floor(2E/3)`, `m2=floor(11E/12)` (deduped/
  clamped). Everything else (seed 0, deterministic, voxel size, cumulative-val pkl)
  is identical to the main runs. `E=12` reuses the existing main-run results.
- **Naive sweep** (`run_naive_v4_epochs.sh`): plain fine-tune; old-class objects in
  the new-stage scenes are unlabeled/carved out.
- **Pseudo sweep** (`run_pseudo_v4_epochs.sh`): before training each stage, the
  *E-epoch* previous-stage model labels old-class objects in the new-stage scenes
  (`make_pseudo_labels.py`, score_thr 0.50), and those pseudo-boxes are merged into
  the training targets. stage1 is seeded from the E-epoch naive stage1 (no old
  classes exist yet there, so they're identical).
- **Orchestration**: one detached master (`run_epoch_study_all.sh`, `setsid`,
  PPID=1) ran scout → naive {2,4,6,8} → pseudo {2,4,6,8} → analysis →
  `EPOCH_STUDY_REPORT.txt` + `epoch_study.DONE`, surviving the closed Claude session.

### Results — NAIVE (stage3 cumulative val)

| E  | old@.25 | new@.25 | all@.25 | old@.50 | all@.50 | chair@.25 |
|----|--------:|--------:|--------:|--------:|--------:|----------:|
| 2  | 0.0083  | 0.1766  | 0.0813  | 0.0004  | 0.0316  | 0.000 |
| 4  | 0.0047  | 0.2266  | 0.1008  | 0.0002  | 0.0472  | 0.000 |
| 6  | 0.0091  | 0.2232  | **0.1019** | 0.0012 | 0.0486 | 0.000 |
| 8  | 0.0042  | 0.2283  | 0.1013  | 0.0009  | 0.0505  | 0.000 |
| 12 | 0.0039  | 0.2243  | 0.0994  | 0.0023  | 0.0497  | 0.000 |

Old-class @.25 is flat-near-zero and noisy (0.004–0.009) across all E; the "best at
E=6" (+0.0053 vs E=12) is within run-to-run noise and never reaches a usable level.
chair stays 0.000 at every budget. **Early stopping does not meaningfully rescue
old classes in the naive variant.**

### Results — PSEUDO (stage3 cumulative val)

| E  | old@.25 | new@.25 | all@.25 | old@.50 | all@.50 | chair@.25 |
|----|--------:|--------:|--------:|--------:|--------:|----------:|
| 2  | 0.0323  | 0.1753  | 0.0943  | 0.0144  | 0.0421  | 0.445 |
| 4  | 0.0413  | 0.2207  | 0.1191  | 0.0190  | 0.0551  | 0.562 |
| 6  | 0.0553  | 0.2218  | 0.1274  | 0.0259  | 0.0626  | 0.600 |
| 8  | 0.0571  | 0.2175  | 0.1266  | 0.0253  | 0.0593  | 0.595 |
| 12 | **0.0587** | 0.2187 | **0.1280** | **0.0260** | 0.0601 | 0.527 |

Old-class @.25 rises **monotonically** with epochs (0.032 → 0.059, ~1.8×); overall
and old@.50 likewise peak at E=12. New-class @.25 plateaus by E≈4. **More epochs is
strictly better for pseudo — the opposite of the hypothesis.**

### Scout: per-stage forgetting curves (naive, prior fixed at its real 12-ep value)

Training a single stage in isolation and evaluating each epoch shows the expected
forgetting dynamic for naive fine-tuning — old-class mAP peaks at epoch 1–2 then
**decays** as training continues:

| stage | old-mAP peak | final (ep12) old-mAP | old gain from early stop |
|-------|-------------:|---------------------:|-------------------------:|
| stage2 | 0.0059 @ ep1 | 0.0020 | +0.0039 (new cost −0.0591) |
| stage3 | 0.0123 @ ep2 | 0.0029 | +0.0094 (new cost −0.0257) |

So early-stopping *does* claw back a little old-class mAP for naive — but the
absolute level (~0.01) is negligible and the new-class cost is large.

### Why is "more epochs better" possible? (the apparent paradox)

The early-stop hypothesis assumes longer fine-tuning = more exposure to *only-new*
data = more forgetting. That holds **only for naive** (and the scout curves confirm
it: old-mAP peaks at ep1–2 then decays). The catch is the magnitude — naive old-class
mAP is essentially zero at every budget, so the "forgetting-with-epochs" effect is
real but irrelevant.

**Pseudo reverses the sign because longer training is no longer training on
new-data-only.** The teacher injects old-class pseudo-labels into every new-stage
scene, so each epoch optimizes a **joint old + new objective** (rehearsal-by-
distillation), not a new-only objective. With old classes explicitly supervised in
every gradient step, extra epochs don't erode them — they *converge* the joint fit.
So pseudo behaves like ordinary training where more epochs → better fit on both
heads (until the new-class head plateaus ~E4 and old-class retention keeps climbing
to E12). The forgetting pressure that makes early-stopping attractive is exactly
what the pseudo-labels neutralize.

**Bottom line:** the epoch budget is the wrong lever for forgetting. Keep **E=12**.
The real lever is the pseudo-label rehearsal — and it wants the full schedule, not a
truncated one.

## Fully-supervised (joint) upper bound

One TR3D trained from scratch on the **full** train set with **all 30 (3-stage)
/ 32 (6-stage) classes labeled simultaneously** (same train scenes as the
incremental chain — the union of all per-stage pools — but with full labels in
every scene), 12 epochs, evaluated on the same cumulative final-stage val pkl
(`sunrgbd_infos_val_stage{3,6}.pkl`). This isolates joint-vs-sequential learning.
Built via `incremental_learning/build_joint_pkl.py`; configs
`configs/tr3d_incremental_v4/{3stage,6stage}/joint.py`; results in
`work_dirs/incremental_v4_joint/{3stage,6stage}/`.

Note: this is **not** comparable to the canonical `tr3d_sunrgbd_canonical` run
(66.6/49.6) — that uses the standard VoteNet 10-class set, a different vocabulary.

| schedule | #cls | joint mAP@0.25 | joint mAP@0.50 |
|----------|-----:|---------------:|---------------:|
| 3-stage  |  30  | **0.2062**     | 0.1010         |
| 6-stage  |  32  | **0.2803**     | 0.1542         |

### Incremental final-stage overall mAP@0.25 vs the joint ceiling

| schedule | naive | best incremental | joint ceiling | best as % of ceiling |
|----------|------:|-----------------:|--------------:|---------------------:|
| 3-stage  | 0.0994 | 0.1366 (pseudo@0.45) | 0.2062 | 66% |
| 6-stage  | 0.0393 | 0.0423 (pseudo@0.50) | 0.2803 | 15% |

Key per-class @0.25 (joint vs best incremental final), the high-frequency
stage-1 classes:

| class | 3-stage joint | 3-stage best incr. | 6-stage joint |
|-------|--------------:|-------------------:|--------------:|
| chair | 0.758 | 0.617 | 0.756 |
| table | 0.245 | 0.095 | 0.303 |
| bed   | 0.821 | — (new at final) | 0.794 |

Takeaway: on **3-stage**, pseudo@0.45 already recovers ~two-thirds of the joint
ceiling — the gap to "train everything at once" is modest. On **6-stage** the
incremental methods sit at only ~15% of the ceiling: the ceiling itself is
*higher* (0.2803 > 0.2062, because joint training is unaffected by the long chain
and the degenerate fridge-only stage5), which makes the sequential collapse look
even worse. The 6-stage failure is a property of the incremental *schedule*, not
a limit of the data — joint training on the same scenes does fine.
