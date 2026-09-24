# Object-level memory bank findings

**Checkpoint date:** 2026-09-10
**Scope:** SUN RGB-D 40-class, five-stage 8x5 frequency-order protocol,
seed 201, stages 2--5 resumed from the same released stage-1 checkpoint.

## Outcome

The object-level bank is implemented end to end and measurably prevents
catastrophic forgetting. The preferred tested policy is height-aware replay
with one attempted pasted object per natural training scene.

| Run | Stage 2 | Stage 3 | Stage 4 | Stage 5 mAP@.25 | Stage 5 mAP@.50 |
|---|---:|---:|---:|---:|---:|
| No-memory fine-tuning | .1704 | .0982 | .0869 | .0314 | .0123 |
| Object replay, floor placement, 3 at p=.7 | .1891 | .1091 | .1065 | .0682 | .0280 |
| Object replay, height-aware, 3 at p=.7 | **.2017** | .1173 | .1251 | .0803 | .0372 |
| Object replay, height-aware, 1 at p=1.0 | .1957 | **.1344** | **.1257** | **.0808** | **.0387** |
| Scene replay, random, no pseudo/review | **.3492** | **.2676** | **.2369** | **.1929** | **.1063** |

The low-dose object policy improves final mAP@.25 over no-memory fine-tuning by
.0494 absolute (+157% relative). It matches the heavier replay dose at the
final stage and improves novel-stage plasticity (.1037 versus .0890), so the
SUN RGB-D config now defaults to one attempted paste per sample.

The completed matched-mechanism scene control uses the same seed and released
stage-1 checkpoint and also disables pseudo labels and reviewing. Random scene
replay reaches .1929 final mAP@.25, exceeding random object replay by **.1121
absolute**. Its final cohort APs are s1=.3447, s2=.1672, s3=.1388, s4=.2067,
s5=.1072. The nearly identical novel-cohort result (.1072 versus .1037) shows
that most of the gap is old-class retention rather than reduced plasticity.

## What is implemented and validated

- Self-contained, box-local object crops; replay does not reopen source scenes.
- Balanced per-class storage, deterministic selection, pickle persistence,
  and metadata-only JSON sidecars.
- Correct local/world transforms, bottom-centred output boxes, yaw-aware
  placement envelopes, and collision rejection against natural and pasted
  boxes.
- Source-floor-relative height preservation for elevated objects, with a
  backward-compatible v2 state loader and an explicit floor-placement
  ablation switch.
- Wiring before point sampling/geometric augmentation in the incremental
  dataset pipeline.
- Strict stage and prior-class validation when resuming from a saved bank.
- Final-stage persistence. The tested stage-5 artifact has 800 objects,
  exactly 20 for every class, 3,000,371 cropped points, and all height offsets.
- SUN RGB-D and ScanNet S5 configs. ScanNet geometry was audited, but a real
  ScanNet run is not possible on this machine because extracted point data is
  absent.
- Offline bank visualization: quota/source-diversity dashboard and a projected
  crop gallery for all 40 classes. The first gallery exposes sparse and
  planar/background-dominated crops that aggregate counts alone conceal.
- Object source-seat reviewing groundwork: original carrier scenes can be
  deduplicated for LD evaluation, their `ld_drop` weights transfer to all
  corresponding crops, and insertion samples objects by those weights.
- Full test suite: 103 passed on 2026-09-10.

## Interpretation

Height preservation fixed a real placement error and improved final mAP@.25
from .0682 to .0803. Reducing replay from an expectation of 2.1 attempted
objects per sample to exactly one did not reduce final retention and produced
better novel-cohort AP. This indicates that replay dose, rather than bank
capacity, was limiting plasticity in the heavier run.

The matched random-scene control establishes that the present object replay
mechanism retains substantially less old-class knowledge than whole scenes.
This comparison holds seed, initial checkpoint, pseudo-label policy, and
reviewing policy fixed, but uses each representation's configured native
budget (528 scene seats versus 800 object crops), not an equal-byte budget.
The likely causes now include lost object context, variable crop support, and
the fact that one pasted crop supplies far less supervised content than a
replayed scene.

The released complete LDMR result is .2503 final mAP@.25, but it additionally
combines Design-2 selection, intra-stage reviewing, and pseudo labels. The
matched experiments below now isolate the first two mechanisms for crops.

## LDMR-aligned extension result (2026-09-10 14:25 +08)

The no-review Object Design-2 run completed at .0667 mAP@.25 / .0251 mAP@.50.
Its random-selection control, resumed from the exact same from-scratch stage-1
checkpoint but with a rebuilt random bank, completed at **.0931 mAP@.25 / .0450
mAP@.50**. The selection-policy difference is .0264. A September 23 artifact audit
confirmed that the initial banks share only 2 of 160 object identities, so this
comparison includes stage-1 bank selection rather than isolating later updates.
The earlier checkpoint confound is resolved.

The no-pseudo Design-2+reviewing ablation also completed cleanly. Its stage
2--5 mAP@.25 trajectory is .1793, .1103, .1064, **.0632**. Reviewing does not
rescue Design-2 crop selection: it is .0035 below Design-2 without reviewing
and .0299 below matched random selection. Its final cohort AP@.25 is s1=.0445,
s2=.0132, s3=.0345, s4=.1271, s5=.0969. Review statistics and weighted
sampling summaries exist at every stage, making this a measured negative
result rather than a missing mechanism.

The pseudo-label combination remains implemented but is intentionally not run
in the current task window. Its guarded launcher expected `epoch_1.pth` while
the five-segment schedule produces `epoch_5.pth`; the guard is fixed for future
use. A new full run would not complete in the remaining task time and is not
needed to close the object-bank baseline and visualization deliverable.

The completed stage-5 Design-2 bank has also been visualized and audited. It
contains 800 annotated crops, 20/class, and exactly 20 distinct source scenes
inside every class; across classes it covers 725 source scenes versus 701 for
the random bank. Design-2 selected fewer cropped points overall (2,542,744
versus 3,000,371) and a lower median crop size (1,430 versus 1,687 points).
It reduced the most extreme sparse tail (3 versus 9 crops below 20 points; 35
versus 43 below 100), but slightly increased crops below 500 points (189 versus
175). Visual inspection still shows planar/background-heavy examples, so
source-scene learning dynamics and diversity do not by themselves provide an
object-crop quality criterion. The visualizer now records global
distinct/duplicate source assignments, mean point count, and counts below
20/50/100/500 points directly in `summary.json`.

The visualization deliverable is complete. Alongside both bank audits and
40-class crop galleries, deterministic before/after views show real crops
inserted into three real SUN RGB-D scenes, including XY boxes and side-view
height placement. A compact results figure summarizes the replay trajectories
and the matched selection/reviewing ablation. A direct random-versus-Design-2
figure compares support distributions, sparse tails, source diversity, and
per-class median crop support.

## Reproducibility artifacts

- Preferred low-dose run:
  `incremental_logs/sunrgbd_s5_object_memory_heightaware_1obj_sunrgbd40_s5_freqorder_20260908_151502/`
- No-memory control:
  `incremental_logs/sunrgbd_s5_no_memory_finetune_sunrgbd40_s5_freqorder_20260908_151502/`
- Heavy height-aware run:
  `incremental_logs/sunrgbd_s5_object_memory_heightaware_sunrgbd40_s5_freqorder_20260908_111255/`
- Original floor-placement run:
  `incremental_logs/sunrgbd_s5_object_memory_sunrgbd40_s5_freqorder_20260907_161225/`
- Matched random scene-memory control:
  `incremental_logs/sunrgbd_s5_scene_memory_random_matched_s201_20260909_sunrgbd40_s5_freqorder_20260909_222706/`
- Random-bank visualization:
  `visualizations/object_memory_random_stage5/`
- Design-2-bank visualization:
  `visualizations/object_memory_design2_stage5/`
- Exact-stage-1 random object control:
  `incremental_logs/sunrgbd_s5_object_memory_random_matched_scratch_s201_20260910_sunrgbd40_s5_freqorder_20260910_100729/`
- Design-2 plus reviewing:
  `incremental_logs/sunrgbd_s5_object_memory_design2_reviewing_resume3_s201_20260910_sunrgbd40_s5_freqorder_20260910_081411/`
- Real pasted-scene views:
  `visualizations/object_memory_pasted_scenes/`
- Experimental summary figure:
  `visualizations/object_memory_progress_summary.png`
- Direct bank-comparison figure and JSON:
  `visualizations/object_memory_bank_comparison/`
