# Week 3: verified cosine schedule comparison

> Historical seed-201 analysis. Seed 202 is now complete; final interpretation is
> in `WEEK_3_WRAP_UP.md` and measurements in `WEEK_3_COSINE_REPLICATION.md`.
> Replay adds no final mAP@.25 at reported precision under cosine in seed 202.
> Experimentation closed September 24 around 17:00 SGT.

Both corrected runs completed on September 24: pseudo-only at 13:09:43 SGT and 25% replay at 13:22:32 SGT. Exit codes are zero. Final checkpoints, four correctly scoped stage metrics, completion markers, and the replay bank were verified. Logged LR decays in every trained stage, from approximately .001 to .0001. Host GPUs were idle at 13:29 SGT.

## Final stage, seed 201

Values are mAP fractions. All runs share the released stage-1 checkpoint.

| Policy | mAP@.25 | Old @.25 | New @.25 | mAP@.50 |
|---|---:|---:|---:|---:|
| pseudo_only | 0.23521 | 0.26854 | 0.10187 | 0.13286 |
| pseudo_cosine_v2 | 0.23635 | 0.26949 | 0.10380 | 0.14394 |
| dose25 | 0.24612 | 0.28184 | 0.10323 | 0.14031 |
| dose25_cosine_v2 | 0.25157 | 0.28598 | 0.11391 | 0.15251 |

## Paired differences by stage

Cosine minus the original schedule; differences are fractions, so .01 equals one percentage point.

| Replay | Stage | mAP@.25 | Old @.25 | New @.25 | mAP@.50 |
|---|---:|---:|---:|---:|---:|
| pseudo_only | 2 | +0.00792 | +0.01409 | +0.00175 | +0.01191 |
| pseudo_only | 3 | +0.00649 | +0.01616 | -0.01288 | +0.00892 |
| pseudo_only | 4 | -0.00468 | +0.00144 | -0.02305 | +0.00228 |
| pseudo_only | 5 | +0.00114 | +0.00095 | +0.00193 | +0.01108 |
| dose25 | 2 | +0.00904 | +0.01856 | -0.00048 | +0.00587 |
| dose25 | 3 | +0.01488 | +0.01580 | +0.01304 | +0.01735 |
| dose25 | 4 | +0.01339 | +0.01486 | +0.00895 | +0.01162 |
| dose25 | 5 | +0.00545 | +0.00414 | +0.01068 | +0.01220 |

## Interpretation and next step

The strongest measured final result this week is 25% replay with cosine: .25157 mAP@.25 and .15251 mAP@.50. Relative to its matched constant-LR run, the gains are +.00545 and +.01220, respectively. Both final old-class (+.00414) and new-class (+.01069) mAP@.25 improve. Overall mAP improves at both thresholds in all four trained stages for this replay policy.

Pseudo-only gains only +.00114 final mAP@.25, although final mAP@.50 rises +.01108. Its stage-4 mAP@.25 falls by .00468, driven by new classes. Cosine is therefore not uniformly beneficial across all policy/stage/cohort combinations.

At the final stage, replay adds .01522 mAP@.25 under cosine versus .01091 under the original schedule. The difference of these effects is +.00431, a descriptive single-seed interaction, not established synergy.

The replay-dose result has three continuation seeds (mean final gain +.00975; follow-up seeds alone +.00917). The cosine result has only seed 201. Shared stage-1 initialization, stochastic inference, evolving teachers/banks, and the changed mean LR limit interpretation. Do not change the default or claim statistical significance from this pair.

The original 09:13 cosine runs did not apply the LR override and were deliberately stopped. Their artifacts remain preserved but are excluded from this comparison and the live measured-results table. The corrected runs restarted from stage 1 after fixing stage-config propagation.

Final update: seed 202 is complete. See `WEEK_3_COSINE_REPLICATION.md` and
`WEEK_3_WRAP_UP.md`; the seed-201 interpretation above is preliminary.
