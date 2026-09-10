# 09 — Object-level memory replay for LDMR

This experiment replaces LDMR's referenced full-scene memory with a
self-contained bank of cropped 3D objects. Each stored crop contains its point
features, class, oriented box geometry, source identity, and source-floor
offset. During later incremental stages, one old-class crop is sampled and
inserted into each current SUN RGB-D training scene before point sampling and
geometric augmentation.

The implementation is based on LDMR commit `ab67f3d` and targets the SUN RGB-D
40-class, five-stage 8×5 frequency-order protocol with seed 201.

## Main result

| Method | Stage 2 | Stage 3 | Stage 4 | Stage 5 mAP@.25 |
|---|---:|---:|---:|---:|
| No-memory fine-tuning | .1704 | .0982 | .0869 | .0314 |
| Object replay, floor placement, heavy dose | .1891 | .1091 | .1065 | .0682 |
| Object replay, height-aware, heavy dose | .2017 | .1173 | .1251 | .0803 |
| **Object replay, height-aware, one crop** | .1957 | **.1344** | **.1257** | **.0808** |
| Random scene replay, matched mechanism | **.3492** | **.2676** | **.2369** | **.1929** |

The preferred object policy improves final mAP@.25 by `.0494` over sequential
fine-tuning and therefore measurably limits catastrophic forgetting. Whole
scene replay remains `.1121` higher. Novel-cohort AP is similar for object and
scene replay, so most of that gap is old-class retention rather than reduced
plasticity.

## Matched selection and reviewing ablation

The faithful Design-2 trajectory run required training stage 1 from scratch.
To avoid confounding selection with a different initial model, the random
continuation uses exactly the same stage-1 checkpoint and bank.

| Continuation after matched stage 1 | Final mAP@.25 | Final mAP@.50 |
|---|---:|---:|
| Random object updates | **.0931** | **.0450** |
| Design-2 object updates | .0667 | .0251 |
| Design-2 + source-scene reviewing | .0632 | .0257 |

Scene-derived learning dynamics improve bank source diversity and reduce the
most extreme sparse tail, but do not improve object replay accuracy. The crop
galleries show that an axis-aligned ground-truth box can still contain sparse,
planar, or background-heavy support. Scene-level dynamics are therefore not a
sufficient object-quality criterion.

## Implementation highlights

- Self-contained, gravity-centred object crops with versioned pickle state and
  metadata-only JSON sidecars.
- Correct local/world transforms, bottom-centred output boxes, yaw-aware
  placement envelopes, and collision rejection.
- Source-floor-relative height preservation, including backward-compatible
  loading for older floor-placement banks.
- Exact per-class quotas, deterministic random or Design-2 selection, and
  diversity-aware source-scene tie-breaking.
- Source-scene review statistics transferred into weighted, without-replacement
  object sampling.
- Stage-3+ resume validation and final-stage bank persistence.
- SUN RGB-D and ScanNet configurations. Real-data evaluation here is SUN RGB-D
  only because extracted ScanNet point clouds were unavailable.
- **103 tests pass** in the working LDMR environment.

## Repository contents

| Path | Contents |
|---|---|
| `OBJECT_MEMORY_FINDINGS.md` | Complete experimental results and interpretation |
| `OBJECT_MEMORY_LDMR_AUDIT.md` | Mechanism-by-mechanism comparison with scene-memory LDMR |
| `results/` | Machine-readable experiment and bank summaries |
| `visualizations/object_memory_progress_summary.png` | Main replay and matched-ablation results |
| `visualizations/random_vs_design2_bank.png` | Direct bank-quality comparison |
| `visualizations/pasted_scene_examples.png` | Real SUN RGB-D scenes before and after crop insertion |
| `visualizations/*_exemplar_gallery.png` | One stored crop for every SUN RGB-D class |
| `ldmr_overlay/` | Modified files, configs, analysis tools, and tests relative to LDMR `ab67f3d` |

The overlay is not a standalone copy of LDMR. Apply the paths under
`ldmr_overlay/` to a checkout of the upstream repository at the commit above.
Large checkpoints, training logs, datasets, Python environments, and object
bank pickle files are intentionally excluded.

## Reproducing the offline figures

From the LDMR checkout after applying the overlay:

```bash
MPLBACKEND=Agg python tools/analysis_tools/visualize_object_memory_bank.py \
  /path/to/object_memory_bank_stage_5.pkl \
  --output-dir /path/to/output

MPLBACKEND=Agg python tools/analysis_tools/visualize_object_memory_insertions.py \
  /path/to/object_memory_bank_stage_5.pkl \
  --output-dir /path/to/output
```

The saved bank is sufficient for the bank dashboard/gallery. The insertion
view additionally requires the locally prepared SUN RGB-D point files and
40-class metadata expected by the LDMR config.
