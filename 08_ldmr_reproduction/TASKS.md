# LDMR reproduction checklist

Completed July 23, 2026; documentation refreshed September 24.

- [x] Build and verify Python 3.9 / CUDA 11.3 / MinkowskiEngine environment.
- [x] Validate released SUN RGB-D 40-class metadata and point-cloud paths.
- [x] Download and verify all 18 released SUN RGB-D checkpoints.
- [x] Evaluate all stages of the 3-, 5- and 10-stage protocols.
- [x] Confirm final mAP@0.25: .2935, .2503 and .1938, respectively.
- [x] Record the all-40 versus seen-class averaging distinction at earlier stages.
- [x] Publish results, scripts and environment notes.

Full results: `FINDINGS.md` and `logs/SWEEP_REPORT.txt`.
Follow-up work: `../09_object_level_memory/` and `../10_object_memory_pseudo/`.
ScanNet evaluation and full independent training reproduction remain outside
these completed experiments.
