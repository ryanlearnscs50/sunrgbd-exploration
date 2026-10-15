# Current implementation validation

October 15, 2026. Scope: LDMR scene memory and the TR3D_OBJ-style object-memory
extension, with a UniDet3D adapter for the original ScanNet-35 S5 protocol.

| Check | Evidence |
|---|---|
| Scene selection | Ten real stage selections match pristine LDMR |
| Online reviewing | All 32 audited weight decisions match pristine LDMR |
| SUN learning-dynamics GT accounting | All 5,285 training scenes and 24,377 annotations checked; native CUDA overlap checks agree on zero-height cases |
| SUN crop extraction | Every valid annotation matches reference crop membership; all 3,979 retained final-bank crops checked |
| ScanNet rich crops | All 700 stage-1 crops match reference point selection and semantic/instance/superpoint rows |
| Five-stage ScanNet object banks | 700 / 1,400 / 2,100 / 2,800 / 3,245 retained crops |
| Five-stage SUN object banks | 800 / 1,600 / 2,400 / 3,200 / 3,979 retained crops |
| Real native training batch | Eight scenes, 737,034 points, 86 pseudo boxes, 22 inserted objects; finite loss and 223 finite gradient tensors, including nonzero backbone/classifier gradients |
| Regression suite after implementation changes | 611 passed, 16 dependency warnings |

The crop budget is a cap. Classes with fewer eligible objects retain the available
objects without duplicating stored entries. Zero-volume crops are excluded while
the original scene-level GT counting semantics are preserved.

The [portable evidence summary](results/current_validation.json) records the
source revisions and checks. Release checks and source hashes are recorded at
the Week 5 package root.

## Remaining acceptance work

Full-sequence detection accuracy for the current named recipes is not yet measured.
That is the principal remaining acceptance gate. Original-detector baseline
training and UniDet3D transfer must be assessed separately, with their declared
initialization, supervision, optimization and exposure.

Exact within-stage optimizer/sampler/RNG recovery, distributed training/evaluation,
and a fresh-machine installation are not validated. Completed-stage continuation
is supported under the integrity checks described in the adapter guide.
UniDet3D full-method support is ScanNet-35 S5; SUN and S3/S10 host training are
outside the validated scope. Test counts do not establish defect-free software.
