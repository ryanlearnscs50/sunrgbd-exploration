# Reusable scene and object memory codebase

This implementation supports LDMR scene replay and a TR3D_OBJ-style object-memory
extension in one maintained codebase. The TR3D runtime and UniDet3D runtime use
separate environments and share memory-bank policies and the original class maps.

Start with the [method guide](reference_methods.md), which defines the four
detector/memory paths, named recipes, training exposure and explicit adaptations.
The [validation report](unidet3d_validation.md) describes the evidence and limits.

## Setup and execution

1. Install the [legacy environment](legacy_install.md); install the separate
   [UniDet3D environment](../adapters/environment/README.md) for host experiments.
2. Prepare data following the [memory guide](memory_banks.md) and
   [adapter guide](../adapters/ldmr_unidet3d/README.md). Use the original
   SUN40/ScanNet35 label maps; host superpoints must align with every point row.
3. Run a named method from [reference_methods.md](reference_methods.md) into a
   new work directory. Archive the source, resolved configuration and environment.
4. Validate checkpoints, metrics, class coverage, bank provenance and reviewing
   receipts. Preserve all artifacts needed for completed-stage continuation.

The legacy suite runs from the codebase directory:

```bash
OMP_NUM_THREADS=2 UNIDET3D_ROOT=/path/to/UniDet3D python -m pytest tests -q
```

Run host tests from outside that directory to avoid importing legacy MMDetection3D:

```bash
cd /tmp
UNIDET3D_ROOT=/path/to/UniDet3D \
PYTHONPATH=/path/to/codebase/adapters:/path/to/UniDet3D \
OMP_NUM_THREADS=2 /path/to/host/bin/python -m pytest /path/to/codebase/adapters/tests -q
```

Source pins: LDMR `ab67f3d7a3e6e98c6f66182832bb37e8656cceaa`,
TR3D_OBJ `cd667a180c3cbf4eae29cf779ea907592400266f`,
UniDet3D `940a730a09711b0bf266fd972504da29a83b91f6`.

The implementation has substantive source and real-data validation. Full-schedule
accuracy remains an open acceptance gate; no detector-performance ranking is
claimed. Exact within-stage resume, distributed execution and fresh-machine
installation remain unverified.
