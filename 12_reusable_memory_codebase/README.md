# Reusable scene and object memory codebase

One maintained implementation now supports LDMR scene memory and a TR3D_OBJ-style
object-memory extension, with TR3D and a UniDet3D adapter. The work focuses on
reusable training entry points, shared policies, reference fidelity, data integrity,
and reproducible validation.

## Ongoing codebase validation

The [Week 6 validation report](validation/week6/README.md) continues this same
reliability effort. Fresh TR3D SUN40 S5 training reaches 23.90 AP25 / 14.11 AP50;
full-schedule UniDet3D ScanNet35 S5 reaches 38.82 / 24.94. Six-domain fine-tuning
is in progress, with pseudo labeling and LDMR queued. The report separates
completed accuracy, intermediate observations and pending comparisons.

The `codebase/` directory and release manifests remain the Week 5 source snapshot.
The Week 6 report records later work in the maintained research checkout; it does
not imply that the archived source includes the newer dataset adapters. Sections
below describe the Week 5 release and its validation scope at that time.

## Start here

- [Codebase and setup](codebase/README.md)
- [Four detector/memory paths and commands](codebase/docs/reference_methods.md)
- [Researcher handoff](codebase/docs/researcher_handoff.md)
- [Week 6 reliability and full training results](validation/week6/README.md)
- [Week 5 implementation validation](codebase/docs/unidet3d_validation.md)
- [Source-to-port decisions](codebase/adapters/ldmr_unidet3d/REFERENCE_DECISIONS.md)
- [Release checks](RELEASE_CHECKS.json) and [file hashes](SOURCE_MANIFEST.json)

## Week 5 implementation release

| Area | Delivered |
|---|---|
| Shared codebase | Scene/object mode selection, named policies and common bank implementations |
| LDMR scene path | Frozen-teacher pseudo supervision, Design-2 selection and online reviewing |
| Object extension | Self-contained crops, bounded replacement, reference replay settings and geometry |
| UniDet3D integration | Original ScanNet-35 S5 vocabulary/stages, rich crops, explicit box targets and classifier expansion |
| Data and state integrity | Point/mask alignment, checkpoint validation, source manifests and completed-stage continuation |
| Efficiency | Bounded source-cloud caching, bounded object replacement, and early stopping of unused object-ranking work while preserving selection |
| Reuse | Separate environment guides, command examples, regression suites and source notices |

The bank lifecycle includes versioned state, atomic checkpoint replacement,
capacity and class-coverage checks, and explicit compatibility checks on restore.
Scene banks preserve annotation visibility; object banks retain self-contained
crops with source identity and coordinate-frame metadata. Stage transitions
preserve sparse-layer metadata and copy learned classifier rows by class name.

Training entry points validate settings and protect existing output directories.
The host sequence records source/data identities, validates train/validation
separation, manages child-process cleanup, and checks completed-stage artifacts
before continuing. Teacher boxes and pasted-object geometry reach the native
loss through explicit box targets. Evaluation supports seen-class scope and
machine-readable metrics, with checkpoint and metric auditing tools included.

## Week 5 validation snapshot

Ten scene selections and 32 reviewing decisions match pristine LDMR. Real-data
crop extraction and complete five-stage bank population have been checked on
SUN RGB-D and ScanNet. A real eight-scene UniDet3D batch with pseudo and object
supervision passes native CUDA loss/backward with finite, nonzero gradients.
The implementation regression suite passed 611 tests after the latest changes.

## Validation limits at the Week 5 release

Full-schedule accuracy of the current named recipes remains pending. The code
has substantive implementation validation; a fully validated training recipe
also requires completed accuracy evaluation. Exact within-stage resume,
distributed execution and fresh-machine installation remain unverified.
UniDet3D full-method support is ScanNet-35 S5; its optimizer and detector differ
from TR3D, and equal accuracy is not assumed.

## Source package

`codebase/` is a complete source snapshot, not a patch overlay. Run legacy commands
from that directory. Native binaries, datasets, model weights, environments and
machine-specific runtime artifacts are obtained or built separately. Upstream
license and attribution files are retained. See the manifest for file identities.
