# Week 5: reusable scene and object memory codebase

One maintained implementation now supports LDMR scene memory and a TR3D_OBJ-style
object-memory extension, with TR3D and a UniDet3D adapter. The work focuses on
reusable training entry points, shared policies, reference fidelity, data integrity,
and reproducible validation.

## Start here

- [Codebase and setup](codebase/README.md)
- [Four detector/memory paths and commands](codebase/docs/reference_methods.md)
- [Researcher handoff](codebase/docs/researcher_handoff.md)
- [Current validation evidence](codebase/docs/unidet3d_validation.md)
- [Source-to-port decisions](codebase/adapters/ldmr_unidet3d/REFERENCE_DECISIONS.md)
- [Release checks](RELEASE_CHECKS.json) and [file hashes](SOURCE_MANIFEST.json)

## Delivered this week

| Area | Delivered |
|---|---|
| Shared codebase | Scene/object mode selection, named policies and common bank implementations |
| LDMR scene path | Frozen-teacher pseudo supervision, Design-2 selection and online reviewing |
| Object extension | Self-contained crops, bounded replacement, reference replay settings and geometry |
| UniDet3D integration | Original ScanNet-35 S5 vocabulary/stages, rich crops, explicit box targets and classifier expansion |
| Data and state integrity | Point/mask alignment, checkpoint validation, source manifests and completed-stage continuation |
| Reuse | Separate environment guides, command examples, regression suites and source notices |

## What is validated

Ten scene selections and 32 reviewing decisions match pristine LDMR. Real-data
crop extraction and complete five-stage bank population have been checked on
SUN RGB-D and ScanNet. A real eight-scene UniDet3D batch with pseudo and object
supervision passes native CUDA loss/backward with finite, nonzero gradients.
The implementation regression suite passed 611 tests after the latest changes.

## Remaining validation

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
