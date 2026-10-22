# Experiment protocols and validation

## Class and dataset increments

Class increments reveal new categories within a fixed dataset: SUN40 or ScanNet35. Dataset increments move between six native datasets in the fixed order ScanNet, ARKitScenes, S3DIS, MultiScan, 3RScan and ScanNet++. Dataset ScanNet uses 18 native classes. Repeated category names share exact-name union identities; synonyms are not silently merged. The union vocabulary is known from initialization, but future-domain training samples are unavailable.

## Three method paths

Fine-tuning trains on the current domain and transfers the preceding detector while starting a fresh optimizer. Evaluation covers all domains seen so far.

Pseudo labeling freezes the preceding detector as teacher. It predicts on current-domain geometry and retains previously learned names absent from the current native annotations. Original LDMR confidence, NMS, cap and visible-GT filtering govern selection. Pseudo boxes merge before affine augmentation; teacher inputs exclude GT boxes and semantic or instance masks. Matched UniDet3D fine-tuning and pseudo paths use explicit box targets with elastic warps disabled.

LDMR adds saved native scene replay, periodic learning-dynamics measurements, original Design-2 bank updates and online reviewing. Replay retains the saved domain's own labels and geometry. Reviewing changes main-process sampler membership at defined boundaries so persistent data-loader workers see the update. The optimizer continues through review boundaries within a stage.

## Schedules and transfer choices

| Task | Schedule | Batch | Seed | Scene bank |
| --- | --- | --- | --- | --- |
| Completed TR3D SUN40 S5 | Original SUN stage exposure 5/2/2/1/1 | Original recipe | 200 | Original recipe |
| Completed UniDet3D ScanNet35 S5 | 180 passes per stage | 8 | 201 | 120 |
| Dataset UniDet3D FT/pseudo/LDMR | 180 passes per stage | 8 | 200 | 120 for LDMR |
| Dataset TR3D FT/pseudo/LDMR | 12 epochs × 15 repeats per stage | 4 | 200 | 120 for LDMR |
| Queued UniDet3D SUN40 S5 comparisons | 180 passes per stage | 8 | 201 | 528 for LDMR |

The dataset paths retain the original detector families and training frameworks. Their domain routing, union labels, replay geometry and sequential schedules are new experimental choices. They do not reproduce native six-dataset joint training. TR3D's review boundaries round to real epochs [0,2,4,7,9,12], whereas native UniDet3D uses [0,36,72,108,144,180]. Equal passes do not imply equal optimizer steps across backbones.

TR3D batch 16 failed with CUDA out-of-memory on ARKit. A fresh full sequence at batch 4 is the current recovery; batch statistics and optimizer-step counts change. The failed attempt is preserved locally. TR3D LDMR retains two empty-GT ScanNet scenes as negative examples; the earlier FT/pseudo loader could refetch another scene at these indices. That difference limits an exact causal comparison.

## Validation completed

The completed UniDet3D S5 audit covers all five 180-pass stages, finite saved model and optimizer states, exact learned-weight transfer and actual loaded initialization at subsequent stages. Full validation histories were checked for class coverage, finite values and agreement between class AP and their mean. TR3D SUN final metrics come from re-evaluation of the freshly trained final checkpoint.

All six prepared dataset inputs were audited. Native replay CPU pipelines passed across all five domain transitions on 32 transformed samples, including the five S3DIS training areas. Teacher and pseudo-supervision diagnostics include actual TR3D six-stage execution and separate positive-target backward checks. Small diagnostic AP is excluded from benchmark results.

The SUN host CPU preflight found and fixed two integration faults: serialized alignment needed conversion to an array, and validation attempted to read training-only annotations. The corrected full-data input/model preflight passed. SUN host GPU training remains queued, so CPU acceptance is not full detector acceptance.

## Result acceptance

A completed process alone cannot produce a benchmark entry. The workflow requires complete expected stages, checkpoint and optimizer checks, actual transfer verification, full validation coverage, and a sealed result tied to source and input identities. Dataset results additionally require evaluation of all six domains from the final stage. The average is the unweighted mean of six available domain scores; missing domains leave the average unavailable.

The completed class scores, intermediate dataset observations, diagnostic execution checks and queued experiments are separate evidence categories. Single-seed results do not establish statistical superiority, and component checks do not establish an error-free implementation.

## Remaining evidence

Full six-domain FT, pseudo and LDMR outcomes are pending on both backbones. ScanNet35 S3/S10 and joint, and SUN40 S5 host comparisons are queued. Additional seeds and broader class-baseline coverage remain future work. The next decisive comparison is retention after matched complete sequences, together with incoming-domain learning.
