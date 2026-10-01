# Week 4 research briefing

Experimental phase closed. Six full ten-stage runs passed the completion audit. Four matched stage-2 diagnostics also passed.

Reducing random object memory from 100 to 20 crops per class reduces the final bank from 3,979 to 800 objects (79.89%). Mean final mAP@0.25 falls from 15.6815% to 14.7520%, a loss of 0.9295 percentage points. This does not meet the provisional 0.5-point retention tolerance. Two seeds show a storage–accuracy tradeoff; they do not establish statistical equivalence.

## Selection result

In the bounded stage-2 comparison, largest-point-count selection changes AP25 by +0.870 and +1.080 pp for seeds 200 and 201 (mean +0.9750 pp). Each pair shares the exact same saved stage-1 checkpoint and resets the continuation seed. The comparison covers eight seen classes and uses 80 replay objects; it does not provide a final ten-stage selection result. Pseudo-label boxes and scores also differ within each pair, so the gain cannot be attributed solely to crop selection. Full details and old/new AP are in [the recovery report](../../WEEK_4_SELECTION_RECOVERY.md).

The dense 800-object bank uses 251.70 MiB, 3.60 times random B=20. The criterion changes point/byte costs even at the same object count.

## Scope and conclusion

Six full runs cover B=100/50/20 and two independent full-training seeds. The full selector pair failed at stage-1 population; repaired partial continuations remain separate from final ten-stage results. No verified reference ten-stage object score was identified. The study establishes a tradeoff, not equivalence.

Present the protocol, final accuracy, storage costs, seed/class variation and selection limits in that order. The slide PDF follows this sequence.

[Full weekly writeup](../../WEEK_4_WRAP_UP.md) · [Slides](WEEK_4_RESEARCH_SLIDES.pdf) · [Speaker notes](WEEK_4_SPEAKER_NOTES.md)
