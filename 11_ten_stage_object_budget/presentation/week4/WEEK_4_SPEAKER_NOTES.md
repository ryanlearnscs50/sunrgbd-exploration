# Week 4 presentation notes

Suggested delivery: approximately 8 minutes including time to discuss the figures.

## Slide 1 Object memory budget study

This week I tested how much stored object memory we can remove in ten-stage
incremental 3D detection, and what accuracy we give up. The main result is that
reducing the bank from roughly four thousand to eight hundred objects saves
about eighty percent of stored objects. Final AP at IoU 0.25 decreases from
15.68 to 14.75 percent on average: a loss of 0.93 percentage points.

That is a substantial storage saving, but it does not meet the provisional
half-point accuracy-loss tolerance. I will separate that completed budget
study from a shorter experiment on how the objects are selected.

## Slide 2 Protocol and comparison scope

The experiment uses forty SUN RGB-D classes, introduced four at a time over
ten stages. I compared caps of one hundred, fifty and twenty crops per class.
Each condition has two complete training runs, including independently trained
first stages. The schedule, pseudo supervision and configured insertion dose
stay fixed across the budget sweep.

This is different from last week's five-stage work, which changed how often
replay was attempted while leaving the bank capacity fixed. The reference
repositories establish the object budget and ten-stage split, but I could not
identify a verified ten-stage object-memory score to reproduce. The appropriate
claim here is a local ten-stage extension and controlled budget comparison.

## Slide 3 Final accuracy across budgets

The crosses show individual seeds, and the error bars show the sample standard
deviation over two runs. They are not confidence intervals. Fifty and twenty
objects per class give almost the same mean AP25, so these results do not
establish a smooth accuracy-versus-budget curve.

For twenty objects per class, the paired losses are 1.754 points in one seed
and 0.105 in the other. That variation matters: a one-point tolerance would
accept the mean but would not accept both seeds. At the original half-point
tolerance, neither reduced budget passes even on the mean. Two seeds are
useful evidence of variability, but insufficient for a claim of equivalence.

## Slide 4 Stage trajectory and forgetting

Each point on the stage curve evaluates a growing set of seen classes.
The class mix changes as harder or rarer classes arrive. A decrease in this
curve therefore cannot, by itself, be called forgetting.

The separate forgetting measure follows individual classes: it takes each
class's best earlier AP minus its final AP, then averages over the first
thirty-six classes. It is signed, so improvement beyond an earlier peak is
negative forgetting. Final old-class and new-class AP are also reported
separately, over thirty-six and four classes. These measurements help distinguish
retention from learning the final cohort.

## Slide 5 Object count and byte cost

The random twenty-object banks occupy about seventy MiB, compared with about
three hundred and twenty-eight MiB for the baseline. Training time changes
little, from around 6.8 to 6.7 hours under the fixed schedule. The main measured
benefit is bank storage.

The alternative selector chooses crops with the most points. Its offline
forty-class bank still has eight hundred objects, but occupies about two hundred
and fifty-two MiB: 3.6 times the random small bank. Against the baseline, that
is only about twenty-three percent fewer bytes despite eighty percent fewer
objects. Object count, point count and bytes answer different resource questions.
The offline dense bank has no associated full ten-stage detector score.

## Slide 6 Class effects and seed variability

The class-level view shows where the aggregate difference comes from. The
recycle-bin class loses about nineteen AP points on average. Since final mAP
equally weights forty classes, that contributes about 0.48 points to the overall
0.93-point loss: just over half of it.

Eleven classes lose in both seeds and seven gain in both. That suggests useful
classes to inspect next, but it does not establish why they change. For example,
this evidence does not yet show that a class needs more stored objects, or that
its selected crops are worse. Those remain hypotheses for a later experiment.

## Slide 7 Selection repair and short diagnostic

The original full selector runs failed after first-stage training because the
dataset population wrapper rejected the selector. I repaired that path and
checked its output against exhaustive ranking for all forty classes. The
population regression tests and existing focused tests pass.

The follow-up compares only stage two, using the same first-stage checkpoint
within each seed pair. Largest-point-count selection is associated with gains
of 0.87 and 1.08 AP25 points. However, a content audit also found differences
in the regenerated pseudo labels. I therefore cannot attribute the gain solely
to selection. It is an encouraging early-stage observation, not a replacement
for the missing full ten-stage comparison.

## Slide 8 Conclusions and next experiment

The completed result is a measurable storage–accuracy tradeoff: about eighty
percent fewer objects at a mean cost of 0.93 AP25 points. Retention within half
a point has not been demonstrated, and two seeds do not establish equivalence.

My next experimental priority would be a complete ten-stage selector comparison
that reuses the same cached pseudo labels within each pair. A third matched
full-training seed would then strengthen the variability estimate. The acceptable
accuracy-loss tolerance should be fixed before making a retention claim. For
this week, the completed results, failed attempts and remaining gaps are all
recorded, and no additional training is queued.
