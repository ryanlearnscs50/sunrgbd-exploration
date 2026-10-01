"""Close Week 4 from measured artifacts and export a research report and slide PDF."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics as st
import subprocess
import sys
import textwrap
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'week4_runs/selection_recovery'
PRESENT = ROOT / 'presentation/week4'


def atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def finalize(preview=False):
    from week4_selection_recovery_report import report
    diagnostic = json.loads((STATE / 'audit.json').read_text()) if preview else report()
    subprocess.run([sys.executable, str(ROOT / 'week4_analyze_results.py')], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / 'week4_explanatory_analysis.py')], check=True, cwd=ROOT)
    analysis = json.loads((ROOT / 'week4_runs/analysis.json').read_text())
    extra = json.loads((PRESENT / 'week4_explanatory_summary.json').read_text())
    groups = {b: [r for r in analysis['runs'] if r['budget'] == b] for b in (100, 50, 20)}
    assert all(len(v) == 2 for v in groups.values())
    complete = diagnostic['audited_complete']
    stamp = datetime.now(timezone.utc).isoformat()
    status = ('Preview while bounded diagnostics are running.' if preview else
              'Experimental phase closed. Six full ten-stage runs passed the completion audit. '
              + ('Four matched stage-2 diagnostics also passed.' if complete else
                 'The bounded selection diagnostic is incomplete; only audited results are reported.'))
    main_result = ('Reducing random object memory from 100 to 20 crops per class reduces the final bank '
                   'from 3,979 to 800 objects (79.89%). Mean final mAP@0.25 falls from 15.6815% to '
                   '14.7520%, a loss of 0.9295 percentage points. This does not meet the provisional '
                   '0.5-point retention tolerance. Two seeds show a storage–accuracy tradeoff; '
                   'they do not establish statistical equivalence.')
    criterion = ('Both full ten-stage largest-point-count runs failed after completing stage-1 training. '
                 'Their bank implementation supported the selector, but the SUN RGB-D population wrapper '
                 'rejected it. The preflight tested the isolated ranking function and missed the actual '
                 'population path. The repaired path scans all eligible crops and retains a bounded top-20 '
                 'set with stable ties. Four new population regression cases and 22 prior checks passed. '
                 'An independent exhaustive ranking confirmed the selected identities and point counts '
                 'for all 40 classes. Failed run evidence is retained.')
    if complete:
        effects = [p['delta25_pp'] for p in diagnostic['paired_deltas']]
        short_result = (f'In the bounded stage-2 comparison, largest-point-count selection changes AP25 '
                        f'by {effects[0]:+.3f} and {effects[1]:+.3f} pp for seeds 200 and 201 '
                        f'(mean {st.mean(effects):+.4f} pp). Each pair shares the exact same saved '
                        'stage-1 checkpoint and resets the continuation seed. The comparison covers '
                        'eight seen classes and uses 80 replay objects; it does not provide a final '
                        'ten-stage selection result. Pseudo-label boxes and scores also differ within each pair, '
                        'so the gain cannot be attributed solely to crop selection. Full details and old/new AP are in '
                        '[the recovery report](WEEK_4_SELECTION_RECOVERY.md).')
    else:
        short_result = ('The bounded stage-2 comparison is not yet fully audited. Partial results and '
                        'terminal statuses are in [the recovery report](WEEK_4_SELECTION_RECOVERY.md). '
                        'No final ten-stage selection effect can be claimed.')
    lines = ['# Week 4 ten stage object memory study', '', status, '', f'Evidence refresh: {stamp}.', '',
             main_result, '', '## Question and experimental design', '',
             'This study extends the earlier five-stage work to ten stages, uses the reference '
             'object budget, reduces the number of stored objects, repeats across two seeds, '
             'and tests whether another selection criterion makes a smaller bank useful.', '',
             'The completed sweep uses SUN RGB-D with 40 frequency-ordered classes introduced four '
             'at a time. For each budget (100, 50 and 20 objects per class), seeds 200 and 201 train '
             'from scratch through all ten stages. Stage 1 uses six epochs and later stages one epoch '
             'each, with dataset repeat 15, batch size 16 and AdamW LR 0.001. LR falls to 0.0001 for '
             'the sixth initial epoch and remains 0.001 in later stages. Random selection, pseudo '
             'supervision, insertion probability 0.7, up to three candidate pastes and all placement '
             'settings stay fixed. Only the per-class and total object caps change.', '',
             'Week 3 reduced the probability of replay while leaving stored capacity fixed. Week 4 '
             'changes stored capacity while holding the configured replay dose fixed. Accepted paste '
             'counts were not logged, so equal configured dose is not a measurement of equal accepted pastes.', '',
             '## Accuracy and stored object budget', '',
             'AP is reported as a percentage; losses are percentage points. SD is the sample standard '
             'deviation of two full runs, not a confidence interval.', '',
             '| Objects per class | Actual final objects | Final AP25 mean ± SD | Final AP50 mean ± SD | Mean AP25 change | Bank MiB mean |',
             '|---|---|---|---|---|---|']
    for b, pair in groups.items():
        ap25 = [100*r['final_map25'] for r in pair]
        ap50 = [100*r['final_map50'] for r in pair]
        lines.append(f'| {b} | {pair[0]["final_objects"]:,} | {st.mean(ap25):.4f} ± {st.stdev(ap25):.3f} | '
                     f'{st.mean(ap50):.4f} ± {st.stdev(ap50):.3f} | '
                     f'{st.mean(r["delta_vs_baseline25_pp"] for r in pair):+.4f} | '
                     f'{st.mean(r["resources"][-1]["pickle_bytes"] for r in pair)/2**20:.2f} |')
    lines += ['', 'The B=20 paired changes are −1.754 and −0.105 pp. B=50 changes are −2.318 '
              'and +0.440 pp. B=50 and B=20 have nearly the same mean, and the seed ordering reverses '
              'at B=50. These observations do not support a precise monotonic budget curve.', '',
              'At the provisional 0.5 pp loss tolerance, neither reduced budget passes even on the '
              'mean. At 1 pp, both means pass but neither passes in both seeds. At 2 pp, B=20 passes '
              'in both seeds. These are descriptive sensitivity checks, not a revised acceptance '
              'criterion or an equivalence test. No budget smaller than B=100 has demonstrated '
              'retention within 0.5 pp in this study.', '',
              '## What storage and accuracy measurements mean', '',
              'The 3,979-object final baseline falls below its nominal 4,000 cap because mirror, '
              'laptop and towel provide fewer eligible crops. The final bank is written after adding '
              'the last cohort; stage-10 training instead uses the stage-9 bank: 3,600 old objects '
              'for B=100 and 720 for B=20. These are separate quantities.', '',
              'Random B=20 occupies roughly 69–71 MiB versus 327–329 MiB for B=100, but total run '
              'time only changes from about 6.8 to 6.7 hours. The measured benefit is mainly bank '
              'storage under the fixed training schedule. Runtime includes evaluation, pseudo-label '
              'generation and bank construction; it is not a controlled throughput benchmark.', '',
              'Final AP25 gives equal weight to each of the 40 classes. recycle_bin loses '
              '19.1705 class AP points on average at B=20, contributing 0.4793 pp to the overall '
              '0.9295 pp loss. Eleven classes lose in both seeds and seven gain in both. This '
              'decomposition identifies where the aggregate change occurs; it does not identify its cause.', '',
              'The stage-average metric averages seen-class mAP across ten different class scopes. '
              'A falling stage curve alone is not a forgetting estimate. The separate forgetting '
              'measure averages prior-peak-minus-final AP for the first 36 classes; an improvement '
              'beyond the prior peak contributes a negative value. See the full analysis for both '
              'metrics and final old/new AP.', '',
              '## Selection criterion and recovery', '', criterion, '', short_result, '',
              f'The offline 40-class largest-point-count bank contains 800 crops, 10,990,248 points '
              f'and {extra["offline_bank_mib"]:.2f} MiB. It uses '
              f'{extra["offline_byte_ratio_vs_random20"]:.2f} times the mean bytes of random B=20. '
              f'Against random B=100 it saves 79.89% of objects but only '
              f'{extra["offline_byte_reduction_vs_b100_percent"]:.2f}% of bytes. This is bank-construction '
              'evidence, with no full ten-stage detector score. More points are not by themselves '
              'proof of better semantic quality or class representativeness.', '',
              '## Relation to the reference work', '',
              'The retained reference audit establishes the nominal 100-object-per-class budget '
              'and the ten-stage class split, but does not identify a verified ten-stage object-memory '
              'score to reproduce. The separate 19.38% ten-stage scene-memory result has a different '
              'protocol and budget unit. Local box-origin/collision fixes, uniform subset selection, '
              'object sampling without replacement and the absence of extra crop yaw remain '
              'implementation differences. This is a local ten-stage extension and controlled '
              'object-budget study, not a confirmed numerical reproduction.', '',
              'Reference commits: TR3D_OBJ `cd667a180c3cbf4eae29cf779ea907592400266f`; '
              'LDMR_backup `1c58607f04b86f1b66233b39a07b631f624f201c`. The local '
              '[protocol audit](WEEK_4_PROTOCOL_AUDIT.md) and '
              '[training plan](WEEK_4_TRAINING_PLAN.md) retain the source evidence.', '',
              '## Conclusion and next experiment', '',
              'Random B=20 is a substantial storage reduction with a measured mean cost of about '
              '0.93 pp AP25. It is a candidate when that loss is acceptable, not a demonstrated '
              'same-performance replacement under the provisional 0.5 pp criterion. Agree on an '
              'acceptable loss before selecting a deployment budget. A third matched full-training '
              'seed and a completed ten-stage selector comparison would strengthen the evidence. '
              'Targeting classes with consistent losses is a hypothesis for a later study, not a '
              'validated budget-allocation rule. No additional full runs are queued.', '',
              '## Evidence and reproducibility', '',
              '- [Full numerical analysis](WEEK_4_ANALYSIS.md) and [budget comparison](WEEK_4_BUDGET_COMPARISON.md).',
              '- [Selection repair and bounded results](WEEK_4_SELECTION_RECOVERY.md).',
              '- [Storage and class diagnostics](WEEK_4_EXPLANATORY_DIAGNOSTICS.md).',
              '- [Research slide PDF](presentation/week4/WEEK_4_RESEARCH_SLIDES.pdf) and [speaker notes](presentation/week4/WEEK_4_SPEAKER_NOTES.md).',
              '- Machine-readable full-run results: `week4_runs/analysis.json`; bounded results: `week4_runs/selection_recovery/audit.json`.',
              '- Full-run completion audits check exits, class scopes and metric means, bank caps/seed/selector, finite training losses, LR, and nonempty checkpoints/pseudo artifacts. Model payloads are not deserialized by that audit.', '']
    report_path = ROOT / ('WEEK_4_WRAP_UP_PREVIEW.md' if preview else 'WEEK_4_WRAP_UP.md')
    atomic(report_path, '\n'.join(lines))

    make_slides(groups, diagnostic, extra, preview)
    if not preview:
        brief = ['# Week 4 research briefing', '', status, '', main_result, '',
                 '## Selection result', '', short_result.replace('(WEEK_4_SELECTION_RECOVERY.md)', '(../../WEEK_4_SELECTION_RECOVERY.md)'), '',
                 f'The dense 800-object bank uses {extra["offline_bank_mib"]:.2f} MiB, '
                 f'{extra["offline_byte_ratio_vs_random20"]:.2f} times random B=20. '
                 'The criterion changes point/byte costs even at the same object count.', '',
                 '## Scope and conclusion', '',
                 'Six full runs cover B=100/50/20 and two independent full-training seeds. '
                 'The full selector pair failed at stage-1 population; repaired partial continuations '
                 'remain separate from final ten-stage results. No verified reference ten-stage '
                 'object score was identified. The study establishes a tradeoff, not equivalence.', '',
                 'Present the protocol, final accuracy, storage costs, seed/class variation and '
                 'selection limits in that order. The slide PDF follows this sequence.', '',
                 '[Full weekly writeup](../../WEEK_4_WRAP_UP.md) · [Slides](WEEK_4_RESEARCH_SLIDES.pdf) · '
                 '[Speaker notes](WEEK_4_SPEAKER_NOTES.md)', '']
        atomic(PRESENT / 'WEEK_4_RESEARCH_BRIEFING.md', '\n'.join(brief))
        atomic(STATE / 'finalization.json', json.dumps(dict(status='complete', at=stamp,
               full_runs=6, bounded_diagnostic_complete=complete, experimental_phase_closed=True), indent=2)+'\n')
        with (ROOT / 'WEEK_4_MEMORY.md').open('a') as out:
            out.write(f'\n- Weekly finalization at {stamp}: six full runs audited; bounded stage-2 diagnostic '
                      f'complete={complete}. Wrote WEEK_4_WRAP_UP.md, research briefing, speaker notes and '
                      'WEEK_4_RESEARCH_SLIDES.pdf. Experimental phase closed; no further training queued. '
                      'Full ten-stage selector comparison remains missing after the overnight failure.\n')
    print('WEEK4_FINALIZATION', 'preview' if preview else 'complete', str(report_path), flush=True)


def make_slides(groups, diagnostic, extra, preview):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    slides = [
        ('Week 4 object memory budget study', None,
         ['Ten stages · 40 SUN RGB-D classes · two full-training seeds',
          '3,979 → 800 stored objects: 79.89% fewer',
          'Final AP25: 15.6815% → 14.7520% (−0.9295 pp)',
          'The smaller bank does not meet the provisional 0.5 pp loss tolerance.'],
         'Open with the measured tradeoff. Say percentage points, not percent, for the AP change. The object reduction uses actual final occupancy.'),
        ('Protocol and comparison scope', None,
         ['Ten frequency-ordered stages, four new classes per stage.',
          'B=100, 50 and 20; seeds 200 and 201 train from scratch.',
          'Fixed schedule, pseudo supervision and configured insertion dose.',
          'Week 3 changed replay probability; Week 4 changes stored capacity.',
          'Reference budget and split matched; no verified S10 object score.'],
         'Explain that each budget condition contains two complete independent training runs. A scene entry is not an object crop. Refer to the protocol audit for remaining implementation differences.'),
        ('Final accuracy across budgets', 'week4_final_accuracy.png',
         ['B=20 paired AP25 changes: −1.754 and −0.105 pp.',
          'Neither reduced budget passes the provisional 0.5 pp mean tolerance.'],
         'Crosses show seeds and error bars show sample SD over only two runs. They are not confidence intervals. B=50 and B=20 means are similar; do not claim a precise monotonic budget curve.'),
        ('Stage trajectory and forgetting', 'week4_stage_accuracy.png',
         ['Seen-class scope grows at each stage.', 'Raw mAP decline alone is not a forgetting measure.'],
         'Stage-average AP averages ten different class scopes. The separate signed forgetting metric is prior peak minus final AP over the first 36 classes. Final old/new AP uses 36 and 4 classes.'),
        ('Equal object counts have different byte costs', 'week4_bank_costs.png',
         [f'Dense B=20 bank: {extra["offline_bank_mib"]:.2f} MiB; {extra["offline_byte_ratio_vs_random20"]:.2f}× random B=20.',
          'Dense bank is offline GT construction; no full S10 detector score.'],
         'Object count is the primary budget measure. Random B=20 saves about 79% of bytes, but dense B=20 saves only 23.25% versus B=100. The offline bank ranking was checked exhaustively for all 40 classes.'),
        ('Class effects and seed variability', 'week4_cohort_and_seed_effects.png',
         ['recycle_bin contributes −0.4793 pp to the total −0.9295 pp AP25 change.',
          'Descriptive decomposition; the cause of the loss is not established.'],
         'The scatter compares per-class B=20 minus B=100 effects in the two seeds. Eleven classes lose in both and seven gain in both. Class-level changes can be much larger than the overall mean.'),
    ]
    if diagnostic['audited_complete']:
        effects = [p['delta25_pp'] for p in diagnostic['paired_deltas']]
        recovery_lines = ['Full selector runs failed at stage-1 bank population; integration path repaired.',
                          'Same stage-1 checkpoint within each pair; stage 2 only, eight seen classes.',
                          f'Largest minus random AP25: {effects[0]:+.3f}, {effects[1]:+.3f} pp.',
                          f'Mean early-stage change: {st.mean(effects):+.4f} pp.',
                          'Pseudo labels differ too; the gain is not isolated to the selector.',
                          'This does not establish final ten-stage performance.']
    else:
        recovery_lines = ['Full selector runs failed at stage-1 bank population; integration path repaired.',
                          '26 focused tests passed; all-class offline crop ranking verified.',
                          'Bounded stage-2 diagnostic is not yet fully audited.',
                          'No final ten-stage selector result is available.']
    slides += [
        ('Selection repair and bounded diagnostic', None, recovery_lines,
         'Be explicit about the overnight failure and preflight gap. The partial continuation is a separate matched diagnostic, not a completed S10 replication. Original failure logs and checkpoint lineage are preserved.'),
        ('What we can conclude', None,
         ['Random B=20 gives large storage savings at about 0.93 pp mean AP25 cost.',
          'Retention within 0.5 pp has not been demonstrated.',
          'Two seeds do not establish statistical equivalence.',
          'Next: finish S10 selection with shared pseudo labels; add a full seed.',
          'This week closes with measured results and the missing comparison stated.'],
         'Prioritize a complete ten-stage selector comparison with shared cached pseudo labels, then a third matched seed; specify the acceptable AP loss before evaluating retention. No future runs are queued. Do not claim an exact Peisheng numerical reproduction.'),
    ]
    target = PRESENT / ('WEEK_4_RESEARCH_SLIDES_PREVIEW.pdf' if preview else 'WEEK_4_RESEARCH_SLIDES.pdf')
    notes = ['# Week 4 speaker notes', '', 'Suggested duration: 8–10 minutes.', '']
    with PdfPages(target) as pdf:
        for i, (title, figure, bullets, note) in enumerate(slides, 1):
            fig = plt.figure(figsize=(13.333, 7.5), facecolor='white')
            fig.text(.045, .935, title, fontsize=25, weight='bold', color='#18344a')
            if figure:
                ax = fig.add_axes([.06, .20, .88, .65])
                ax.imshow(plt.imread(PRESENT / figure))
                ax.axis('off')
                for y, bullet in zip((.14, .095), bullets):
                    fig.text(.065, y, bullet, fontsize=13, color='#263238')
            else:
                y = .78
                for bullet in bullets:
                    wrapped = textwrap.fill(bullet, 82)
                    fig.text(.075, y, '• ' + wrapped, fontsize=20, color='#263238', va='top', linespacing=1.45)
                    y -= .115 + .045*wrapped.count('\n')
            fig.text(.045, .035, 'Week 4 · SUN RGB-D incremental object memory' + (' · PREVIEW' if preview else ''), fontsize=10, color='#617480')
            fig.text(.95, .035, str(i), fontsize=10, color='#617480', ha='right')
            pdf.savefig(fig)
            fig.savefig(PRESENT / f'week4_slide_{i:02d}{"_preview" if preview else ""}.png', dpi=100)
            plt.close(fig)
            notes += [f'## Slide {i} {title}', '', note, '']
    atomic(PRESENT / ('WEEK_4_SPEAKER_NOTES_PREVIEW.md' if preview else 'WEEK_4_SPEAKER_NOTES.md'), '\n'.join(notes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/ldmr-week4-mpl')
    if args.wait:
        while not ((STATE / 'completion.json').exists() and (STATE / 'report.exit_code').exists()):
            if (STATE / 'controller_error.json').exists():
                break
            time.sleep(30)
    if not args.preview:
        assert (STATE / 'completion.json').exists(), 'Training controller has not finished; use --preview'
    finalize(args.preview)
