"""Audited Week 4 analysis and exportable figures, including optional selection pair."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics as st

from summarize_week4_budgets import audit

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'week4_runs'
FIGURES = ROOT / 'figures/week4'


def atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text)
    tmp.replace(path)


def enrich(budget, seed, strategy='random'):
    run_id = f'object{budget}_s{seed}' if strategy == 'random' else f'object20_largest_s{seed}'
    r = audit(budget, seed, run_id, strategy)
    r.update(run_id=run_id, strategy=strategy,
             label=f'Random B={budget}' if strategy == 'random' else 'Most points B=20')
    state = STATE / run_id
    manifest = json.loads((state / 'manifest.json').read_text())
    stem = Path(manifest['work_dir_stem'])
    directory, = list(stem.parent.glob(stem.name + '_*'))
    r['work_dir'] = str(directory)
    scores = []
    resources = []
    peak_memory = 0
    for stage in range(1, 11):
        m = json.loads((directory / f'memory_bank/scores/stage_{stage}_metrics.json').read_text())
        scores.append({c['model_idx']: c for c in m['classes']})
        p = directory / f'object_memory_bank/object_memory_bank_stage_{stage}.json'
        bank = json.loads(p.read_text())
        objects = [x for bucket in bank['exemplars'].values() for x in bucket]
        assert all(x['point_count'] >= bank['config']['min_points'] for x in objects)
        resources.append(dict(stage=stage, objects=len(objects), points=sum(x['point_count'] for x in objects),
                              pickle_bytes=p.with_suffix('.pkl').stat().st_size,
                              unique_source_scenes=len({x['scene_id'] for x in objects})))
        if strategy == 'largest_point_count':
            for bucket in bank['exemplars'].values():
                # Selection sorts descending where support exceeds the cap.
                # When all candidates fit, order is intentionally unchanged.
                assert len(bucket) <= 20
        for path in (directory / f'checkpoints/stage_{stage}').glob('*.log.json'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get('mode') == 'train':
                    peak_memory = max(peak_memory, row.get('memory', 0))
    classes = []
    for i, last in sorted(scores[-1].items()):
        first_stage = i // 4 + 1
        record = dict(model_idx=i, name=last['name'], introduced_stage=first_stage)
        for suffix, key in [('25', 'AP_0.25'), ('50', 'AP_0.50')]:
            trajectory = [s[i][key] for s in scores[first_stage-1:]]
            record[f'final_ap{suffix}'] = trajectory[-1]
            record[f'introduction_ap{suffix}'] = trajectory[0]
            # Signed prior-peak minus final, excluding final cohort in aggregation.
            record[f'forgetting{suffix}'] = max(trajectory[:-1]) - trajectory[-1] if len(trajectory) > 1 else None
        classes.append(record)
    r['classes'] = classes
    r['resources'] = resources
    for suffix in ('25', '50'):
        r[f'final_old_map{suffix}'] = st.mean(c[f'final_ap{suffix}'] for c in classes[:36])
        r[f'final_new_map{suffix}'] = st.mean(c[f'final_ap{suffix}'] for c in classes[36:])
        r[f'forgetting{suffix}'] = st.mean(c[f'forgetting{suffix}'] for c in classes[:36])
        r[f'backward_transfer{suffix}'] = st.mean(c[f'final_ap{suffix}'] - c[f'introduction_ap{suffix}'] for c in classes[:36])
    begin = datetime.fromisoformat((state / 'started_at').read_text().strip())
    end = datetime.fromisoformat((state / 'ended_at').read_text().strip())
    r['duration_hours'] = (end-begin).total_seconds()/3600
    r['peak_logged_memory_mb'] = peak_memory
    return r


def plot(groups):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    colors = ['#2864a0', '#cd7b27', '#23856a', '#9456a2']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, suffix in zip(axes, ('25', '50')):
        for idx, (label, pair) in enumerate(groups.items()):
            ys = [r[f'final_map{suffix}']*100 for r in pair]
            ax.errorbar(idx, st.mean(ys), yerr=st.stdev(ys), fmt='o', color=colors[idx], capsize=5)
            ax.scatter([idx-.08, idx+.08], ys, color=colors[idx], marker='x', alpha=.8)
        ax.set_xticks(list(range(len(groups))))
        ax.set_xticklabels(list(groups), rotation=15, ha='right')
        ax.set_ylabel(f'Final mAP@.{suffix} (%)')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Ten stages, two full-training seeds | mean and sample SD; crosses are seeds')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES / f'week4_final_accuracy.{ext}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, suffix in zip(axes, ('25', '50')):
        for idx, (label, pair) in enumerate(groups.items()):
            ys = [[s[f'map{suffix}']*100 for s in r['stages']] for r in pair]
            ax.plot(range(1, 11), [st.mean(v) for v in zip(*ys)], label=label, color=colors[idx])
            for y in ys:
                ax.plot(range(1, 11), y, alpha=.22, linewidth=.8, color=colors[idx])
        ax.set_xlabel('Stage (4 additional classes per stage)')
        ax.set_ylabel(f'Seen-class mAP@.{suffix} (%)')
        ax.set_xticks(range(1, 11))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Mean stage trajectory; faint lines are individual seeds')
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIGURES / f'week4_stage_accuracy.{ext}', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-selection', action='store_true')
    args = parser.parse_args()
    runs = [enrich(b, s) for b in (100, 50, 20) for s in (200, 201)]
    exits = [STATE / f'object20_largest_s{s}/exit_code' for s in (200, 201)]
    selection_complete = all(p.exists() and p.read_text().strip() == '0' for p in exits)
    selection_failed = any(p.exists() and p.read_text().strip() != '0' for p in exits)
    selection_note = ('The full ten-stage point-count selection pair failed at stage-1 bank population '
                      'and is excluded from numerical conclusions. The population-path repair and '
                      'separate matched stage-2 diagnostics are documented in WEEK_4_SELECTION_RECOVERY.md; '
                      'they do not replace the missing ten-stage result.' if selection_failed else
                      'The point-count selection pair is pending and excluded from numerical conclusions.')
    if args.require_selection:
        assert selection_complete, 'Selection pair has not completed successfully'
    if selection_complete:
        runs += [enrich(20, s, 'largest_point_count') for s in (200, 201)]
    groups = {}
    for r in runs:
        groups.setdefault(r['label'], []).append(r)
    baseline = {r['seed']: r for r in groups['Random B=100']}
    random20 = {r['seed']: r for r in groups['Random B=20']}
    for r in runs:
        b = baseline[r['seed']]
        r['object_reduction_percent'] = 100*(1-r['final_objects']/b['final_objects'])
        for suffix in ('25', '50'):
            r[f'delta_vs_baseline{suffix}_pp'] = 100*(r[f'final_map{suffix}']-b[f'final_map{suffix}'])
            r[f'delta_vs_random20_{suffix}_pp'] = 100*(r[f'final_map{suffix}']-random20[r['seed']][f'final_map{suffix}'])
    report = dict(updated_at=datetime.now(timezone.utc).isoformat(), audited=True,
                  selection_complete=selection_complete, selection_failed=selection_failed, runs=runs)
    atomic(STATE / 'analysis.json', json.dumps(report, indent=2)+'\n')
    with (STATE / 'week4_per_class.csv').open('w', newline='') as out:
        fields = ['run_id', 'seed', 'budget', 'strategy'] + list(runs[0]['classes'][0])
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            for c in r['classes']:
                writer.writerow(dict({k: r[k] for k in fields[:4]}, **c))
    lines = ['# Week 4 object budget and selection analysis', '',
             f'Updated: {report["updated_at"]}. {len(runs)} completed runs passed the ten-stage artifact audit.', '',
             'The random-selection budget sweep reduces final storage from 3,979 to 800 objects '
             '(79.89%) with a mean final mAP@.25 loss of 0.9295 percentage points. '
             'This exceeds the provisional 0.5 pp tolerance. Two seeds show the tradeoff; '
             'they do not establish statistical equivalence.', '',
             ('The point-count selection pair is complete and included below.' if selection_complete else
              selection_note), '',
             '## Final accuracy and object count', '',
             'AP values are percentages; deltas are percentage points. Each condition has full-training seeds 200 and 201. '
             'Sample SD describes those two runs, not a confidence interval.', '',
             '| Condition | Final objects | Final AP25 mean ± SD | Final AP50 mean ± SD | Mean paired AP25 delta vs B100 | Worst seed AP25 delta |',
             '|---|---|---|---|---|---|']
    for label, pair in groups.items():
        def mean_sd(key):
            vals = [100*r[key] for r in pair]
            return f'{st.mean(vals):.3f} ± {st.stdev(vals):.3f}'
        deltas = [r['delta_vs_baseline25_pp'] for r in pair]
        lines.append(f'| {label} | {pair[0]["final_objects"]} | {mean_sd("final_map25")} | '
                     f'{mean_sd("final_map50")} | {st.mean(deltas):+.4f} | {min(deltas):+.3f} |')
    lines += ['', '## Paired seed details', '',
              '| Condition | Seed | Final AP25 | Delta vs B100 | Delta vs random B20 | Stage avg AP25 | Stage avg AP50 | Final old AP25 | Final new AP25 | Forgetting AP25 |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for r in runs:
        lines.append(f'| {r["label"]} | {r["seed"]} | {100*r["final_map25"]:.3f} | '
                     f'{r["delta_vs_baseline25_pp"]:+.3f} | {r["delta_vs_random20_25_pp"]:+.3f} | '
                     f'{100*r["stage_average_map25"]:.3f} | {100*r["stage_average_map50"]:.3f} | '
                     f'{100*r["final_old_map25"]:.3f} | {100*r["final_new_map25"]:.3f} | {100*r["forgetting25"]:.3f} |')
    lines += ['', 'Stage average is the unweighted mean of seen-class mAP over ten stages. '
              'Old/new means cover the first 36/final 4 classes. Forgetting is the mean, over the first '
              '36 classes, of maximum AP from introduction through stage 9 minus stage-10 AP. '
              'It is signed: an improvement beyond the prior peak contributes negatively. '
              'The evaluation set grows with stages, so raw stage-mAP decline alone is not a forgetting measure.', '',
              '## Object budget and resource context', '',
              '| Condition | Seed | Replay objects at stage 10 | Final objects | Final points | Final bank MiB | Hours | Peak logged GPU MB |',
              '|---|---|---|---|---|---|---|---|']
    for r in runs:
        final = r['resources'][-1]
        lines.append(f'| {r["label"]} | {r["seed"]} | {r["stage10_old_objects"]} | {r["final_objects"]} | '
                     f'{final["points"]:,} | {final["pickle_bytes"]/2**20:.2f} | {r["duration_hours"]:.2f} | {r["peak_logged_memory_mb"]} |')
    lines += ['', 'Final banks include the last cohort added after training; the stage-9 bank supplies '
              'stage-10 replay. Bank MiB is the final pickle file size. Runtime includes all stages, '
              'evaluation, pseudo labels and bank extraction. GPU memory is the maximum recorded training-log value, '
              'not a hardware-wide peak. Object count is the primary budget measure; denser crops can increase '
              'points and bytes at the same object count. Configured insertion probability/count stays fixed, '
              'but accepted paste counts are not measured.', '',
              '## Sensitivity to the performance tolerance', '',
              'The following descriptive checks use final AP25 relative to the seed-matched random B100 baseline. '
              'A pass means observed loss is within a chosen tolerance; it is not an equivalence test. '
              'The 0.5 pp convention was proposed before these runs; 1.0 and 2.0 pp show sensitivity, '
              'not a revised acceptance criterion chosen after seeing results.', '',
              '| Condition | Object reduction | Tolerance pp | Mean loss within tolerance | Both seed losses within tolerance |',
              '|---|---|---|---|---|']
    for label, pair in groups.items():
        if label == 'Random B=100':
            continue
        deltas = [r['delta_vs_baseline25_pp'] for r in pair]
        for tolerance in (.5, 1., 2.):
            lines.append(f'| {label} | {pair[0]["object_reduction_percent"]:.2f}% | {tolerance:.1f} | '
                         f'{"yes" if st.mean(deltas) >= -tolerance else "no"} | '
                         f'{"yes" if min(deltas) >= -tolerance else "no"} |')
    if selection_complete:
        pair = groups['Most points B=20']
        delta = st.mean(r['delta_vs_random20_25_pp'] for r in pair)
        gap = st.mean(r['delta_vs_baseline25_pp'] for r in pair)
        lines += ['', '## Selection criterion result', '',
                  f'At B=20, choosing crops with the most points changes final AP25 by {delta:+.4f} pp '
                  f'on average versus random selection, and by {gap:+.4f} pp versus random B100. '
                  'Only the selection strategy differs in the resolved configs; all runs start from scratch. '
                  'Check both paired seed effects and point/byte costs before interpreting the mean. '
                  'This comparison tests one alternative criterion; learning-dynamics selection was not tested in this S10 study.']
    lines += ['', '## Scope and evidence for the research discussion', '',
              'This is a local ten-stage extension at the reference object budget, not an exact reproduction '
              'of a verified Peisheng ten-stage object score. The retained reference audit did not identify '
              'such a target. The separate 19.38% ten-stage scene-memory result uses a different protocol; '
              'scene entries and object crops are different budget units. See WEEK_4_PROTOCOL_AUDIT.md '
              'and WEEK_4_TRAINING_PLAN.md for source commits and remaining implementation differences.', '',
              'Two independent full-training seeds are available per completed condition. Equal seed numbers '
              'support paired comparisons but do not guarantee bitwise deterministic training. '
              'A third matched seed and a pre-agreed performance tolerance would strengthen a final retention claim. '
              'No B=10/5 runs were started because B=20 already exceeded the provisional loss tolerance.', '',
              'Audits verify successful exits and completion markers, all metric class scopes/means, '
              'nonempty checkpoints/banks/pseudo files, budget/seed/selector settings, finite logged losses '
              'and LR schedules. Pickle/model payloads are not deserialized by the audit.', '',
              'Machine-readable evidence: `week4_runs/analysis.json` and `week4_runs/week4_per_class.csv`. '
              'Exportable figures: `figures/week4/week4_final_accuracy.png` and '
              '`figures/week4/week4_stage_accuracy.png`, with PDF copies. '
              'Failed full selection runs remain excluded. See WEEK_4_SELECTION_RECOVERY.md for the separately scoped diagnostic.', '']
    plot(groups)
    atomic(ROOT / 'WEEK_4_ANALYSIS.md', '\n'.join(lines))
    print(f'Audited {len(runs)} runs, wrote analysis, per-class CSV and PNG/PDF figures', flush=True)


if __name__ == '__main__':
    main()
