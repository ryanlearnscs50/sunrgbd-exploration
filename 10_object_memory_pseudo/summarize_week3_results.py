"""Summarize the completed first Week 3 pair from persisted validation metrics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RUNS = {
    'Object, no pseudo': 'sunrgbd_s5_object_memory_heightaware_1obj_sunrgbd40_s5_freqorder_20260908_151502',
    'Pseudo only': 'week3_pseudo_only_s201_sunrgbd40_s5_freqorder_20260923_131248',
    'Object + pseudo': 'week3_object_pseudo_s201_sunrgbd40_s5_freqorder_20260923_131248',
}


def main():
    report = {}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for label, directory in RUNS.items():
        metrics = []
        for stage in range(2, 6):
            path = ROOT / 'incremental_logs' / directory / 'memory_bank/scores' / f'stage_{stage}_metrics.json'
            data = json.loads(path.read_text())
            assert data['evaluated_at_stage'] == stage
            assert len(data['classes']) == stage * 8
            cohorts = [sum(c['AP_0.25'] for c in data['classes']
                           if cohort * 8 <= c['model_idx'] < (cohort + 1) * 8) / 8
                       for cohort in range(stage)]
            metrics.append(dict(stage=stage, map25=data['mAP_0.25'],
                                map50=data['mAP_0.50'], cohort_map25=cohorts))
        report[label] = dict(run_dir=directory, metrics=metrics)
        axes[0].plot(range(2, 6), [m['map25'] * 100 for m in metrics], 'o-', label=label)
        axes[1].plot(range(1, 6), [v * 100 for v in metrics[-1]['cohort_map25']], 'o-', label=label)
    for ax, title, xlabel, ticks in zip(axes,
            ['Seen-class validation performance', 'Final performance by arrival cohort'],
            ['Training stage', 'Class arrival stage'], [range(2, 6), range(1, 6)]):
        ax.set(title=title, xlabel=xlabel, ylabel='mAP@0.25 (%)', xticks=list(ticks))
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Week 3: pseudo supervision and object replay (seed 201)')
    fig.tight_layout()
    output = ROOT / 'visualizations/week3_pseudo_comparison.png'
    fig.savefig(output, dpi=170)
    plt.close(fig)
    (ROOT / 'week3_runs/first_pair_results.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Week 3 first experiment results', '',
             'SUN RGB-D 40-class S5; same released stage-1 checkpoint and seed 201.', '',
             '| Policy | Stage 2 | Stage 3 | Stage 4 | Stage 5 |', '|---|---:|---:|---:|---:|']
    for label, result in report.items():
        lines.append('| ' + label + ' | ' + ' | '.join(f'{m["map25"]:.4f}' for m in result['metrics']) + ' |')
    obj = report['Object + pseudo']['metrics'][-1]['map25']
    prior = report['Object, no pseudo']['metrics'][-1]['map25']
    pseudo = report['Pseudo only']['metrics'][-1]['map25']
    lines += ['', f'Pseudo supervision changes object replay by {obj-prior:+.4f} absolute mAP.',
              f'Object replay changes pseudo-only performance by {obj-pseudo:+.4f}.', '',
              'Both current runs exited successfully. The no-pseudo result is a historical matched control.',
              'One seed does not establish significance. Later teachers and random augmentation streams',
              'differ between policies. These are full policy comparisons.', '',
              'Dose sweep and seed replication are complete; see WEEK_3_REPLICATION.md for the latest results.',
              'No improved replay default is established yet.', '',
              '![Stage trajectories and cohort retention](visualizations/week3_pseudo_comparison.png)', '']
    (ROOT / 'WEEK_3_RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:15]))


if __name__ == '__main__':
    main()
