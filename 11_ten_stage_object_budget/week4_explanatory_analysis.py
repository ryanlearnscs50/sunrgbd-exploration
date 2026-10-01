"""Measured cohort and bank-cost evidence for the Week 4 research explanation."""
import csv
import json
from pathlib import Path
import statistics as st

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'presentation/week4'


def savefig(fig, name):
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=180)
    plt.close(fig)


def main():
    data = json.loads((ROOT / 'week4_runs/analysis.json').read_text())
    runs = data['runs']
    assert len(runs) == 6 and all(r['strategy'] == 'random' for r in runs)
    offline = json.loads((ROOT / 'week4_runs/selection_bank_audit/audit.json').read_text())
    assert offline['status'] == 'passed'
    groups = {b: [r for r in runs if r['budget'] == b] for b in (100, 50, 20)}
    baseline = groups[100]
    paired_classes = []
    cohort_rows = []
    for i in range(40):
        row = dict(class_id=i, name=baseline[0]['classes'][i]['name'], cohort=i//4+1)
        for b, pair in groups.items():
            vals = [100*r['classes'][i]['final_ap25'] for r in pair]
            row[f'b{b}_mean_ap25'] = st.mean(vals)
            if b != 100:
                changes = [100*(r['classes'][i]['final_ap25']-base['classes'][i]['final_ap25'])
                           for r, base in zip(pair, baseline)]
                row[f'b{b}_delta_s200'] = changes[0]
                row[f'b{b}_delta_s201'] = changes[1]
                row[f'b{b}_mean_delta'] = st.mean(changes)
        paired_classes.append(row)
    for c in range(10):
        row = dict(cohort=c+1, classes=', '.join(r['name'] for r in paired_classes[c*4:c*4+4]))
        for b in groups:
            row[f'b{b}_final_ap25'] = st.mean(r[f'b{b}_mean_ap25'] for r in paired_classes[c*4:c*4+4])
        row['b20_delta'] = row['b20_final_ap25']-row['b100_final_ap25']
        cohort_rows.append(row)
    for name, rows in [('week4_class_budget_effects', paired_classes), ('week4_cohort_budget_effects', cohort_rows)]:
        with (OUT / f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = dict(
        offline_bank_objects=offline['objects'], offline_bank_points=offline['points'],
        offline_bank_mib=offline['pickle_bytes']/2**20,
        offline_byte_reduction_vs_b100_percent=100*(1-offline['pickle_bytes']/st.mean(r['resources'][-1]['pickle_bytes'] for r in baseline)),
        offline_byte_ratio_vs_random20=offline['pickle_bytes']/st.mean(r['resources'][-1]['pickle_bytes'] for r in groups[20]),
        b20_both_seeds_class_loss_count=sum(r['b20_delta_s200'] < 0 and r['b20_delta_s201'] < 0 for r in paired_classes),
        b20_both_seeds_class_gain_count=sum(r['b20_delta_s200'] > 0 and r['b20_delta_s201'] > 0 for r in paired_classes),
        largest_mean_class_losses=sorted(paired_classes, key=lambda r: r['b20_mean_delta'])[:5],
        cohorts=cohort_rows,
        limitations='Descriptive diagnostics, not significance tests. Offline dense banks have no full S10 detector score.')
    (OUT / 'week4_explanatory_summary.json').write_text(json.dumps(summary, indent=2)+'\n')

    labels = ['Random\nB=100', 'Random\nB=50', 'Random\nB=20', 'Most points\nB=20\n(offline bank)']
    colors = ['#2864a0', '#cd7b27', '#23856a', '#9456a2']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    quantities = [('objects', 'Stored objects', 1), ('points', 'Stored points (million)', 1e6),
                  ('pickle_bytes', 'Bank file size (MiB)', 2**20)]
    for ax, (key, title, scale) in zip(axes, quantities):
        vals = [st.mean(r['resources'][-1][key] for r in groups[b])/scale for b in (100, 50, 20)]
        vals.append(offline[key]/scale)
        bars = ax.bar(range(4), vals, color=colors)
        bars[-1].set_hatch('//')
        for i, v in enumerate(vals):
            ax.text(i, v, f'{v:.2f}' if scale != 1 else f'{v:.0f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(vals)*1.18)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(title)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Same object count does not imply the same point or byte budget\nRandom banks: mean of two trained runs; dense bank: audited GT construction only', fontsize=12)
    savefig(fig, 'week4_bank_costs')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    x = np.arange(1, 11)
    for b, color, offset in zip((100, 50, 20), colors, (-.25, 0, .25)):
        axes[0].bar(x+offset, [r[f'b{b}_final_ap25'] for r in cohort_rows], width=.25, color=color, label=f'B={b}')
    axes[0].set(xlabel='Introduction cohort (4 classes per cohort)', ylabel='Final-stage AP25 (%)', xticks=x)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=.2)
    a = np.array([r['b20_delta_s200'] for r in paired_classes])
    b = np.array([r['b20_delta_s201'] for r in paired_classes])
    axes[1].scatter(a, b, color=colors[2], alpha=.8)
    axes[1].axvline(0, color='#777777', linewidth=.8)
    axes[1].axhline(0, color='#777777', linewidth=.8)
    axes[1].set(xlabel='Seed 200 class AP25 change (pp)', ylabel='Seed 201 class AP25 change (pp)')
    for i in np.argsort(np.abs(a)+np.abs(b))[-5:]:
        axes[1].annotate(paired_classes[int(i)]['name'], (a[i], b[i]), xytext=(4, 4), textcoords='offset points', fontsize=8)
    axes[1].grid(alpha=.2)
    fig.suptitle('Budget effects vary across class cohorts and seeds\nLeft: two-seed means; right: B=20 minus B=100 per class; descriptive evidence')
    savefig(fig, 'week4_cohort_and_seed_effects')

    lines = ['# Week 4 explanatory diagnostics', '',
             'These diagnostics explain the measured storage tradeoff and where accuracy changes occur. '
             'They do not establish a causal mechanism or statistical equivalence.', '',
             '## Equal crop counts and unequal storage', '',
             f'The exhaustive production-path audit selects 800 dense crops containing {offline["points"]:,} points '
             f'and occupying {summary["offline_bank_mib"]:.2f} MiB. This is '
             f'{summary["offline_byte_ratio_vs_random20"]:.2f} times the mean random B=20 bank size. '
             f'Relative to random B=100, object count falls by 79.89% but bytes fall by only '
             f'{summary["offline_byte_reduction_vs_b100_percent"]:.2f}%. '
             'This bank was built offline from training GT; no full ten-stage detector score is attached to it.', '',
             'The selection audit independently enumerated every eligible crop and compared stable top-20 '
             'identities and point counts for all 40 classes against the production output. Dense support is '
             'the selection criterion; it is not a measured guarantee of representativeness or semantic quality.', '',
             '## Final accuracy by introduction cohort', '',
             'Each row averages four class APs at stage 10, then averages the two full-training seeds. '
             'All values are percentages, except the delta in percentage points.', '',
             '| Cohort | Classes | B100 AP25 | B50 AP25 | B20 AP25 | B20 minus B100 |',
             '|---|---|---|---|---|---|']
    for r in cohort_rows:
        lines.append(f'| {r["cohort"]} | {r["classes"]} | {r["b100_final_ap25"]:.3f} | '
                     f'{r["b50_final_ap25"]:.3f} | {r["b20_final_ap25"]:.3f} | {r["b20_delta"]:+.3f} |')
    lines += ['', f'B=20 loses AP25 in both seeds for {summary["b20_both_seeds_class_loss_count"]}/40 classes '
              f'and gains in both for {summary["b20_both_seeds_class_gain_count"]}/40. '
              'Other classes have opposing effects or a zero change. These counts are descriptive and use no significance threshold. '
              'Classwise effects can vary widely even when aggregate mAP changes by less than one percentage point.', '',
              'The largest mean class loss is recycle_bin: −19.1705 AP25 points '
              '(seed changes −19.482 and −18.859). With equal class weighting over 40 classes, '
              'this contributes −0.4792625 pp to the total −0.9295 pp mAP change. '
              'This is an arithmetic decomposition, not evidence of the mechanism causing the loss.', '',
              'Artifacts: `presentation/week4/week4_bank_costs.png`, '
              '`presentation/week4/week4_cohort_and_seed_effects.png`, PDF copies, class/cohort CSV files '
              'and `week4_explanatory_summary.json`. Underlying accuracy comes only from the six audited full runs.', '']
    (ROOT / 'WEEK_4_EXPLANATORY_DIAGNOSTICS.md').write_text('\n'.join(lines))
    print(json.dumps({k: v for k, v in summary.items() if k != 'cohorts'}))


if __name__ == '__main__':
    main()
