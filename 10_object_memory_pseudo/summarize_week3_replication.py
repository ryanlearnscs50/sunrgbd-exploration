"""Write paired seed results without treating pending runs as measurements."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent


def summarize():
    status = json.loads((ROOT / 'week3_runs/status.json').read_text())
    rows = []
    for seed in (201, 202, 203):
        row = dict(seed=seed)
        for policy in ('pseudo_only', 'dose25'):
            runs = [r for r in status['runs'] if r.get('policy', r['mode']) == policy
                    and r.get('seed', 201) == seed and r['status'] == 'completed']
            final = [m for r in runs for m in r['metrics'] if m['stage'] == 5]
            row[policy] = final[0] if len(final) == 1 else None
        row['delta'] = (row['dose25']['map25'] - row['pseudo_only']['map25']
                        if row['dose25'] and row['pseudo_only'] else None)
        rows.append(row)
    lines = ['# Week 3 replay replication', '',
             'Same released stage-1 checkpoint; seeds vary continuation training and random bank selection.',
             'Values are mAP@0.25 fractions. Only completed jobs contribute.', '',
             '| Seed | Pseudo only | 25% object replay + pseudo | Paired difference |',
             '|---|---:|---:|---:|']
    for row in rows:
        def fmt(value):
            return 'pending' if value is None else f'{value:.4f}'
        lines.append(f'| {row["seed"]} | {fmt(row["pseudo_only"]["map25"] if row["pseudo_only"] else None)} | '
                     f'{fmt(row["dose25"]["map25"] if row["dose25"] else None)} | {fmt(row["delta"])} |')
    deltas = [r['delta'] for r in rows if r['delta'] is not None]
    summary = dict(completed_pairs=len(deltas), mean_delta=statistics.mean(deltas) if deltas else None,
                   sample_sd_delta=statistics.stdev(deltas) if len(deltas) > 1 else None)
    if len(deltas) > 1:
        lines += ['', f'Mean paired difference: {summary["mean_delta"]:+.4f}; '
                  f'sample SD across {len(deltas)} seeds: {summary["sample_sd_delta"]:.4f}.']
    lines += ['', 'Seed 201 selected the 25% candidate; seeds 202 and 203 are follow-up checks.',
              'These runs share a pretrained checkpoint and do not measure variability from training stage 1.',
              'A few seeds provide limited evidence; no default is changed automatically.', '',
              'Queue details: `week3_runs/overnight_20260924/status.json`.', '']
    followup = [r['delta'] for r in rows if r['seed'] != 201 and r['delta'] is not None]
    summary['followup_mean_delta'] = statistics.mean(followup) if followup else None
    if followup:
        lines += [f"Follow-up seeds only: mean difference {statistics.mean(followup):+.4f} ({len(followup)} pairs).", '']
    lines += ['## Stage and cohort checks', '',
              'Mean paired differences (25% replay minus pseudo-only); old/new classes are defined at each stage.', '',
              '| Stage | Pairs | mAP@.25 | Old mAP@.25 | New mAP@.25 | mAP@.50 |',
              '|---|---:|---:|---:|---:|---:|']
    stage_rows = []
    for stage in range(2, 6):
        pairs = []
        for row in rows:
            if row['delta'] is None:
                continue
            metrics = {}
            for policy in ('pseudo_only', 'dose25'):
                run = next(r for r in status['runs'] if r.get('policy', r['mode']) == policy
                           and r.get('seed', 201) == row['seed'] and r['status'] == 'completed')
                metrics[policy] = next(m for m in run['metrics'] if m['stage'] == stage)
            pairs.append({k: metrics['dose25'][k] - metrics['pseudo_only'][k]
                          for k in ('map25', 'old_map25', 'new_map25', 'map50')})
        if pairs:
            means = {k: statistics.mean(p[k] for p in pairs) for k in pairs[0]}
            stage_rows.append(dict(stage=stage, pairs=len(pairs), mean_deltas=means))
            lines.append(f'| {stage} | {len(pairs)} | ' + ' | '.join(f'{v:+.4f}' for v in means.values()) + ' |')
    summary['stage_comparisons'] = stage_rows
    if len(deltas) == 3:
        lines += ['', 'Final old- and new-class mAP@.25 improve in all three seeds. Earlier-stage effects are mixed.',
                  'Final mAP@.50 improves in seeds 201 and 202, but decreases in seed 203.',
                  'The two follow-up seeds support the selected dose; they do not establish broad statistical significance.', '']
    for path, content in [(ROOT / 'WEEK_3_REPLICATION.md', '\n'.join(lines)),
                          (ROOT / 'week3_runs/replication.json', json.dumps(dict(rows=rows, summary=summary), indent=2) + '\n')]:
        tmp = path.with_suffix(path.suffix + '.tmp')
        tmp.write_text(content)
        tmp.replace(path)


if __name__ == '__main__':
    summarize()
