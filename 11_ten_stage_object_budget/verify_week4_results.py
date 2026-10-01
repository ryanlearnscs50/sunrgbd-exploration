"""Recompute published Week 4 numbers using only JSON and the standard library.

Raw per-stage metrics are checked against the derived summary so an error in
aggregation is detectable independently of the original training environment.
This is evidence validation, not a replacement for the host completion audit.
"""
import json
import math
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent


def read(relative):
    return json.loads((ROOT / relative).read_text())


def close(actual, expected, tolerance=1e-9):
    assert math.isfinite(actual) and abs(actual - expected) <= tolerance, (actual, expected)


def main():
    analysis = read('week4_runs/analysis.json')
    runs = analysis['runs']
    assert analysis['audited'] and len(runs) == 6
    assert {(r['budget'], r['seed']) for r in runs} == {
        (b, s) for b in (100, 50, 20) for s in (200, 201)}
    for run in runs:
        assert run['strategy'] == 'random'
        state = ROOT / 'week4_runs' / run['run_id']
        assert (state / 'exit_code').read_text().strip() == '0'
        metrics = [json.loads((state / 'metrics' / f'stage_{s}_metrics.json').read_text())
                   for s in range(1, 11)]
        for stage, metric in enumerate(metrics, 1):
            assert metric['evaluated_at_stage'] == stage
            classes = metric['classes']
            assert sorted(c['model_idx'] for c in classes) == list(range(4 * stage))
            for suffix in ('25', '50'):
                field = 'AP_0.' + suffix
                assert all(0 <= c[field] <= 1 for c in classes)
                close(mean(c[field] for c in classes), metric['m' + field], 1e-5)
                close(metric['m' + field], run['stages'][stage - 1]['map' + suffix])
        for suffix in ('25', '50'):
            field = 'AP_0.' + suffix
            close(metrics[-1]['m' + field], run['final_map' + suffix])
            close(mean(m['m' + field] for m in metrics), run['stage_average_map' + suffix])
            final = {c['model_idx']: c[field] for c in metrics[-1]['classes']}
            close(mean(final[i] for i in range(36)), run['final_old_map' + suffix])
            close(mean(final[i] for i in range(36, 40)), run['final_new_map' + suffix])
            forgetting = []
            for i in range(36):
                prior = [next(c[field] for c in m['classes'] if c['model_idx'] == i)
                         for m in metrics[i // 4:9]]
                forgetting.append(max(prior) - final[i])
            close(mean(forgetting), run['forgetting' + suffix])
        assert run['final_objects'] == run['resources'][-1]['objects']
        assert run['stage10_old_objects'] == run['resources'][-2]['objects']
        baseline = next(r for r in runs if r['budget'] == 100 and r['seed'] == run['seed'])
        close(100 * (run['final_map25'] - baseline['final_map25']), run['delta_vs_baseline25_pp'])
    groups = {b: [r for r in runs if r['budget'] == b] for b in (100, 50, 20)}
    print('budget,mean_AP25_percent,sample_SD_pp,mean_bank_MiB')
    for budget, pair in groups.items():
        vals = [100 * r['final_map25'] for r in pair]
        print(f'{budget},{mean(vals):.4f},{stdev(vals):.4f},'
              f'{mean(r["resources"][-1]["pickle_bytes"] for r in pair)/2**20:.4f}')
    close(100 * mean(r['final_map25'] for r in groups[100]), 15.6815)
    close(100 * mean(r['final_map25'] for r in groups[20]), 14.7520)
    diagnostic = read('week4_runs/selection_recovery/audit.json')
    assert diagnostic['audited_complete'] and not diagnostic['audit_errors']
    assert len(diagnostic['runs']) == 4
    for pair in diagnostic['paired_deltas']:
        rows = {r['strategy']: r for r in diagnostic['runs'] if r['seed'] == pair['seed']}
        a, b = rows['random'], rows['largest_point_count']
        assert a['checkpoint_sha256'] == b['checkpoint_sha256']
        close(100 * (b['map25'] - a['map25']), pair['delta25_pp'])
    close(mean(p['delta25_pp'] for p in diagnostic['paired_deltas']), 0.975)
    offline = read('week4_runs/selection_bank_audit/audit.json')
    assert offline['status'] == 'passed' and offline['objects'] == 800
    assert offline['points'] == 10990248
    for seed in (200, 201):
        assert (ROOT / f'week4_runs/object20_largest_s{seed}/exit_code').read_text().strip() == '1'
    print('PASS: 60 stage metrics, six full runs, four diagnostic runs; two failed full selector runs excluded.')


if __name__ == '__main__':
    main()
