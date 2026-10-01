"""Audit completed seed pairs and report object count versus accuracy."""
import argparse
import json
import math
from pathlib import Path
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'week4_runs'


def audit(budget, seed, run_id=None, selection_strategy='random'):
    state = STATE / (run_id or f'object{budget}_s{seed}')
    manifest = json.loads((state / 'manifest.json').read_text())
    assert (state / 'exit_code').read_text().strip() == '0'
    assert manifest['budget_per_class'] == budget and manifest['seed'] == seed
    stem = Path(manifest['work_dir_stem'])
    dirs = list(stem.parent.glob(stem.name + '_*'))
    assert len(dirs) == 1
    run = dirs[0]
    console = (state / 'console.log').read_text()
    assert 'Explicit mapping incremental learning completed!' in console
    assert f'WEEK4_EXPERIMENT_EXIT seed={seed} code=0' in console
    assert 'Traceback (most recent call last)' not in console
    stages = []
    for stage in range(1, 11):
        metrics = json.loads((run / f'memory_bank/scores/stage_{stage}_metrics.json').read_text())
        assert metrics['evaluated_at_stage'] == stage
        assert sorted(c['model_idx'] for c in metrics['classes']) == list(range(4*stage))
        for key in ('AP_0.25', 'AP_0.50'):
            values = [c[key] for c in metrics['classes']]
            assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
            assert abs(statistics.mean(values) - metrics['m'+key]) < 1e-5
        checkpoint = run / f'checkpoints/stage_{stage}/epoch_{6 if stage == 1 else 1}.pth'
        assert checkpoint.stat().st_size > 0
        bank_path = run / f'object_memory_bank/object_memory_bank_stage_{stage}.json'
        bank = json.loads(bank_path.read_text())
        assert bank_path.with_suffix('.pkl').stat().st_size > 0
        assert bank['stage_id'] == stage
        assert bank['config']['exemplars_per_class'] == budget
        assert bank['config']['max_total_exemplars'] == budget*40
        assert bank['config']['random_seed'] == seed
        assert bank['config']['selection_strategy'] == selection_strategy
        assert sorted(map(int, bank['exemplars'])) == list(range(4*stage))
        assert all(0 < len(v) <= budget for v in bank['exemplars'].values())
        epochs = set()
        for path in (run / f'checkpoints/stage_{stage}').glob('*.log.json'):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if row.get('mode') == 'train':
                    epochs.add(row['epoch'])
                    expected = .0001 if stage == 1 and row['epoch'] == 6 else .001
                    assert math.isclose(row['lr'], expected, rel_tol=1e-6)
                    assert math.isfinite(row['loss'])
        assert epochs == (set(range(1, 7)) if stage == 1 else {1})
        if stage > 1:
            assert (run / f'pseudo_labels/stage_{stage}_pseudo_only_conf50_pseudo_labels.pkl').stat().st_size > 0
        stages.append(dict(stage=stage, map25=metrics['mAP_0.25'], map50=metrics['mAP_0.50'],
                           stored_objects=sum(len(v) for v in bank['exemplars'].values())))
    return dict(budget=budget, seed=seed, stages=stages, final_map25=stages[-1]['map25'],
                final_map50=stages[-1]['map50'], final_objects=stages[-1]['stored_objects'],
                stage10_old_objects=stages[-2]['stored_objects'],
                stage_average_map25=statistics.mean(s['map25'] for s in stages),
                stage_average_map50=statistics.mean(s['map50'] for s in stages))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-budget', type=int, choices=(100, 50, 20), default=100)
    args = parser.parse_args()
    results = []
    for budget in (100, 50, 20):
        paths = [STATE / f'object{budget}_s{s}/exit_code' for s in (200, 201)]
        if all(p.exists() for p in paths):
            results.extend(audit(budget, seed) for seed in (200, 201))
        elif budget == args.require_budget:
            raise RuntimeError(f'B={budget} pair incomplete')
    baselines = {r['seed']: r for r in results if r['budget'] == 100}
    assert set(baselines) == {200, 201}
    for r in results:
        baseline = baselines[r['seed']]
        r['object_reduction_percent'] = 100*(1-r['final_objects']/baseline['final_objects'])
        r['delta_final_map25_pp'] = 100*(r['final_map25']-baseline['final_map25'])
        r['delta_final_map50_pp'] = 100*(r['final_map50']-baseline['final_map50'])
    updated = datetime.now(timezone.utc).isoformat()
    output = dict(updated_at=updated, validation='passed', runs=results)
    (STATE / 'budget_comparison.json').write_text(json.dumps(output, indent=2)+'\n')
    lines = ['# Week 4 — Object-count budget comparison', '', f'Updated: {updated}', '',
             'Only completed, audited seed pairs are shown. AP is percent; deltas are percentage points relative to the matching B=100 seed.', '',
             '| Cap/class | Seed | Stage-10 old objects | Final objects | Object reduction | Final @.25 | Delta @.25 | Final @.50 | Delta @.50 | Stage avg @.25 | Stage avg @.50 |',
             '|---|---|---|---|---|---|---|---|---|---|---|']
    for r in results:
        lines.append(f"| {r['budget']} | {r['seed']} | {r['stage10_old_objects']} | {r['final_objects']} | {r['object_reduction_percent']:.2f}% | {100*r['final_map25']:.3f} | {r['delta_final_map25_pp']:+.3f} | {100*r['final_map50']:.3f} | {r['delta_final_map50_pp']:+.3f} | {100*r['stage_average_map25']:.3f} | {100*r['stage_average_map50']:.3f} |")
    lines += ['', '## Two-seed averages', '', '| Cap/class | Final @.25 mean ± sample SD | Final @.50 mean ± sample SD | Mean paired delta @.25 |', '|---|---|---|---|']
    for budget in sorted({r['budget'] for r in results}, reverse=True):
        pair = [r for r in results if r['budget'] == budget]
        def stats(key):
            values = [100*r[key] for r in pair]
            return f'{statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}'
        lines.append(f"| {budget} | {stats('final_map25')} | {stats('final_map50')} | {statistics.mean(r['delta_final_map25_pp'] for r in pair):+.3f} |")
    lines += ['', 'Stage average is the unweighted mean of the ten seen-class stage mAPs. Final objects include the last cohort added after training; stage-10 old objects describe the bank available during final-stage replay.', '',
              'Audit checks: successful exit/completion markers, all ten metric scopes and class means, nonempty terminal checkpoints/bank pickles/pseudo-label files, bank caps and seed, actual LR schedule and finite logged losses. Model/bank pickle payloads are not deserialized.', '',
              'These are full-training seeds under the fixed local object+pseudo protocol. Two seeds provide descriptive variability, not statistical equivalence. Scene-memory entry counts are different units; this report makes no equal-size comparison to original LDMR. Selection criteria remain a later phase.', '']
    (ROOT / 'WEEK_4_BUDGET_COMPARISON.md').write_text('\n'.join(lines))
    print(f'Audited {len(results)} runs; wrote WEEK_4_BUDGET_COMPARISON.md', flush=True)


if __name__ == '__main__':
    main()
