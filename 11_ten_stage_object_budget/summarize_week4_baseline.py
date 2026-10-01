"""Audit completed Week 4 baselines without model execution or pickle loading."""
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from collect_week4_status import collect

ROOT = Path(__file__).resolve().parent


def main():
    collect()
    status = json.loads((ROOT / 'week4_runs/status.json').read_text())
    results = []
    for run in status['runs']:
        assert run['artifact_checks_passed'], run['run_id']
        directory = Path(run['run_dir'])
        state = ROOT / 'week4_runs' / run['run_id']
        console = (state / 'console.log').read_text()
        assert 'Explicit mapping incremental learning completed!' in console
        assert f"WEEK4_EXPERIMENT_EXIT seed={run['seed']} code=0" in console
        assert 'Traceback (most recent call last)' not in console
        stage_metrics = []
        peak_memory = 0
        epochs_lr = {}
        for stage in range(1, 11):
            metrics = json.loads((directory / f'memory_bank/scores/stage_{stage}_metrics.json').read_text())
            assert metrics['evaluated_at_stage'] == stage
            assert sorted(c['model_idx'] for c in metrics['classes']) == list(range(4 * stage))
            for key in ('AP_0.25', 'AP_0.50'):
                values = [c[key] for c in metrics['classes']]
                assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
                assert abs(statistics.mean(values) - metrics['m' + key]) < 1e-5
            stage_metrics.append(metrics)
            checkpoint = directory / f'checkpoints/stage_{stage}/epoch_{6 if stage == 1 else 1}.pth'
            assert checkpoint.is_file() and checkpoint.stat().st_size > 0
            bank = run['bank_stats'][stage - 1]
            assert bank['stage'] == stage and bank['pickle_bytes'] > 0
            assert sorted(map(int, bank['per_class_counts'])) == list(range(4 * stage))
            assert all(0 < n <= 100 for n in bank['per_class_counts'].values())
            assert bank['config']['exemplars_per_class'] == 100
            assert bank['config']['random_seed'] == run['seed']
            rates = {}
            for log in (directory / f'checkpoints/stage_{stage}').glob('*.log.json'):
                for line in log.read_text().splitlines():
                    row = json.loads(line)
                    if row.get('mode') != 'train':
                        continue
                    epoch = int(row['epoch'])
                    rates.setdefault(epoch, set()).add(row['lr'])
                    expected = .0001 if stage == 1 and epoch == 6 else .001
                    assert math.isclose(row['lr'], expected, rel_tol=1e-6)
                    assert math.isfinite(row['loss'])
                    peak_memory = max(peak_memory, row.get('memory', 0))
            assert sorted(rates) == list(range(1, 7) if stage == 1 else range(1, 2))
            epochs_lr[stage] = {e: sorted(v) for e, v in rates.items()}
            if stage > 1:
                pseudo = directory / f'pseudo_labels/stage_{stage}_pseudo_only_conf50_pseudo_labels.pkl'
                assert pseudo.is_file() and pseudo.stat().st_size > 0
        final = run['metrics'][-1]
        names = {str(c['model_idx']): c['name'] for c in stage_metrics[-1]['classes']}
        deficits = {names[k]: n for k, n in run['bank_stats'][-1]['per_class_counts'].items() if n < 100}
        start = datetime.fromisoformat((state / 'started_at').read_text().strip())
        end = datetime.fromisoformat((state / 'ended_at').read_text().strip())
        results.append(dict(seed=run['seed'], final=final,
                            stage_average_map25=statistics.mean(m['map25'] for m in run['metrics']),
                            stage_average_map50=statistics.mean(m['map50'] for m in run['metrics']),
                            duration_hours=(end-start).total_seconds()/3600,
                            ended_at=end.isoformat(), peak_logged_memory_mb=peak_memory,
                            final_bank=run['bank_stats'][-1], stage10_replay_bank=run['bank_stats'][-2],
                            below_cap_classes=deficits, epochs_lr=epochs_lr))
    assert {r['seed'] for r in results} == {200, 201}
    output = dict(updated_at=status['updated_at'], validation='passed', runs=results)
    (ROOT / 'week4_runs/baseline_summary.json').write_text(json.dumps(output, indent=2) + '\n')
    lines = ['# Week 4 — Completed ten-stage B=100 baseline', '',
             'Chunk 2 complete. Both independent full-training seeds exited 0 and passed the artifact audit.', '',
             'All AP values are percentages. Stage average is the unweighted mean of the ten seen-class stage mAPs.', '',
             '| Seed | Final @.25 | Final @.50 | Stage avg @.25 | Stage avg @.50 | Final old @.25 | Final new @.25 | Hours |',
             '|---|---|---|---|---|---|---|---|']
    for r in results:
        f = r['final']
        lines.append(f"| {r['seed']} | {100*f['map25']:.3f} | {100*f['map50']:.3f} | {100*r['stage_average_map25']:.3f} | {100*r['stage_average_map50']:.3f} | {100*f['old_map25']:.3f} | {100*f['new_map25']:.3f} | {r['duration_hours']:.2f} |")
    means = {k: statistics.mean(r['final'][k] for r in results)*100 for k in ('map25', 'map50')}
    sds = {k: statistics.stdev(r['final'][k] for r in results)*100 for k in means}
    lines += ['', f"Final mean ± sample SD: **{means['map25']:.3f} ± {sds['map25']:.3f} @.25**, **{means['map50']:.3f} ± {sds['map50']:.3f} @.50** (n=2; descriptive, not a confidence interval).", '',
              '## Actual bank occupancy', '',
              '| Seed | Old objects available at stage 10 | Final stored objects | Final pickle MiB | Final crop points | Peak logged training memory MB |',
              '|---|---|---|---|---|---|']
    for r in results:
        b = r['final_bank']
        lines.append(f"| {r['seed']} | {r['stage10_replay_bank']['total_objects']} | {b['total_objects']} | {b['pickle_bytes']/2**20:.2f} | {b['total_points']:,} | {r['peak_logged_memory_mb']} |")
    lines += ['', 'Below-cap final classes (actual stored objects):']
    for r in results:
        lines.append(f"- Seed {r['seed']}: " + ', '.join(f'{k}={v}' for k, v in r['below_cap_classes'].items()) + '.')
    lines += ['', 'The final bank includes the final four classes after training; it is not the replay budget used in stage 10. Pickle size is stored bytes; logged training memory is not whole-device peak usage.', '',
              '## Validation and limits', '',
              '- Twenty correctly scoped stage metric files: 4→40 class indices, finite AP values, and class means agree with reported mAP.',
              '- Twenty nonempty terminal checkpoints, twenty bank summaries and nonempty bank pickles; per-class counts respect B=100. Pickle/checkpoint payloads were not deserialized in this audit.',
              '- Actual logged stage-1 LR is .001 in epochs 1–5 and .0001 in epoch 6; stages 2–10 use .001. All logged losses are finite.',
              '- Nine nonempty generated pseudo-label files per seed. Object bank construction and insertion configuration were validated before launch; accepted paste counts are not measured by this summary.',
              '- Both completion markers and exit codes agree. No traceback in either console.', '',
              '## Interpretation and next part', '',
              'This establishes the local ten-stage object+pseudo baseline at the reference nominal budget. No verified Peisheng ten-stage object result was found in the retained reference audit, so a numerical reproduction match cannot be claimed. His five-stage object results use a different stage protocol.', '',
              f"The released 19.38% ten-stage scene-memory score exceeds the local mean by {19.38-means['map25']:.3f} percentage points @.25. This is contextual only: memory type and implementation differ; it does not isolate a causal gap.", '',
              'Next chunk: reduced stored-object budgets, initially B=50 and B=20, with unchanged replay dose and placement. Pair seeds 200/201. If reusing stage-1 checkpoints, use each matching seed and record continuation/RNG lineage; do not describe these as new from-scratch seeds. Later chunks address selection criteria and the budget/performance conclusion.', '',
              'Stop here before launching reduced budgets. Full protocol and known implementation differences: `WEEK_4_TRAINING_PLAN.md`. Per-stage evidence: `WEEK_4_RUN_STATUS.md`; machine-readable summary: `week4_runs/baseline_summary.json`. Reproduce with `summarize_week4_baseline.py` in a detached job.', '']
    (ROOT / 'WEEK_4_BASELINE_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(validation='passed', final_mean_percent=means, final_sample_sd_pp=sds)))


if __name__ == '__main__':
    main()
