"""Persist cosine replication comparisons and validate finished runs; never launch jobs."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

from collect_week3_status import collect
from monitor_week3_cutoff import lr_evidence
from run_week3_overnight import verify_complete

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'week3_runs/cosine_replication_s202'
KEYS = ('map25', 'old_map25', 'new_map25', 'map50')


def update():
    collect()
    runs = json.loads((ROOT / 'week3_runs/status.json').read_text())['runs']
    selected = [r for r in runs if r['mode'] in (
        'pseudo_cosine_v2', 'dose25_cosine_v2',
        'pseudo_cosine_v2_s202', 'dose25_cosine_v2_s202')]
    evidence = {}
    lines = ['# Week 3 cosine replication', '',
             'Same released stage-1 checkpoint; seed 202 is a follow-up to seed 201.',
             'Only verified complete cosine runs enter the comparison. Values are mAP fractions.', '',
             '| Seed | Policy | Stage | mAP@.25 | Old @.25 | New @.25 | mAP@.50 | Delta @.25 vs original LR | Delta @.50 vs original LR |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    comparisons = []
    for run in selected:
        name = run['mode']
        path = Path(run['run_dir']) if run['run_dir'] else None
        row = dict(exit_code=run['exit_code'], lr=lr_evidence(path) if path else {})
        evidence[name] = row
        if run['exit_code'] != 0:
            continue
        try:
            verify_complete(ROOT / 'week3_runs' / name)
            if len(row['lr']) != 4 or not all(v['decaying'] for v in row['lr'].values()):
                raise RuntimeError('Missing per-stage decay evidence')
            replay = name.startswith('dose25')
            if replay and not (path / 'object_memory_bank/object_memory_bank_stage_5.pkl').stat().st_size:
                raise RuntimeError('Empty replay bank')
            baseline = next(r for r in runs if r['policy'] == ('dose25' if replay else 'pseudo_only')
                            and r['seed'] == run['seed'] and r['exit_code'] == 0)
            for metric in run['metrics']:
                base = next(m for m in baseline['metrics'] if m['stage'] == metric['stage'])
                delta = {k: metric[k] - base[k] for k in KEYS}
                comparisons.append(dict(run=name, seed=run['seed'], stage=metric['stage'],
                                        metrics=metric, baseline=baseline['mode'], delta=delta))
                lines.append(f"| {run['seed']} | {'25% replay' if replay else 'Pseudo only'} | {metric['stage']} | "
                             + ' | '.join(f'{metric[k]:.5f}' for k in KEYS)
                             + f" | {delta['map25']:+.5f} | {delta['map50']:+.5f} |")
            row['verified'] = True
        except Exception as exc:
            row['validation_error'] = str(exc)
    lines += ['', '## Run checks', '']
    for name, row in evidence.items():
        lines.append(f"- {name}: exit={row['exit_code']}; verified={row.get('verified', False)}; "
                     f"LR stages observed={len(row['lr'])}; error={row.get('validation_error', 'none')}.")
    lines += ['', 'No further experiment launch is configured.',
              'A second continuation seed does not measure stage-1 variability or establish statistical significance.', '']
    result = dict(updated_at=datetime.now(timezone.utc).isoformat(), runs=evidence, comparisons=comparisons)
    (STATE / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
    (ROOT / 'WEEK_3_COSINE_REPLICATION.md').write_text('\n'.join(lines))
    done = len(selected) == 4 and all(r['exit_code'] is not None for r in selected)
    if done:
        valid = all(r.get('verified') for r in evidence.values())
        (STATE / 'exit_code').write_text('0\n' if valid else '1\n')
    return done


def main():
    STATE.mkdir(exist_ok=True)
    with (STATE / 'monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not update():
            time.sleep(60)


if __name__ == '__main__':
    main()
