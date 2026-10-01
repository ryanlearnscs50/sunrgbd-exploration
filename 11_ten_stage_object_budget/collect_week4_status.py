"""Persist Week 4 status; watch only, never launch or stop trainers."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'week4_runs'


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def collect(budget=100):
    runs = []
    status_name = 'status' if budget == 100 else f'object{budget}_status'
    report_name = 'WEEK_4_RUN_STATUS' if budget == 100 else f'WEEK_4_OBJECT{budget}_RUN_STATUS'
    for path in sorted(STATE.glob(f'object{budget}_s*/manifest.json')):
        manifest = read_json(path)
        state = path.parent
        stem = Path(manifest['work_dir_stem'])
        dirs = sorted(stem.parent.glob(stem.name + '_*'))
        run_dir = dirs[-1] if len(dirs) == 1 else None
        exit_path = state / 'exit_code'
        code = int(exit_path.read_text()) if exit_path.exists() else None
        pid_path = state / 'trainer.pid'
        pid = int(pid_path.read_text()) if pid_path.exists() else None
        alive = False
        if pid:
            try:
                alive = b'train_incremental_scene.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
            except OSError:
                pass
        metrics = []
        checkpoints = {}
        banks = []
        lr = {}
        if run_dir:
            for stage in range(1, 11):
                data = read_json(run_dir / 'memory_bank/scores' / f'stage_{stage}_metrics.json')
                if data and data.get('evaluated_at_stage') == stage:
                    classes = data.get('classes', [])
                    old = [c['AP_0.25'] for c in classes if c['model_idx'] < (stage - 1) * 4]
                    new = [c['AP_0.25'] for c in classes if (stage - 1) * 4 <= c['model_idx'] < stage * 4]
                    metrics.append(dict(stage=stage, map25=data.get('mAP_0.25'), map50=data.get('mAP_0.50'),
                                        old_map25=sum(old)/len(old) if old else None,
                                        new_map25=sum(new)/len(new) if new else None))
                ckpts = [p for p in (run_dir / 'checkpoints' / f'stage_{stage}').rglob('*.pth')
                         if p.is_file() and p.stat().st_size > 0]
                checkpoints[stage] = sorted(str(p) for p in ckpts)
                bank = run_dir / 'object_memory_bank' / f'object_memory_bank_stage_{stage}.json'
                if bank.exists():
                    data = read_json(bank) or {}
                    buckets = data.get('exemplars', {})
                    counts = {k: len(v) for k, v in buckets.items()}
                    pkl = bank.with_suffix('.pkl')
                    banks.append(dict(stage=stage, config=data.get('config'),
                                      per_class_counts=counts, total_objects=sum(counts.values()),
                                      total_points=sum(x.get('point_count', 0) for v in buckets.values() for x in v),
                                      pickle_bytes=pkl.stat().st_size if pkl.exists() else None))
                rates = []
                for log in (run_dir / 'checkpoints' / f'stage_{stage}').rglob('*.log.json'):
                    for line in log.read_text(errors='replace').splitlines():
                        try:
                            row = json.loads(line)
                            if row.get('mode') == 'train' and isinstance(row.get('lr'), (int, float)):
                                rates.append(row['lr'])
                        except ValueError:
                            pass
                if rates:
                    lr[stage] = dict(min=min(rates), max=max(rates), count=len(rates))
        valid = (code == 0 and len(metrics) == 10 and all(checkpoints.values()) and len(banks) == 10)
        status = ('finished; artifact checks passed' if valid else 'exit 0; artifact checks incomplete') if code == 0 else (
            'failed' if code is not None else 'running' if alive else 'startup or missing process; inspect')
        runs.append(dict(**manifest, status=status, trainer_pid=pid, exit_code=code,
                         run_dir=str(run_dir) if run_dir else None, metrics=metrics,
                         checkpoint_paths=checkpoints, bank_stats=banks, logged_lr=lr,
                         artifact_checks_passed=valid))
    report = dict(updated_at=datetime.now(timezone.utc).isoformat(), runs=runs)
    tmp = STATE / f'{status_name}.json.tmp'
    tmp.write_text(json.dumps(report, indent=2) + '\n')
    tmp.replace(STATE / f'{status_name}.json')
    lines = [f'# Week 4 — Ten-stage B={budget} status', '', f'Updated: {report["updated_at"]}', '',
             f'B={budget}/class; two independent full-training seeds. This watcher launches no further runs.', '',
             '| Seed | GPU | Status | Completed metric stages | Trainer PID |',
             '|---|---|---|---|---|']
    for run in runs:
        lines.append(f'| {run["seed"]} | {run["gpu"]} | {run["status"]} | {len(run["metrics"])} / 10 | {run["trainer_pid"]} |')
    lines += ['', '## Measured metrics', '', 'AP values below are percentages.', '',
              '| Seed | Stage | mAP@.25 | mAP@.50 | Old @.25 | New @.25 |', '|---|---|---|---|---|---|']
    def fmt(x):
        return '—' if x is None else f'{100*x:.3f}'
    for run in runs:
        for m in run['metrics']:
            lines.append(f'| {run["seed"]} | {m["stage"]} | {fmt(m["map25"])} | {fmt(m["map50"])} | {fmt(m["old_map25"])} | {fmt(m["new_map25"])} |')
    lines += ['', f'Detailed metrics, bank statistics, checkpoint paths and logged LR: `week4_runs/{status_name}.json`.',
              'Recovery and interpretation: `WEEK_4_MEMORY.md` and `WEEK_4_TRAINING_PLAN.md`.', '']
    md = ROOT / f'{report_name}.md.tmp'
    md.write_text('\n'.join(lines))
    md.replace(ROOT / f'{report_name}.md')
    return len(runs) == 2 and all(r['exit_code'] is not None for r in runs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--budget', type=int, choices=(100, 50, 20), default=100)
    args = parser.parse_args()
    STATE.mkdir(exist_ok=True)
    lock_name = 'watch.lock' if args.budget == 100 else f'object{args.budget}_watch.lock'
    with (STATE / lock_name).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.monotonic() + 24 * 3600
        while True:
            complete = collect(args.budget)
            if complete or not args.watch or time.monotonic() >= deadline:
                break
            time.sleep(60)


if __name__ == '__main__':
    main()
