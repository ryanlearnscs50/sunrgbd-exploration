"""Detached, bounded replication queue; stop on failure and preserve every run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

from collect_week3_status import collect
from summarize_week3_replication import summarize

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'week3_runs/overnight_20260924'
SEEDS = (202, 203)
POLICIES = ('pseudo_only', 'dose25')


def stamp():
    return datetime.now(timezone.utc).isoformat()


def verify_complete(state):
    manifest = json.loads((state / 'manifest.json').read_text())
    stem = Path(manifest['work_dir_stem'])
    runs = list(stem.parent.glob(stem.name + '_*'))
    if len(runs) != 1:
        raise RuntimeError(f'Ambiguous or missing run: {state}')
    run = runs[0]
    for stage in range(2, 6):
        data = json.loads((run / f'memory_bank/scores/stage_{stage}_metrics.json').read_text())
        if data['evaluated_at_stage'] != stage or len(data['classes']) != 8 * stage:
            raise RuntimeError(f'Incomplete metric scope: {run}, stage {stage}')
    checkpoint = run / 'checkpoints/stage_5/epoch_1.pth'
    if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
        raise RuntimeError(f'Missing final checkpoint: {run}')
    if manifest['mode'] == 'dose25' and not (run / 'object_memory_bank/object_memory_bank_stage_5.pkl').is_file():
        raise RuntimeError(f'Missing final object bank: {run}')
    if 'INCREMENTAL TRAINING COMPLETED' not in (state / 'console.log').read_text():
        raise RuntimeError(f'Missing training completion marker: {state}')


def main():
    STATE.mkdir(exist_ok=True)
    with (STATE / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (STATE / 'started_at').exists():
            raise RuntimeError('Queue already started; inspect artifacts, do not restart blindly.')
        (STATE / 'started_at').write_text(stamp())
        (STATE / 'pid').write_text(str(os.getpid()))
        deadline = time.monotonic() + 8 * 3600
        snapshot = json.loads((STATE / 'source_hashes.json').read_text())
        result = dict(started_at=stamp(), state='running', seeds=list(SEEDS), policies=list(POLICIES))
        code = 1
        try:
            for seed in SEEDS:
                if time.monotonic() > deadline - 3 * 3600:
                    raise RuntimeError('Insufficient overnight window to start another pair.')
                for path, expected in snapshot.items():
                    if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
                        raise RuntimeError(f'Source changed since plan: {path}')
                print(f'{stamp()} Launching matched seed {seed}', flush=True)
                subprocess.run([sys.executable, str(ROOT / 'launch_week3_experiments.py'),
                                '--modes', *POLICIES, '--seed', str(seed)], cwd=ROOT, check=True)
                states = [ROOT / 'week3_runs' / f'{mode}_s{seed}' for mode in POLICIES]
                result.update(active_seed=seed)
                (STATE / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
                while True:
                    collect()
                    summarize()
                    exits = [(int((s / 'exit_code').read_text()) if (s / 'exit_code').exists() else None)
                             for s in states]
                    if any(c is not None and c != 0 for c in exits):
                        raise RuntimeError(f'Seed {seed} failed: {exits}; no further runs queued.')
                    if all(c == 0 for c in exits):
                        for state in states:
                            verify_complete(state)
                        print(f'{stamp()} Verified completion seed {seed}', flush=True)
                        break
                    for state, exit_code in zip(states, exits):
                        if exit_code is None:
                            pid = json.loads((state / 'manifest.json').read_text())['supervisor_pid']
                            cmdline = Path(f'/proc/{pid}/cmdline').read_bytes()
                            if b'run_week3_experiment.sh' not in cmdline:
                                raise RuntimeError(f'Supervisor disappeared: {state}')
                    if time.monotonic() >= deadline:
                        raise RuntimeError('Queue monitoring deadline reached; active trainers remain detached.')
                    time.sleep(60)
            code = 0
            result['state'] = 'completed'
        except Exception as exc:
            result.update(state='stopped', error=str(exc))
            print(f'{stamp()} Queue stopped: {exc}', flush=True)
        finally:
            result['ended_at'] = stamp()
            (STATE / 'status.json').write_text(json.dumps(result, indent=2) + '\n')
            (STATE / 'exit_code').write_text(str(code) + '\n')
            collect()
            summarize()
        return code


if __name__ == '__main__':
    sys.exit(main())
