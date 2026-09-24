"""Monitor the corrected pair; stop only its training sessions by 16:45 SGT."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from run_week3_overnight import verify_complete

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'week3_runs/cutoff_20260924'
MODES = ('pseudo_cosine_v2', 'dose25_cosine_v2')
DEADLINE = datetime.fromisoformat('2026-09-24T16:45:00+08:00')

def stamp():
    return datetime.now(timezone.utc).isoformat()

def lr_evidence(run):
    result = {}
    for stage in range(2, 6):
        rates = []
        for log in (run / f'checkpoints/stage_{stage}').glob('*.log.json'):
            for line in log.read_text().splitlines():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue  # The trainer may be appending the final line.
                if row.get('mode') == 'train' and 'lr' in row:
                    rates.append(row['lr'])
        if rates:
            result[str(stage)] = dict(first=rates[0], last=rates[-1], count=len(rates),
                                     decaying=min(rates) < max(rates))
    return result

def main():
    STATE.mkdir(exist_ok=True)
    with (STATE / 'monitor.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifests = {m: json.loads((ROOT / 'week3_runs' / m / 'manifest.json').read_text()) for m in MODES}
        while True:
            rows = {}
            for mode, manifest in manifests.items():
                state = ROOT / 'week3_runs' / mode
                exits = state / 'exit_code'
                code = int(exits.read_text()) if exits.exists() else None
                runs = list(Path(manifest['work_dir_stem']).parent.glob(Path(manifest['work_dir_stem']).name + '_*'))
                evidence = lr_evidence(runs[0]) if len(runs) == 1 else {}
                rows[mode] = dict(exit_code=code, lr=evidence)
                if code == 0:
                    try:
                        verify_complete(state)
                        if len(evidence) != 4 or not all(x['decaying'] for x in evidence.values()):
                            raise RuntimeError('Completed without verified stage-wise LR decay')
                        if mode.startswith('dose25') and not (runs[0] / 'object_memory_bank/object_memory_bank_stage_5.pkl').exists():
                            raise RuntimeError('Missing final object bank')
                    except Exception as exc:
                        rows[mode]['validation_error'] = str(exc)
            done = all(row['exit_code'] is not None for row in rows.values())
            deadline = datetime.now(timezone.utc) >= DEADLINE
            if deadline and not done:
                for mode, row in rows.items():
                    if row['exit_code'] is not None:
                        continue
                    state = ROOT / 'week3_runs' / mode
                    pid = int((state / 'trainer.pid').read_text())
                    path = Path(f'/proc/{pid}/cmdline')
                    cmd = path.read_bytes() if path.exists() else b''
                    if b'train_incremental_scene.py' in cmd and manifests[mode]['config'].encode() in cmd:
                        sid = manifests[mode]['supervisor_pid']
                        assert os.getsid(pid) == sid and os.getpgid(pid) == sid
                        (state / 'cutoff_stop.json').write_text(json.dumps(dict(time=stamp(),reason='16:45 SGT writeup cutoff'))+'\n')
                        os.kill(pid, signal.SIGTERM)
                        time.sleep(5)
                        # Only this launched session; remove any remaining loader workers.
                        try:
                            os.killpg(sid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        time.sleep(2)
                        try:
                            os.killpg(sid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        if not (state / 'exit_code').exists():
                            (state / 'exit_code').write_text('143\n')
                continue
            status = dict(updated_at=stamp(),deadline=DEADLINE.isoformat(),runs=rows,
                          state='finished' if done else 'monitoring')
            (STATE / 'status.json').write_text(json.dumps(status,indent=2)+'\n')
            subprocess.run([sys.executable,str(ROOT/'collect_week3_status.py')],cwd=ROOT,
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            if done:
                (STATE / 'exit_code').write_text('0\n' if all(r['exit_code']==0 and 'validation_error' not in r for r in rows.values()) else '1\n')
                print(stamp(),'Pair ended; monitoring stopped.',flush=True)
                return
            time.sleep(30)

if __name__ == '__main__':
    main()
