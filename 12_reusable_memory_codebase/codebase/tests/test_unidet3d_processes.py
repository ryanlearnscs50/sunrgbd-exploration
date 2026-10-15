"""Exercise real supervisor/worker processes; never touch training jobs or GPUs."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ADAPTERS = Path(__file__).resolve().parents[1] / 'adapters'
sys.path.insert(0, str(ADAPTERS))
from ldmr_unidet3d.processes import run_managed


def wait_for(predicate, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError('process did not reach expected state')


def running(pid):
    path = Path(f'/proc/{pid}/stat')
    if not path.exists():
        return False
    return path.read_text().rsplit(')', 1)[1].split()[0] != 'Z'


@pytest.mark.skipif(sys.platform != 'linux', reason='Linux process-tree audit')
@pytest.mark.parametrize('signum', [signal.SIGTERM, signal.SIGINT])
def test_supervisor_interrupt_stops_stubborn_descendants_only(tmp_path, signum):
    worker = tmp_path / 'worker.py'
    worker.write_text('''import json, os, signal, subprocess, sys, time
from pathlib import Path
signal.signal(signal.SIGTERM, signal.SIG_IGN)
child = subprocess.Popen([sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'])
Path('workers.json').write_text(json.dumps([os.getpid(), child.pid]))
Path('ready').touch()
time.sleep(60)
''')
    supervisor = tmp_path / 'supervisor.py'
    supervisor.write_text('''import json, os, sys
from pathlib import Path
from ldmr_unidet3d.processes import run_managed, termination_handler
state = {'status': 'running'}
def started(pid):
    state['pid'] = pid
try:
    with termination_handler(), open('child.log', 'w') as log:
        run_managed([sys.executable, 'worker.py'], stdout=log, env=dict(os.environ),
                    on_started=started, grace_seconds=.3)
    state['status'] = 'complete'
except BaseException:
    state['status'] = 'failed'
    raise
finally:
    Path('status.json').write_text(json.dumps(state))
''')
    env = dict(os.environ, PYTHONPATH=str(ADAPTERS))
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
    process = subprocess.Popen([sys.executable, str(supervisor)], cwd=tmp_path,
                               env=env, start_new_session=True)
    workers = []
    try:
        wait_for(lambda: (tmp_path / 'ready').exists())
        workers = json.loads((tmp_path / 'workers.json').read_text())
        os.kill(process.pid, signum)  # Signal only the supervisor, not its group.
        time.sleep(.1)
        os.kill(process.pid, signum)  # Repeated interruption must not abort cleanup.
        assert process.wait(timeout=10) == 128 + signum
        wait_for(lambda: all(not running(pid) for pid in workers))
        assert unrelated.poll() is None
        state = json.loads((tmp_path / 'status.json').read_text())
        assert state == dict(status='failed', pid=workers[0])
    finally:
        for pid in [process.pid, *workers]:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        unrelated.terminate()
        unrelated.wait()


def test_recording_failure_reaps_already_created_child(tmp_path):
    children = []
    def fail_to_record(pid):
        children.append(pid)
        raise OSError('status disk unavailable')
    with (tmp_path / 'child.log').open('w') as log:
        with pytest.raises(OSError, match='status disk unavailable'):
            run_managed([sys.executable, '-c', 'import time; time.sleep(60)'],
                        stdout=log, env=dict(os.environ), on_started=fail_to_record,
                        grace_seconds=.3)
    assert len(children) == 1 and not running(children[0])
