"""POSIX subprocess ownership for interruptible training supervisors."""
from contextlib import contextmanager
import os
import signal
import subprocess
import time


class SequenceInterrupted(SystemExit):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(128 + signum)


@contextmanager
def termination_handler():
    """Let the supervisor's existing finally block record SIGTERM failure."""
    def interrupt(signum, frame):
        raise SequenceInterrupted(signum)
    previous = signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _stop_group(process, grace_seconds):
    """Terminate only the new child session, including workers/collectors."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader without assuming descendants exited.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_managed(command, *, stdout, env, on_started, grace_seconds=5):
    """Wait for a child, cleaning its group before propagating interruption.

    Signals are deferred during Popen so interruption cannot lose ownership
    between creating a child and recording its PID. Repeated signals cannot
    interrupt cleanup. Use in the main thread of a POSIX supervisor.
    """
    pending = []
    def defer(signum, frame):
        pending.append(signum)
    previous = {sig: signal.signal(sig, defer) for sig in (signal.SIGINT, signal.SIGTERM)}
    process = None
    try:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stdout,
                                   stderr=subprocess.STDOUT, env=env, start_new_session=True)
        on_started(process.pid)
        while True:
            if pending:
                raise SequenceInterrupted(pending[0])
            try:
                code = process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                continue
            if pending:
                raise SequenceInterrupted(pending[0])
            return code
    except BaseException:
        if process is not None:
            _stop_group(process, grace_seconds)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
