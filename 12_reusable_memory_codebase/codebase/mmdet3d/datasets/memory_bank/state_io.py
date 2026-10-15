"""Atomic writes for trusted local memory-bank artifacts."""

import os
from pathlib import Path
import tempfile


def atomic_write(path, write, binary=False):
    """Replace one artifact only after serialization and fsync succeed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb' if binary else 'w') as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
