#!/usr/bin/env python3
"""Public training entry point for LDMR scene or object memory.

The historical train_incremental_scene.py entry point remains supported.
"""

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.train_incremental_scene import main


if __name__ == '__main__':
    main()
