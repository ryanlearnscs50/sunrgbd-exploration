#!/usr/bin/env python3
"""Check resolved control/pseudo configurations; no detector plugins required.

Use the host interpreter for MMEngine. Checkpoints, logs and source/input
integrity require separate audits; configuration equality alone is insufficient.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmr_unidet3d.paired_audit import compare_configs, read_config
from ldmr_unidet3d.sequence import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True, type=Path)
    parser.add_argument('--pseudo', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refusing to overwrite evidence')
    paths = [args.control.resolve(), args.pseudo.resolve()]
    hashes = {str(p): sha256(p) for p in paths}
    report = compare_configs(*(read_config(p) for p in paths))
    if hashes != {str(p): sha256(p) for p in paths}:
        raise ValueError('configuration changed during audit')
    report['config_sha256'] = hashes
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print('Paired configuration audit passed')


if __name__ == '__main__':
    main()
