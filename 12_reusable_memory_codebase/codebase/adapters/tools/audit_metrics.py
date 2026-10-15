#!/usr/bin/env python3
"""Audit completed host validation logs without importing a detector runtime.

Strict full-class coverage audit: undefined AP is an error here, even though the
standalone evaluator can serialize it as null. Run audit_checkpoints.py separately
to verify saved weights, source/data integrity and stage transfers.
"""
import argparse
import ast
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.sequence import sha256, stage_directory


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_numbers(value, path):
    if isinstance(value, dict):
        for key, child in value.items():
            finite_numbers(child, f'{path}.{key}')
    elif isinstance(value, list):
        for index, child in enumerate(value):
            finite_numbers(child, f'{path}[{index}]')
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), f'nonfinite scalar: {path}')


def validation_records(text, names, epochs, scenes, interval=1):
    require(type(interval) is int and interval > 0 and epochs % interval == 0,
            'invalid validation interval')
    records = []
    pattern = re.compile(r'Epoch\(val\)\s+\[(\d+)\]\[(\d+)/(\d+)\]\s+scannet:\s*(\{.*\})')
    for line in text.splitlines():
        if 'Epoch(val)' not in line or 'scannet:' not in line:
            continue
        match = pattern.search(line)
        require(match is not None, 'malformed validation record')
        epoch, done, total = map(int, match.group(1, 2, 3))
        require(done == total == scenes, 'validation scene count mismatch')
        metrics = ast.literal_eval(match.group(4))
        require(isinstance(metrics, dict) and bool(metrics), 'empty validation metrics')
        require(all(isinstance(k, str) and type(v) in (int, float)
                    and math.isfinite(v) and 0 <= v <= 1
                    for k, v in metrics.items()), 'invalid validation metric')
        for threshold in ('0.25', '0.50'):
            for suffix, mean_key in ((f'_AP_{threshold}', f'mAP_{threshold}'),
                                     (f'_rec_{threshold}', f'mAR_{threshold}')):
                found = {k[:-len(suffix)] for k in metrics if k.endswith(suffix)}
                require(found == set(names), f'class coverage mismatch: {suffix}')
                mean = sum(metrics[n + suffix] for n in names) / len(names)
                require(mean_key in metrics and abs(mean - metrics[mean_key]) < 1e-7,
                        f'inconsistent mean: {mean_key}')
        records.append(dict(epoch=epoch, validation_scenes=scenes, metrics=metrics))
    require([r['epoch'] for r in records] == list(range(interval, epochs + 1, interval)),
            'missing, duplicate or out-of-order validation epochs')
    return records


def training_logs(training):
    """Select the runner session owning scalar history, excluding collector logs."""
    training = Path(training)
    scalars = list(training.glob('*/vis_data/scalars.json'))
    require(len(scalars) == 1, f'{training}: ambiguous or missing scalar sessions')
    session = scalars[0].parent.parent
    logs = list(session.glob('*.log'))
    require(len(logs) == 1, f'{session}: ambiguous or missing runner logs')
    return logs[0], scalars[0]


def audit_run(run, validation_scenes):
    require(type(validation_scenes) is int and validation_scenes > 0,
            'validation scenes must be a positive integer')
    run = Path(run)
    args = json.loads((run / 'arguments.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    protocol = ScanNet35Protocol(args['stage_setting'])
    last = args['last_stage'] or len(protocol.stages)
    protocol.seen_count(last)
    require(type(args['epochs']) is int and args['epochs'] > 0, 'invalid epoch count')
    require(status['status'] == 'complete' and status['protocol_id'] == protocol.identity
            and status['memory_mode'] == args['memory_mode'], 'sequence not complete or mismatched')
    require(status['completed_stages'] == list(range(1, last + 1)), 'incomplete stage inventory')
    require(bool(status['commands']) and all(c.get('returncode') == 0 for c in status['commands']),
            'unsuccessful sequence commands')
    stages, hashes = [], {}
    for stage in range(1, last + 1):
        training = stage_directory(run, stage, status) / 'training'
        log, scalar = training_logs(training)
        logs, scalars = [log], [scalar]
        records = validation_records(logs[0].read_text(), protocol.names[:protocol.seen_count(stage)],
                                     args['epochs'], validation_scenes,
                                     args.get('validation_interval', 1))
        lines = scalars[0].read_text().splitlines()
        require(bool(lines), 'empty scalar history')
        for number, line in enumerate(lines, 1):
            row = json.loads(line)
            require(isinstance(row, dict) and bool(row), f'invalid scalar row {number}')
            finite_numbers(row, f'{scalars[0]}:{number}')
        for path in logs + scalars:
            hashes[str(path.resolve())] = sha256(path)
        stages.append(dict(stage=stage, validations=records, scalar_rows=len(lines)))
    return dict(scope='Validation coverage, means and finite scalar history; not checkpoint or accuracy parity validation',
                run=str(run.resolve()), protocol_id=protocol.identity,
                memory_mode=args['memory_mode'], stages=stages, evidence_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--validation-scenes', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refusing to overwrite evidence')
    report = audit_run(args.run, args.validation_scenes)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Validated metrics for {len(report["stages"])} stages')


if __name__ == '__main__':
    main()
