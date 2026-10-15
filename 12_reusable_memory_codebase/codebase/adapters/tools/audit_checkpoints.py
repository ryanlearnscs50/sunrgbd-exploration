#!/usr/bin/env python3
"""Read-only audit of completed run_sequence model and optimizer artifacts.

Run with the isolated host interpreter on trusted local checkpoints (saved
metadata can require MMEngine while unpickling).
This checks stage transfer and finite saved state, not detector accuracy,
training equivalence, dataset fidelity or exact interrupted-training resume.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.checkpoints import expand_checkpoint, validate_checkpoint_state
from ldmr_unidet3d.sequence import stage_directory, verify_stage, sha256, verify_hashes, stage_operations


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_tree(value, path='optimizer'):
    """Inspect nested optimizer tensors and scalar values without modifying them."""
    import math
    if isinstance(value, torch.Tensor):
        require(bool(torch.isfinite(value).all()), f'nonfinite {path}')
    elif isinstance(value, dict):
        for key, child in value.items():
            finite_tree(child, f'{path}.{key}')
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            finite_tree(child, f'{path}[{index}]')
    elif isinstance(value, float):
        require(math.isfinite(value), f'nonfinite {path}')


def audit_run(directory):
    directory = Path(directory)
    arguments = json.loads((directory / 'arguments.json').read_text())
    status = json.loads((directory / 'status.json').read_text())
    protocol = ScanNet35Protocol(arguments['stage_setting'])
    last = arguments['last_stage'] or len(protocol.stages)
    protocol.seen_count(last)
    require(status['status'] == 'complete', 'sequence has not completed')
    require(status['protocol_id'] == protocol.identity, 'sequence protocol mismatch')
    mode = arguments['memory_mode']
    require(mode in ('none', 'scene', 'object') and status['memory_mode'] == mode,
            'sequence memory mode mismatch')
    require(status['completed_stages'] == list(range(1, last + 1)),
            'completed stages differ from requested stages')
    expected_labels = [f's{stage}_{operation}' for stage in range(1, last + 1)
                       for operation in stage_operations(stage, mode, arguments.get('pseudo', False))]
    require([entry['label'] for entry in status['commands']] == expected_labels,
            'sequence command inventory is incomplete or out of order')
    require(all(entry.get('returncode') == 0 for entry in status['commands']),
            'a sequence command did not succeed')
    stages, previous = [], None
    for stage in range(1, last + 1):
        if 'artifact_hashes' in status or status.get('inherited_stages'):
            verify_stage(directory, status, stage, arguments['epochs'], mode, len(protocol.stages), arguments.get('pseudo', False), arguments.get('ld_epochs'), arguments.get('reviewing', False))
        training = stage_directory(directory, stage, status) / 'training'
        path = training / f'epoch_{arguments["epochs"]}.pth'
        checkpoint = torch.load(path, map_location='cpu')
        # Schema consistency against preceding-stage expansion is checked below;
        # this does not build the host model to prove its complete architecture.
        validate_checkpoint_state(checkpoint, checkpoint.get('state_dict', {}), protocol, stage)
        require(checkpoint['meta'].get('epoch') == arguments['epochs'],
                f'stage {stage} did not save the requested epoch')
        iterations = checkpoint['meta'].get('iter')
        require(type(iterations) is int and iterations > 0,
                f'stage {stage} has no optimizer iterations')
        optimizer = checkpoint.get('optimizer')
        require(isinstance(optimizer, dict) and bool(optimizer.get('state'))
                and bool(optimizer.get('param_groups')), f'stage {stage} has no optimizer state')
        finite_tree(optimizer)
        entry = dict(stage=stage, epoch=arguments['epochs'], iterations=iterations,
                     checkpoint_sha256=sha256(path),
                     finite_model_and_optimizer=True)
        if previous is not None:
            initialized = torch.load(training / f'initialized_stage_{stage}.pth', map_location='cpu')
            validate_checkpoint_state(initialized, checkpoint['state_dict'], protocol, stage)
            require('optimizer' not in initialized, 'stage initialization contains old optimizer state')
            expected = expand_checkpoint(previous, initialized['state_dict'], protocol, stage)
            require(all(torch.equal(value, initialized['state_dict'][name])
                        for name, value in expected['state_dict'].items()),
                    f'stage {stage} transfer changed learned weights or classifier rows')
            receipt_path = training / 'loaded_initialization.json'
            require(receipt_path.is_file(), f'stage {stage} lacks actual loaded initialization evidence')
            receipt = json.loads(receipt_path.read_text())
            require(receipt.get('loaded_state_exact') is True
                    and receipt.get('stage') == stage
                    and receipt.get('protocol_id') == protocol.identity
                    and receipt.get('matched_tensors') == len(initialized['state_dict'])
                    and receipt.get('checkpoint_sha256') == sha256(training / f'initialized_stage_{stage}.pth'),
                    f'stage {stage} loaded initialization evidence mismatch')
            entry['learned_weight_transfer_exact'] = True
            entry['loaded_initialization_verified'] = True
        stages.append(entry)
        previous = checkpoint
    source_hashes = json.loads((directory / 'source_hashes.json').read_text())
    require(bool(source_hashes), 'empty source manifest')
    changed = [name for name, digest in source_hashes.items()
               if not Path(name).is_file() or sha256(name) != digest]
    inputs_path = directory / 'input_hashes.json'
    if inputs_path.is_file():
        verify_hashes(json.loads(inputs_path.read_text()), 'input data')
    return dict(scope='Saved checkpoint/optimizer and stage-transfer audit; not accuracy validation',
                run=str(directory.resolve()), protocol_id=protocol.identity,
                memory_mode=mode, stages=stages,
                input_files_verified=inputs_path.is_file(),
                source_files_match_now=not changed, changed_or_missing_sources=changed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--require-source-match', action='store_true',
                        help='fail if original source paths no longer match their recorded hashes')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refusing to overwrite evidence')
    report = audit_run(args.run)
    if args.require_source_match:
        require(report['source_files_match_now'], 'recorded source files changed or are missing')
    # Audit completes before publication; exclusive creation preserves old evidence.
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(f'Checked {len(report["stages"])} stages; source match: {report["source_files_match_now"]}')


if __name__ == '__main__':
    main()
