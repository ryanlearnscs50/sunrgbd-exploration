"""Integrity and read-only inheritance for completed incremental stages.

No detector/framework imports. Continuation starts a fresh next-stage process;
it does not recover a partial stage or its optimizer/RNG state.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_hashes(hashes, description):
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError(f'missing {description} integrity manifest')
    for name, digest in hashes.items():
        path = Path(name)
        if not path.is_file() or sha256(path) != digest:
            raise ValueError(f'{description} changed or missing: {name}')


def input_hashes(manifests, datasets):
    """Hash manifests and all referenced point/mask files with bounded buffers."""
    paths = {Path(path).absolute() for path in manifests}
    for records in datasets:
        for record in records:
            paths.add(Path(record['lidar_points']['lidar_path']).absolute())
            for key in ('pts_instance_mask_path', 'pts_semantic_mask_path', 'super_pts_path'):
                paths.add(Path(record[key]).absolute())
    return {str(path): sha256(path) for path in sorted(paths)}


def stage_directory(run, stage, status):
    """Inherited stages remain at their original absolute paths, without copies."""
    inherited = status.get('inherited_stages', {})
    if str(stage) in inherited:
        path = Path(inherited[str(stage)])
        if not path.is_absolute():
            raise ValueError('inherited stage path must be absolute')
        return path
    return Path(run).absolute() / f'stage_{stage}'


def stage_operations(stage, mode, pseudo=False):
    return (["train_infos", "val_infos"]
            + (["pseudo"] if pseudo and stage > 1 else [])
            + ["training"] + (["bank"] if mode != "none" else []))


def stage_files(directory, stage, epochs, mode, total_stages, pseudo=False, ld_epochs=None, reviewing=False):
    """Required checkpoint, config, metadata and replay files for a stage."""
    directory = Path(directory)
    files = [directory / 'train.pkl', directory / 'val.pkl',
             directory / 'training' / f'epoch_{epochs}.pth',
             directory / 'training' / 'resolved_stage.py']
    if pseudo and stage > 1:
        files.append(directory / 'pseudo.json')
    if stage > 1:
        files += [directory / 'training' / f'initialized_stage_{stage}.pth',
                  directory / 'training' / 'loaded_initialization.json']
    if mode != 'none':
        files += [directory / 'bank' / 'bank.pkl', directory / 'bank' / 'provenance.json']
        if mode == 'object':
            files.append(directory / 'bank' / 'crops.pkl')
        elif stage < total_stages:
            files.append(directory / 'bank' / 'replay_records.pkl')
    if ld_epochs is not None:
        collection = directory / 'training' / 'ld_collection'
        files += [directory / 'bank' / 'selection.json', collection / 'complete.json',
                  collection / 'context.json', collection / 'scores.json']
        for epoch in ld_epochs:
            files += [collection / f'snapshot_{epoch}.json',
                      directory / 'training' / 'ld_checkpoints' / f'epoch_{epoch}.pth']
    if reviewing:
        from .reviewing import review_boundaries
        review_dir = directory / 'training' / 'reviewing'
        files += [review_dir / f'epoch_{e}_sampling.json' for e in range(1, epochs + 1)]
        if stage > 1:
            boundaries = review_boundaries(epochs)
            for k, epoch in enumerate(boundaries[1:-1], 1):
                collection = review_dir / f'review_{k}'
                files += [review_dir / f'review_{k}.json', review_dir / f'review_{k}.log']
                files += [collection / name for name in ('complete.json', 'context.json', 'scores.json')]
                files += [collection / f'snapshot_{e}.json' for e in boundaries[:k + 1]]
    return files


def seal_stage(directory, stage, epochs, mode, total_stages, pseudo=False, ld_epochs=None, reviewing=False):
    return {str(path.absolute()): sha256(path)
            for path in stage_files(directory, stage, epochs, mode, total_stages, pseudo, ld_epochs, reviewing)}


def verify_stage(run, status, stage, epochs, mode, total_stages, pseudo=False, ld_epochs=None, reviewing=False):
    recorded = status.get('artifact_hashes', {}).get(str(stage))
    expected = {str(path.absolute()) for path in stage_files(
        stage_directory(run, stage, status), stage, epochs, mode, total_stages, pseudo, ld_epochs, reviewing)}
    if not isinstance(recorded, dict) or set(recorded) != expected:
        raise ValueError(f'stage {stage} has no complete artifact integrity manifest')
    verify_hashes(recorded, f'stage {stage} artifact')


def continuation(previous_run, arguments, sources, inputs, protocol, last_stage):
    """Return an inherited prefix only after settings and artifact checks pass.

    The caller must supply a fresh output directory. Legacy sequences without
    integrity manifests are intentionally not eligible for automatic recovery.
    """
    previous_run = Path(previous_run).absolute()
    old_arguments = json.loads((previous_run / 'arguments.json').read_text())
    status = json.loads((previous_run / 'status.json').read_text())
    if status.get('status') not in ('failed', 'complete'):
        raise ValueError('continuation requires a stopped failed or complete sequence')
    if status.get('protocol_id') != protocol.identity:
        raise ValueError('continuation protocol mismatch')
    mode = arguments['memory_mode']
    if status.get('memory_mode') != mode:
        raise ValueError('continuation memory mode mismatch')
    keys = ('host_python', 'ldmr_python', 'upstream_root', 'train_info', 'val_info',
            'stage_setting', 'memory_mode', 'epochs', 'seed', 'batch_size', 'workers',
            'scene_budget', 'objects_per_class', 'max_objects')
    for key in keys:
        if key not in old_arguments or str(old_arguments[key]) != str(arguments[key]):
            raise ValueError(f'continuation setting changed: {key}')
    for key, default in (('target_mode', 'masks'), ('pseudo', False), ('ld_epochs', None), ('reviewing', False), ('recipe', 'custom'),
                         ('object_policy', 'ldmr_weighted'), ('object_selector', 'random'),
                         ('min_points', 5), ('validation_interval', 1)):
        if old_arguments.get(key, default) != arguments.get(key, default):
            raise ValueError(f'continuation setting changed: {key}')
    pseudo = arguments.get('pseudo', False)
    old_sources = json.loads((previous_run / 'source_hashes.json').read_text())
    if old_sources != sources:
        raise ValueError('continuation requires identical source paths and hashes')
    verify_hashes(sources, 'source')
    if not (previous_run / 'input_hashes.json').is_file():
        raise ValueError('sequence predates input integrity manifests; automatic continuation is unsupported')
    old_inputs = json.loads((previous_run / 'input_hashes.json').read_text())
    if not inputs or old_inputs != inputs:
        raise ValueError('continuation input paths or contents changed')
    # The caller has just computed `inputs` from the actual files.
    stages = status.get('completed_stages')
    if (not isinstance(stages, list) or not stages
            or any(type(stage) is not int for stage in stages)
            or stages != list(range(1, len(stages) + 1)) or stages[-1] >= last_stage):
        raise ValueError('continuation needs a contiguous completed prefix before last stage')
    labels = [f's{stage}_{operation}' for stage in stages
              for operation in stage_operations(stage, mode, pseudo)]
    commands = status.get('commands', [])[:len(labels)]
    if ([entry.get('label') for entry in commands] != labels
            or any(entry.get('returncode') != 0 for entry in commands)):
        raise ValueError('completed prefix has missing or unsuccessful commands')
    inherited, artifacts = {}, {}
    for stage in stages:
        verify_stage(previous_run, status, stage, arguments['epochs'], mode, len(protocol.stages), pseudo, arguments.get('ld_epochs'), arguments.get('reviewing', False))
        inherited[str(stage)] = str(stage_directory(previous_run, stage, status))
        artifacts[str(stage)] = deepcopy(status['artifact_hashes'][str(stage)])
    return dict(completed_stages=list(stages), commands=deepcopy(commands),
                inherited_stages=inherited, artifact_hashes=artifacts,
                continued_from=str(previous_run))
