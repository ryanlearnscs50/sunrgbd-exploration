"""Recovery trusts verified completed artifacts, never a success flag alone."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.sequence import (continuation, input_hashes, seal_stage,
    sha256, stage_directory, stage_files, verify_stage)


@pytest.fixture(params=['none', 'scene', 'object'])
def previous(tmp_path, request):
    mode = request.param
    protocol = ScanNet35Protocol()
    arguments = dict(host_python='/host/python', ldmr_python='/legacy/python',
        upstream_root='/upstream', train_info='/train.pkl', val_info='/val.pkl',
        stage_setting=protocol.stage_setting, memory_mode=mode, epochs=1,
        seed=200, batch_size=1, workers=0, scene_budget=4,
        objects_per_class=2, max_objects=70, last_stage=1, work_dir=str(tmp_path))
    source = tmp_path / 'source.py'
    source.write_text('# unchanged implementation\n')
    sources = {str(source): sha256(source)}
    inputs = {'/train.pkl': 'training-digest', '/val.pkl': 'validation-digest'}
    folder = tmp_path / 'stage_1'
    for file in stage_files(folder, 1, 1, mode, 5):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'original artifact')
    operations = ['train_infos', 'val_infos', 'training'] + (['bank'] if mode != 'none' else [])
    status = dict(status='complete', protocol_id=protocol.identity, memory_mode=mode,
        completed_stages=[1], commands=[dict(label=f's1_{op}', returncode=0) for op in operations],
        artifact_hashes={'1': seal_stage(folder, 1, 1, mode, 5)}, inherited_stages={})
    for name, value in [('arguments', arguments), ('status', status),
                        ('source_hashes', sources), ('input_hashes', inputs)]:
        (tmp_path / (name + '.json')).write_text(json.dumps(value))
    return tmp_path, arguments, sources, inputs, protocol


def recover(previous):
    return continuation(*previous, last_stage=2)


def update_status(path, operation):
    status = json.loads((path / 'status.json').read_text())
    operation(status)
    (path / 'status.json').write_text(json.dumps(status))


def test_continuation_reuses_only_completed_prefix_without_writes(previous):
    path, arguments, *_ = previous
    update_status(path, lambda state: state.update(status='failed', commands=state['commands'] + [
        dict(label='s2_train_infos', returncode=0), dict(label='s2_training', returncode=1)]))
    before = {str(p): p.read_bytes() for p in path.rglob('*') if p.is_file()}
    recovered = recover(previous)
    assert recovered['completed_stages'] == [1]
    assert all(command['label'].startswith('s1_') for command in recovered['commands'])
    assert recovered['inherited_stages'] == {'1': str(path / 'stage_1')}
    assert stage_directory('/new/run', 1, recovered) == path / 'stage_1'
    assert stage_directory('/new/run', 2, recovered) == Path('/new/run/stage_2')
    assert before == {str(p): p.read_bytes() for p in path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage', ['active', 'empty_prefix', 'noncontiguous',
    'bool_stage', 'protocol', 'mode', 'failed_command', 'missing_command',
    'missing_artifact_hash', 'checkpoint_changed', 'source_changed', 'source_path',
    'inputs_changed', 'old_format', 'settings', 'no_new_stages'])
def test_recovery_rejects_mismatched_or_incomplete_runs(previous, damage):
    path, arguments, sources, inputs, protocol = previous
    if damage == 'checkpoint_changed':
        (path / 'stage_1/training/epoch_1.pth').write_bytes(b'corrupted')
    elif damage == 'source_changed':
        (path / 'source.py').write_text('# changed\n')
    elif damage == 'source_path':
        sources['/unexpected.py'] = 'digest'
    elif damage == 'inputs_changed':
        inputs['/train.pkl'] = 'different'
    elif damage == 'old_format':
        (path / 'input_hashes.json').unlink()
    elif damage == 'settings':
        arguments['seed'] += 1
    elif damage == 'no_new_stages':
        with pytest.raises(ValueError, match='prefix'):
            continuation(*previous, last_stage=1)
        return
    else:
        def change(state):
            if damage == 'active': state['status'] = 'running'
            elif damage == 'empty_prefix': state['completed_stages'] = []
            elif damage == 'noncontiguous': state['completed_stages'] = [2]
            elif damage == 'bool_stage': state['completed_stages'] = [True]
            elif damage == 'protocol': state['protocol_id'] = 'unrelated'
            elif damage == 'mode': state['memory_mode'] = 'unrelated'
            elif damage == 'failed_command': state['commands'][0]['returncode'] = 1
            elif damage == 'missing_command': state['commands'].pop()
            elif damage == 'missing_artifact_hash': state['artifact_hashes']['1'].popitem()
        update_status(path, change)
    with pytest.raises(ValueError):
        recover(previous)


def test_chained_continuation_retains_original_artifact_paths(previous, tmp_path):
    original, arguments, sources, inputs, protocol = previous
    inherited = recover(previous)
    second = tmp_path / 'second_run'
    second.mkdir()
    inherited.update(status='failed', protocol_id=protocol.identity, memory_mode=arguments['memory_mode'])
    for name, value in [('status', inherited), ('arguments', arguments),
                        ('source_hashes', sources), ('input_hashes', inputs)]:
        (second / (name + '.json')).write_text(json.dumps(value))
    chained = continuation(second, arguments, sources, inputs, protocol, 2)
    assert chained['inherited_stages']['1'] == str(original / 'stage_1')
    assert chained['continued_from'] == str(second)
    assert chained['artifact_hashes'] == inherited['artifact_hashes']


def test_all_referenced_data_content_is_hashed_once(tmp_path):
    paths = [tmp_path / f'{i}.bin' for i in range(5)]
    for i, path in enumerate(paths): path.write_bytes(bytes([i]))
    record = dict(lidar_points=dict(lidar_path=str(paths[1])),
                  pts_instance_mask_path=str(paths[2]),
                  pts_semantic_mask_path=str(paths[3]), super_pts_path=str(paths[4]))
    before = input_hashes([paths[0]], [[record, copy.deepcopy(record)]])
    assert len(before) == 5
    paths[4].write_bytes(b'different superpoint groups')
    after = input_hashes([paths[0]], [[record]])
    assert before[str(paths[4])] != after[str(paths[4])]
    assert all(before[str(p)] == after[str(p)] for p in paths[:4])


def test_relative_inherited_paths_are_rejected():
    with pytest.raises(ValueError, match='absolute'):
        stage_directory('/run', 1, {'inherited_stages': {'1': '../old/stage_1'}})


@pytest.mark.parametrize('key,value', [('target_mode', 'boxes'), ('pseudo', True)])
def test_continuation_rejects_supervision_change(previous, key, value):
    previous[1][key] = value
    with pytest.raises(ValueError, match=key):
        recover(previous)


def test_pseudo_artifacts_and_command_are_required_for_recovery(previous):
    from ldmr_unidet3d.sequence import stage_operations
    path, arguments, sources, inputs, protocol = previous
    mode = arguments['memory_mode']
    arguments.update(target_mode='boxes', pseudo=True)
    (path / 'arguments.json').write_text(json.dumps(arguments))
    folder = path / 'stage_2'
    for file in stage_files(folder, 2, 1, mode, 5, pseudo=True):
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b'stage two')
    status = json.loads((path / 'status.json').read_text())
    status['completed_stages'].append(2)
    status['commands'] += [dict(label=f's2_{op}', returncode=0)
                           for op in stage_operations(2, mode, True)]
    status['artifact_hashes']['2'] = seal_stage(folder, 2, 1, mode, 5, True)
    (path / 'status.json').write_text(json.dumps(status))
    recovered = continuation(*previous, last_stage=3)
    assert recovered['completed_stages'] == [1, 2]
    (folder / 'pseudo.json').write_bytes(b'corrupt teacher predictions')
    with pytest.raises(ValueError, match='artifact changed'):
        continuation(*previous, last_stage=3)
    (folder / 'pseudo.json').write_bytes(b'stage two')
    status['commands'] = [e for e in status['commands'] if e['label'] != 's2_pseudo']
    (path / 'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError, match='commands'):
        continuation(*previous, last_stage=3)


def test_ld_stage_seal_requires_measurements_and_selection(tmp_path):
    stage = tmp_path / 'stage_1'
    for path in stage_files(stage, 1, 2, 'scene', 5, ld_epochs=[0, 1, 2]):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'original')
    status = dict(artifact_hashes={'1':seal_stage(stage, 1, 2, 'scene', 5, ld_epochs=[0, 1, 2])})
    verify_stage(tmp_path, status, 1, 2, 'scene', 5, ld_epochs=[0, 1, 2])
    (stage / 'training/ld_collection/snapshot_1.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed or missing'):
        verify_stage(tmp_path, status, 1, 2, 'scene', 5, ld_epochs=[0, 1, 2])


def test_continuation_cannot_switch_random_run_to_ld(previous):
    path, arguments, sources, inputs, protocol = previous
    arguments = dict(arguments, ld_epochs=[0,1])
    with pytest.raises(ValueError, match='ld_epochs'):
        continuation(path, arguments, sources, inputs, protocol, last_stage=2)
