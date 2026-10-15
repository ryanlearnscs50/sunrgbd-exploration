"""Evidence audits must reject damaged artifacts even after a successful exit."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

TOOL = Path(__file__).resolve().parents[1] / 'adapters/tools/audit_checkpoints.py'
spec = importlib.util.spec_from_file_location('checkpoint_audit', TOOL)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def run(tmp_path):
    protocol = audit.ScanNet35Protocol()
    args = dict(stage_setting=protocol.stage_setting, last_stage=2, epochs=1, memory_mode='none')
    (tmp_path / 'arguments.json').write_text(json.dumps(args))
    status = dict(status='complete', protocol_id=protocol.identity, memory_mode='none',
                  completed_stages=[1, 2], commands=[
                      dict(label=f's{s}_{op}', returncode=0) for s in (1, 2)
                      for op in ('train_infos', 'val_infos', 'training')])
    (tmp_path / 'status.json').write_text(json.dumps(status))
    source = tmp_path / 'source.py'
    source.write_text('# preserved experiment source\n')
    (tmp_path / 'source_hashes.json').write_text(json.dumps({
        str(source): hashlib.sha256(source.read_bytes()).hexdigest()}))
    previous = None
    for stage in (1, 2):
        folder = tmp_path / f'stage_{stage}/training'
        folder.mkdir(parents=True)
        rows = protocol.seen_count(stage) + 1
        state = {'decoder.outs_cls.2.weight': torch.ones(rows, 3),
                 'decoder.outs_cls.2.bias': torch.zeros(rows), 'backbone.weight': torch.ones(2)}
        if previous:
            initialized = audit.expand_checkpoint(previous, state, protocol, stage)
            torch.save(initialized, folder / f'initialized_stage_{stage}.pth')
            state = initialized['state_dict']
            (folder / 'loaded_initialization.json').write_text(json.dumps(dict(
                stage=stage, protocol_id=protocol.identity, loaded_state_exact=True,
                matched_tensors=len(state),
                checkpoint_sha256=audit.sha256(folder / f'initialized_stage_{stage}.pth'))))
        checkpoint = dict(state_dict=state, meta=dict(ldmr_protocol_id=protocol.identity,
                          ldmr_stage=stage, epoch=1, iter=5), optimizer=dict(
                              state={0: dict(step=torch.tensor(5), exp_avg=torch.ones(2))},
                              param_groups=[dict(lr=.001, params=[0])]))
        torch.save(checkpoint, folder / 'epoch_1.pth')
        previous = checkpoint
    return tmp_path


def test_success_and_changed_sources_are_distinct(run):
    report = audit.audit_run(run)
    assert len(report['stages']) == 2
    assert report['stages'][1]['learned_weight_transfer_exact']
    assert report['source_files_match_now']
    (run / 'source.py').write_text('# later edit\n')
    report = audit.audit_run(run)
    assert not report['source_files_match_now']
    assert report['changed_or_missing_sources'] == [str(run / 'source.py')]


def test_audit_resolves_and_verifies_inherited_stage_artifacts(run):
    from ldmr_unidet3d.sequence import seal_stage
    inherited = run / 'original_stage_1'
    (run / 'stage_1').rename(inherited)
    status = json.loads((run / 'status.json').read_text())
    status['inherited_stages'] = {'1': str(inherited)}
    status['artifact_hashes'] = {}
    for stage, directory in [(1, inherited), (2, run / 'stage_2')]:
        (directory / 'train.pkl').write_bytes(b'train')
        (directory / 'val.pkl').write_bytes(b'val')
        (directory / 'training/resolved_stage.py').write_text('# config\n')
        status['artifact_hashes'][str(stage)] = seal_stage(directory, stage, 1, 'none', 5)
    (run / 'status.json').write_text(json.dumps(status))
    assert len(audit.audit_run(run)['stages']) == 2
    (inherited / 'train.pkl').write_bytes(b'changed')
    with pytest.raises(ValueError, match='artifact changed'):
        audit.audit_run(run)


def test_audit_checks_recorded_input_bytes_when_available(run):
    source = run / 'input.bin'
    source.write_bytes(b'original points')
    (run / 'input_hashes.json').write_text(json.dumps({str(source): audit.sha256(source)}))
    assert audit.audit_run(run)['input_files_verified']
    source.write_bytes(b'changed points')
    with pytest.raises(ValueError, match='input data changed'):
        audit.audit_run(run)


@pytest.mark.parametrize('damage', ['unfinished', 'missing_command', 'failed_command',
                                  'missing_stage', 'optimizer_nan', 'optimizer_inf',
                                  'no_optimizer', 'wrong_epoch', 'zero_iterations',
                                  'transfer', 'classifier_transfer', 'old_optimizer'])
def test_success_status_does_not_hide_damaged_artifacts(run, damage):
    if damage in ('unfinished', 'missing_command', 'failed_command', 'missing_stage'):
        path = run / 'status.json'
        status = json.loads(path.read_text())
        if damage == 'unfinished':
            status['status'] = 'running'
        elif damage == 'missing_command':
            status['commands'].pop()
        elif damage == 'failed_command':
            status['commands'][-1]['returncode'] = 1
        else:
            status['completed_stages'].pop()
        path.write_text(json.dumps(status))
    else:
        filename = ('initialized_stage_2.pth' if damage in
                    ('transfer', 'classifier_transfer', 'old_optimizer') else 'epoch_1.pth')
        path = run / 'stage_2/training' / filename
        checkpoint = torch.load(path)
        if damage == 'optimizer_nan':
            checkpoint['optimizer']['state'][0]['exp_avg'][0] = float('nan')
        elif damage == 'optimizer_inf':
            checkpoint['optimizer']['param_groups'][0]['lr'] = float('inf')
        elif damage == 'no_optimizer':
            del checkpoint['optimizer']
        elif damage == 'wrong_epoch':
            checkpoint['meta']['epoch'] = 0
        elif damage == 'zero_iterations':
            checkpoint['meta']['iter'] = 0
        elif damage == 'transfer':
            checkpoint['state_dict']['backbone.weight'][0] += 1
        elif damage == 'classifier_transfer':
            checkpoint['state_dict']['decoder.outs_cls.2.weight'][-1, 0] += 1
        else:
            checkpoint['optimizer'] = {}
        torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        audit.audit_run(run)


def test_audit_requires_actual_loaded_initialization_evidence(run):
    (run / 'stage_2/training/loaded_initialization.json').unlink()
    with pytest.raises(ValueError, match='actual loaded initialization'):
        audit.audit_run(run)
