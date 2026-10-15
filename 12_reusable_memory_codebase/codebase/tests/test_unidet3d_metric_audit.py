"""A successful training exit must not hide incomplete or corrupt metrics."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'metric_audit', Path(__file__).resolve().parents[1] / 'adapters/tools/audit_metrics.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def metrics(names):
    result = {}
    for threshold in ('0.25', '0.50'):
        for suffix, mean in (('AP', 'mAP'), ('rec', 'mAR')):
            result.update({f'{name}_{suffix}_{threshold}': .25 for name in names})
            result[f'{mean}_{threshold}'] = .25
    return result


def line(values, epoch=1, count=312):
    return f'Epoch(val) [{epoch}][{count}/{count}] scannet: {values!r}  time: 0.2\n'


@pytest.mark.parametrize('fault', ['missing_class', 'extra_class', 'wrong_mean', 'bool', 'negative', 'inf'])
def test_rejects_incomplete_or_invalid_metrics(fault):
    values = metrics(['chair', 'door'])
    if fault == 'missing_class':
        del values['door_AP_0.25']
    elif fault == 'extra_class':
        values['future_AP_0.25'] = .25
    elif fault == 'wrong_mean':
        values['mAP_0.25'] = .5
    else:
        values['door_AP_0.25'] = {'bool': True, 'negative': -.1, 'inf': float('inf')}[fault]
    with pytest.raises(ValueError):
        audit.validation_records(line(values), ['chair', 'door'], 1, 312)


@pytest.mark.parametrize('epochs', [[], [1, 1], [2, 1], [1, 3]])
def test_requires_each_validation_epoch_in_order(epochs):
    with pytest.raises(ValueError, match='epochs'):
        audit.validation_records(''.join(line(metrics(['chair']), e) for e in epochs), ['chair'], 2, 312)


def test_scene_count_and_nested_nonfinite_scalars():
    with pytest.raises(ValueError, match='scene count'):
        audit.validation_records(line(metrics(['chair']), count=311), ['chair'], 1, 312)
    with pytest.raises(ValueError, match='nonfinite'):
        audit.finite_numbers({'losses': [1., float('nan')]}, 'history')


def test_reference_validation_uses_nominal_epochs():
    epochs = list(range(15, 181, 15))
    logs = ''.join(line(metrics(['chair']), epoch) for epoch in epochs)
    actual = audit.validation_records(logs, ['chair'], 180, 312, interval=15)
    assert [record['epoch'] for record in actual] == epochs
    with pytest.raises(ValueError, match='epochs'):
        audit.validation_records(logs, ['chair'], 180, 312)
    with pytest.raises(ValueError, match='epochs'):
        audit.validation_records(logs.replace(line(metrics(['chair']), 90), ''),
                                 ['chair'], 180, 312, interval=15)


def test_real_layout_and_inherited_stage_logs(tmp_path):
    protocol = audit.ScanNet35Protocol()
    source = tmp_path / 'original_stage'
    session = source / 'training/session'
    (session / 'vis_data').mkdir(parents=True)
    (session / 'session.log').write_text(line(metrics(protocol.names[:7])))
    scalar = session / 'vis_data/scalars.json'
    scalar.write_text(json.dumps({'loss': 0., 'iter': 1}) + '\n')
    (tmp_path / 'arguments.json').write_text(json.dumps(dict(
        stage_setting=protocol.stage_setting, last_stage=1, epochs=1, memory_mode='none')))
    (tmp_path / 'status.json').write_text(json.dumps(dict(
        status='complete', protocol_id=protocol.identity, memory_mode='none',
        completed_stages=[1], commands=[dict(returncode=0)], inherited_stages={'1': str(source)})))
    result = audit.audit_run(tmp_path, 312)
    assert result['stages'][0]['validations'][0]['metrics']['mAP_0.25'] == .25
    assert len(result['evidence_sha256']) == 2
    scalar.write_text('{"loss": NaN}\n')
    with pytest.raises(ValueError, match='nonfinite'):
        audit.audit_run(tmp_path, 312)


def test_review_collectors_do_not_create_runner_sessions(tmp_path):
    session = tmp_path / 'session'
    (session / 'vis_data').mkdir(parents=True)
    log = session / 'session.log'
    scalar = session / 'vis_data/scalars.json'
    log.write_text('runner')
    scalar.write_text('{}\n')
    reviewing = tmp_path / 'reviewing'
    reviewing.mkdir()
    for index in range(1, 5):
        (reviewing / f'review_{index}.log').write_text('collector')
    assert audit.training_logs(tmp_path) == (log, scalar)
    other = tmp_path / 'other/vis_data'
    other.mkdir(parents=True)
    (other / 'scalars.json').write_text('{}\n')
    with pytest.raises(ValueError, match='scalar sessions'):
        audit.training_logs(tmp_path)
