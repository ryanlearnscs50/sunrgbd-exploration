"""A paired experiment must not silently accept additional interventions."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from ldmr_unidet3d.paired_audit import compare_configs


def pair():
    control = dict(work_dir='/control', load_from='/control/initialized_stage_2.pth', resume=False,
        ldmr_protocol=dict(stage=2, stage_setting='scannet35_s5_freqorder', target_mode='boxes',
                           pseudo_labels=False, pseudo_artifact=None),
        train_dataloader=dict(dataset=dict(datasets=[dict(type='LDMRScanNetDataset',
            ann_file='train.pkl', pipeline=[dict(type='GlobalAlignment'), dict(type='LDMRInsertObjects')])])),
        val_dataloader=dict(ann_file='val.pkl'), randomness=dict(seed=200))
    pseudo = deepcopy(control)
    pseudo.update(work_dir='/pseudo', load_from='/pseudo/initialized_stage_2.pth')
    pseudo['ldmr_protocol'].update(pseudo_labels=True, pseudo_artifact=dict(path='pseudo.json'))
    pseudo['train_dataloader']['dataset']['datasets'][0]['pipeline'].insert(1,
        dict(type='LDMRMergePseudoBoxes', stage=2, stage_setting='scannet35_s5_freqorder', path='pseudo.json'))
    return control, pseudo


def test_pair_accepts_only_expected_intervention_without_mutation():
    a, b = pair()
    original = deepcopy((a, b))
    assert compare_configs(a, b)['configuration_matched']
    assert (a, b) == original


@pytest.mark.parametrize('fault', ['seed', 'validation', 'pipeline', 'train_info',
    'resume', 'initialization', 'missing_merge', 'duplicate_merge', 'order',
    'artifact', 'pseudo_flag', 'unexpected_key'])
def test_pair_rejects_uncontrolled_differences(fault):
    a, b = pair()
    dataset = b['train_dataloader']['dataset']['datasets'][0]
    pipeline = dataset['pipeline']
    if fault == 'seed': b['randomness']['seed'] = 201
    elif fault == 'validation': b['val_dataloader']['ann_file'] = 'other.pkl'
    elif fault == 'pipeline': pipeline[-1]['probability'] = .5
    elif fault == 'train_info': dataset['ann_file'] = 'other.pkl'
    elif fault == 'resume': b['resume'] = True
    elif fault == 'initialization': b['load_from'] = '/other.pth'
    elif fault == 'missing_merge': pipeline.pop(1)
    elif fault == 'duplicate_merge': pipeline.append(deepcopy(pipeline[1]))
    elif fault == 'order': pipeline[0], pipeline[1] = pipeline[1], pipeline[0]
    elif fault == 'artifact': pipeline[1]['path'] = 'other.json'
    elif fault == 'pseudo_flag': b['ldmr_protocol']['pseudo_labels'] = 1
    elif fault == 'unexpected_key': b['extra'] = None
    with pytest.raises(ValueError): compare_configs(a, b)
