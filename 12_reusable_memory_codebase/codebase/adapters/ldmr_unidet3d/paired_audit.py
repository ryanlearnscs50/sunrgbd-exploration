"""Configuration checks for a box-target control versus pseudo-label pair.

These checks complement checkpoint, metric and source audits. They do not
establish identical stochastic trajectories or an isolated supervision effect:
pseudo boxes can also change collision-aware object placement.
"""
from copy import deepcopy
from pathlib import Path


def read_config(path):
    """Read trusted resolved configuration without importing detector plugins."""
    from mmengine.config import Config
    return Config.fromfile(str(path), import_custom_modules=False).to_dict()


def compare_configs(control, pseudo):
    """Reject every configuration difference outside the intended intervention.

Initialization paths are checked for their stage-local naming here; callers
must separately verify that their checkpoint tensors are identical.
"""
    arms = [deepcopy(control), deepcopy(pseudo)]
    for index, cfg in enumerate(arms):
        context = cfg['ldmr_protocol']
        stage = context['stage']
        if type(stage) is not int or stage < 2 or context['target_mode'] != 'boxes':
            raise ValueError('paired pseudo audit requires later-stage box targets')
        if context['pseudo_labels'] is not bool(index):
            raise ValueError('expected control without pseudo and intervention with pseudo')
        artifact = context['pseudo_artifact']
        if (index == 0 and artifact is not None) or (index == 1 and not isinstance(artifact, dict)):
            raise ValueError('invalid paired pseudo artifact configuration')
        expected = Path(cfg['work_dir']) / f'initialized_stage_{stage}.pth'
        if Path(cfg['load_from']).resolve() != expected.resolve() or cfg['resume'] is not False:
            raise ValueError('pair must start from stage-local initialization without resume')
        datasets = cfg['train_dataloader']['dataset']['datasets']
        if len(datasets) != 1 or datasets[0]['type'] != 'LDMRScanNetDataset':
            raise ValueError('expected a single ScanNet training dataset')
        pipeline = datasets[0]['pipeline']
        merges = [i for i, step in enumerate(pipeline) if step['type'] == 'LDMRMergePseudoBoxes']
        if len(merges) != index:
            raise ValueError('missing or duplicate pseudo merge transform')
        if index:
            position = merges[0]
            if position == 0 or pipeline[position - 1]['type'] != 'GlobalAlignment':
                raise ValueError('pseudo merge must immediately follow alignment')
            expected_merge = dict(type='LDMRMergePseudoBoxes', stage=stage,
                                  stage_setting=context['stage_setting'], **artifact)
            if pipeline[position] != expected_merge:
                raise ValueError('pseudo transform differs from recorded artifact configuration')
            pipeline.pop(position)
        context['pseudo_labels'] = False
        context['pseudo_artifact'] = None
        for key in ('work_dir', 'load_from'):
            cfg.pop(key)
    if arms[0] != arms[1]:
        keys = sorted(k for k in arms[0].keys() | arms[1].keys()
                      if arms[0].get(k) != arms[1].get(k) or (k in arms[0]) != (k in arms[1]))
        raise ValueError(f'paired configurations differ outside pseudo intervention: {keys}')
    return dict(configuration_matched=True,
                allowed_differences=['work_dir', 'load_from (tensor equality audited separately)',
                                     'pseudo artifact metadata', 'pseudo merge transform'],
                limitation='Pseudo boxes can change collision-aware object placement; '
                           'this compares complete pseudo policies, not supervision alone.')
