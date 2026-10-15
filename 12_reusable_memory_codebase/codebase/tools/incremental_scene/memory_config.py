"""Resolve the public memory switch into the trainer's existing configuration.

This module has no detector imports, so mode resolution is independently testable.
It does not build datasets, access checkpoints, or change replay algorithms.
"""

from collections.abc import Mapping


MEMORY_MODES = ('scene', 'object', 'none')


def normalize_memory_mode(config, override=None):
    """Normalize in place and return the selected mode.

    Precedence: CLI override > top-level memory_mode > legacy flags/defaults.
    Explicit modes override inherited legacy flags but require their own bank
    configuration. Inactive bank settings are retained for later mode switches.
    Legacy configurations preserve the trainer's original mode inference.
    """
    mode = override if override is not None else config.get('memory_mode')
    explicit = mode is not None
    if explicit:
        if mode not in MEMORY_MODES:
            raise ValueError('memory_mode must be scene, object, or none; '
                             f'got {mode!r}.')
    else:
        objects = bool(config.get(
            'use_object_memory', config.get('object_memory_config') is not None))
        scenes = (config.get('scene_memory_config', {}) is not None
                  and bool(config.get('use_scene_memory', True)))
        if objects and scenes:
            raise ValueError('Object-level and scene-level replay cannot be '
                             'enabled together. Set memory_mode explicitly or '
                             'enable only one legacy memory flag.')
        mode = 'object' if objects else 'scene' if scenes else 'none'

    if mode != 'none':
        key = f'{mode}_memory_config'
        # Only legacy scene configurations may use implicit constructor defaults.
        default = {} if mode == 'scene' and not explicit else None
        if not isinstance(config.get(key, default), Mapping):
            raise ValueError(f'memory_mode={mode!r} requires a {key} mapping. '
                             'Use a config defining that bank; mode switching '
                             'does not invent budgets or selection policies.')
    if mode == 'object':
        train = config.get('data', {}).get('train', {})
        while isinstance(train, Mapping) and 'dataset' in train:
            train = train['dataset']
        dataset_type = train.get('type', '') if isinstance(train, Mapping) else ''
        scannet = (str(config.get('stage_setting', '')).startswith('scannet')
                   or dataset_type == 'IncrementalScanNetDataset')
        if (scannet and config['object_memory_config'].get('selection_strategy')
                == 'learning_dynamics_design2'):
            raise ValueError('ScanNet object Design-2 population is not '
                             'implemented; use a supported object selector or '
                             'scene memory. See docs/memory_banks.md.')
    config['memory_mode'] = mode
    config['use_scene_memory'] = mode == 'scene'
    config['use_object_memory'] = mode == 'object'
    if mode == 'scene' and 'scene_memory_config' not in config:
        config['scene_memory_config'] = {}
    return mode


def memory_config_summary(config):
    """Describe a normalized configuration without claiming runtime validation."""
    mode = config['memory_mode']
    bank = config.get(f'{mode}_memory_config', {}) or {}
    return dict(
        memory_mode=mode,
        use_scene_memory=config['use_scene_memory'],
        use_object_memory=config['use_object_memory'],
        selection_strategy=(bank.get('selection_strategy',
                            'balanced' if mode == 'scene' else 'random')
                            if mode != 'none' else None),
        validation_scope='memory configuration only; no dataset/model execution',
    )
