"""Versioned global-budget scene checkpoints, separate from inspection JSON.

Pickles are trusted local artifacts only. No dataset/model object or unselected
candidate snapshot is persisted. Selector implementations remain unchanged.
"""

import copy
import pickle
from numbers import Integral

import numpy as np

from .state_io import atomic_write


def _validate(bank, stage_id):
    if bank.use_legacy_mode:
        raise ValueError('Scene checkpoints support global-budget mode only.')
    if not isinstance(stage_id, Integral) or isinstance(stage_id, bool) or stage_id < 1:
        raise ValueError('Scene checkpoint requires a positive completed stage.')
    if not isinstance(bank.memory_scenes, dict):
        raise ValueError('Invalid scene checkpoint memory_scenes.')
    count = 0
    for scene_id, data in bank.memory_scenes.items():
        if not isinstance(scene_id, (str, Integral)) or not isinstance(data, dict):
            raise ValueError('Invalid scene checkpoint entry.')
        stages = data.get('stages')
        if not isinstance(stages, dict) or not stages:
            raise ValueError('Scene checkpoint entry has no stages.')
        if bank.enforce_unique_scene_ids and len(stages) != 1:
            raise ValueError('Scene checkpoint violates unique-scene policy.')
        for saved_stage, entry in stages.items():
            if (not isinstance(saved_stage, Integral) or isinstance(saved_stage, bool)
                    or not 1 <= saved_stage <= stage_id):
                raise ValueError('Invalid or future scene save stage.')
            snapshot = entry.get('snapshot', {})
            if snapshot.get('save_stage') != saved_stage:
                raise ValueError('Scene snapshot stage mismatch.')
            if not isinstance(snapshot.get('data_info'), dict):
                raise ValueError('Scene checkpoint lacks replay annotations.')
            present = set(snapshot.get('present_classes', []))
            if not present.issubset(set(snapshot.get('seen_classes', []))):
                raise ValueError('Scene snapshot contains unseen classes.')
            if not np.isfinite(float(entry.get('importance', float('nan')))):
                raise ValueError('Scene checkpoint has nonfinite importance.')
            count += 1
    if count > bank.memory_budget:
        raise ValueError('Scene checkpoint exceeds its memory budget.')


def save_checkpoint(bank, filepath, stage_id, context=None):
    _validate(bank, stage_id)
    payload = dict(format_version=1, memory_level='scene', stage_id=stage_id,
                   config=bank._checkpoint_config,
                   context=copy.deepcopy(context),
                   memory_scenes=bank.memory_scenes)
    atomic_write(filepath, lambda f: pickle.dump(
        payload, f, protocol=pickle.HIGHEST_PROTOCOL), binary=True)


def load_checkpoint(cls, filepath, expected_stage=None, expected_config=None,
                    expected_context=None):
    with open(filepath, 'rb') as stream:
        payload = pickle.load(stream)
    if (not isinstance(payload, dict) or payload.get('format_version') != 1
            or payload.get('memory_level') != 'scene'):
        raise ValueError('Unsupported scene checkpoint format; inspection JSON '
                         'cannot be used as a checkpoint.')
    if expected_stage is not None and payload.get('stage_id') != expected_stage:
        raise ValueError('Scene checkpoint stage does not precede resume stage.')
    if expected_context is not None and payload.get('context') != expected_context:
        raise ValueError('Scene checkpoint dataset/class/stage protocol mismatch.')
    config = payload.get('config')
    if not isinstance(config, dict):
        raise ValueError('Scene checkpoint has no constructor configuration.')
    config = copy.deepcopy(config)
    if expected_config is not None:
        current = cls(**expected_config)._checkpoint_config
        # Output diagnostics may move; selection inputs must not silently change.
        ignored = {'debug_mode', 'metrics_dir'}
        mismatches = [key for key in set(config) | set(current)
                      if key not in ignored and config.get(key) != current.get(key)]
        if mismatches:
            raise ValueError('Scene checkpoint configuration mismatch: '
                             + ', '.join(sorted(mismatches)))
        config.update({key: current[key] for key in ignored})
    bank = cls(**config)
    bank.memory_scenes = payload.get('memory_scenes')
    _validate(bank, payload.get('stage_id'))
    bank.loaded_stage_id = payload['stage_id']
    bank._recompute_class_distribution()
    bank._sync_active_stage_stats()
    bank.total_scenes_stored = bank._count_total_unique_scenes()
    return bank
