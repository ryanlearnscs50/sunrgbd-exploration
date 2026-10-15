"""Use the maintained LDMR mapping without importing its vendored mmdet3d."""
import hashlib
import json
from numbers import Integral
from pathlib import Path
import runpy

import numpy as np


class ScanNet35Protocol:
    def __init__(self, stage_setting='scannet35_s5_freqorder'):
        source = (Path(__file__).resolve().parents[2] / 'configs' / '_base_' /
                  'class_mappings' / 'scannet_dynamic_head_mappings.py')
        mapping = runpy.run_path(str(source))
        self.stages = mapping['get_stage_definitions'](
            strategy='frequency', stage_setting=stage_setting)
        self.names = tuple(n for s in self.stages for n in s['class_names'])
        self.raw_ids = tuple(n for s in self.stages for n in s['nyu40_ids'])
        self.stage_setting = stage_setting
        identity = dict(names=self.names, raw_ids=self.raw_ids,
                        stages=[s['class_indices'] for s in self.stages])
        self.identity = hashlib.sha256(json.dumps(
            identity, sort_keys=True).encode()).hexdigest()

    def _stage(self, stage):
        if (not isinstance(stage, Integral) or isinstance(stage, bool)
                or not 1 <= stage <= len(self.stages)):
            raise ValueError('stage must be a valid one-based LDMR stage')
        return self.stages[stage - 1]

    def seen_count(self, stage):
        return self._stage(stage)['class_indices'][-1] + 1

    def visible_classes(self, stage, replay_from_stage=None, evaluation=False):
        current = self._stage(stage)
        if evaluation:
            if replay_from_stage is not None:
                raise ValueError('evaluation does not accept a replay snapshot')
            return tuple(range(self.seen_count(stage)))
        if replay_from_stage is not None:
            self._stage(replay_from_stage)
            if replay_from_stage >= stage:
                raise ValueError('replay snapshot must precede the current stage')
            # LDMR snapshots retain all seen classes at the save stage.
            return tuple(range(self.seen_count(replay_from_stage)))
        return tuple(current['class_indices'])

    def map_raw(self, raw, stage, replay_from_stage=None, evaluation=False):
        raw = np.asarray(raw)
        if raw.ndim != 1 or raw.dtype.kind not in 'iu':
            raise ValueError('raw NYU40 semantic labels must be a 1-D integer array')
        labels = np.full(raw.shape, -1, dtype=np.int64)
        for index in self.visible_classes(stage, replay_from_stage, evaluation):
            labels[raw == self.raw_ids[index]] = index
        return labels

    def decoder_classes(self, stage):
        """Dataset output order is LDMR order; encoder sorts internal rows."""
        return list(self.names[:self.seen_count(stage)])


def remap_classifier(old_weight, old_bias, old_classes,
                     new_weight, new_bias, new_classes):
    """Copy learned classes and no-object by name into an initialized new head.

    Arrays or torch tensors are accepted. New rows retain the host's own
    initialization. Return copies; never mutate the source checkpoint. This
    migrates only the final classification layer, not optimizer state.
    """
    old_classes, new_classes = list(old_classes), list(new_classes)
    for classes in (old_classes, new_classes):
        if len(set(classes)) != len(classes) or 'no_obj' in classes:
            raise ValueError('class names must be unique and exclude no_obj')
    if not set(old_classes).issubset(new_classes):
        raise ValueError('new vocabulary must retain every learned class')
    old_rows, new_rows = sorted(old_classes) + ['no_obj'], sorted(new_classes) + ['no_obj']
    if (old_weight.ndim != 2 or new_weight.ndim != 2
            or old_weight.shape != (len(old_rows), new_weight.shape[1])
            or new_weight.shape[0] != len(new_rows)
            or old_bias.shape != (len(old_rows),)
            or new_bias.shape != (len(new_rows),)):
        raise ValueError('classifier shape does not match the vocabulary')
    is_torch = hasattr(new_weight, 'detach')
    if any(hasattr(x, 'detach') != is_torch for x in
           (old_weight, old_bias, new_weight, new_bias)):
        raise ValueError('classifier arrays must use the same backend')
    weight = new_weight.detach().clone() if is_torch else new_weight.copy()
    bias = new_bias.detach().clone() if is_torch else new_bias.copy()
    for src, name in enumerate(old_rows):
        dst = new_rows.index(name)
        weight[dst] = old_weight[src].detach() if is_torch else old_weight[src]
        bias[dst] = old_bias[src].detach() if is_torch else old_bias[src]
    return weight, bias
