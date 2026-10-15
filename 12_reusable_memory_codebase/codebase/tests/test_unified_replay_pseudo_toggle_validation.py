from __future__ import annotations

import pytest
from mmcv import Config

from tools.train_incremental_scene import _validate_unified_replay_pseudo_cfg_or_raise


def _base_cfg():
    return Config(
        dict(
            data=dict(
                train=dict(
                    type='RepeatDataset',
                    dataset=dict(
                        type='IncrementalSUNRGBDDataset',
                    ),
                    use_pseudo_labels=False,
                    pseudo_label_config=dict(),
                )
            )
        )
    )


def test_unified_replay_toggle_requires_pseudo_enabled():
    cfg = _base_cfg()
    cfg.data.train.use_pseudo_labels = False
    cfg.data.train.pseudo_label_config = dict(apply_to_memory_scenes=True)
    with pytest.raises(ValueError, match='requires use_pseudo_labels=True'):
        _validate_unified_replay_pseudo_cfg_or_raise(cfg)


def test_unified_replay_toggle_rejects_legacy_dataset_keys():
    cfg = _base_cfg()
    cfg.data.train.dataset.use_memory_pseudo_labels = True
    with pytest.raises(ValueError, match='Deprecated memory-pseudo interface'):
        _validate_unified_replay_pseudo_cfg_or_raise(cfg)


def test_unified_replay_toggle_rejects_legacy_memory_block():
    cfg = _base_cfg()
    cfg.MEMORY = dict(ENRICH_PSEUDO_ON_UPDATE=True)
    with pytest.raises(ValueError, match='Deprecated memory-pseudo interface'):
        _validate_unified_replay_pseudo_cfg_or_raise(cfg)


def test_unified_replay_toggle_accepts_valid_config():
    cfg = _base_cfg()
    cfg.data.train.use_pseudo_labels = True
    cfg.data.train.pseudo_label_config = dict(apply_to_memory_scenes=True)
    _validate_unified_replay_pseudo_cfg_or_raise(cfg)

