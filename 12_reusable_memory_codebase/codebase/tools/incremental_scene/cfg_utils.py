"""Config-access helpers for incremental scene training."""

from __future__ import annotations

from copy import deepcopy


DETECTOR_CONFIG_SECTIONS = (
    'data', 'model', 'optimizer', 'optimizer_config', 'lr_config',
    'checkpoint_config', 'log_config', 'evaluation', 'runner', 'workflow',
    'dist_params', 'fp16',
)


def merge_incremental_options(config, options):
    """Accept legacy base_config.* CLI paths as aliases of public sections.

    LDMR configs export the detector settings twice via locals().update().
    MMCV inheritance breaks their shared identity; normalize explicit aliases
    before resolving the detector config so a path/batch override cannot vanish.
    """
    normalized = {}
    for key, value in (options or {}).items():
        canonical = key
        if key.startswith('base_config.'):
            suffix = key[len('base_config.'):]
            if suffix.split('.')[0] in DETECTOR_CONFIG_SECTIONS:
                canonical = suffix
        if canonical in normalized:
            raise ValueError(f'duplicate detector option through base_config alias: {canonical}')
        normalized[canonical] = value
    config.merge_from_dict(normalized)


def resolve_detector_config(base, incremental=None):
    """Use public detector sections; stage class/epoch policy is applied later."""
    resolved = deepcopy(base)
    if incremental is not None:
        for section in DETECTOR_CONFIG_SECTIONS:
            if section in incremental:
                resolved[section] = deepcopy(incremental[section])
    return resolved


def cfg_get(cfg_node, key: str, default=None):
    if cfg_node is None:
        return default
    if isinstance(cfg_node, dict):
        return cfg_node.get(key, default)
    try:
        return cfg_node.get(key, default)
    except Exception:
        return getattr(cfg_node, key, default)


def cfg_has_key(cfg_node, key: str) -> bool:
    if cfg_node is None:
        return False
    if isinstance(cfg_node, dict):
        return key in cfg_node
    try:
        return key in cfg_node
    except Exception:
        return hasattr(cfg_node, key)


def unwrap_train_dataset_cfg(train_cfg):
    ds_cfg = cfg_get(train_cfg, 'dataset', None)
    if ds_cfg is None:
        return None
    cur = ds_cfg
    while True:
        nxt = cfg_get(cur, 'dataset', None)
        if nxt is None:
            return cur
        cur = nxt


def validate_unified_replay_pseudo_cfg_or_raise(incremental_cfg) -> None:
    """Validate unified replay-scene pseudo interface and reject legacy keys."""
    train_cfg = None
    data_cfg = cfg_get(incremental_cfg, 'data', None)
    if data_cfg is not None:
        train_cfg = cfg_get(data_cfg, 'train', None)
    outer_ds_cfg = cfg_get(train_cfg, 'dataset', None) if train_cfg is not None else None
    inner_ds_cfg = unwrap_train_dataset_cfg(train_cfg) if train_cfg is not None else None

    deprecated_hits = []
    for node in (outer_ds_cfg, inner_ds_cfg):
        if cfg_has_key(node, 'use_memory_pseudo_labels'):
            deprecated_hits.append('data.train.dataset.use_memory_pseudo_labels')
        if cfg_has_key(node, 'memory_pseudo_label_config'):
            deprecated_hits.append('data.train.dataset.memory_pseudo_label_config')

    memory_cfg = cfg_get(incremental_cfg, 'MEMORY', None)
    for legacy_key in (
            'ENRICH_PSEUDO_ON_UPDATE',
            'ENRICH_PSEUDO_CKPT',
            'ENRICH_PSEUDO_CONF_THRESH',
            'ENRICH_PSEUDO_NMS_THRESH',
            'ENRICH_PSEUDO_MAX_PER_SCENE'):
        if cfg_has_key(memory_cfg, legacy_key):
            deprecated_hits.append(f'MEMORY.{legacy_key}')

    if deprecated_hits:
        deprecated_hits = sorted(set(deprecated_hits))
        raise ValueError(
            "Deprecated memory-pseudo interface is no longer supported. "
            "Use unified pseudo_label_config.apply_to_memory_scenes "
            "(with use_pseudo_labels=True). "
            f"Found deprecated keys: {deprecated_hits}"
        )

    # Resolve canonical pseudo toggles from train cfg first, then top-level.
    use_pseudo_labels = False
    if cfg_has_key(train_cfg, 'use_pseudo_labels'):
        use_pseudo_labels = bool(cfg_get(train_cfg, 'use_pseudo_labels', False))
    elif cfg_has_key(incremental_cfg, 'use_pseudo_labels'):
        use_pseudo_labels = bool(cfg_get(incremental_cfg, 'use_pseudo_labels', False))

    pseudo_cfg = cfg_get(train_cfg, 'pseudo_label_config', None)
    if pseudo_cfg is None:
        pseudo_cfg = cfg_get(incremental_cfg, 'pseudo_label_config', None)
    apply_to_memory = bool(cfg_get(pseudo_cfg, 'apply_to_memory_scenes', False))

    if apply_to_memory and not use_pseudo_labels:
        raise ValueError(
            "pseudo_label_config.apply_to_memory_scenes=True requires "
            "use_pseudo_labels=True."
        )
