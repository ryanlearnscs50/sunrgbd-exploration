"""Shared pseudo-label replay policy helpers for incremental datasets."""

from __future__ import annotations

from typing import Any, Tuple


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def is_memory_or_merged_scene(info: Any) -> bool:
    """Return True when a data_info entry is replay/merged memory content."""
    if not isinstance(info, dict):
        return False
    return bool(info.get('is_replay', False)) or bool(info.get('is_merged', False))


def resolve_replay_pseudo_policy(*,
                                 use_pseudo_labels: Any,
                                 pseudo_label_config: Any,
                                 evaluation_mode: bool,
                                 stage_id: int,
                                 dataset_name: str) -> Tuple[bool, bool]:
    """Resolve unified pseudo-label policy for natural/replay scenes.

    Returns:
        (use_pseudo_labels_effective, apply_to_memory_scenes_effective)
    """
    use_pseudo = bool(use_pseudo_labels)
    apply_to_memory = bool(_cfg_get(pseudo_label_config, 'apply_to_memory_scenes', False))

    if apply_to_memory and not use_pseudo:
        raise ValueError(
            f"{dataset_name}: "
            "pseudo_label_config.apply_to_memory_scenes=True requires "
            "use_pseudo_labels=True."
        )

    # Never enable pseudo injection for eval or stage-1.
    if bool(evaluation_mode) or int(stage_id) <= 1:
        return False, False

    return use_pseudo, apply_to_memory
