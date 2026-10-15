"""Strict strategy/config validation for LD Design-1 and Design-2."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional


LD_DESIGN1_STRATEGY = "learning_dynamics_design1"
LD_DESIGN2_STRATEGY = "learning_dynamics_design2"

LD_DESIGN_SHARED_KEYS = frozenset(
    {
        "q_metric",
        "stage1_scores_mode",
        "stage1_scores_file",
        "stage1_stats_dir",
        "min_add_lower_bound",
        "force_accept_until_lower_bound",
        "allow_missing_seat_terms",
        "supply_scaling_mode",
        "supply_cap",
        "use_class_balance",
    }
)
LD_DESIGN1_ONLY_KEYS = frozenset({"use_compatibility_kernel", "compatibility_weight"})
LD_DESIGN2_ONLY_KEYS = frozenset(
    {"w_max", "redundancy_lambda", "redundancy_topk", "min_class_quota"}
)

LD_DESIGN1_ALLOWED_KEYS = LD_DESIGN_SHARED_KEYS | LD_DESIGN1_ONLY_KEYS
LD_DESIGN2_ALLOWED_KEYS = LD_DESIGN_SHARED_KEYS | LD_DESIGN2_ONLY_KEYS

LD_DESIGN_STRATEGIES = frozenset({LD_DESIGN1_STRATEGY, LD_DESIGN2_STRATEGY})


def _as_dict(value: Any, *, key_name: str, context: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(
            f"{context}.{key_name} must be a dict when provided, got {type(value).__name__}."
        )
    return dict(value)


def _format_keys(keys) -> str:
    return ", ".join(sorted(str(k) for k in keys))


def _strip_meta_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Drop config-system keys that may appear after CLI merge."""
    out = dict(cfg)
    out.pop("_delete_", None)
    return out


def _build_migration_error(*, context: str) -> ValueError:
    return ValueError(
        "Legacy mixed LD config is no longer supported: "
        f"{context}.selection_strategy='learning_dynamics_design1' with "
        f"{context}.learning_dynamics_design1.design_version=2. "
        "Migrate to explicit Design-2 config by setting "
        f"{context}.selection_strategy='learning_dynamics_design2' and moving "
        "Design-2 keys into "
        f"{context}.learning_dynamics_design2."
    )


def validate_scene_memory_ld_strategy_config(
    scene_memory_config: Mapping[str, Any], *, context: str = "scene_memory_config"
) -> Dict[str, Any]:
    """Validate strict LD Design-1/Design-2 ownership and key allowlists.

    Returns:
      A dict containing:
        - selection_strategy
        - active_ld_block_key (None for non-design strategies)
        - active_ld_config (dict; empty for non-design strategies)
    """
    if not isinstance(scene_memory_config, Mapping):
        raise ValueError(
            f"{context} must be a dict-like mapping, got {type(scene_memory_config).__name__}."
        )

    strategy = str(scene_memory_config.get("selection_strategy", "")).strip().lower()
    d1_raw = _as_dict(
        scene_memory_config.get("learning_dynamics_design1", None),
        key_name="learning_dynamics_design1",
        context=context,
    )
    d2_raw = _as_dict(
        scene_memory_config.get("learning_dynamics_design2", None),
        key_name="learning_dynamics_design2",
        context=context,
    )
    d1_cfg = _strip_meta_keys(d1_raw or {})
    d2_cfg = _strip_meta_keys(d2_raw or {})

    # Legacy mixed config guard (explicit migration error).
    if (
        strategy == LD_DESIGN1_STRATEGY
        and isinstance(d1_raw, dict)
        and "design_version" in d1_cfg
    ):
        try:
            design_version = int(d1_cfg.get("design_version"))
        except Exception:
            design_version = None
        if design_version == 2:
            raise _build_migration_error(context=context)

    if strategy == LD_DESIGN1_STRATEGY:
        if d1_raw is None:
            raise ValueError(
                f"{context}.selection_strategy='{LD_DESIGN1_STRATEGY}' requires "
                f"{context}.learning_dynamics_design1 block."
            )
        if "q_metric" not in d1_cfg:
            raise ValueError(
                f"{context}.selection_strategy='{LD_DESIGN1_STRATEGY}' requires "
                f"{context}.learning_dynamics_design1.q_metric to be explicitly set "
                "(no implicit fallback)."
            )
        if d2_raw is not None:
            raise ValueError(
                f"{context}.selection_strategy='{LD_DESIGN1_STRATEGY}' forbids "
                f"{context}.learning_dynamics_design2 block."
            )
        forbidden_keys = set(d1_cfg.keys()) & (
            LD_DESIGN2_ONLY_KEYS | {"design_version"}
        )
        if forbidden_keys:
            raise ValueError(
                f"{context}.learning_dynamics_design1 has Design-2-only/legacy keys: "
                f"{_format_keys(forbidden_keys)}."
            )
        unknown = set(d1_cfg.keys()) - LD_DESIGN1_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"{context}.learning_dynamics_design1 has unknown/forbidden keys: "
                f"{_format_keys(unknown)}. Allowed keys: "
                f"{_format_keys(LD_DESIGN1_ALLOWED_KEYS)}."
            )
        return dict(
            selection_strategy=strategy,
            active_ld_block_key="learning_dynamics_design1",
            active_ld_config=dict(d1_cfg),
        )

    if strategy == LD_DESIGN2_STRATEGY:
        if d2_raw is None:
            raise ValueError(
                f"{context}.selection_strategy='{LD_DESIGN2_STRATEGY}' requires "
                f"{context}.learning_dynamics_design2 block."
            )
        if "q_metric" not in d2_cfg:
            raise ValueError(
                f"{context}.selection_strategy='{LD_DESIGN2_STRATEGY}' requires "
                f"{context}.learning_dynamics_design2.q_metric to be explicitly set "
                "(no implicit fallback)."
            )
        if d1_raw is not None:
            raise ValueError(
                f"{context}.selection_strategy='{LD_DESIGN2_STRATEGY}' forbids "
                f"{context}.learning_dynamics_design1 block."
            )
        forbidden_keys = set(d2_cfg.keys()) & (
            LD_DESIGN1_ONLY_KEYS | {"design_version"}
        )
        if forbidden_keys:
            raise ValueError(
                f"{context}.learning_dynamics_design2 has Design-1-only/legacy keys: "
                f"{_format_keys(forbidden_keys)}."
            )
        unknown = set(d2_cfg.keys()) - LD_DESIGN2_ALLOWED_KEYS
        if unknown:
            raise ValueError(
                f"{context}.learning_dynamics_design2 has unknown/forbidden keys: "
                f"{_format_keys(unknown)}. Allowed keys: "
                f"{_format_keys(LD_DESIGN2_ALLOWED_KEYS)}."
            )
        return dict(
            selection_strategy=strategy,
            active_ld_block_key="learning_dynamics_design2",
            active_ld_config=dict(d2_cfg),
        )

    return dict(
        selection_strategy=strategy,
        active_ld_block_key=None,
        active_ld_config={},
    )


def get_ld_design_stage1_filenames(selection_strategy: str) -> Dict[str, str]:
    """Return strategy-specific Stage-1 score filenames for LD design strategies."""
    strategy = str(selection_strategy).strip().lower()
    if strategy == LD_DESIGN1_STRATEGY:
        return dict(
            score_filename="learning_dynamics_design1_scores.json",
            recomputed_filename="learning_dynamics_design1_scores_recomputed.json",
            trajectories_filename="learning_dynamics_design1_q_trajectories.json",
        )
    if strategy == LD_DESIGN2_STRATEGY:
        return dict(
            score_filename="learning_dynamics_design2_scores.json",
            recomputed_filename="learning_dynamics_design2_scores_recomputed.json",
            trajectories_filename="learning_dynamics_design2_q_trajectories.json",
        )
    raise ValueError(
        "get_ld_design_stage1_filenames supports only strategies "
        f"{sorted(LD_DESIGN_STRATEGIES)}, got {selection_strategy!r}."
    )
