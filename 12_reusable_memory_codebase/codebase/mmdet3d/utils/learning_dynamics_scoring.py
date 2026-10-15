"""Learning-dynamics scoring utilities for incremental memory selection (SUNRGBD).

This module is intentionally lightweight (pure-Python) so it can be used from
training scripts and unit tests without pulling in heavy training dependencies.

Definitions (per seat/scene s, class c, checkpoint index k):
  - q_{s,c}(k): F1@IoUτ (precision-sensitive), higher is better
  - Forgetness (old classes): cumulative drop across k
    F_{s,c} = Σ_{k=1..K} max(0, q(k-1) - q(k))
  - Replay priority (new classes), policy-driven:
    - default: slow_saturation
    - legacy_between:
      U_{s,c} = (1 - q_end) * max(0, (q(k_end)-q(k_start)) / (k_end-k_start))
    - slow_saturation:
      U_{s,c} = 1[q_end>=tau_q] * competence(q_end) * G * S_slow
      where G = Σ max(0, q(k)-q(k-1)-delta), and S_slow rewards later gains.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

SeatKey = Tuple[str, int]  # (scene_id, save_stage)
Q_METRIC = "f1"
Q_FORMULA = "2TP/(2TP+FP+FN+eps)"
REPLAY_POLICY_DEFAULT = "slow_saturation"
REPLAY_POLICY_ALLOWED = ("legacy_between", "slow_saturation")


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


def normalize_replay_priority_policy_type(value: Any) -> str:
    """Normalize replay-priority policy identifier.

    Returns:
      - 'slow_saturation' for trajectory-shape-aware replay priority
      - 'legacy_between' for legacy 2-point slope replay priority
    """
    raw = '' if value is None else str(value)
    v = raw.strip().lower()
    if v in ('', 'default'):
        return str(REPLAY_POLICY_DEFAULT)
    if v in ('legacy_between', 'legacy', 'between'):
        return 'legacy_between'
    if v in ('slow_saturation', 'slow-saturation', 'slow'):
        return 'slow_saturation'
    raise ValueError(
        "Invalid replay_priority_policy.type. "
        "Supported: ['slow_saturation', 'legacy_between']. "
        f"Got: {value!r}"
    )


def normalize_reviewing_weight_policy_type(value: Any) -> str:
    """Normalize reviewing.weight_policy.type to a canonical identifier.

    Returns:
      - 'ap_drop' for the legacy AP-drop path ("drop_dominant_sum")
      - 'ld_drop' for the LD-style q-drop path (F1/recall)
      - 'fixed' for constant per-seat reviewing weights
    """
    raw = '' if value is None else str(value)
    v = raw.strip().lower()
    if v in ('', 'drop_dominant_sum', 'ap_drop', 'apdrop', 'legacy'):
        return 'ap_drop'
    if v in ('ld_f1_drop', 'ld_drop', 'f1_drop'):
        return 'ld_drop'
    if v in ('fixed', 'constant'):
        return 'fixed'
    raise ValueError(
        "Invalid reviewing.weight_policy.type. "
        "Supported: ['drop_dominant_sum' (legacy AP-drop), "
        "'ld_drop' (LD q drop), 'fixed' (constant per-seat weight); "
        "legacy aliases: 'ld_f1_drop', 'f1_drop', 'constant']. "
        f"Got: {value!r}"
    )


def validate_sunrgbd_ld_reviewing_design_consistency(
        *,
        learning_dynamics_selection: bool,
        reviewing_enabled: bool,
        reviewing_weight_policy_type: Any,
        ld_iou_mode: Any,
        reviewing_weight_iou_thr: Any,
) -> None:
    """Fail fast if a single run mixes old vs new aggregation designs.

    In this repo:
      - Learning-dynamics (LD) memory updates use within-seat q aggregation.
      - Legacy reviewing sampling uses scene→class→scene AP-drop aggregation.

    When LD selection is enabled AND reviewing is enabled, we allow:
      - ld_drop: coupled to LD q drop and must match LD IoU.
      - fixed: constant per-seat baseline (decoupled from LD drop).
    Legacy AP-drop remains disallowed in this mixed setting.
    """
    if not (bool(learning_dynamics_selection) and bool(reviewing_enabled)):
        return

    pol = normalize_reviewing_weight_policy_type(reviewing_weight_policy_type)
    if pol == 'ap_drop':
        raise ValueError(
            "Mixed experiment design detected: "
            "scene_memory_config.selection_strategy='learning_dynamics' requires "
            "reviewing.weight_policy.type in ['ld_drop', 'fixed'] when reviewing.enabled=True. "
            f"Got reviewing.weight_policy.type={reviewing_weight_policy_type!r}."
        )
    if pol == 'fixed':
        return

    ld = _normalize_iou_mode(ld_iou_mode)
    wi = _normalize_iou_mode(reviewing_weight_iou_thr)
    if ld != wi:
        raise ValueError(
            "Mixed experiment design detected: LD and reviewing must use the same IoU threshold. "
            f"Got SCORING.LD_IOU_MODE={ld!r} but reviewing.weight_iou_thr={wi!r}."
        )


def beta_smoothed_recall(*,
                         tp: float,
                         fn: float,
                         alpha: float = 1.0,
                         beta: float = 1.0) -> float:
    """Smoothed recall using a Beta(α,β) prior.

    q = (TP + α) / (TP + FN + α + β)
    """
    tp = float(tp)
    fn = float(fn)
    alpha = float(alpha)
    beta = float(beta)
    # IMPORTANT: for per-(scene,class) recall, "no GT" should not receive an
    # artificial 0.5 prior. Treat TP=FN=0 as q=0.
    if tp + fn <= 0.0:
        return 0.0
    denom = tp + fn + alpha + beta
    if denom <= 0.0:
        return 0.0
    q = (tp + alpha) / denom
    if q < 0.0:
        return 0.0
    if q > 1.0:
        return 1.0
    return float(q)


def f1_score(*, tp: float, fp: float, fn: float, eps: float = 1e-9) -> float:
    """F1 = 2TP / (2TP + FP + FN + eps), clamped to [0,1]."""
    tp = float(tp)
    fp = float(fp)
    fn = float(fn)
    eps = float(eps)
    if eps <= 0.0:
        eps = 1e-9
    denom = 2.0 * tp + fp + fn + eps
    if denom <= 0.0:
        return 0.0
    q = (2.0 * tp) / denom
    if q < 0.0:
        return 0.0
    if q > 1.0:
        return 1.0
    return float(q)


def _normalize_iou_mode(mode: Any) -> str:
    """Return canonical LD IoU threshold string.

    Historical note:
      - Older runs used the string mode "avg_0.25_0.50" to average two IoUs.
        This repo no longer supports that averaging; use a single IoU threshold.
    """
    allowed = (0.25, 0.50, 0.75, 0.80, 0.90)
    raw = str(mode).strip().lower()
    if 'avg' in raw:
        raise ValueError(
            "LD IoU averaging modes are no longer supported. "
            "Use a single threshold in "
            "['0.25', '0.50', '0.75', '0.80', '0.90']."
        )
    raw = raw.replace('_', '.')
    try:
        thr = float(raw)
    except Exception as e:
        raise ValueError(
            "Invalid LD IoU threshold. Expected one of "
            "['0.25', '0.50', '0.75', '0.80', '0.90'], "
            f"got '{mode}'."
        ) from e
    for a in allowed:
        if abs(float(thr) - float(a)) < 1e-6:
            return f"{float(a):.2f}"
    raise ValueError(
        "Invalid LD IoU threshold. Expected one of "
        "['0.25', '0.50', '0.75', '0.80', '0.90'], "
        f"got '{mode}'."
    )


def _stat_key_with_fallback(stat: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if not isinstance(stat, Mapping):
        return default
    if key in stat:
        return stat.get(key, default)
    # Legacy JSON-friendly variants (e.g. "tp_0.25")
    if key.endswith('_025'):
        return stat.get(key[:-4] + '_0.25', default)
    return default


def _q_from_stat(stat: Mapping[str, Any], *, iou_mode: str, eps: float) -> Optional[float]:
    """Compute q from a per-class stat dict.

    IMPORTANT: q is only defined when gt_count > 0.
    """
    if not isinstance(stat, Mapping):
        return None
    gt_count = _safe_float(stat.get('gt_count', 0.0), default=0.0)
    if gt_count <= 0.0:
        return None

    # Validate iou_mode even though match stats are already computed at that IoU.
    _normalize_iou_mode(iou_mode)
    tp = _get_stat(stat, 'tp', 0.0)
    fp = _get_stat(stat, 'fp', 0.0)
    fn = _get_stat(stat, 'fn', 0.0)
    return float(f1_score(tp=tp, fp=fp, fn=fn, eps=float(eps)))


def cumulative_drop(q_traj: Sequence[float]) -> float:
    """Σ max(0, q(k-1) - q(k)) over k=1..K (always ≥ 0)."""
    if not q_traj:
        return 0.0
    total = 0.0
    prev = float(q_traj[0])
    for q in q_traj[1:]:
        q = float(q)
        total += max(0.0, prev - q)
        prev = q
    return float(max(0.0, total))


def replay_priority_between(q_traj: Sequence[float], *, k_start: int,
                            k_end: int) -> float:
    """Legacy 2-point replay priority (backward-compatible path only).

    Formula: (1-q_end) * max(0, (q(k_end)-q(k_start)) / (k_end-k_start))

    Returns 0 if indices are invalid or if slope <= 0.
    """
    if not q_traj:
        return 0.0
    k_start = int(k_start)
    k_end = int(k_end)
    if k_start < 0 or k_end <= k_start:
        return 0.0
    if k_end >= len(q_traj):
        return 0.0
    q0 = float(q_traj[k_start])
    q1 = float(q_traj[k_end])
    denom = float(k_end - k_start)
    if denom <= 0.0:
        return 0.0
    m = (q1 - q0) / denom
    if m <= 0.0:
        return 0.0
    return float(max(0.0, (1.0 - q1) * m))


def replay_priority_slow_saturation(
        q_traj: Sequence[float],
        *,
        gt_count: float,
        delta: float = 0.002,
        tau_q: float = 0.02,
        use_competence: bool = True,
        slow_factor: str = 'centroid',
        eps: float = 1e-9) -> float:
    """Replay priority that favors slower saturation among learnable classes.

    Formula:
      U = 1[q_end >= tau_q] * competence(q_end) * G * S_slow
      G = Σ_{k=1..K} max(0, q_k - q_{k-1} - delta)
      S_slow (centroid) = clamp((k_bar - 1) / max(1, K - 1), 0, 1)
      k_bar = Σ k*Δq_k / (G + eps)
    """
    if _safe_float(gt_count, default=0.0) <= 0.0:
        return 0.0
    if not q_traj or len(q_traj) < 2:
        return 0.0

    eps = _safe_float(eps, default=1e-9)
    if eps <= 0.0:
        eps = 1e-9
    try:
        delta = float(delta)
    except Exception as e:
        raise ValueError(
            f"Invalid replay-priority delta for slow_saturation: {delta!r}"
        ) from e
    if delta < 0.0:
        raise ValueError(
            f"replay-priority delta must be >= 0 for slow_saturation, got {delta}."
        )
    try:
        tau_q = float(tau_q)
    except Exception as e:
        raise ValueError(
            f"Invalid replay-priority tau_q for slow_saturation: {tau_q!r}"
        ) from e
    if tau_q < 0.0 or tau_q > 1.0:
        raise ValueError(
            "replay-priority tau_q must be in [0, 1] for slow_saturation, "
            f"got {tau_q}."
        )

    mode = str(slow_factor).strip().lower()
    if mode != 'centroid':
        raise ValueError(
            "Invalid replay_priority_policy.slow_factor. "
            "Supported: ['centroid']."
        )

    q_vals = [_clamp01(_safe_float(q, default=0.0)) for q in q_traj]
    K = len(q_vals) - 1
    if K <= 0:
        return 0.0

    gains = []
    for k in range(1, len(q_vals)):
        dq = float(q_vals[k]) - float(q_vals[k - 1]) - float(delta)
        gains.append(float(max(0.0, dq)))
    G = float(sum(gains))
    if G <= 0.0:
        return 0.0

    weighted_k_sum = 0.0
    for k, dq in enumerate(gains, start=1):
        weighted_k_sum += float(k) * float(dq)
    k_bar = float(weighted_k_sum / float(G + eps))
    s_slow = float((k_bar - 1.0) / float(max(1, K - 1)))
    s_slow = _clamp01(s_slow)

    q_end = float(q_vals[-1])
    gate = 1.0 if float(q_end) >= float(tau_q) else 0.0
    competence = float(q_end) if bool(use_competence) else 1.0

    u = float(gate) * float(competence) * float(G) * float(s_slow)
    if not (u == u) or u <= 0.0:
        return 0.0
    return float(u)


def _normalize_replay_priority_policy(policy: Optional[Mapping[str, Any]], *,
                                      eps: float) -> Dict[str, Any]:
    """Normalize replay priority policy settings with defaults."""
    cfg = dict(policy or {})
    p_type = normalize_replay_priority_policy_type(
        cfg.get('type', REPLAY_POLICY_DEFAULT)
    )
    out = dict(type=str(p_type), eps=float(eps))

    if str(p_type) == 'legacy_between':
        return out

    try:
        delta = float(cfg.get('delta', 0.002))
    except Exception as e:
        raise ValueError(
            "Invalid replay_priority_policy.delta. Expected float >= 0."
        ) from e
    if delta < 0.0:
        raise ValueError(
            f"Invalid replay_priority_policy.delta={delta}. Must be >= 0."
        )
    try:
        tau_q = float(cfg.get('tau_q', 0.02))
    except Exception as e:
        raise ValueError(
            "Invalid replay_priority_policy.tau_q. Expected float in [0, 1]."
        ) from e
    if tau_q < 0.0 or tau_q > 1.0:
        raise ValueError(
            f"Invalid replay_priority_policy.tau_q={tau_q}. Must be in [0, 1]."
        )
    uc_raw = cfg.get('use_competence', True)
    if isinstance(uc_raw, bool):
        use_competence = bool(uc_raw)
    elif isinstance(uc_raw, (int, float)) and float(uc_raw) in (0.0, 1.0):
        use_competence = bool(int(uc_raw))
    else:
        raise ValueError(
            "Invalid replay_priority_policy.use_competence. "
            "Expected bool."
        )
    slow_factor = str(cfg.get('slow_factor', 'centroid')).strip().lower()
    if slow_factor != 'centroid':
        raise ValueError(
            "Invalid replay_priority_policy.slow_factor. "
            "Supported: ['centroid']."
        )
    out.update(
        delta=float(delta),
        tau_q=float(tau_q),
        use_competence=bool(use_competence),
        slow_factor=str(slow_factor),
    )
    return out


def _as_int_list(values: Iterable[int]) -> List[int]:
    return [int(x) for x in values]


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
        if out != out:  # NaN
            return float(default)
        return float(out)
    except Exception:
        return float(default)


def _get_stat(stats: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    if not isinstance(stats, Mapping):
        return float(default)
    return _safe_float(stats.get(key, default), default=default)


def _object_weight_for_class(stats: Mapping[str, Any], *,
                             object_count_cap: Optional[int]) -> float:
    """Weight a_{s,c} based on GT count (optionally capped).

    For this repo's current learning-dynamics scoring:
      a_{s,c} = min(gt_count_{s,c}, cap)
    """
    w = _safe_float(
        stats.get('gt_count', 0.0) if isinstance(stats, Mapping) else 0.0,
        default=0.0,
    )
    if w <= 0.0:
        return 0.0
    if object_count_cap is not None:
        cap = int(object_count_cap)
        if cap > 0:
            w = min(w, float(cap))
    return float(w)


def _parse_seat_id(seat_id: Any) -> Optional[SeatKey]:
    if isinstance(seat_id, tuple) and len(seat_id) == 2:
        return (str(seat_id[0]), int(seat_id[1]))
    if isinstance(seat_id, str):
        # Legacy: "{scene_id}_stage{save_stage}"
        if '_stage' in seat_id:
            base, maybe_stage = seat_id.rsplit('_stage', 1)
            try:
                return (str(base), int(maybe_stage))
            except Exception:
                return None
    return None


def _iter_seat_metrics(seats_obj: Any) -> Iterable[Tuple[SeatKey, Mapping[int, Mapping[str, Any]]]]:
    """Yield (SeatKey, per_class_stats) pairs from multiple supported layouts."""
    # New format: list of seat records
    if isinstance(seats_obj, list):
        for seat in seats_obj:
            if not isinstance(seat, Mapping):
                continue
            scene_id = seat.get('scene_id', None)
            save_stage = seat.get('save_stage', None)
            if scene_id is None or save_stage is None:
                continue
            try:
                key = (str(scene_id), int(save_stage))
            except Exception:
                continue
            per_cls = seat.get('classes', None)
            if not isinstance(per_cls, Mapping):
                continue
            # Normalize class keys to int
            out = {}
            for cid, stat in per_cls.items():
                try:
                    out[int(cid)] = stat if isinstance(stat, Mapping) else {}
                except Exception:
                    continue
            yield key, out
        return

    # Legacy format: {seat_id: {class_id: stats}}
    if isinstance(seats_obj, Mapping):
        for seat_id, per_cls in seats_obj.items():
            key = _parse_seat_id(seat_id)
            if key is None:
                continue
            if not isinstance(per_cls, Mapping):
                continue
            out = {}
            for cid, stat in per_cls.items():
                try:
                    out[int(cid)] = stat if isinstance(stat, Mapping) else {}
                except Exception:
                    continue
            yield key, out


def build_recall_trajectories(
        metrics_by_k: Mapping[int, Any],
        *,
        iou_mode: str = '0.50',
        alpha: float = 1.0,
        beta: float = 1.0,
        eps: float = 1e-9,
) -> Dict[str, Dict[int, Dict[int, List[float]]]]:
    """Convert per-checkpoint TP/FP/FN metrics into q_{s,c}(k) trajectories.

    `iou_mode` is a *single IoU threshold* (string) used for validation/metadata.
    The provided TP/FP/FN stats are assumed to have been computed using IoU >= τ
    with τ equal to `iou_mode`.

    Supported input formats:
      - New: {k: [ {'scene_id': str, 'save_stage': int, 'classes': {cid: {...}}}, ... ]}
      - Legacy: {k: {seat_id: {cid: {...}}}}

    Returns:
      {scene_id: {save_stage: {class_id: [q(k=0), ..., q(k=K)]}}}
    """
    if not isinstance(metrics_by_k, Mapping) or not metrics_by_k:
        return {}

    ks = sorted(int(k) for k in metrics_by_k.keys())
    k_to_idx = {k: i for i, k in enumerate(ks)}
    k_len = int(len(ks))

    out: Dict[str, Dict[int, Dict[int, List[float]]]] = {}
    for k, seats_obj in metrics_by_k.items():
        ki = k_to_idx.get(int(k), None)
        if ki is None:
            continue
        for (scene_id, save_stage), per_cls in _iter_seat_metrics(seats_obj):
            out.setdefault(str(scene_id), {}).setdefault(int(save_stage), {})
            for cid, stat in per_cls.items():
                # iou_mode is validated, but stats already reflect the selected IoU.
                q = _q_from_stat(stat, iou_mode=str(iou_mode), eps=float(eps))
                if q is None:
                    continue
                traj = out[str(scene_id)][int(save_stage)].setdefault(int(cid), [0.0] * k_len)
                traj[ki] = float(q)
    return out


def validate_fixed_seat_coverage(metrics_by_k: Mapping[int, Any]) -> None:
    """Reject dropped/duplicated seats and changed GT in a fixed evaluation pool.

    Controllers must call this before scoring checkpoint snapshots of the same
    pool. Empty memory pools (e.g. stage 1) are valid if empty at every step.
    This does not impose fixed coverage on historical standalone scorer APIs.
    """
    if not isinstance(metrics_by_k, Mapping) or not metrics_by_k:
        raise ValueError('LD coverage validation requires checkpoint snapshots')
    baseline = None
    for k, seats in metrics_by_k.items():
        if not isinstance(seats, list):
            raise ValueError(f'LD snapshot {k} must contain a seat list')
        coverage = {}
        for seat in seats:
            if not isinstance(seat, Mapping):
                raise ValueError(f'Invalid LD seat at checkpoint {k}')
            sid, stage = seat.get('scene_id'), seat.get('save_stage')
            if (not isinstance(sid, str) or not sid or type(stage) is not int
                    or stage < 1):
                raise ValueError(f'Invalid LD seat identity at checkpoint {k}')
            key = (sid, stage)
            if key in coverage:
                raise ValueError(f'Duplicate LD seat {key} at checkpoint {k}')
            classes = seat.get('classes')
            if not isinstance(classes, Mapping):
                raise ValueError(f'Invalid LD classes at checkpoint {k}')
            counts = {}
            for cid, stat in classes.items():
                if type(cid) is int and cid >= 0:
                    label = cid
                elif (isinstance(cid, str) and cid.isascii() and cid.isdecimal()
                      and str(int(cid)) == cid):
                    label = int(cid)
                else:
                    raise ValueError(f'Invalid LD class key at checkpoint {k}')
                if label in counts or not isinstance(stat, Mapping):
                    raise ValueError(f'Duplicate/invalid LD class at checkpoint {k}')
                count = stat.get('gt_count')
                if (isinstance(count, bool) or not isinstance(count, (int, float))
                        or not np.isfinite(count) or count < 0 or int(count) != count):
                    raise ValueError(f'Invalid LD GT count at checkpoint {k}')
                counts[label] = int(count)
            coverage[key] = counts
        if baseline is None:
            baseline = coverage
        elif coverage != baseline:
            raise ValueError(f'LD seat/class coverage or GT changed at checkpoint {k}')


def compute_learning_dynamics_scores(
        metrics_by_k: Mapping[int, Any],
        *,
        old_classes: Sequence[int],
        new_classes: Sequence[int],
        iou_mode: str = '0.50',
        alpha: float = 1.0,
        beta: float = 1.0,
        slope_k_start: int,
        slope_k_end: int,
        object_count_cap: int = 20,
        eps: float = 1e-9,
        replay_priority_policy: Optional[Mapping[str, Any]] = None,
        return_trajectories: bool = False,
) -> Dict[str, Any]:
    """Compute seat-level and class-level learning-dynamics scores.

    Returns a dict containing:
      - forgetness_by_seat: {scene_id: {save_stage: float}}
      - replay_priority_by_seat: {scene_id: {save_stage: float}}
      - forgetness_by_class: {class_id: float}
      - replay_priority_by_class: {class_id: float}
      - (optional) q_trajectories: {scene_id: {save_stage: {class_id: [q...]}}}
    """
    old_classes_i = sorted(set(_as_int_list(old_classes)))
    new_classes_i = sorted(set(_as_int_list(new_classes)))
    iou_mode_norm = _normalize_iou_mode(iou_mode)
    iou_thr = float(iou_mode_norm)
    replay_policy = _normalize_replay_priority_policy(
        replay_priority_policy, eps=float(eps)
    )

    if not isinstance(metrics_by_k, Mapping) or not metrics_by_k:
        return dict(
            forgetness_by_seat={},
            replay_priority_by_seat={},
            forgetness_by_class={},
            replay_priority_by_class={},
            q_trajectories={} if return_trajectories else None,
            meta=dict(
                iou_mode=iou_mode_norm,
                iou_thr=float(iou_thr),
                eps=float(eps),
                # alpha/beta kept for backward compatibility (legacy recall q); unused.
                alpha=float(alpha),
                beta=float(beta),
                slope_k_start=int(slope_k_start),
                slope_k_end=int(slope_k_end),
                slope_window=int(slope_k_end) - int(slope_k_start),
                object_count_cap=int(object_count_cap),
                replay_priority_policy=dict(replay_policy),
            ),
        )

    trajectories = build_recall_trajectories(
        metrics_by_k,
        iou_mode=str(iou_mode_norm),
        alpha=float(alpha),
        beta=float(beta),
        eps=float(eps),
    )

    # Also keep per-seat per-class weights from k=0 stats (stable across k for GT).
    k0 = sorted(int(k) for k in metrics_by_k.keys())[0]
    weights_by_seat_class: Dict[SeatKey, Dict[int, float]] = {}
    for seat_key, per_cls in _iter_seat_metrics(metrics_by_k.get(k0, {})):
        for cid, stat in per_cls.items():
            w = _object_weight_for_class(stat, object_count_cap=int(object_count_cap))
            if w <= 0.0:
                continue
            weights_by_seat_class.setdefault(seat_key, {})[int(cid)] = float(w)

    forgetness_by_seat: Dict[str, Dict[int, float]] = {}
    replay_priority_by_seat: Dict[str, Dict[int, float]] = {}
    forgetness_by_class: Dict[int, float] = {int(c): 0.0 for c in old_classes_i}
    replay_priority_by_class: Dict[int, float] = {int(c): 0.0 for c in new_classes_i}

    for scene_id, by_stage in trajectories.items():
        for save_stage, per_cls_traj in by_stage.items():
            seat_key = (str(scene_id), int(save_stage))
            fs = 0.0
            us = 0.0

            # Old-class forgetness.
            for c in old_classes_i:
                traj = per_cls_traj.get(int(c), None)
                if traj is None:
                    continue
                a = float(weights_by_seat_class.get(seat_key, {}).get(int(c), 0.0))
                if a <= 0.0:
                    continue
                f_sc = cumulative_drop(traj)
                fs += a * float(f_sc)
                forgetness_by_class[int(c)] = float(
                    forgetness_by_class.get(int(c), 0.0) + a * float(f_sc)
                )

            # New-class replay priority.
            for c in new_classes_i:
                traj = per_cls_traj.get(int(c), None)
                if traj is None:
                    continue
                a = float(weights_by_seat_class.get(seat_key, {}).get(int(c), 0.0))
                if a <= 0.0:
                    continue
                if str(replay_policy.get('type', REPLAY_POLICY_DEFAULT)) == 'slow_saturation':
                    u_sc = replay_priority_slow_saturation(
                        traj,
                        gt_count=float(a),
                        delta=float(replay_policy.get('delta', 0.002)),
                        tau_q=float(replay_policy.get('tau_q', 0.02)),
                        use_competence=bool(
                            replay_policy.get('use_competence', True)
                        ),
                        slow_factor=str(
                            replay_policy.get('slow_factor', 'centroid')
                        ),
                        eps=float(replay_policy.get('eps', eps)),
                    )
                else:
                    u_sc = replay_priority_between(
                        traj,
                        k_start=int(slope_k_start),
                        k_end=int(slope_k_end),
                    )
                us += a * float(u_sc)
                replay_priority_by_class[int(c)] = float(
                    replay_priority_by_class.get(int(c), 0.0) + a * float(u_sc)
                )

            fs = float(max(0.0, fs)) if (fs == fs) else 0.0
            us = float(max(0.0, us)) if (us == us) else 0.0
            forgetness_by_seat.setdefault(str(scene_id), {})[int(save_stage)] = fs
            replay_priority_by_seat.setdefault(str(scene_id), {})[int(save_stage)] = us

    return dict(
        forgetness_by_seat=forgetness_by_seat,
        replay_priority_by_seat=replay_priority_by_seat,
        forgetness_by_class={int(k): float(v) for k, v in forgetness_by_class.items()},
        replay_priority_by_class={int(k): float(v) for k, v in replay_priority_by_class.items()},
        q_trajectories=trajectories if return_trajectories else None,
        meta=dict(
            q_metric=str(Q_METRIC),
            q_formula=str(Q_FORMULA),
            old_classes=old_classes_i,
            new_classes=new_classes_i,
            iou_mode=iou_mode_norm,
            iou_thr=float(iou_thr),
            eps=float(eps),
            # alpha/beta kept for backward compatibility (legacy recall q); unused.
            alpha=float(alpha),
            beta=float(beta),
            slope_k_start=int(slope_k_start),
            slope_k_end=int(slope_k_end),
            slope_window=int(slope_k_end) - int(slope_k_start),
            object_count_cap=int(object_count_cap),
            replay_priority_policy=dict(replay_policy),
        ),
    )


def _normalize_design1_q_metric(q_metric: Any) -> str:
    raw = 'f1' if q_metric is None else str(q_metric)
    out = raw.strip().lower()
    if out not in ('f1', 'recall'):
        raise ValueError(
            "learning_dynamics_design1 q_metric must be one of ['f1', 'recall'], "
            f"got {q_metric!r}"
        )
    return str(out)


def _q_formula_for_metric(q_metric: str) -> str:
    if str(q_metric) == 'recall':
        return 'TP/(TP+FN+eps)'
    return str(Q_FORMULA)


def _q_from_stat_by_metric(
        stat: Mapping[str, Any],
        *,
        q_metric: str,
        eps: float) -> Optional[float]:
    if not isinstance(stat, Mapping):
        return None
    gt_count = _safe_float(stat.get('gt_count', 0.0), default=0.0)
    if gt_count <= 0.0:
        return None
    tp = _get_stat(stat, 'tp', 0.0)
    fp = _get_stat(stat, 'fp', 0.0)
    fn = _get_stat(stat, 'fn', 0.0)
    if str(q_metric) == 'recall':
        denom = float(tp + fn + float(eps))
        if denom <= 0.0:
            return 0.0
        return float(_clamp01(float(tp / denom)))
    return float(f1_score(tp=tp, fp=fp, fn=fn, eps=float(eps)))


def _build_q_trajectories_design1(
        metrics_by_k: Mapping[int, Any],
        *,
        q_metric: str,
        eps: float) -> Dict[str, Dict[int, Dict[int, List[float]]]]:
    if not isinstance(metrics_by_k, Mapping) or not metrics_by_k:
        return {}
    ks = sorted(int(k) for k in metrics_by_k.keys())
    k_to_idx = {k: i for i, k in enumerate(ks)}
    k_len = int(len(ks))
    out: Dict[str, Dict[int, Dict[int, List[float]]]] = {}
    for k, seats_obj in metrics_by_k.items():
        ki = k_to_idx.get(int(k), None)
        if ki is None:
            continue
        for (scene_id, save_stage), per_cls in _iter_seat_metrics(seats_obj):
            out.setdefault(str(scene_id), {}).setdefault(int(save_stage), {})
            for cid, stat in per_cls.items():
                q = _q_from_stat_by_metric(
                    stat,
                    q_metric=str(q_metric),
                    eps=float(eps),
                )
                if q is None:
                    continue
                traj = out[str(scene_id)][int(save_stage)].setdefault(
                    int(cid),
                    [0.0] * k_len,
                )
                traj[int(ki)] = float(q)
    return out


def _compute_design1_class_need(
        metrics_by_k: Mapping[int, Any],
        *,
        class_ids: Sequence[int],
        q_metric: str,
        eps: float,
        design_version: int = 1) -> Tuple[Dict[int, float], Dict[int, float], Dict[int, float]]:
    """Compute class need = ((1-q_cur)+max(0,q_best-q_cur))*q_best, normalized.

    Args:
        design_version: 1 = original (gt-weighted aggregation),
                        2 = per-scene aggregation (Bug C.1 fix) + single-checkpoint fallback (Bug C.4).
    """
    class_ids_i = sorted(set(_as_int_list(class_ids)))
    if not class_ids_i:
        return {}, {}, {}

    ks = sorted(int(k) for k in metrics_by_k.keys()) if isinstance(metrics_by_k, Mapping) else []
    if not ks:
        return {int(c): 0.0 for c in class_ids_i}, {}, {}
    k_to_idx = {k: i for i, k in enumerate(ks)}
    n_k = int(len(ks))
    num = {int(c): np.zeros((n_k,), dtype=float) for c in class_ids_i}
    den = {int(c): np.zeros((n_k,), dtype=float) for c in class_ids_i}

    for k, seats_obj in (metrics_by_k.items() if isinstance(metrics_by_k, Mapping) else []):
        ki = k_to_idx.get(int(k), None)
        if ki is None:
            continue
        for _, per_cls in _iter_seat_metrics(seats_obj):
            for cid, stat in per_cls.items():
                cid_i = int(cid)
                if cid_i not in num:
                    continue
                gt = _safe_float(stat.get('gt_count', 0.0), default=0.0)
                if gt <= 0.0:
                    continue
                q = _q_from_stat_by_metric(stat, q_metric=str(q_metric), eps=float(eps))
                if q is None:
                    continue
                if int(design_version) >= 2:
                    # Bug C.1 fix: per-scene equal weight (each scene with
                    # gt>0 contributes one unit so scenes with many objects
                    # do not dominate class-level q statistics).
                    num[cid_i][int(ki)] += float(q)
                    den[cid_i][int(ki)] += 1.0
                else:
                    # Design 1 original: gt-count weighted aggregation.
                    num[cid_i][int(ki)] += float(q) * float(gt)
                    den[cid_i][int(ki)] += float(gt)

    class_q_cur: Dict[int, float] = {}
    class_q_best: Dict[int, float] = {}
    class_need_raw: Dict[int, float] = {}
    for cid in class_ids_i:
        q_series = np.zeros((n_k,), dtype=float)
        mask = den[cid] > 0.0
        if mask.any():
            q_series[mask] = num[cid][mask] / den[cid][mask]
        q_cur = float(q_series[-1]) if q_series.size > 0 else 0.0
        q_best = float(np.max(q_series)) if q_series.size > 0 else 0.0
        q_cur = float(_clamp01(q_cur))
        q_best = float(_clamp01(q_best))
        class_q_cur[int(cid)] = q_cur
        class_q_best[int(cid)] = q_best
        deficit = float(max(0.0, 1.0 - q_cur))
        forgetting = float(max(0.0, q_best - q_cur))
        class_need_raw[int(cid)] = float((deficit + forgetting) * q_best)

    total = float(sum(class_need_raw.values()))
    if total <= 0.0:
        if int(design_version) >= 2 and len(class_ids_i) > 0:
            # Bug C.4 fix: when only 1 checkpoint exists (K=1) q_cur==q_best
            # for all classes, making all needs 0.  Fall back to uniform need
            # so unary scores are not all zero.
            import warnings
            warnings.warn(
                "LD design 2: class_need total is 0 (likely single-checkpoint "
                f"K={n_k}). Falling back to uniform need over "
                f"{len(class_ids_i)} classes.",
                stacklevel=2,
            )
            uniform = 1.0 / float(len(class_ids_i))
            class_need = {int(cid): uniform for cid in class_ids_i}
        else:
            class_need = {int(cid): 0.0 for cid in class_ids_i}
    else:
        class_need = {int(cid): float(v / total) for cid, v in class_need_raw.items()}
    return class_need, class_q_cur, class_q_best


def _compute_design1_seat_class_terms(
        trajectories: Mapping[str, Mapping[int, Mapping[int, Sequence[float]]]],
        *,
        class_ids: Sequence[int]) -> Dict[str, Dict[int, Dict[int, Dict[str, float]]]]:
    class_set = set(_as_int_list(class_ids))
    out: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {}
    for scene_id, by_stage in (trajectories.items() if isinstance(trajectories, Mapping) else []):
        sid = str(scene_id)
        for save_stage, per_cls in (by_stage.items() if isinstance(by_stage, Mapping) else []):
            try:
                st = int(save_stage)
            except Exception:
                continue
            for cid, traj in (per_cls.items() if isinstance(per_cls, Mapping) else []):
                cid_i = int(cid)
                if class_set and cid_i not in class_set:
                    continue
                if traj is None:
                    continue
                q = [float(_clamp01(_safe_float(x, default=0.0))) for x in list(traj)]
                if not q:
                    continue
                r_start = float(q[0])
                r_end = float(q[-1])
                r_best = float(max(q))
                g = 0.0
                for i in range(1, len(q)):
                    g += float(max(0.0, q[i] - q[i - 1]))
                d = float(max(0.0, r_best - r_end))
                u = float(g * r_best + d)
                out.setdefault(sid, {}).setdefault(int(st), {})[cid_i] = dict(
                    g=float(g),
                    r_best=float(r_best),
                    d=float(d),
                    u=float(max(0.0, u)),
                    r_start=float(r_start),
                    r_end=float(r_end),
                )
    return out


def compute_learning_dynamics_design1_scores(
        metrics_by_k: Mapping[int, Any],
        *,
        class_ids: Sequence[int],
        new_classes: Optional[Sequence[int]] = None,
        q_metric: str = 'f1',
        eps: float = 1e-9,
        design_version: int = 1) -> Dict[str, Any]:
    """Compute Design-1/2 class need and per-seat terms from TP/FP/FN trajectories.

    Args:
        design_version: 1 = original Design-1 behaviour (backward compatible).
                        2 = Design-2 with per-scene aggregation and single-checkpoint fallback.
    """
    q_metric_norm = _normalize_design1_q_metric(q_metric)
    class_ids_i = sorted(set(_as_int_list(class_ids)))
    new_classes_i = sorted(set(_as_int_list(new_classes or [])))
    dv = max(1, int(design_version))

    if not isinstance(metrics_by_k, Mapping) or not metrics_by_k:
        return dict(
            q_metric=str(q_metric_norm),
            eps=float(eps),
            class_ids=class_ids_i,
            new_classes=new_classes_i,
            class_need={int(c): 0.0 for c in class_ids_i},
            class_q_current={},
            class_q_best={},
            seat_class_terms={},
        )

    trajectories = _build_q_trajectories_design1(
        metrics_by_k,
        q_metric=str(q_metric_norm),
        eps=float(eps),
    )
    class_need, class_q_cur, class_q_best = _compute_design1_class_need(
        metrics_by_k,
        class_ids=class_ids_i,
        q_metric=str(q_metric_norm),
        eps=float(eps),
        design_version=dv,
    )
    seat_class_terms = _compute_design1_seat_class_terms(
        trajectories,
        class_ids=class_ids_i,
    )

    return dict(
        q_metric=str(q_metric_norm),
        eps=float(eps),
        class_ids=class_ids_i,
        new_classes=new_classes_i,
        class_need={int(k): float(v) for k, v in class_need.items()},
        class_q_current={int(k): float(v) for k, v in class_q_cur.items()},
        class_q_best={int(k): float(v) for k, v in class_q_best.items()},
        seat_class_terms=seat_class_terms,
    )


def topk_seats(score_by_seat: Mapping[str, Mapping[Union[int, str], float]],
              k: int,
              *,
              seed: int = 0) -> List[Dict[str, Any]]:
    """Top-k seats with seeded-random tie-breaking (reproducible)."""
    k = int(k)
    if k <= 0:
        return []
    # Deterministic base ordering for deterministic RNG assignment.
    items = []
    for scene_id, by_stage in (score_by_seat.items() if isinstance(score_by_seat, Mapping) else []):
        if not isinstance(by_stage, Mapping):
            continue
        for save_stage, score in by_stage.items():
            try:
                st = int(save_stage)
            except Exception:
                continue
            sc = _safe_float(score, default=0.0)
            items.append((str(scene_id), int(st), float(sc)))
    items.sort(key=lambda x: (x[0], x[1]))

    try:
        import numpy as np
        rng = np.random.RandomState(int(seed))
        items_with_r = [(sid, st, sc, float(rng.rand())) for sid, st, sc in items]
    except Exception:
        items_with_r = [(sid, st, sc, 0.0) for sid, st, sc in items]

    items_with_r.sort(key=lambda x: (-x[2], x[3]))
    out = []
    for sid, st, sc, _ in items_with_r[:k]:
        out.append(dict(scene_id=str(sid), save_stage=int(st), score=float(sc)))
    return out


def compute_reviewing_entry_weights_ld_drop(
        prev_seats: Any,
        curr_seats: Any,
        *,
        old_classes: Sequence[int],
        iou_mode: str,
        q_metric: str = 'f1',
        object_count_cap: int = 20,
        eps: float = 1e-9,
        eta: float = 5.0,
        normalize_by_gt_weight: bool = True,
        w_entry_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Compute SUNRGBD reviewing entry weights from per-seat q drop (LD-style).

    This derives per-entry sampling weights for memory seats at a reviewing
    update point k by comparing two consecutive per-seat stat snapshots:
      prev (k-1) vs curr (k)

    The seat-local drop signal is:
      D_s = Σ_c a_{s,c} * max(0, q_prev - q_curr)
    where:
      q is selected by `q_metric` in {'f1', 'recall'} at IoU τ:
        - f1: 2TP/(2TP+FP+FN+eps)
        - recall: TP/(TP+FN+eps)
      a_{s,c} = min(gt_count_{s,c}, cap)

    If normalize_by_gt_weight=True, D_s is divided by Σ_c a_{s,c} (per seat).

    Args:
      prev_seats / curr_seats: seat stats in the same JSON-friendly layout as
        `learning_dynamics/stage_<t>/memory_stats_k*.json` ("seats" list), or
        the legacy `metrics/learning_dynamics/*/memory_stats_k*.json` layout, or
        the legacy mapping format accepted by `compute_learning_dynamics_scores`.
      old_classes: class indices to include in the aggregation.
      iou_mode: string IoU threshold (validated; metadata only; stats are assumed
        to already be computed at this IoU).
      q_metric: q definition used to compute drop ('f1' or 'recall').
      w_entry_max: optional hard cap on the returned sampling weight.

    Returns:
      Dict with:
        - weights_by_uid: {replay_unique_id: weight}
        - seat_drop_by_uid: {replay_unique_id: raw_drop}
        - seat_denom_by_uid: {replay_unique_id: gt_weight_sum}
    """
    q_metric_norm = _normalize_design1_q_metric(q_metric)
    old_classes_i = sorted(set(_as_int_list(old_classes)))
    eps = float(eps)
    if eps <= 0.0:
        eps = 1e-9

    if str(q_metric_norm) == 'f1':
        metrics_by_k = {0: prev_seats, 1: curr_seats}
        scores = compute_learning_dynamics_scores(
            metrics_by_k,
            old_classes=list(old_classes_i),
            new_classes=[],
            iou_mode=str(iou_mode),
            alpha=1.0,
            beta=1.0,
            slope_k_start=0,
            slope_k_end=1,
            object_count_cap=int(object_count_cap),
            eps=float(eps),
            return_trajectories=False,
        )
        drop_by_seat = scores.get('forgetness_by_seat', {}) or {}
    else:
        drop_by_seat = {}
        prev_by_seat = {
            tuple(seat_key): per_cls
            for seat_key, per_cls in _iter_seat_metrics(prev_seats)
        }
        curr_by_seat = {
            tuple(seat_key): per_cls
            for seat_key, per_cls in _iter_seat_metrics(curr_seats)
        }
        for seat_key, prev_per_cls in prev_by_seat.items():
            scene_id, save_stage = str(seat_key[0]), int(seat_key[1])
            curr_per_cls = curr_by_seat.get((scene_id, save_stage), {})
            raw_drop = 0.0
            for c in old_classes_i:
                stat_prev = prev_per_cls.get(int(c), None)
                if not isinstance(stat_prev, Mapping):
                    continue
                w = _object_weight_for_class(
                    stat_prev, object_count_cap=int(object_count_cap)
                )
                if w <= 0.0:
                    continue
                q_prev = _q_from_stat_by_metric(
                    stat_prev,
                    q_metric=str(q_metric_norm),
                    eps=float(eps),
                )
                if q_prev is None:
                    continue
                q_curr = 0.0
                stat_curr = (
                    curr_per_cls.get(int(c), None)
                    if isinstance(curr_per_cls, Mapping) else None
                )
                if isinstance(stat_curr, Mapping):
                    q_curr_raw = _q_from_stat_by_metric(
                        stat_curr,
                        q_metric=str(q_metric_norm),
                        eps=float(eps),
                    )
                    if q_curr_raw is not None:
                        q_curr = float(q_curr_raw)
                raw_drop += float(w) * float(
                    max(0.0, float(q_prev) - float(q_curr))
                )
            drop_by_seat.setdefault(str(scene_id), {})[int(save_stage)] = float(
                max(0.0, raw_drop)
            )

    denom_by_seat: Dict[SeatKey, float] = {}
    # Use prev stats for stable gt_count-based weights.
    for seat_key, per_cls in _iter_seat_metrics(prev_seats):
        denom = 0.0
        for c in old_classes_i:
            stat = per_cls.get(int(c), None)
            if not isinstance(stat, Mapping):
                continue
            w = _object_weight_for_class(stat, object_count_cap=int(object_count_cap))
            denom += float(w)
        if denom > 0.0:
            denom_by_seat[seat_key] = float(denom)

    weights_by_uid: Dict[str, float] = {}
    seat_drop_by_uid: Dict[str, float] = {}
    seat_denom_by_uid: Dict[str, float] = {}

    eta = float(eta)
    if not (eta == eta) or eta < 0.0:
        eta = 0.0
    w_entry_max_f = None
    if w_entry_max is not None:
        try:
            w_entry_max_f = float(w_entry_max)
        except Exception:
            w_entry_max_f = None
    if w_entry_max_f is not None and (not (w_entry_max_f == w_entry_max_f) or w_entry_max_f <= 1.0):
        w_entry_max_f = None

    for (scene_id, save_stage), denom in denom_by_seat.items():
        raw_drop = float(drop_by_seat.get(str(scene_id), {}).get(int(save_stage), 0.0))
        if not (raw_drop == raw_drop) or raw_drop < 0.0:
            raw_drop = 0.0
        norm_drop = raw_drop
        if bool(normalize_by_gt_weight):
            norm_drop = raw_drop / float(denom + eps)

        w = 1.0 + eta * float(max(0.0, norm_drop))
        w = float(max(1.0, w))
        if w_entry_max_f is not None:
            w = float(min(w, w_entry_max_f))

        uid = f"{str(scene_id)}_stage{int(save_stage)}"
        weights_by_uid[str(uid)] = float(w)
        seat_drop_by_uid[str(uid)] = float(raw_drop)
        seat_denom_by_uid[str(uid)] = float(denom)

    return dict(
        q_metric=str(q_metric_norm),
        q_formula=str(_q_formula_for_metric(str(q_metric_norm))),
        weights_by_uid=weights_by_uid,
        seat_drop_by_uid=seat_drop_by_uid,
        seat_denom_by_uid=seat_denom_by_uid,
        normalize_by_gt_weight=bool(normalize_by_gt_weight),
        object_count_cap=int(object_count_cap),
        eps=float(eps),
        eta=float(eta),
        w_entry_max=float(w_entry_max_f) if w_entry_max_f is not None else None,
        iou_mode=str(_normalize_iou_mode(iou_mode)),
    )


def compute_reviewing_entry_weights_ld_f1_drop(
        prev_seats: Any,
        curr_seats: Any,
        *,
        old_classes: Sequence[int],
        iou_mode: str,
        q_metric: str = 'f1',
        object_count_cap: int = 20,
        eps: float = 1e-9,
        eta: float = 5.0,
        normalize_by_gt_weight: bool = True,
        w_entry_max: Optional[float] = None,
) -> Dict[str, Any]:
    """Backward-compatible alias for compute_reviewing_entry_weights_ld_drop()."""
    return compute_reviewing_entry_weights_ld_drop(
        prev_seats,
        curr_seats,
        old_classes=old_classes,
        iou_mode=iou_mode,
        q_metric=q_metric,
        object_count_cap=object_count_cap,
        eps=eps,
        eta=eta,
        normalize_by_gt_weight=normalize_by_gt_weight,
        w_entry_max=w_entry_max,
    )
