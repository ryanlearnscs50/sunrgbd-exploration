"""Helpers for SUNRGBD reviewing / learning-dynamics orchestration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def compute_review_segment_times(*,
                                 stage_epochs: int,
                                 repeat_times: int,
                                 review_fractions: List[float]) -> List[int]:
    """Map review fractions to integer RepeatDataset-time budgets."""
    stage_epochs = int(stage_epochs)
    repeat_times = int(repeat_times)
    assert stage_epochs >= 1 and repeat_times >= 1
    total_units = int(stage_epochs * repeat_times)
    assert total_units >= 1

    fracs = [float(x) for x in (review_fractions or [])]
    for f in fracs:
        assert 0.0 < f < 1.0, f
    fracs = sorted(fracs)

    boundaries = []
    for f in fracs:
        b = int(np.floor(float(f) * float(total_units)))
        b = max(1, min(total_units - 1, b))
        if not boundaries or b > boundaries[-1]:
            boundaries.append(b)

    ends = boundaries + [total_units]
    segs = []
    prev = 0
    for e in ends:
        seg = int(e - prev)
        if seg <= 0:
            continue
        segs.append(seg)
        prev = int(e)
    assert sum(segs) == total_units, (segs, total_units)
    return segs


def resolve_effective_ld_reviewing_params(*,
                                          reviewing_ld_object_count_cap: Optional[Any],
                                          reviewing_ld_w_entry_max: Optional[Any],
                                          ld_object_count_cap: Optional[Any],
                                          learning_dynamics_enabled: bool) -> Tuple[int, Optional[float]]:
    effective_ld_cap = reviewing_ld_object_count_cap
    if effective_ld_cap is None:
        effective_ld_cap = int(ld_object_count_cap) if learning_dynamics_enabled else 20
    else:
        effective_ld_cap = int(effective_ld_cap)
    if effective_ld_cap <= 0:
        effective_ld_cap = 20

    effective_w_entry_max = reviewing_ld_w_entry_max
    if effective_w_entry_max is not None:
        try:
            effective_w_entry_max = float(effective_w_entry_max)
        except Exception:
            effective_w_entry_max = None

    return int(effective_ld_cap), effective_w_entry_max


def build_review_weight_policy(*,
                               reviewing_weight_policy: str,
                               alpha_drop: float,
                               beta_ap: float,
                               gamma: float,
                               w_max: float,
                               eta: float,
                               fixed_review_weight: Optional[float],
                               drop_clamp_min: float,
                               reviewing_ld_q_metric: str,
                               reviewing_ld_q_formula: str,
                               ld_iou_thr: float,
                               ld_iou_mode: str,
                               ld_eps: float,
                               effective_ld_cap: int,
                               reviewing_ld_normalize_by_gt_weight: bool,
                               effective_w_entry_max: Optional[float]) -> Tuple[str, Dict[str, Any]]:
    if reviewing_weight_policy == 'ap_drop':
        return (
            'AP',
            dict(
                type='drop_dominant_sum',
                alpha_drop=float(alpha_drop),
                beta_ap=float(beta_ap),
                gamma=float(gamma),
                w_max=float(w_max),
                eta=float(eta),
                drop_clamp_min=float(drop_clamp_min),
            ),
        )
    if reviewing_weight_policy == 'fixed':
        return (
            'fixed_weight',
            dict(
                type='fixed',
                fixed_value=float(fixed_review_weight),
            ),
        )

    metric = 'q_recall_drop' if str(reviewing_ld_q_metric) == 'recall' else 'q_f1_drop'
    return (
        metric,
        dict(
            type='ld_drop',
            q_metric=str(reviewing_ld_q_metric),
            q_formula=str(reviewing_ld_q_formula),
            iou_thr=float(ld_iou_thr),
            iou_mode=str(ld_iou_mode),
            eps=float(ld_eps),
            object_count_cap=int(effective_ld_cap),
            normalize_by_gt_weight=bool(reviewing_ld_normalize_by_gt_weight),
            eta=float(eta),
            w_entry_max=float(effective_w_entry_max) if effective_w_entry_max is not None else None,
        ),
    )


def build_reviewing_eval_payload(*,
                                 stage_id: int,
                                 review_k: int,
                                 eval_iou_thrs: List[float],
                                 weight_iou_thr: float,
                                 reviewing_weight_policy: str,
                                 weight_metric: str,
                                 weight_policy_desc: Dict[str, Any],
                                 sampling_mode: str,
                                 memory_share_max: float,
                                 seed_offset: int) -> Dict[str, Any]:
    return {
        'stage_id': int(stage_id),
        'review_k': int(review_k),
        'split': 'train(memory_bank_subset)',
        'metric': f"AP/AR@{','.join(f'{float(x):.2f}' for x in eval_iou_thrs)}",
        'eval_iou_thrs': [float(x) for x in eval_iou_thrs],
        'weight_iou_thr': float(weight_iou_thr),
        'weight_policy_type': str(reviewing_weight_policy),
        'weight_metric': str(weight_metric),
        'weight_policy': dict(weight_policy_desc),
        'sampling': dict(
            mode=str(sampling_mode),
            memory_share_max=float(memory_share_max),
            seed_offset=int(seed_offset),
        ),
        'by_intro_stage': {},
    }
