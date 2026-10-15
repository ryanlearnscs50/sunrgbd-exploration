"""Compatibility-weighted memory bank updates.

This module keeps unary and compatibility terms on comparable scales:
- unary is percentile-normalized to [0, 1]
- compatibility uses MEAN positive dot similarity, not SUM

`lambda_compat` controls the mix:
- `lambda_compat=0`: pure unary ranking
- `lambda_compat=1`: pure compatibility (with unary seed fallback at stage-1)

For stage-ratio updates, this module also provides deterministic, ratio-targeted
swap behavior with explicit shortfall reporting when candidate supply is
insufficient.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch


def _stable_argsort(values: torch.Tensor, descending: bool) -> torch.Tensor:
    """Deterministic argsort, preferring stable sort when supported."""
    try:
        return torch.argsort(values, descending=descending, stable=True)
    except TypeError:
        return torch.argsort(values, descending=descending)


def _normalize_embeddings(emb: torch.Tensor) -> torch.Tensor:
    """L2-normalize row embeddings with safe zero handling."""
    if emb.ndim != 2:
        raise ValueError(f"Expected 2D embedding tensor, got shape={tuple(emb.shape)}")
    if emb.numel() == 0:
        return emb
    norms = torch.linalg.norm(emb, dim=1, keepdim=True)
    safe_norms = torch.where(norms > 0, norms, torch.ones_like(norms))
    return emb / safe_norms


def _largest_remainder_allocation(weights: Mapping[int, int],
                                  total: int) -> Dict[int, int]:
    """Allocate `total` integers proportionally to `weights`.

    Uses largest-remainder with deterministic tie-break: stage id ascending.
    """
    total = int(total)
    keys = sorted(int(k) for k in weights.keys())
    if total <= 0:
        return {k: 0 for k in keys}

    clean_weights: Dict[int, int] = {}
    for k in keys:
        w = int(weights.get(k, 0))
        if w < 0:
            raise ValueError(f"Weight must be >=0, got stage={k}, weight={w}")
        clean_weights[k] = w

    denom = int(sum(clean_weights.values()))
    if denom <= 0:
        raise ValueError(
            f"Cannot allocate total={total} with non-positive weight sum: {clean_weights}"
        )

    base: Dict[int, int] = {}
    remainders: Dict[int, float] = {}
    used = 0
    for k in keys:
        raw = float(total) * float(clean_weights[k]) / float(denom)
        flo = int(raw)
        base[k] = flo
        remainders[k] = float(raw - flo)
        used += flo

    remaining = int(total - used)
    if remaining > 0:
        order = sorted(
            keys,
            key=lambda stage: (-remainders[stage], int(stage)),
        )
        for stage in order[:remaining]:
            base[stage] += 1

    if int(sum(base.values())) != int(total):
        raise RuntimeError(
            "Largest-remainder allocation sum mismatch: "
            f"expected={int(total)}, actual={int(sum(base.values()))}, alloc={base}"
        )
    return base


def _count_stages(stage_ids: torch.Tensor) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    if stage_ids.numel() == 0:
        return counts
    for value in stage_ids.detach().cpu().tolist():
        stage = int(value)
        counts[stage] = int(counts.get(stage, 0)) + 1
    return counts


def compute_percentile_rank(u_raw: torch.Tensor) -> torch.Tensor:
    """Return deterministic percentile ranks in [0, 1], shape-preserving."""
    if not isinstance(u_raw, torch.Tensor):
        raise TypeError(f"u_raw must be torch.Tensor, got {type(u_raw)}")
    flat = u_raw.reshape(-1)
    n = int(flat.numel())
    if n == 0:
        return torch.empty_like(u_raw, dtype=torch.float32)

    work = flat.to(dtype=torch.float32)
    if n == 1:
        return torch.zeros_like(work).reshape(u_raw.shape)

    order = _stable_argsort(work, descending=False)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(n, device=order.device, dtype=order.dtype)
    out = ranks.to(dtype=torch.float32) / float(n - 1)
    return out.reshape(u_raw.shape)


def compute_comp_mean(E_cand: torch.Tensor, E_bank: torch.Tensor) -> torch.Tensor:
    """Mean compatibility for each candidate against current bank."""
    if not isinstance(E_cand, torch.Tensor) or not isinstance(E_bank, torch.Tensor):
        raise TypeError("E_cand and E_bank must be torch.Tensor")
    if E_cand.ndim != 2 or E_bank.ndim != 2:
        raise ValueError(
            "Expected 2D tensors: "
            f"E_cand.shape={tuple(E_cand.shape)}, E_bank.shape={tuple(E_bank.shape)}"
        )
    if E_cand.shape[1] != E_bank.shape[1]:
        raise ValueError(
            f"Embedding dim mismatch: {int(E_cand.shape[1])} vs {int(E_bank.shape[1])}"
        )

    n_cand = int(E_cand.shape[0])
    n_bank = int(E_bank.shape[0])
    if n_cand == 0:
        return torch.empty((0,), dtype=torch.float32, device=E_cand.device)
    if n_bank == 0:
        return torch.zeros((n_cand,), dtype=torch.float32, device=E_cand.device)

    cand = _normalize_embeddings(E_cand.to(dtype=torch.float32))
    bank = _normalize_embeddings(E_bank.to(dtype=torch.float32))
    k = torch.clamp(cand @ bank.t(), min=0.0)
    return k.mean(dim=1)


def compute_class_balance_weights(count_bank: torch.Tensor) -> torch.Tensor:
    """Diminishing-returns class weight: 1/sqrt(1 + count_bank[c]).

    This is the Design-1 (legacy) formula.  Design-2 uses
    :func:`compute_class_balance_weights_v2` instead.
    """
    if not isinstance(count_bank, torch.Tensor):
        raise TypeError(f"count_bank must be torch.Tensor, got {type(count_bank)}")
    if count_bank.ndim != 1:
        raise ValueError(
            f"count_bank must be 1D tensor, got shape={tuple(count_bank.shape)}"
        )
    if count_bank.numel() == 0:
        return torch.empty_like(count_bank, dtype=torch.float32)
    count = count_bank.to(dtype=torch.float32)
    if not bool(torch.all(torch.isfinite(count)).item()):
        raise ValueError("count_bank contains non-finite values")
    if not bool(torch.all(count >= 0.0).item()):
        raise ValueError("count_bank must be non-negative")
    return torch.rsqrt(1.0 + count)


def compute_class_balance_weights_v2(
    count_bank: torch.Tensor,
    *,
    w_max: float = 10.0,
) -> torch.Tensor:
    """Stronger inverse-count class balance weight for LD Design 2.

    Formula: w[c] = min(1 / (1 + count_bank[c]), w_max)

    Uses ``1/(1+count)`` instead of Design-1's ``1/sqrt(1+count)`` so the
    correction fully counters the linear object-count bias introduced by
    supply weighting.  ``w_max`` caps weights for zero/rare classes to
    prevent a single missing class from dominating the score (default 10
    means a zero-count class is weighted at most 10x a common class).

    Args:
        count_bank: 1-D tensor of per-class object counts in the bank
            (must use **raw** counts, not scaled supply values).
        w_max: Upper bound on any single class weight.  Default 10.0 is
            chosen to be large enough to meaningfully boost rare classes
            while staying stable (scores remain O(10) of the median).
    """
    if not isinstance(count_bank, torch.Tensor):
        raise TypeError(f"count_bank must be torch.Tensor, got {type(count_bank)}")
    if count_bank.ndim != 1:
        raise ValueError(
            f"count_bank must be 1D tensor, got shape={tuple(count_bank.shape)}"
        )
    if count_bank.numel() == 0:
        return torch.empty_like(count_bank, dtype=torch.float32)
    count = count_bank.to(dtype=torch.float32)
    if not bool(torch.all(torch.isfinite(count)).item()):
        raise ValueError("count_bank contains non-finite values")
    if not bool(torch.all(count >= 0.0).item()):
        raise ValueError("count_bank must be non-negative")
    w = 1.0 / (1.0 + count)
    if float(w_max) > 0.0:
        w = torch.clamp(w, max=float(w_max))
    return w


def compute_redundancy_penalty(
    E_cand: torch.Tensor,
    E_bank: torch.Tensor,
    *,
    topk: int = 5,
) -> torch.Tensor:
    """Per-candidate redundancy score based on top-k mean similarity to bank.

    Higher value ⇒ candidate is more redundant (similar to existing bank).
    Uses top-k mean instead of max to be less sensitive to a single near-
    duplicate outlier while still penalising concentrated similarity.

    Args:
        E_cand: (N_cand, D) candidate embeddings.
        E_bank: (N_bank, D) bank embeddings.
        topk: Number of nearest bank neighbours to average.  K=5 is a
            pragmatic default: large enough to smooth noise, small enough
            to stay sensitive.  Clamped to bank size internally.

    Returns:
        (N_cand,) tensor in [0, 1].
    """
    if not isinstance(E_cand, torch.Tensor) or not isinstance(E_bank, torch.Tensor):
        raise TypeError("E_cand and E_bank must be torch.Tensor")
    if E_cand.ndim != 2 or E_bank.ndim != 2:
        raise ValueError(
            "Expected 2D tensors: "
            f"E_cand.shape={tuple(E_cand.shape)}, E_bank.shape={tuple(E_bank.shape)}"
        )

    n_cand = int(E_cand.shape[0])
    n_bank = int(E_bank.shape[0])
    if n_cand == 0:
        return torch.empty((0,), dtype=torch.float32, device=E_cand.device)
    if n_bank == 0:
        return torch.zeros((n_cand,), dtype=torch.float32, device=E_cand.device)

    cand = _normalize_embeddings(E_cand.to(dtype=torch.float32))
    bank = _normalize_embeddings(E_bank.to(dtype=torch.float32))
    sim = torch.clamp(cand @ bank.t(), min=0.0)  # (n_cand, n_bank)
    k = max(1, min(int(topk), n_bank))
    topk_vals, _ = torch.topk(sim, k=k, dim=1)  # (n_cand, k)
    return topk_vals.mean(dim=1)


def stage1_greedy_fill(u_norm: torch.Tensor, E_all: torch.Tensor, K: int,
                       lambda_compat: float) -> torch.Tensor:
    """Greedy stage-1 fill using incremental mean-compatibility updates."""
    if not isinstance(u_norm, torch.Tensor) or not isinstance(E_all, torch.Tensor):
        raise TypeError("u_norm and E_all must be torch.Tensor")
    if u_norm.ndim != 1:
        raise ValueError(f"u_norm must be 1D, got shape={tuple(u_norm.shape)}")
    if E_all.ndim != 2:
        raise ValueError(f"E_all must be 2D, got shape={tuple(E_all.shape)}")
    if int(u_norm.shape[0]) != int(E_all.shape[0]):
        raise ValueError(
            "u_norm/E_all size mismatch: "
            f"{int(u_norm.shape[0])} vs {int(E_all.shape[0])}"
        )
    if not (0.0 <= float(lambda_compat) <= 1.0):
        raise ValueError(f"lambda_compat must be in [0,1], got {lambda_compat}")

    n = int(u_norm.shape[0])
    k = max(0, int(K))
    if n == 0 or k == 0:
        return torch.empty((0,), dtype=torch.long, device=u_norm.device)

    max_take = min(n, k)
    emb = _normalize_embeddings(E_all.to(dtype=torch.float32))
    unary = u_norm.to(dtype=torch.float32)

    selected_mask = torch.zeros((n,), dtype=torch.bool, device=u_norm.device)
    comp_vec = torch.zeros((n,), dtype=torch.float32, device=u_norm.device)
    selected: List[int] = []
    selected_count = 0

    for _ in range(max_take):
        if selected_count == 0 and abs(float(lambda_compat) - 1.0) <= 1e-12:
            score = unary.clone()
        else:
            score = (1.0 - float(lambda_compat)) * unary + float(lambda_compat) * comp_vec
        score = score.masked_fill(selected_mask, float('-inf'))
        best_idx = int(torch.argmax(score).item())
        if torch.isneginf(score[best_idx]):
            break

        selected_mask[best_idx] = True
        selected.append(best_idx)

        k_vec = torch.clamp(emb @ emb[best_idx], min=0.0)
        comp_vec = (
            comp_vec * float(selected_count) + k_vec
        ) / float(selected_count + 1)
        selected_count += 1

    return torch.tensor(selected, dtype=torch.long, device=u_norm.device)


def ratio_targeted_swap_update(
    bank_indices: torch.Tensor,
    bank_stage_ids: torch.Tensor,
    E_bank: torch.Tensor,
    u_bank_norm: torch.Tensor,
    cand_indices: torch.Tensor,
    cand_stage_ids: torch.Tensor,
    E_cand: torch.Tensor,
    u_cand_norm: torch.Tensor,
    lambda_compat: float,
    target_stage_counts: Mapping[int, int],
    current_stage_id: int,
    *,
    use_class_balance: bool = False,
    bank_unary_base: Optional[torch.Tensor] = None,
    cand_unary_base: Optional[torch.Tensor] = None,
    bank_supply_counts: Optional[torch.Tensor] = None,
    cand_supply_counts: Optional[torch.Tensor] = None,
    design_version: int = 1,
    w_max: float = 10.0,
    min_class_quota: int = 0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Ratio-targeted Stage>=2 update with proportional old-stage evictions."""
    if not (0.0 <= float(lambda_compat) <= 1.0):
        raise ValueError(f"lambda_compat must be in [0,1], got {lambda_compat}")
    if not isinstance(target_stage_counts, Mapping):
        raise TypeError("target_stage_counts must be a mapping")

    if bank_indices.ndim != 1 or bank_stage_ids.ndim != 1 or u_bank_norm.ndim != 1:
        raise ValueError("bank_indices/bank_stage_ids/u_bank_norm must be 1D tensors")
    if cand_indices.ndim != 1 or cand_stage_ids.ndim != 1 or u_cand_norm.ndim != 1:
        raise ValueError("cand_indices/cand_stage_ids/u_cand_norm must be 1D tensors")
    if E_bank.ndim != 2 or E_cand.ndim != 2:
        raise ValueError("E_bank and E_cand must be 2D tensors")

    n_bank = int(bank_indices.shape[0])
    n_cand = int(cand_indices.shape[0])
    if int(bank_stage_ids.shape[0]) != n_bank or int(u_bank_norm.shape[0]) != n_bank:
        raise ValueError("Bank tensor sizes are inconsistent")
    if int(cand_stage_ids.shape[0]) != n_cand or int(u_cand_norm.shape[0]) != n_cand:
        raise ValueError("Candidate tensor sizes are inconsistent")
    if int(E_bank.shape[0]) != n_bank or int(E_cand.shape[0]) != n_cand:
        raise ValueError("Embedding row count does not match index count")
    if n_bank > 0 and int(E_bank.shape[1]) <= 0:
        raise ValueError(f"Invalid E_bank shape: {tuple(E_bank.shape)}")
    if n_cand > 0 and int(E_cand.shape[1]) <= 0:
        raise ValueError(f"Invalid E_cand shape: {tuple(E_cand.shape)}")
    if n_bank > 0 and n_cand > 0 and int(E_bank.shape[1]) != int(E_cand.shape[1]):
        raise ValueError(
            f"Embedding dim mismatch: {int(E_bank.shape[1])} vs {int(E_cand.shape[1])}"
        )

    current_stage_id = int(current_stage_id)
    target_counts: Dict[int, int] = {}
    for k, v in target_stage_counts.items():
        stage = int(k)
        count = int(v)
        if count < 0:
            raise ValueError(
                f"target_stage_counts must be non-negative, got stage={stage}, count={count}"
            )
        target_counts[stage] = count
    if current_stage_id not in target_counts:
        raise ValueError(
            f"target_stage_counts must include current_stage_id={current_stage_id}, "
            f"but got keys={sorted(target_counts.keys())}"
        )

    bank_tokens = bank_indices.to(dtype=torch.long).clone()
    bank_stages = bank_stage_ids.to(dtype=torch.long).clone()
    bank_u = u_bank_norm.to(dtype=torch.float32).clone()
    bank_emb = _normalize_embeddings(E_bank.to(dtype=torch.float32).clone())

    cand_tokens_full = cand_indices.to(dtype=torch.long)
    cand_stages_full = cand_stage_ids.to(dtype=torch.long)
    cand_u_full = u_cand_norm.to(dtype=torch.float32)
    cand_emb_full = _normalize_embeddings(E_cand.to(dtype=torch.float32))

    current_mask = cand_stages_full == int(current_stage_id)
    cand_tokens = cand_tokens_full[current_mask]
    cand_stages = cand_stages_full[current_mask]
    cand_u = cand_u_full[current_mask]
    cand_emb = cand_emb_full[current_mask]

    bank_base = None
    cand_base = None
    bank_supply = None
    cand_supply = None
    class_balance_vec = None
    if bool(use_class_balance):
        if bank_unary_base is None or cand_unary_base is None:
            raise ValueError(
                "use_class_balance=True requires bank_unary_base and cand_unary_base."
            )
        if bank_supply_counts is None or cand_supply_counts is None:
            raise ValueError(
                "use_class_balance=True requires bank_supply_counts and cand_supply_counts."
            )
        if bank_unary_base.ndim != 2 or cand_unary_base.ndim != 2:
            raise ValueError(
                "bank_unary_base/cand_unary_base must be 2D tensors when "
                "use_class_balance=True."
            )
        if bank_supply_counts.ndim != 2 or cand_supply_counts.ndim != 2:
            raise ValueError(
                "bank_supply_counts/cand_supply_counts must be 2D tensors when "
                "use_class_balance=True."
            )
        if int(bank_unary_base.shape[0]) != n_bank:
            raise ValueError(
                f"bank_unary_base row mismatch: {int(bank_unary_base.shape[0])} vs {n_bank}"
            )
        if int(cand_unary_base.shape[0]) != n_cand:
            raise ValueError(
                f"cand_unary_base row mismatch: {int(cand_unary_base.shape[0])} vs {n_cand}"
            )
        if int(bank_supply_counts.shape[0]) != n_bank:
            raise ValueError(
                f"bank_supply_counts row mismatch: {int(bank_supply_counts.shape[0])} vs {n_bank}"
            )
        if int(cand_supply_counts.shape[0]) != n_cand:
            raise ValueError(
                f"cand_supply_counts row mismatch: {int(cand_supply_counts.shape[0])} vs {n_cand}"
            )

        class_dim = int(bank_unary_base.shape[1])
        if class_dim <= 0:
            raise ValueError(
                f"class dimension must be >0, got bank_unary_base.shape={tuple(bank_unary_base.shape)}"
            )
        for name, tensor in (
                ("cand_unary_base", cand_unary_base),
                ("bank_supply_counts", bank_supply_counts),
                ("cand_supply_counts", cand_supply_counts)):
            if int(tensor.shape[1]) != class_dim:
                raise ValueError(
                    f"{name} class dim mismatch: {int(tensor.shape[1])} vs {class_dim}"
                )

        bank_base = bank_unary_base.to(dtype=torch.float32).clone()
        cand_base_full = cand_unary_base.to(dtype=torch.float32)
        bank_supply = bank_supply_counts.to(dtype=torch.float32).clone()
        cand_supply_full = cand_supply_counts.to(dtype=torch.float32)

        cand_base = cand_base_full[current_mask].clone()
        cand_supply = cand_supply_full[current_mask].clone()
        class_balance_vec = torch.sum(bank_supply, dim=0)
        if not bool(torch.all(torch.isfinite(class_balance_vec)).item()):
            raise ValueError("bank_supply_counts produced non-finite class counts")
        if not bool(torch.all(class_balance_vec >= 0.0).item()):
            raise ValueError("bank_supply_counts produced negative class counts")

    counts_before = _count_stages(bank_stages)
    target_current = int(target_counts.get(current_stage_id, 0))
    current_before = int(counts_before.get(current_stage_id, 0))
    required_add_t = max(0, target_current - current_before)

    feasible_candidates = int(cand_tokens.shape[0])
    add_t = min(required_add_t, feasible_candidates)
    shortfall_t = int(required_add_t - add_t)

    target_total = int(sum(target_counts.values()))
    direct_add_capacity = max(0, target_total - int(bank_tokens.shape[0]))
    direct_add_count = min(int(add_t), int(direct_add_capacity))
    evict_total = int(add_t - direct_add_count)

    surplus_by_stage: Dict[int, int] = {}
    for stage, count in counts_before.items():
        stage_i = int(stage)
        if stage_i == current_stage_id:
            continue
        target_i = int(target_counts.get(stage_i, 0))
        surplus = max(0, int(count) - target_i)
        if surplus > 0:
            surplus_by_stage[stage_i] = int(surplus)

    surplus_total = int(sum(surplus_by_stage.values()))
    if evict_total > surplus_total:
        raise RuntimeError(
            "Cannot allocate required evictions from surplus stages: "
            f"evict_total={evict_total}, surplus_total={surplus_total}, "
            f"surplus_by_stage={surplus_by_stage}, counts_before={counts_before}, "
            f"target_counts={target_counts}."
        )

    eviction_plan = _largest_remainder_allocation(surplus_by_stage, evict_total)
    remaining_plan = {int(k): int(v) for k, v in eviction_plan.items()}

    cand_order = list(range(int(cand_tokens.shape[0])))
    if not bool(use_class_balance):
        cand_order.sort(
            key=lambda i: (
                -float(cand_u[i].item()),
                int(cand_tokens[i].item()),
            ))

    actions: List[Dict[str, Any]] = []
    inserted_tokens: List[int] = []
    evicted_tokens: List[int] = []
    inserted = 0
    direct_remaining = int(direct_add_count)
    remaining_candidates = torch.ones(
        (int(cand_tokens.shape[0]),), dtype=torch.bool, device=cand_tokens.device
    )

    def _compute_dynamic_unary():
        if not bool(use_class_balance):
            return bank_u, cand_u, None, bank_u, cand_u
        assert bank_base is not None and cand_base is not None
        assert class_balance_vec is not None
        # Design 2: stronger 1/(1+count) balance with w_max cap.
        if int(design_version) >= 2:
            w_bal = compute_class_balance_weights_v2(
                class_balance_vec, w_max=float(w_max)
            )
        else:
            w_bal = compute_class_balance_weights(class_balance_vec)
        u_bank_dyn = (
            torch.mv(bank_base, w_bal)
            if int(bank_base.shape[0]) > 0
            else torch.empty((0,), dtype=torch.float32, device=w_bal.device)
        )
        u_cand_dyn = (
            torch.mv(cand_base, w_bal)
            if int(cand_base.shape[0]) > 0
            else torch.empty((0,), dtype=torch.float32, device=w_bal.device)
        )
        # Design 2: min per-class quota boost for candidates.
        if int(design_version) >= 2 and int(min_class_quota) > 0 and cand_supply is not None:
            under_quota = (class_balance_vec < float(min_class_quota))
            if bool(torch.any(under_quota).item()):
                boost = torch.mv(
                    cand_supply.clamp(min=0.0),
                    under_quota.to(dtype=torch.float32),
                )
                if float(boost.max().item()) > 0.0:
                    u_cand_dyn = u_cand_dyn + boost
        return (
            compute_percentile_rank(u_bank_dyn),
            compute_percentile_rank(u_cand_dyn),
            w_bal,
            u_bank_dyn,
            u_cand_dyn,
        )

    loop_order = cand_order if not bool(use_class_balance) else [0] * int(cand_tokens.shape[0])
    for ci_seed in loop_order:
        if inserted >= int(add_t):
            break

        bank_u_work, cand_u_work, w_bal, bank_u_raw_work, cand_u_raw_work = (
            _compute_dynamic_unary()
        )
        if bool(use_class_balance):
            cand_rank = cand_u_work.masked_fill(~remaining_candidates, float('-inf'))
            ci = int(torch.argmax(cand_rank).item())
            if torch.isneginf(cand_rank[ci]):
                break
        else:
            ci = int(ci_seed)
            if not bool(remaining_candidates[ci].item()):
                continue

        tok = cand_tokens[ci]
        stg = cand_stages[ci]
        u_s = cand_u_work[ci]
        e_s = cand_emb[ci]
        remaining_candidates[ci] = False

        if direct_remaining > 0:
            bank_tokens = torch.cat([bank_tokens, tok.view(1)], dim=0)
            bank_stages = torch.cat([bank_stages, stg.view(1)], dim=0)
            bank_u = torch.cat([bank_u_work, u_s.view(1)], dim=0)
            bank_emb = torch.cat([bank_emb, e_s.view(1, -1)], dim=0)
            if bool(use_class_balance):
                assert bank_base is not None and cand_base is not None
                assert bank_supply is not None and cand_supply is not None
                assert class_balance_vec is not None
                bank_base = torch.cat([bank_base, cand_base[ci].view(1, -1)], dim=0)
                bank_supply = torch.cat(
                    [bank_supply, cand_supply[ci].view(1, -1)], dim=0
                )
                class_balance_vec = class_balance_vec + cand_supply[ci]
            inserted += 1
            direct_remaining -= 1
            inserted_tokens.append(int(tok.item()))
            actions.append(
                dict(
                    reason='direct_add',
                    candidate_token=int(tok.item()),
                    candidate_stage=int(stg.item()),
                    candidate_unary_percentile=float(u_s.item()),
                    candidate_unary_balanced=float(cand_u_raw_work[ci].item()),
                    evicted_token=None,
                    evicted_stage=None,
                    evicted_unary_percentile=None,
                    evicted_unary_balanced=None,
                    swap_delta=None,
                ))
            continue

        elig_mask = torch.zeros(
            (int(bank_tokens.shape[0]),),
            dtype=torch.bool,
            device=bank_tokens.device,
        )
        for stage, remaining in remaining_plan.items():
            if int(remaining) > 0:
                elig_mask = elig_mask | (bank_stages == int(stage))

        if not bool(torch.any(elig_mask).item()):
            break

        kb = int(bank_tokens.shape[0])
        if kb <= 0:
            break

        if kb == 1:
            comp_s_excl = torch.zeros((1,), dtype=torch.float32, device=bank_tokens.device)
            comp_i_excl = torch.zeros((1,), dtype=torch.float32, device=bank_tokens.device)
        else:
            k_sb = torch.clamp(torch.mv(bank_emb, e_s), min=0.0)
            sum_k = torch.sum(k_sb)
            comp_s_excl = (sum_k - k_sb) / float(kb - 1)

            k_bb = torch.clamp(bank_emb @ bank_emb.t(), min=0.0)
            diag = torch.diagonal(k_bb, offset=0)
            row_sum_excl = torch.sum(k_bb, dim=1) - diag
            comp_i_excl = row_sum_excl / float(kb - 1)

        if int(design_version) >= 2:
            # Design 2: redundancy penalty -- higher similarity to bank is
            # penalised, so prefer swapping in less-redundant candidates and
            # evicting more-redundant bank items.
            delta = (
                (1.0 - float(lambda_compat)) * (u_s - bank_u_work)
                + float(lambda_compat) * (comp_i_excl - comp_s_excl)
            )
        else:
            # Design 1: compatibility reward.
            delta = (
                (1.0 - float(lambda_compat)) * (u_s - bank_u_work)
                + float(lambda_compat) * (comp_s_excl - comp_i_excl)
            )
        masked = delta.masked_fill(~elig_mask, float('-inf'))
        best_idx = int(torch.argmax(masked).item())
        if torch.isneginf(masked[best_idx]):
            break

        ev_token = int(bank_tokens[best_idx].item())
        ev_stage = int(bank_stages[best_idx].item())
        if int(remaining_plan.get(ev_stage, 0)) <= 0:
            raise RuntimeError(
                "Internal eviction-plan mismatch: "
                f"ev_stage={ev_stage}, remaining_plan={remaining_plan}"
            )

        bank_tokens[best_idx] = tok
        bank_stages[best_idx] = stg
        bank_u = bank_u_work.clone()
        bank_u[best_idx] = u_s
        bank_emb[best_idx] = e_s
        if bool(use_class_balance):
            assert bank_base is not None and cand_base is not None
            assert bank_supply is not None and cand_supply is not None
            assert class_balance_vec is not None
            old_supply = bank_supply[best_idx].clone()
            bank_base[best_idx] = cand_base[ci]
            bank_supply[best_idx] = cand_supply[ci]
            class_balance_vec = class_balance_vec + cand_supply[ci] - old_supply

        remaining_plan[ev_stage] -= 1
        inserted += 1
        inserted_tokens.append(int(tok.item()))
        evicted_tokens.append(int(ev_token))
        actions.append(
            dict(
                reason='swap',
                candidate_token=int(tok.item()),
                candidate_stage=int(stg.item()),
                candidate_unary_percentile=float(u_s.item()),
                candidate_unary_balanced=float(cand_u_raw_work[ci].item()),
                evicted_token=int(ev_token),
                evicted_stage=int(ev_stage),
                evicted_unary_percentile=float(bank_u_work[best_idx].item()),
                evicted_unary_balanced=float(bank_u_raw_work[best_idx].item()),
                swap_delta=float(masked[best_idx].item()),
            ))

    if inserted != int(add_t):
        raise RuntimeError(
            "ratio_targeted_swap_update failed to complete planned insertions: "
            f"inserted={inserted}, add_t={add_t}, feasible_candidates={feasible_candidates}, "
            f"required_add_t={required_add_t}, remaining_plan={remaining_plan}"
        )

    counts_after = _count_stages(bank_stages)
    if int(shortfall_t) == 0:
        for stage, target in target_counts.items():
            got = int(counts_after.get(int(stage), 0))
            if int(got) != int(target):
                raise RuntimeError(
                    "Exact composition invariant failed without shortfall: "
                    f"stage={int(stage)}, got={int(got)}, target={int(target)}, "
                    f"counts_after={counts_after}, target_counts={target_counts}"
                )

    eviction_actual: Dict[int, int] = {}
    for token in evicted_tokens:
        token_i = int(token)
        ev_stage = None
        # Recover stage from actions (one pass, deterministic).
        for action in actions:
            if int(action.get('evicted_token', -1)) == token_i:
                ev_stage = int(action.get('evicted_stage'))
                break
        if ev_stage is None:
            continue
        eviction_actual[ev_stage] = int(eviction_actual.get(ev_stage, 0)) + 1

    report = dict(
        target_stage_counts={int(k): int(v) for k, v in target_counts.items()},
        counts_before={int(k): int(v) for k, v in counts_before.items()},
        counts_after={int(k): int(v) for k, v in counts_after.items()},
        current_stage_id=int(current_stage_id),
        current_stage_target=int(target_current),
        current_stage_before=int(current_before),
        required_add_t=int(required_add_t),
        feasible_candidates=int(feasible_candidates),
        add_t=int(add_t),
        shortfall_t=int(shortfall_t),
        direct_add_count=int(direct_add_count),
        evict_total=int(evict_total),
        eviction_plan={int(k): int(v) for k, v in eviction_plan.items()},
        eviction_actual={int(k): int(v) for k, v in eviction_actual.items()},
        inserted_count=int(inserted),
        inserted_tokens=[int(x) for x in inserted_tokens],
        evicted_tokens=[int(x) for x in evicted_tokens],
        remaining_eviction_plan={int(k): int(v) for k, v in remaining_plan.items()},
        use_class_balance=bool(use_class_balance),
        design_version=int(design_version),
        class_balance_weights_final=(
            [float(x) for x in (
                compute_class_balance_weights_v2(class_balance_vec, w_max=float(w_max))
                if int(design_version) >= 2
                else compute_class_balance_weights(class_balance_vec)
            ).detach().cpu().tolist()]
            if bool(use_class_balance) and class_balance_vec is not None
            else None
        ),
        actions=actions,
    )
    return bank_tokens.clone(), report


__all__ = [
    'compute_percentile_rank',
    'compute_comp_mean',
    'compute_class_balance_weights',
    'compute_class_balance_weights_v2',
    'compute_redundancy_penalty',
    'stage1_greedy_fill',
    'ratio_targeted_swap_update',
]
