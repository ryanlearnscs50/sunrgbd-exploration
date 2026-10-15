"""Framework-independent original ScanNet coverage-preserving review policy."""
import numpy as np


def review_sample_indices(candidates, cfg):
    target_len = cfg.get('target_length', None)
    if target_len is None:
        raise ValueError('reviewing_sampling.enabled=True requires target_length.')
    target_len = int(target_len)
    if target_len <= 0:
        raise ValueError(f'Invalid reviewing_sampling.target_length={target_len}.')

    weights_by_uid = cfg.get('weights_by_replay_unique_id', {}) or {}
    if not isinstance(weights_by_uid, dict):
        raise ValueError('reviewing_sampling.weights_by_replay_unique_id must be a dict.')

    seed = cfg.get('seed', None)
    if seed is None:
        raise ValueError('reviewing_sampling.enabled=True requires seed.')
    seed = int(seed)

    baseline_inner_len = cfg.get('baseline_inner_len', None)
    if baseline_inner_len is not None:
        baseline_inner_len = int(baseline_inner_len)
        if baseline_inner_len <= 0:
            baseline_inner_len = None

    strict_memory_coverage = bool(cfg.get('strict_memory_coverage', True))

    memory_share_max = cfg.get('memory_share_max', None)
    if memory_share_max is not None:
        memory_share_max = float(memory_share_max)
        if memory_share_max <= 0.0 or memory_share_max >= 1.0:
            memory_share_max = None

    candidates = list(candidates)
    if not candidates:
        raise RuntimeError('reviewing_sampling has no candidates to sample from.')
    if baseline_inner_len is not None:
        target_len = int(max(target_len, baseline_inner_len))

    rng = np.random.RandomState(seed)
    sample_weights = np.ones((len(candidates),), dtype=np.float64)
    is_memory = np.zeros((len(candidates),), dtype=bool)
    memory_indices = []
    natural_indices = []

    for i, info in enumerate(candidates):
        if bool(info.get('is_replay', False)):
            uid = str(info.get('replay_unique_id', ''))
            if uid:
                is_memory[i] = True
                sample_weights[i] = max(1.0, float(weights_by_uid.get(uid, 1.0)))
                memory_indices.append(int(i))
            else:
                natural_indices.append(int(i))
            continue

        if bool(info.get('is_merged', False)):
            uids = info.get('replay_unique_ids', None) or []
            uids = [str(x) for x in uids if str(x)]
            if uids:
                is_memory[i] = True
                w = max(float(weights_by_uid.get(uid, 1.0)) for uid in uids)
                sample_weights[i] = max(1.0, w)
                memory_indices.append(int(i))
            else:
                natural_indices.append(int(i))
            continue

        natural_indices.append(int(i))

    if memory_share_max is not None:
        mem_sum = float(sample_weights[is_memory].sum()) if bool(is_memory.any()) else 0.0
        nat_sum = float(sample_weights[~is_memory].sum()) if bool((~is_memory).any()) else 0.0
        denom = mem_sum + nat_sum
        if denom > 0.0 and nat_sum > 0.0:
            share = mem_sum / denom
            if share > memory_share_max and mem_sum > 0.0:
                scale = (memory_share_max / max(1e-6, (1.0 - memory_share_max))) * (nat_sum / mem_sum)
                sample_weights[is_memory] *= float(scale)

    total_w = float(sample_weights.sum())
    if not np.isfinite(total_w) or total_w <= 0.0:
        raise RuntimeError('Invalid reviewing_sampling weights (sum <= 0).')

    n_mem = int(len(memory_indices))
    if n_mem > int(target_len):
        if strict_memory_coverage:
            raise RuntimeError(
                "reviewing_sampling cannot guarantee memory coverage: "
                f"target_length={int(target_len)} < num_memory_candidates={n_mem}."
            )
        target_len = int(n_mem)

    mem_sum = float(sample_weights[memory_indices].sum()) if n_mem > 0 else 0.0
    nat_sum = float(sample_weights[natural_indices].sum()) if natural_indices else 0.0
    denom = mem_sum + nat_sum

    if denom > 0.0 and mem_sum > 0.0:
        mem_share = float(mem_sum / denom)
    elif n_mem > 0 and not natural_indices:
        mem_share = 1.0
    else:
        mem_share = 0.0

    k_mem_target = int(round(float(target_len) * float(mem_share)))
    k_mem_target = max(0, min(int(target_len), int(k_mem_target)))
    if n_mem > 0:
        k_mem_target = max(int(k_mem_target), int(n_mem))

    k_nat_target = int(max(0, int(target_len) - int(k_mem_target)))
    k_mem_extra = int(max(0, int(k_mem_target) - int(n_mem)))

    selected_indices = []
    if n_mem > 0:
        selected_indices.extend([int(x) for x in memory_indices])

        if k_mem_extra > 0:
            mem_weights = np.asarray(
                [float(sample_weights[int(i)]) for i in memory_indices],
                dtype=np.float64,
            )
            mem_total = float(mem_weights.sum())
            if not np.isfinite(mem_total) or mem_total <= 0.0:
                mem_probs = np.ones((n_mem,), dtype=np.float64) / float(n_mem)
            else:
                mem_probs = mem_weights / mem_total
            extra_idx = rng.choice(
                n_mem,
                size=int(k_mem_extra),
                replace=True,
                p=mem_probs,
            )
            selected_indices.extend([int(memory_indices[int(j)]) for j in extra_idx.tolist()])

    if k_nat_target > 0:
        if natural_indices:
            if k_nat_target <= len(natural_indices):
                nat_idx = rng.choice(
                    len(natural_indices),
                    size=int(k_nat_target),
                    replace=False,
                )
                selected_indices.extend([int(natural_indices[int(j)]) for j in nat_idx.tolist()])
            else:
                nat_weights = np.asarray(
                    [float(sample_weights[int(i)]) for i in natural_indices],
                    dtype=np.float64,
                )
                nat_total = float(nat_weights.sum())
                if not np.isfinite(nat_total) or nat_total <= 0.0:
                    nat_probs = np.ones((len(natural_indices),), dtype=np.float64) / float(len(natural_indices))
                else:
                    nat_probs = nat_weights / nat_total
                nat_idx = rng.choice(
                    len(natural_indices),
                    size=int(k_nat_target),
                    replace=True,
                    p=nat_probs,
                )
                selected_indices.extend([int(natural_indices[int(j)]) for j in nat_idx.tolist()])
        elif n_mem > 0:
            # Fallback: no natural candidates are available.
            mem_weights = np.asarray(
                [float(sample_weights[int(i)]) for i in memory_indices],
                dtype=np.float64,
            )
            mem_total = float(mem_weights.sum())
            if not np.isfinite(mem_total) or mem_total <= 0.0:
                mem_probs = np.ones((n_mem,), dtype=np.float64) / float(n_mem)
            else:
                mem_probs = mem_weights / mem_total
            extra_mem_idx = rng.choice(
                n_mem,
                size=int(k_nat_target),
                replace=True,
                p=mem_probs,
            )
            selected_indices.extend([int(memory_indices[int(j)]) for j in extra_mem_idx.tolist()])

    if len(selected_indices) != int(target_len):
        raise RuntimeError(
            "reviewing_sampling constructed invalid sample length: "
            f"got={len(selected_indices)}, target={int(target_len)}."
        )

    rng.shuffle(selected_indices)


    memory_index_set = set(int(x) for x in memory_indices)
    seen_memory_candidates = set(
        int(j) for j in selected_indices if int(j) in memory_index_set
    )
    debug = {
        'strict_memory_coverage': bool(strict_memory_coverage),
        'memory_candidate_count': int(n_mem),
        'natural_candidate_count': int(len(natural_indices)),
        'memory_target_count': int(k_mem_target),
        'memory_base_count': int(n_mem),
        'memory_extra_count': int(k_mem_extra),
        'natural_target_count': int(k_nat_target),
        'memory_seen_count': int(len(seen_memory_candidates)),
        'memory_never_seen_count': int(max(0, n_mem - len(seen_memory_candidates))),
        'memory_coverage_ratio': (
            float(len(seen_memory_candidates) / float(n_mem))
            if n_mem > 0 else float('nan')
        ),
    }
    return selected_indices, debug
