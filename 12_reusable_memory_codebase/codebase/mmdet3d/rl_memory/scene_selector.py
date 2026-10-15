"""
Greedy scene selector that maps per-class quotas to full scenes.
"""

from typing import Dict, List

from .structures import SceneDescriptor


def select_scenes_simple(
    T_c: Dict[int, int],
    scene_pool: List[SceneDescriptor],
    max_scenes: int,
) -> List[SceneDescriptor]:
    """Greedy selection of scenes to satisfy per-class quotas.

    Algorithm:
      - Maintain remaining quotas Q_c = T_c.
      - Repeatedly select the scene that adds the largest number of objects
        toward unmet quotas (sum_c min(count_c, Q_c)).
      - Stop when max_scenes reached or no scene improves coverage.

    The function is deterministic given ``T_c`` and ``scene_pool``: ties are
    broken by the original order of ``scene_pool``.

    Args:
        T_c: Target number of objects per class.
        scene_pool: Candidate scenes to choose from.
        max_scenes: Maximum number of scenes to select.

    Returns:
        List[SceneDescriptor]: Selected scenes in the order they were chosen.
    """
    if max_scenes <= 0 or not scene_pool or not T_c:
        return []

    # Remaining quotas
    Q_c: Dict[int, int] = {cid: max(0, int(q)) for cid, q in T_c.items()}

    selected: List[SceneDescriptor] = []
    remaining_indices = list(range(len(scene_pool)))

    while len(selected) < max_scenes and remaining_indices:
        best_score = 0
        best_idx_in_remaining = None

        for pos, pool_idx in enumerate(remaining_indices):
            scene = scene_pool[pool_idx]
            # Compute how many objects this scene contributes toward remaining quotas
            score = 0
            for cid, count in scene.class_counts.items():
                quota = Q_c.get(cid, 0)
                if quota <= 0:
                    continue
                score += min(count, quota)

            if score > best_score:
                best_score = score
                best_idx_in_remaining = pos

        if best_idx_in_remaining is None or best_score <= 0:
            # No scene can improve coverage further
            break

        # Select the best scene
        chosen_pos = best_idx_in_remaining
        chosen_pool_idx = remaining_indices.pop(chosen_pos)
        chosen_scene = scene_pool[chosen_pool_idx]
        selected.append(chosen_scene)

        # Update remaining quotas
        for cid, count in chosen_scene.class_counts.items():
            if cid in Q_c and Q_c[cid] > 0:
                used = min(Q_c[cid], count)
                Q_c[cid] -= used

    return selected

