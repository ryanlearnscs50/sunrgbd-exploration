"""Forgetting metrics helpers for incremental learning.

These utilities are intentionally lightweight (pure-Python) so they can be used
from both training scripts and unit tests without pulling in heavy training
dependencies.
"""

from typing import Any, Dict, Iterable, Optional, Tuple


def _ap_maps_from_stage_metrics_json(
        stage_metrics: dict) -> Tuple[Dict[int, float], Dict[int, float]]:
    """Extract per-class AP maps from `memory_bank/scores/stage_{t}_metrics.json` payloads.

    Returns:
      (ap25_by_idx, ap50_by_idx)
    """
    ap25 = {}
    ap50 = {}
    if not isinstance(stage_metrics, dict):
        return ap25, ap50
    for c in stage_metrics.get('classes', []) or []:
        if not isinstance(c, dict):
            continue
        try:
            idx = int(c.get('model_idx'))
        except Exception:
            continue
        try:
            ap25[idx] = float(c.get('AP_0.25', 0.0))
        except Exception:
            ap25[idx] = 0.0
        try:
            ap50[idx] = float(c.get('AP_0.50', 0.0))
        except Exception:
            ap50[idx] = 0.0
    return ap25, ap50


def calculate_forgetting_metrics_from_stage_metrics_json(
        previous_stage_metrics_json: dict,
        current_stage_metrics_json: dict,
        previous_stage_classes: Iterable[int],
        mappings: Dict[str, Any],
        logger,
        *,
        previous_stage_id: Optional[int] = None,
        current_stage_id: Optional[int] = None,
        verbose: bool = False) -> Dict[str, Any]:
    """Compute forgetting metrics from structured `stage_{t}_metrics.json` artifacts.

    This is more robust than relying on string-keyed log dicts because it matches
    by `model_idx` rather than by class-name keys.
    """
    ap_prev25, ap_prev50 = _ap_maps_from_stage_metrics_json(
        previous_stage_metrics_json)
    ap_curr25, ap_curr50 = _ap_maps_from_stage_metrics_json(
        current_stage_metrics_json)

    model_idx_to_name = (mappings or {}).get('model_idx_to_name', {}) or {}

    per_class_forgetting = {}
    for class_idx in previous_stage_classes:
        class_idx = int(class_idx)
        class_name = model_idx_to_name.get(class_idx, f"class_{class_idx}")

        prev_ap_25 = float(ap_prev25.get(class_idx, 0.0))
        curr_ap_25 = float(ap_curr25.get(class_idx, 0.0))
        prev_ap_50 = float(ap_prev50.get(class_idx, 0.0))
        curr_ap_50 = float(ap_curr50.get(class_idx, 0.0))

        forgetting_25 = curr_ap_25 - prev_ap_25
        forgetting_50 = curr_ap_50 - prev_ap_50
        forgetness_25 = max(prev_ap_25 - curr_ap_25, 0.0)
        forgetness_50 = max(prev_ap_50 - curr_ap_50, 0.0)

        per_class_forgetting[class_idx] = {
            'name': class_name,
            'prev_AP_0.25': prev_ap_25,
            'curr_AP_0.25': curr_ap_25,
            'forgetting_0.25': forgetting_25,
            'forgetness_0.25': forgetness_25,
            'prev_AP_0.50': prev_ap_50,
            'curr_AP_0.50': curr_ap_50,
            'forgetting_0.50': forgetting_50,
            'forgetness_0.50': forgetness_50,
        }

    if not per_class_forgetting:
        return {}

    avg_forgetting_25 = sum(v['forgetting_0.25'] for v in per_class_forgetting.values()) / len(per_class_forgetting)
    avg_forgetting_50 = sum(v['forgetting_0.50'] for v in per_class_forgetting.values()) / len(per_class_forgetting)
    avg_forgetness_25 = sum(v['forgetness_0.25'] for v in per_class_forgetting.values()) / len(per_class_forgetting)
    avg_forgetness_50 = sum(v['forgetness_0.50'] for v in per_class_forgetting.values()) / len(per_class_forgetting)

    worst_forgetting_25 = min(per_class_forgetting.values(), key=lambda x: x['forgetting_0.25'])
    best_retention_25 = max(per_class_forgetting.values(), key=lambda x: x['forgetting_0.25'])

    # Match legacy behavior from `calculate_forgetting_metrics(...)`.
    num_degraded = sum(1 for v in per_class_forgetting.values() if v['forgetting_0.25'] < 0)
    num_improved = sum(1 for v in per_class_forgetting.values() if v['forgetting_0.25'] > 0)
    num_stable = sum(1 for v in per_class_forgetting.values() if abs(v['forgetting_0.25']) < 0.01)

    forgetting_metrics = {
        'previous_stage_id': int(previous_stage_id) if previous_stage_id is not None else None,
        'current_stage_id': int(current_stage_id) if current_stage_id is not None else None,
        'conventions': {
            'forgetting_delta': 'forgetting = curr_AP - prev_AP (negative => forgetting, positive => improvement)',
            'forgetness_drop': 'forgetness = max(prev_AP - curr_AP, 0.0) (positive drop only)',
        },
        'per_class': {int(k): v for k, v in per_class_forgetting.items()},
        'average_forgetting_0.25': float(avg_forgetting_25),
        'average_forgetting_0.50': float(avg_forgetting_50),
        'average_forgetness_0.25': float(avg_forgetness_25),
        'average_forgetness_0.50': float(avg_forgetness_50),
        'worst_class': worst_forgetting_25,
        'best_class': best_retention_25,
        'num_classes_degraded': int(num_degraded),
        'num_classes_improved': int(num_improved),
        'num_classes_stable': int(num_stable),
    }

    if logger is not None:
        try:
            logger.info("=" * 80)
            logger.info(
                f"Forgetting Analysis (stage {previous_stage_id} -> stage {current_stage_id}):"
            )
            logger.info(
                f"   Average Forgetting (delta, AP@0.25): {avg_forgetting_25:+.4f}"
            )
            logger.info(
                f"   Average Forgetting (delta, AP@0.50): {avg_forgetting_50:+.4f}"
            )
            logger.info(
                f"   Average Forgetness (drop, AP@0.25): {avg_forgetness_25:.4f}"
            )
            logger.info(
                f"   Average Forgetness (drop, AP@0.50): {avg_forgetness_50:.4f}"
            )
            logger.info(
                f"   Classes with degraded performance: {num_degraded}/{len(per_class_forgetting)}"
            )
            logger.info(
                f"   Classes with improved performance: {num_improved}/{len(per_class_forgetting)}"
            )
            logger.info(
                f"   Classes with stable performance: {num_stable}/{len(per_class_forgetting)}"
            )
            logger.info("=" * 80)
        except Exception:
            pass

        if verbose:
            try:
                for class_idx in sorted(per_class_forgetting.keys()):
                    m = per_class_forgetting[int(class_idx)]
                    logger.info(
                        f"   {m['name']}: "
                        f"{m['prev_AP_0.25']:.4f} -> {m['curr_AP_0.25']:.4f} "
                        f"({m['forgetting_0.25']:+.4f})"
                    )
            except Exception:
                pass

    return forgetting_metrics
