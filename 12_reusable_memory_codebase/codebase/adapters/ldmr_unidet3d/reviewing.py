"""Original S5 review boundaries, LD-drop weights and coverage sampling."""
from functools import lru_cache
from pathlib import Path
import runpy

from .learning_dynamics import reference_functions
from .trajectories import validate_trajectory

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def reference_reviewing():
    return (
        runpy.run_path(str(ROOT / 'tools/incremental_scene/reviewing_ld.py')),
        runpy.run_path(str(ROOT / 'mmdet3d/utils/reviewing_sampling.py')),
        runpy.run_path(str(ROOT / 'configs/incremental/scannet/'
                          'tr3d_dynamic_head_s5_scene_memory_ld_design2_reviewing.py'))['reviewing'])


def review_boundaries(epochs, stage_setting='scannet35_s5_freqorder'):
    if stage_setting != 'scannet35_s5_freqorder' or type(epochs) is not int or epochs < 1:
        raise ValueError('reviewing requires positive epochs and the original S5 policy')
    helpers, _, policy = reference_reviewing()
    segments = helpers['compute_review_segment_times'](
        stage_epochs=epochs, repeat_times=1, review_fractions=policy['review_fractions'])
    result = [0]
    for segment in segments:
        result.append(result[-1] + segment)
    return result


def review_weights(snapshots, protocol):
    # Complete coverage and stable GT are required before the original scorer,
    # which otherwise interprets missing observations as zero-quality detections.
    validate_trajectory(snapshots, protocol=protocol,
                        expected_steps=[s['step'] for s in snapshots])
    if len(snapshots) < 2 or snapshots[-1]['stage'] < 2:
        raise ValueError('reviewing needs consecutive measurements and old classes')
    if any(s['q_metric'] != 'f1' or s['iou_thr'] != .25 or s['eps'] != 1e-9 for s in snapshots):
        raise ValueError('reviewing requires original F1/.25 measurement policy')
    stage = snapshots[-1]['stage']
    memory = [[s for s in snapshot['seats'] if s['save_stage'] < stage]
              for snapshot in snapshots[-2:]]
    policy = reference_reviewing()[2]['weight_policy']
    result = reference_functions()['scoring']['compute_reviewing_entry_weights_ld_drop'](
        *memory, old_classes=list(range(protocol.seen_count(stage - 1))),
        iou_mode='0.25', q_metric='f1', eps=1e-9,
        **{key: value for key, value in policy.items() if key != 'type'})
    # Seats without old GT remain neutral, as in the original trainer.
    weights = {f"{s['scene_id']}_stage{s['save_stage']}": 1.0 for s in memory[0]}
    weights.update(result['weights_by_uid'])
    result['weights_by_uid'] = weights
    return result


def review_candidates(records):
    """Translate host replay provenance without changing dataset row order."""
    candidates = []
    for row in records:
        candidate = {}
        if row.get('is_replay'):
            uid = f"{row.get('ldmr_scene_id', row['sample_idx'])}_stage{row['ldmr_replay_from_stage']}"
            if row.get('ldmr_include_current'):
                candidate.update(is_merged=True, replay_unique_ids=[uid])
            else:
                candidate.update(is_replay=True, replay_unique_id=uid)
        candidates.append(candidate)
    return candidates


def review_indices(records, weights, *, seed, stage, review_k):
    import math
    candidates = review_candidates(records)
    expected = {uid for c in candidates for uid in
                ([c['replay_unique_id']] if c.get('is_replay') else c.get('replay_unique_ids', []))}
    if set(weights) != expected or any(not math.isfinite(w) or w < 1 for w in weights.values()):
        raise ValueError('review weights must cover exactly the replay seats and be finite >= 1')
    # Original segment index is review_k + 1; the inner dataset is resampled
    # once per segment, then repeated. Preserve that membership across epochs.
    return reference_reviewing()[1]['review_sample_indices'](candidates, dict(
        target_length=len(records), weights_by_replay_unique_id=weights,
        memory_share_max=.9, strict_memory_coverage=True,
        seed=seed + 9000 + 100 * stage + review_k + 1))


def object_review_weights(crops, weights):
    """Each crop inherits its original carrier-seat weight, in crop order."""
    import math
    seats = [f"{c['scene_id']}_stage{c['stage_id']}" for c in crops]
    if set(weights) != set(seats) or any(not math.isfinite(w) or w < 1 for w in weights.values()):
        raise ValueError('object review weights must cover exactly the carrier seats and be finite >= 1')
    return [float(weights[seat]) for seat in seats]
