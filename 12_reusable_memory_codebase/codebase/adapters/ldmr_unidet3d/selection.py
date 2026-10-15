"""Consume sealed host trajectories with the original ScanNet Design-2 scoring and bank selectors."""
import json
from pathlib import Path
import runpy

from .collection import context_digest, fixed_pool
from .data import stage_records
from .scene import merge_replay_records, replay_records
from .sequence import sha256, verify_hashes
from .trajectories import load_snapshot, score_design2


def reference_scene_policy():
    path = Path(__file__).resolve().parents[2] / 'configs/incremental/scannet/tr3d_dynamic_head_s5_scene_memory_ld_design2_reviewing.py'
    policy = runpy.run_path(str(path))['scene_memory_config']
    # The exporter supplies the actual dataset size and explicit budget.
    return {key: value for key, value in policy.items() if key not in
            ('memory_budget_ratio', 'total_training_scenes', 'debug_mode', 'stage_ratio_counts')}


def reference_object_policy():
    """ScanNet F1/.25 adaptation, not SUN RGB-D recall/.50 parity."""
    scene = reference_scene_policy()
    return dict(selection_strategy='learning_dynamics_design2',
                learning_dynamics_update=scene['learning_dynamics_update'],
                learning_dynamics_design2=scene['learning_dynamics_design2'])


def collection_payload(directory, protocol, stage, records, bank, memory_mode='scene', reviewing=False):
    """Validate the complete natural/replay pool before calling the selector.

    Recompute scores from sealed measurements rather than trusting scores.json.
    The fixed GT, geometry and actual input bytes must match this bank update.
    """
    directory = Path(directory)
    seal = json.loads((directory / 'complete.json').read_text())
    if seal.get('status') != 'complete':
        raise ValueError('LD collection is incomplete')
    artifacts = seal['artifacts']
    if any(Path(name).name != name for name in artifacts):
        raise ValueError('LD artifact names must be local filenames')
    verify_hashes({str(directory / name): digest for name, digest in artifacts.items()}, 'LD artifacts')
    context = json.loads((directory / 'context.json').read_text())
    if (context_digest(context) != seal['measurement_sha256']
            or context['protocol_id'] != protocol.identity or context['stage'] != stage
            or context['expected_steps'] != seal['steps']):
        raise ValueError('LD collection context mismatch')
    if context['q_metric'] != reference_scene_policy()['learning_dynamics_design2']['q_metric']:
        raise ValueError('original ScanNet Design-2 requires f1 collection')
    if memory_mode == 'object':
        from .collection import object_source_pool
        pool = object_source_pool(records, bank.get_all_exemplars(), protocol, stage)
    elif memory_mode == 'scene':
        natural = stage_records(records, protocol, stage)
        replay = replay_records(bank, {r['sample_idx']: r for r in records}, protocol, stage)
        pool = merge_replay_records(natural, replay)
    else:
        raise ValueError('unsupported selection memory mode')
    geometry, seats = fixed_pool(pool, protocol, stage)
    if context['seats'] != seats or context.get('geometry') != geometry:
        raise ValueError('LD collection does not cover the exact selection pool and visible GT')
    for row in geometry:
        for path in (row['lidar_points']['lidar_path'], row['super_pts_path']):
            path = str(Path(path).absolute())
            if context['inputs'].get(path) != sha256(path):
                raise ValueError('LD selection geometry differs from measured input')
    # Includes alignment/manifest and source integrity, not only point payloads.
    verify_hashes(context['inputs'], 'LD inputs')
    verify_hashes(context['sources'], 'LD sources')
    verify_hashes(seal['checkpoints'], 'LD checkpoints')
    snapshots = [load_snapshot(directory / f'snapshot_{step}.json', protocol,
        expected_sha256=artifacts[f'snapshot_{step}.json']) for step in seal['steps']]
    if any(s['measurement_sha256'] != seal['measurement_sha256'] or
           s['checkpoint_sha256'] not in seal['checkpoints'].values() for s in snapshots):
        raise ValueError('LD snapshot provenance mismatch')
    scores = score_design2(snapshots, protocol=protocol, expected_steps=seal['steps'])
    evidence = dict(collection_sha256=sha256(directory / 'complete.json'),
        measurement_sha256=seal['measurement_sha256'], steps=seal['steps'])
    if reviewing:
        from .reviewing import review_boundaries, review_weights
        if memory_mode != 'object' or seal['steps'] != review_boundaries(seal['steps'][-1], protocol.stage_setting):
            raise ValueError('object review export requires the original review schedule')
        # Preserve the last weights actually used during training, not a new
        # decision from the final checkpoint, which no training epoch consumed.
        evidence['reviewing_weights'] = (review_weights(snapshots[:-1], protocol)['weights_by_uid']
                                        if stage > 1 and len(snapshots) > 2 else None)
    return dict(scores, stage_id=stage), evidence
