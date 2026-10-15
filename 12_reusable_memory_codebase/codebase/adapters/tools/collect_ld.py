#!/usr/bin/env python3
"""Collect and score a fixed training pool across saved host stage checkpoints.

Epoch zero must be a saved initialization. This standalone collector does not
train, select a bank, or enable reviewing. Partial output has no completion seal.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--train-info', required=True, type=Path)
    parser.add_argument('--checkpoint', action='append', required=True, help='epoch=/path, repeat in order')
    parser.add_argument('--object-source-info', type=Path, help='full training metadata for object carrier measurement')
    parser.add_argument('--scene-ids', type=Path, help='JSON list selecting a fixed subset of training scenes')
    parser.add_argument('--upstream-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--iou-thr', type=float, default=.25, choices=(.25, .5))
    parser.add_argument('--q-metric', default='recall', choices=('recall', 'f1'))
    parser.add_argument('--reuse-collection', type=Path,
                        help='verified completed prefix; retain its measured observations')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refusing to overwrite collection evidence')
    from ldmr_unidet3d.collection import (checkpoint_schedule, fixed_pool,
        validate_checkpoint_epoch, context_digest)
    from ldmr_unidet3d.sequence import sha256, verify_hashes, input_hashes
    schedule = checkpoint_schedule(args.checkpoint)
    sys.path.insert(0, str(args.upstream_root.resolve()))
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.runner import set_random_seed
    from mmdet3d.registry import DATASETS, MODELS
    from mmdet3d.structures import DepthInstance3DBoxes, Box3DMode
    from ldmr_unidet3d import host, ScanNet35Protocol
    from ldmr_unidet3d.data import validate_stage_manifest
    from ldmr_unidet3d.checkpoints import validate_checkpoint_state
    from ldmr_unidet3d.learning_dynamics import measure_scene
    from ldmr_unidet3d.trajectories import make_snapshot, write_snapshot, score_design2
    from ldmr_unidet3d.collection import reusable_prefix
    from generate_pseudo import inference_dataset
    init_default_scope('mmdet3d')
    cfg = Config.fromfile(str(args.config))
    protocol = ScanNet35Protocol(cfg.ldmr_protocol.stage_setting)
    stage = cfg.ldmr_protocol.stage
    classes = protocol.decoder_classes(stage)
    if cfg.ldmr_protocol.identity != protocol.identity or list(cfg.model.decoder.datasets_classes) != [classes]:
        raise ValueError('collector config protocol/classifier mismatch')
    # Hash before reading; verify again before publishing completion.
    manifest_paths = [args.config, args.train_info] + ([args.scene_ids] if args.scene_ids else [])
    metadata_hashes = {str(p.resolve()): sha256(p) for p in manifest_paths}
    train = validate_stage_manifest(args.train_info, protocol, stage,
                                    cfg.ldmr_protocol.memory_mode, evaluation=False)
    if (cfg.ldmr_protocol.memory_mode == 'object') != bool(args.object_source_info):
        raise ValueError('object collection requires full object source info only in object mode')
    pool = train['data_list']
    if args.object_source_info:
        import pickle
        from ldmr_unidet3d.collection import object_source_pool
        from ldmr_unidet3d.objects import load_crops
        from ldmr_unidet3d.data import verify_point_files, stage_records
        manifest_paths.append(args.object_source_info)
        crop_path = cfg.ldmr_protocol.get('ld_crop_path')
        if (stage > 1) != bool(crop_path):
            raise ValueError('object measurement requires preceding crops at later stages')
        if crop_path:
            manifest_paths.append(Path(crop_path))
        metadata_hashes.update({str(p.resolve()): sha256(p) for p in manifest_paths})
        full = pickle.loads(args.object_source_info.read_bytes())['data_list']
        if stage_records(full, protocol, stage) != train['data_list']:
            raise ValueError('object source metadata differs from natural training pool')
        crops = load_crops(crop_path, protocol, stage - 1) if crop_path else []
        pool = object_source_pool(full, crops, protocol, stage)
        for row in pool:
            verify_point_files(row)
    requested = json.loads(args.scene_ids.read_text()) if args.scene_ids else None
    geometry, seats = fixed_pool(pool, protocol, stage, requested)
    selected = [row for row in pool if str(row['sample_idx']) in {r['sample_idx'] for r in geometry}]
    inputs = input_hashes(manifest_paths, [selected])
    verify_hashes(metadata_hashes, 'LD metadata')
    repo = Path(__file__).resolve().parents[2]
    sources = list((repo / 'adapters').rglob('*.py')) + list((args.upstream_root / 'unidet3d').rglob('*.py'))
    sources += [repo / p for p in ('mmdet3d/core/evaluation/scene_class_stats.py',
        'mmdet3d/utils/learning_dynamics_scoring.py',
        'mmdet3d/utils/reviewing_sampling.py', 'tools/incremental_scene/reviewing_ld.py',
        'configs/incremental/scannet/tr3d_dynamic_head_s5_scene_memory_ld_design2_reviewing.py', 'configs/_base_/class_mappings/scannet_dynamic_head_mappings.py')]
    hashes = {str(p.resolve()): sha256(p) for p in sorted(sources)}
    checkpoints = {str(p): sha256(p) for _, p in schedule}
    # Use an explicit unaugmented, annotation-free pipeline. The training
    # pipeline may contain object insertion, pseudo labels and random transforms.
    base = dict(type='LDMRScanNetDataset', data_root='', data_prefix=dict(
        pts='', pts_instance_mask='', pts_semantic_mask='', sp_pts_mask=''), pipeline=[
        dict(type='LoadPointsFromFile', coord_type='DEPTH', shift_height=False,
             use_color=True, load_dim=6, use_dim=[0, 1, 2, 3, 4, 5]),
        dict(type='LoadAnnotations3D_', with_sp_mask_3d=True),
        dict(type='LDMRCompactSuperpoints'),
        dict(type='GlobalAlignment', rotation_axis=2),
        dict(type='NormalizePointsColor_', color_mean=[127.5, 127.5, 127.5]),
        dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])])
    context = dict(version=1, protocol_id=protocol.identity, stage=stage,
        expected_steps=[s for s, _ in schedule], seed=args.seed, inputs=inputs,
        sources=hashes, inference_dataset=base, geometry=geometry, seats=seats,
        iou_thr=args.iou_thr, q_metric=args.q_metric, eps=1e-9,
        runtime=dict(torch=torch.__version__, cuda=torch.version.cuda))
    digest = context_digest(context)
    reused, reuse_receipt = reusable_prefix(args.reuse_collection, context, checkpoints, protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'context.json').write_text(json.dumps(context, indent=2, allow_nan=False)+'\n')
    snapshots, artifacts = [], {}
    with tempfile.TemporaryDirectory(prefix='ldmr_collect_') as directory:
        dataset = DATASETS.build(inference_dataset(base, geometry, classes, Path(directory)/'inference.pkl'))
        if len(dataset) != len(geometry):
            raise ValueError('LD inference dataset dropped scenes')
        set_random_seed(args.seed)
        model = MODELS.build(cfg.model)
        for epoch, checkpoint_path in schedule:
            verify_hashes(inputs, 'LD inputs')
            verify_hashes(hashes, 'LD sources')
            verify_hashes({str(checkpoint_path): checkpoints[str(checkpoint_path)]}, 'LD checkpoint')
            if epoch in reused:
                # Keep observed statistics exactly. Only the schedule-bound
                # context digest changes; the complete prefix is verified and
                # its original context/artifacts remain linked in the receipt.
                snapshot = dict(reused[epoch], measurement_sha256=digest)
                name = f'snapshot_{epoch}.json'
                artifacts[name] = write_snapshot(args.output / name, snapshot, protocol)
                snapshots.append(snapshot)
                print(f'epoch {epoch}: retained sealed observations', flush=True)
                continue
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            validate_checkpoint_state(checkpoint, model.state_dict(), protocol, stage)
            validate_checkpoint_epoch(checkpoint, stage, epoch)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            del checkpoint
            model.cuda().eval().requires_grad_(False)
            predictions = {}
            with torch.inference_mode():
                for index, row in enumerate(geometry):
                    sid = str(row['sample_idx'])
                    if dataset.get_data_info(index)['ldmr_scene_id'] != sid:
                        raise ValueError('LD dataset changed scene order')
                    set_random_seed(args.seed + index)
                    batch = model.data_preprocessor(pseudo_collate([dataset[index]]), training=False)
                    result = model(**batch, mode='predict')[0].pred_instances_3d
                    boxes = result.bboxes_3d
                    if boxes.with_yaw:
                        raise ValueError('LD requires aligned ScanNet AABBs')
                    predictions[sid] = dict(boxes=torch.cat([boxes.gravity_center, boxes.tensor[:, 3:6]], dim=1).cpu().numpy(),
                        labels=result.labels_3d.cpu().numpy(), scores=result.scores_3d.cpu().numpy())
            measured = [measure_scene(predictions[row['scene_id']],
                [i['bbox_3d'] for i in row['instances']], [i['bbox_label_3d'] for i in row['instances']],
                protocol=protocol, stage=stage, scene_id=row['scene_id'],
                replay_from_stage=None if row['save_stage'] == stage else row['save_stage'],
                box_type_3d=DepthInstance3DBoxes, box_mode_3d=Box3DMode.DEPTH,
                iou_thr=args.iou_thr, q_metric=args.q_metric) for row in seats]
            verify_hashes(inputs, 'LD inputs')
            verify_hashes(hashes, 'LD sources')
            verify_hashes({str(checkpoint_path): checkpoints[str(checkpoint_path)]}, 'LD checkpoint')
            snapshot = make_snapshot(measured, protocol=protocol, stage=stage, step=epoch,
                checkpoint_sha256=checkpoints[str(checkpoint_path)], measurement_sha256=digest,
                iou_thr=args.iou_thr, q_metric=args.q_metric)
            name = f'snapshot_{epoch}.json'
            artifacts[name] = write_snapshot(args.output / name, snapshot, protocol)
            snapshots.append(snapshot)
            print(f'epoch {epoch}: {len(seats)} measured seats from {len(geometry)} scenes', flush=True)
    scores = score_design2(snapshots, protocol=protocol, expected_steps=context['expected_steps'])
    (args.output / 'scores.json').write_text(json.dumps(scores, indent=2, allow_nan=False)+'\n')
    verify_hashes(inputs, 'LD inputs')
    verify_hashes(hashes, 'LD sources')
    verify_hashes(checkpoints, 'LD checkpoints')
    artifacts.update({name: sha256(args.output / name) for name in ('context.json', 'scores.json')})
    report = dict(status='complete', measurement_sha256=digest, artifacts=artifacts,
        checkpoints=checkpoints, steps=context['expected_steps'], scenes=len(geometry), seats=len(seats))
    if reuse_receipt is not None:
        verify_hashes(reuse_receipt['hashes'], 'reused collection')
        report['reused_collection'] = reuse_receipt
    with (args.output / 'complete.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
