#!/usr/bin/env python3
"""Freeze previous-stage host predictions for a prepared training manifest.

Teacher inference receives geometry/superpoints only. No GT semantic/instance
mask is loaded, and no GT boxes or labels enter the teacher dataset. Confidence,
NMS, cap and visible-GT conflict filtering run later in the training transform.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import pickle
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def inference_dataset(base_dataset, records, classes, manifest):
    """Strip annotations/replay identity and use the host inference pipeline."""
    cfg = deepcopy(base_dataset)
    cfg.update(ann_file=str(manifest), test_mode=True, load_eval_anns=False,
               metainfo=dict(classes=classes), ignore_index=len(classes))
    for step in cfg['pipeline']:
        if step['type'] == 'LoadAnnotations3D_':
            step.update(with_bbox_3d=False, with_label_3d=False,
                        with_mask_3d=False, with_seg_3d=False, with_sp_mask_3d=True)
    fields = ('sample_idx', 'lidar_points', 'super_pts_path', 'axis_align_matrix',
              'ldmr_protocol_id', 'ldmr_semantic_space')
    clean = [dict(**{key: deepcopy(row[key]) for key in fields}, instances=[])
             for row in records]
    with open(manifest, 'xb') as stream:
        pickle.dump(dict(metainfo=dict(classes=classes), data_list=clean), stream)
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher-config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--train-info', required=True, type=Path)
    parser.add_argument('--upstream-root', required=True, type=Path)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--stage-setting', default='scannet35_s5_freqorder')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--policy', choices=['ldmr', 'tr3d_obj'], default='ldmr')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refusing to overwrite teacher predictions')
    from ldmr_unidet3d import ScanNet35Protocol
    from ldmr_unidet3d.data import validate_stage_manifest
    from ldmr_unidet3d.pseudo import geometry_digest, write_pseudo
    from ldmr_unidet3d.sequence import sha256
    protocol = ScanNet35Protocol(args.stage_setting)
    protocol.seen_count(args.stage)
    if args.stage == 1:
        parser.error('stage 1 has no previous-stage teacher')
    with args.train_info.open('rb') as stream:
        mode = pickle.load(stream)['metainfo']['ldmr_memory_mode']
    train = validate_stage_manifest(args.train_info, protocol, args.stage, mode, evaluation=False)
    records = train['data_list']
    ids = [str(row['sample_idx']) for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError('teacher input scene IDs must be unique')
    teacher_hash, info_hash, config_hash = map(sha256, (args.checkpoint, args.train_info, args.teacher_config))
    geometry_hash = geometry_digest(records)
    sys.path.insert(0, str(args.upstream_root.resolve()))
    import torch
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmengine.runner import set_random_seed
    from mmdet3d.registry import DATASETS, MODELS
    from ldmr_unidet3d import host
    from ldmr_unidet3d.checkpoints import validate_checkpoint_state
    init_default_scope('mmdet3d')
    set_random_seed(args.seed)
    cfg = Config.fromfile(str(args.teacher_config))
    context = cfg.ldmr_protocol
    protocol.seen_count(context.stage)
    classes = protocol.decoder_classes(args.stage - 1)
    if (context.identity != protocol.identity or context.stage != args.stage - 1
            or context.stage_setting != args.stage_setting
            or list(cfg.model.decoder.datasets_classes) != [classes]):
        raise ValueError('teacher config must describe the preceding protocol stage')
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    epoch = checkpoint.get('meta', {}).get('epoch')
    if type(epoch) is not int or epoch < 1 or epoch != cfg.train_cfg.max_epochs:
        raise ValueError('teacher must be the completed final epoch of its resolved stage config')
    model = MODELS.build(cfg.model)
    validate_checkpoint_state(checkpoint, model.state_dict(), protocol, args.stage - 1)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    model.cuda().eval().requires_grad_(False)
    datasets = cfg.test_dataloader.dataset.datasets
    if len(datasets) != 1 or datasets[0].type != 'LDMRScanNetDataset':
        raise ValueError('teacher requires the pinned single-dataset ScanNet adapter')
    predictions = {}
    with tempfile.TemporaryDirectory(prefix='ldmr_teacher_') as directory:
        dataset_cfg = inference_dataset(datasets[0], records, classes, Path(directory) / 'inference.pkl')
        dataset = DATASETS.build(dataset_cfg)
        if len(dataset) != len(records):
            raise ValueError('teacher inference dropped input scenes')
        with torch.inference_mode():
            for index, scene in enumerate(ids):
                if dataset.get_data_info(index)['ldmr_scene_id'] != scene:
                    raise ValueError('teacher dataset reordered input scene identities')
                inputs = model.data_preprocessor(pseudo_collate([dataset[index]]), training=False)
                result = model(**inputs, mode='predict')[0].pred_instances_3d
                boxes = result.bboxes_3d
                if boxes.with_yaw:
                    raise ValueError('teacher must emit ScanNet AABBs')
                predictions[scene] = dict(
                    boxes=torch.cat([boxes.gravity_center, boxes.tensor[:, 3:6]], dim=1).cpu().numpy(),
                    labels=result.labels_3d.cpu().numpy(), scores=result.scores_3d.cpu().numpy())
                print(f'teacher {index + 1}/{len(ids)}: {scene}, {len(boxes)} predictions', flush=True)
    if ((teacher_hash, info_hash, config_hash) != tuple(map(sha256, (args.checkpoint, args.train_info, args.teacher_config)))
            or geometry_digest(records) != geometry_hash):
        raise ValueError('teacher inputs changed during inference')
    write_pseudo(args.output, predictions, protocol, args.stage, policy_name=args.policy,
                 teacher_sha256=teacher_hash, train_info_sha256=info_hash,
                 geometry_sha256=geometry_hash, teacher_config_sha256=config_hash)


if __name__ == '__main__':
    main()
