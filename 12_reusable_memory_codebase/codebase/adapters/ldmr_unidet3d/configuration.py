"""Build a single-stage host config; no old MMDetection3D imports."""
from copy import deepcopy


def stage_config(base, protocol, stage, *, train_info, val_info, work_dir,
                 memory_mode, epochs, crop_path=None, target_mode='masks',
                 pseudo_path=None, teacher_sha256=None, object_policy="ldmr_weighted",
                 validation_interval=1):
    """Adapt the pinned upstream ScanNet config for controlled memory baselines.

    Input infos must already be converted and stage-filtered. Scene mode's train
    info must additionally contain exported LDMR-selected replay seats. This
    function does not create pseudo labels, select memory or change bank budgets.
    """
    protocol.seen_count(stage)
    if memory_mode not in ('none', 'scene', 'object'):
        raise ValueError('invalid memory mode')
    if target_mode not in ('masks', 'boxes'):
        raise ValueError('target mode must be masks or boxes')
    if pseudo_path is not None and (target_mode != 'boxes' or stage == 1):
        raise ValueError('pseudo supervision requires box targets at stage 2+')
    pseudo_options = None
    if pseudo_path is not None:
        from .pseudo import load_pseudo, geometry_digest
        from .sequence import sha256
        import pickle
        from pathlib import Path
        with open(train_info, 'rb') as stream:
            records = pickle.load(stream)['data_list']
            scene_ids = [row['sample_idx'] for row in records]
        pseudo_options = dict(path=str(Path(pseudo_path).resolve()),
                              teacher_sha256=teacher_sha256,
                              train_info_sha256=sha256(train_info),
                              geometry_sha256=geometry_digest(records),
                              artifact_sha256=sha256(pseudo_path),
                              policy_name='tr3d_obj' if object_policy == 'tr3d_obj' else 'ldmr')
        load_pseudo(protocol=protocol, stage=stage, scene_ids=scene_ids, **pseudo_options)
    if type(epochs) is not int or epochs < 1:
        raise ValueError('epochs must be a positive explicit schedule')
    if (type(validation_interval) is not int or validation_interval < 1
            or epochs % validation_interval):
        raise ValueError('validation interval must be positive and divide the stage epochs')
    if memory_mode == 'object' and stage > 1 and not crop_path:
        raise ValueError('object replay requires a preceding-stage rich crop export')
    if object_policy not in ('ldmr_weighted', 'tr3d_obj'):
        raise ValueError('unknown object replay policy')
    if object_policy == 'tr3d_obj' and memory_mode != 'object':
        raise ValueError('TR3D_OBJ replay requires object memory')
    cfg = deepcopy(base)
    classes = protocol.decoder_classes(stage)
    cfg['custom_imports'] = dict(imports=['unidet3d', 'ldmr_unidet3d.host'], allow_failed_imports=False)
    cfg['model']['type'] = 'LDMRUniDet3D'
    cfg['model']['criterion']['matcher']['type'] = 'LDMRUniMatcher'
    cfg['model']['decoder']['datasets_classes'] = [classes]
    if target_mode == 'boxes':
        cfg['model']['bbox_by_mask'] = [False]
        cfg['model']['target_by_distance'] = [True]
    cfg['load_from'] = None  # never inherit the upstream ScanNet-18 checkpoint
    cfg['resume'] = False
    cfg['work_dir'] = str(work_dir)
    cfg['train_cfg'] = dict(type='EpochBasedTrainLoop', max_epochs=epochs,
                            val_interval=validation_interval)
    cfg.setdefault('default_hooks', {}).setdefault('checkpoint', {}).update(
        interval=validation_interval, save_last=True)
    cfg['param_scheduler'] = dict(type='PolyLR', begin=0, end=epochs, power=.9, by_epoch=True)
    cfg['ldmr_protocol'] = dict(identity=protocol.identity, stage=stage,
                                stage_setting=protocol.stage_setting, memory_mode=memory_mode,
                                pseudo_labels=pseudo_path is not None, reviewing=False,
                                target_mode=target_mode, pseudo_artifact=pseudo_options,
                                object_policy=object_policy)
    cfg.setdefault('custom_hooks', []).append(dict(type='LDMRProtocolCheckpointHook',
        stage=stage, stage_setting=protocol.stage_setting))
    train_pipeline = []
    for transform in cfg['train_pipeline_scannet']:
        transform = deepcopy(transform)
        kind = transform['type']
        if kind == 'PointSegClassMapping':
            continue
        if target_mode == 'boxes' and kind == 'ElasticTransfrom':
            continue
        if kind == 'LoadAnnotations3D_':
            transform.update(with_bbox_3d=True, with_label_3d=True)
            if target_mode == 'boxes' and memory_mode != 'object':
                transform.update(with_mask_3d=False, with_seg_3d=False)
        if kind == 'PointDetClassMappingScanNet':
            transform = dict(type='LDMRBoxTargets' if target_mode == 'boxes' else 'LDMRScanNetTargets', stage=stage,
                             stage_setting=protocol.stage_setting)
        if target_mode == 'boxes' and kind == 'Pack3DDetInputs_':
            transform['keys'] = ['points', 'gt_bboxes_3d', 'gt_labels_3d', 'sp_pts_mask']
        if kind == 'Pack3DDetInputs_':
            # Expose required identities and supervision counts for auditing.
            from_keys = ('lidar_path', 'sample_idx', 'ldmr_scene_id', 'axis_align_matrix',
                         'ldmr_pseudo_count', 'ldmr_inserted_count')
            if target_mode == 'boxes':
                transform['meta_keys'] = from_keys
        train_pipeline.append(transform)
        if kind == 'GlobalAlignment' and pseudo_options is not None:
            train_pipeline.append(dict(type='LDMRMergePseudoBoxes', stage=stage,
                stage_setting=protocol.stage_setting, **pseudo_options))
        if kind == 'GlobalAlignment' and memory_mode == 'object' and stage > 1:
            from .recipes import tr3d_obj_insertion
            train_pipeline.append(dict(type='LDMRInsertObjects', bank_path=str(crop_path),
                stage=stage, stage_setting=protocol.stage_setting,
                **(tr3d_obj_insertion() if object_policy == 'tr3d_obj'
                   else dict(max_exemplars=1, probability=1.0))))
    test_pipeline = []
    for transform in cfg['test_pipeline_scannet']:
        test_pipeline.append(deepcopy(transform))
        if transform['type'] == 'LoadAnnotations3D_':
            test_pipeline.append(dict(type='LDMRCompactSuperpoints'))
    for split, path, pipeline in (
            ('train', train_info, train_pipeline),
            ('val', val_info, test_pipeline)):
        dataset = dict(type='LDMRScanNetDataset', ann_file=str(path), data_root='',
                       data_prefix=dict(pts='', pts_instance_mask='',
                                        pts_semantic_mask='', sp_pts_mask=''),
                       metainfo=dict(classes=classes), pipeline=deepcopy(pipeline),
                       ignore_index=len(classes), test_mode=(split == 'val'))
        cfg[split + '_dataloader']['dataset'] = dict(type='ConcatDataset_', datasets=[dataset])
    cfg['test_dataloader'] = deepcopy(cfg['val_dataloader'])
    evaluator = dict(type='LDMRIndoorMetric', datasets=['scannet'], datasets_classes=[classes])
    cfg['val_evaluator'] = evaluator
    cfg['test_evaluator'] = deepcopy(evaluator)
    return cfg
