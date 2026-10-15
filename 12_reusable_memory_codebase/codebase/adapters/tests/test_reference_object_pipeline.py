"""One complete reference object replay -> native host loss/backward contract."""
import pickle
import numpy as np
import pytest
import torch
from mmengine.dataset import pseudo_collate
from mmdet3d.registry import DATASETS, MODELS
from test_host_runtime import config
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.configuration import stage_config
from ldmr_unidet3d.pseudo import write_pseudo, geometry_digest, load_pseudo
from ldmr_unidet3d.sequence import sha256


@pytest.mark.parametrize('config', ['object'], indirect=True)
def test_reference_object_pipeline_and_pseudo_policy(config):
    train = config['train_dataloader']['dataset']['datasets'][0]['ann_file']
    from pathlib import Path
    train = Path(train); protocol = ScanNet35Protocol()
    records = pickle.loads(train.read_bytes())['data_list']
    artifact = train.parent/'reference_teacher.json'
    write_pseudo(artifact, {'fixture': dict(boxes=[[2.5,2.5,2.5,1,1,1]], labels=[0], scores=[.49])},
        protocol, 2, teacher_sha256='a'*64, train_info_sha256=sha256(train),
        geometry_sha256=geometry_digest(records), teacher_config_sha256='b'*64, policy_name='tr3d_obj')
    with pytest.raises(ValueError, match='policy mismatch'):
        load_pseudo(artifact, protocol, 2, teacher_sha256='a'*64,
            train_info_sha256=sha256(train), geometry_sha256=geometry_digest(records))
    cfg = stage_config(config, protocol, 2, train_info=train, val_info=train.parent/'val.pkl',
        work_dir=train.parent/'reference_run', memory_mode='object', epochs=180,
        crop_path=train.parent/'crops.pkl', target_mode='boxes', object_policy='tr3d_obj',
        pseudo_path=artifact, teacher_sha256='a'*64)
    np.random.seed(0)
    sample = DATASETS.build(cfg['train_dataloader']['dataset'])[0]
    data = sample['data_samples']
    assert data.metainfo['ldmr_pseudo_count'] == 0  # .49 must not pass teammate .50
    assert data.metainfo['ldmr_inserted_count'] == 3  # one crop may be drawn repeatedly
    assert sorted(data.gt_instances_3d.labels_3d.tolist()) == [0, 0, 0, 7]
    model = MODELS.build(cfg['model']).cuda().train()
    model.init_weights()
    batch = model.data_preprocessor(pseudo_collate([sample]), training=True)
    losses = model.loss(batch['inputs'], batch['data_samples'])
    loss = sum(value if torch.is_tensor(value) else sum(value) for value in losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
