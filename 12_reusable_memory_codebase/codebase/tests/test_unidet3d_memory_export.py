"""Exercise exporter CLIs across two stages with synthetic point groups."""
import itertools
from pathlib import Path
import pickle
import subprocess
import sys
import os
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'adapters'))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.data import convert_info
from ldmr_unidet3d.objects import load_crops


def full_infos(tmp_path):
    p = ScanNet35Protocol()
    corners = np.array(list(itertools.product([-1.,1.], repeat=3)), np.float32)
    records = []
    for i in range(3):
        source = tmp_path/f's{i}'
        source.mkdir()
        xyz = np.concatenate([corners+[0,0,1], corners+[4,4,2]]).astype('float32')
        pts = np.column_stack([xyz, np.full((16,3),127,np.float32)])
        pts.tofile(source/'points.bin')
        np.repeat([5,18],8).astype(np.int64).tofile(source/'semantic.bin')
        np.repeat([11,23],8).astype(np.int64).tofile(source/'instance.bin')
        (np.arange(16)//2).astype(np.int64).tofile(source/'sp.bin')
        info = dict(point_cloud=dict(lidar_idx=f's{i}',num_features=6),pts_path='points.bin',
                    pts_instance_mask_path='instance.bin',pts_semantic_mask_path='semantic.bin',
                    annos=dict(**{'class':np.array([5,18])},axis_align_matrix=np.eye(4),
                    gt_boxes_upright_depth=np.array([[0,0,1,2,2,2],[4,4,2,2,2,2]])))
        records.append(convert_info(info,p,data_root=source,superpoint_path=source/'sp.bin',box_label_space='nyu40'))
    path = tmp_path/'all.pkl'
    with path.open('wb') as f: pickle.dump(dict(data_list=records),f)
    return path


@pytest.mark.parametrize('mode', ['scene','object'])
def test_export_cli_two_stages_and_stage_manifest(tmp_path, mode):
    source = full_infos(tmp_path)
    env = dict(os.environ, OMP_NUM_THREADS='2')
    for stage in (1,2):
        output = tmp_path/f'{mode}_{stage}'
        command = [sys.executable,str(REPO/'adapters/tools/export_memory.py'),
                   '--input',str(source),'--output-dir',str(output),
                   '--stage',str(stage),'--memory-mode',mode,'--scene-budget','2',
                   '--objects-per-class','2','--max-objects','4','--min-points','1']
        if stage>1:
            command += ['--previous-bank',str(tmp_path/f'{mode}_1/bank.pkl')]
            if mode=='object': command += ['--previous-crops',str(tmp_path/'object_1/crops.pkl')]
        result = subprocess.run(command,capture_output=True,text=True,env=env,timeout=90)
        assert result.returncode==0, result.stdout+result.stderr
        assert (output/'bank.pkl').is_file()
        if mode=='object':
            crops=load_crops(output/'crops.pkl',ScanNet35Protocol(),stage)
            assert len(crops)==stage*2
        else:
            with (output/'replay_records.pkl').open('rb') as f: replay=pickle.load(f)
            assert len(replay)==2
    command = [sys.executable,str(REPO/'adapters/tools/prepare_stage_infos.py'),
               '--input',str(source),'--output',str(tmp_path/'stage2.pkl'),
               '--stage','2','--split','train','--memory-mode',mode]
    if mode=='scene': command += ['--replay-records',str(tmp_path/'scene_1/replay_records.pkl')]
    result = subprocess.run(command,capture_output=True,text=True,env=env,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr
    with (tmp_path/'stage2.pkl').open('rb') as f: payload=pickle.load(f)
    assert len(payload['data_list'])==3
    if mode=='scene':
        assert sum(r.get('ldmr_include_current',False) for r in payload['data_list'])==2


@pytest.mark.parametrize('existing', [False, True])
def test_sequence_rejects_split_leakage_and_existing_run_before_training(tmp_path, existing):
    source = full_infos(tmp_path)
    output = tmp_path / 'run'
    if existing:
        output.mkdir()
        (output / 'sentinel').write_text('preserve')
    command = [sys.executable, str(REPO/'adapters/tools/run_sequence.py'),
               '--host-python', sys.executable, '--ldmr-python', sys.executable,
               '--upstream-root', str(tmp_path/'unused'), '--train-info', str(source),
               '--val-info', str(source), '--work-dir', str(output),
               '--memory-mode', 'none', '--last-stage', '2']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert ('work directory exists' if existing else 'scene IDs overlap') in result.stderr
    if existing:
        assert (output / 'sentinel').read_text() == 'preserve'
        assert len(list(output.iterdir())) == 1
    else:
        assert not output.exists()


@pytest.mark.parametrize('existing', [False, True])
def test_standalone_stage_rejects_leakage_and_existing_output(tmp_path, existing):
    from ldmr_unidet3d.data import stage_records
    source = full_infos(tmp_path)
    with source.open('rb') as stream:
        records = pickle.load(stream)['data_list']
    protocol = ScanNet35Protocol()
    for split in ('train', 'val'):
        payload = dict(metainfo=dict(ldmr_protocol_id=protocol.identity,
            ldmr_stage=1, ldmr_split=split, ldmr_memory_mode='none'),
            data_list=stage_records(records, protocol, 1, evaluation=split == 'val'))
        with (tmp_path / (split + '.pkl')).open('wb') as stream:
            pickle.dump(payload, stream)
    output = tmp_path / 'run'
    if existing:
        output.mkdir()
        (output / 'sentinel').write_text('preserve')
    command = [sys.executable, str(REPO/'adapters/tools/train_stage.py'),
               '--upstream-root', str(tmp_path/'unused'),
               '--train-info', str(tmp_path/'train.pkl'),
               '--val-info', str(tmp_path/'val.pkl'), '--work-dir', str(output),
               '--memory-mode', 'none', '--stage', '1']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert ('work directory exists' if existing else 'scene IDs overlap') in result.stderr
    if existing:
        assert (output / 'sentinel').read_text() == 'preserve'
        assert len(list(output.iterdir())) == 1
    else:
        assert not output.exists()
