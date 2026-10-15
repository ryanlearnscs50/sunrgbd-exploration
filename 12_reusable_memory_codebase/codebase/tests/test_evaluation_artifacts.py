import json
from pathlib import Path

import numpy as np
import pytest
from mmcv import Config

from tools.evaluation_artifacts import write_evaluation_report


def inputs(tmp_path):
    class Dataset:
        CLASSES = ('old', 'future')
        seen_classes_for_eval = [0]
        ann_file = tmp_path / 'ann.pkl'

        def __len__(self):
            return 3

    Dataset.ann_file.write_bytes(b'annotation fixture')
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'checkpoint fixture')
    config_file = tmp_path / 'config.py'
    config_file.write_text('workers = 4\n')
    config = Config.fromfile(str(config_file))
    config.workers = 2
    return dict(dataset=Dataset(), config=config, checkpoint=checkpoint,
                seed=200, stage=1, trained_classes=1)


def test_report_records_actual_scope_and_refuses_overwrite(tmp_path):
    path = tmp_path / 'report.json'
    args = inputs(tmp_path)
    report = write_evaluation_report(path, {'mAP_0.25': np.float32(.5)}, **args)
    assert report['evaluated_classes'] == ['old']
    assert report['scene_count'] == 3
    assert report['trained_class_count'] == 1
    assert "'workers': 2" in report['resolved_config_repr']
    assert report['annotations'][0]['sha256']
    before = path.read_bytes()
    assert json.loads(before)['metrics']['mAP_0.25'] == .5
    with pytest.raises(FileExistsError):
        write_evaluation_report(path, {'mAP_0.25': .9}, **args)
    assert path.read_bytes() == before


@pytest.mark.parametrize('metrics', [{}, {'other': .2}, {'mAP_0.25': float('nan')},
                                    {'mAP_0.25': float('inf')}, {'mAP_0.25': []}])
def test_invalid_results_cannot_be_published_as_success(tmp_path, metrics):
    path = tmp_path / 'report.json'
    with pytest.raises(ValueError):
        write_evaluation_report(path, metrics, **inputs(tmp_path))
    assert not path.exists()
