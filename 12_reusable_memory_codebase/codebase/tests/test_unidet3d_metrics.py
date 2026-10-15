import json
import numpy as np
import pytest

from adapters.ldmr_unidet3d.metrics import json_metrics


def test_host_nested_metrics_preserve_zero_and_report_undefined_ap():
    values, undefined = json_metrics({'scannet': {'mAP_0.25': np.float32(0),
                                                 'missing_AP': float('nan')}})
    assert values == {'scannet': {'mAP_0.25': 0., 'missing_AP': None}}
    assert undefined == [['scannet', 'missing_AP']]
    assert json.loads(json.dumps(values, allow_nan=False)) == values


@pytest.mark.parametrize('value', [{}, {'scannet': {}}, {'AP': '0.3'}, {'AP': True}, []])
def test_invalid_host_results_fail_instead_of_looking_complete(value):
    with pytest.raises(ValueError):
        json_metrics(value)


@pytest.mark.parametrize('value', [float('inf'), float('-inf'), np.float32('inf')])
def test_infinite_metrics_are_errors_not_missing_class_ap(value):
    with pytest.raises(ValueError, match='infinite metric'):
        json_metrics({'scannet': {'mAP_0.25': value}})
