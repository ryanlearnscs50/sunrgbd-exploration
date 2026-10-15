"""Exercise shared LD measurement with the actual modern host box container."""
import pytest
from mmdet3d.structures import DepthInstance3DBoxes, Box3DMode
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.learning_dynamics import measure_scene


@pytest.mark.parametrize('metric,expected', [('f1', 2/3), ('recall', 1)])
def test_host_box_origin_and_duplicate_matching(metric, expected):
    box = [3, -2, 5, 2, 4, 6]
    result = measure_scene(dict(boxes=[box, box], labels=[0, 0], scores=[.9, .8]),
        [box], [0], protocol=ScanNet35Protocol(), stage=2, replay_from_stage=1,
        scene_id='host', box_type_3d=DepthInstance3DBoxes, box_mode_3d=Box3DMode.DEPTH,
        q_metric=metric)
    row = result['classes'][0]
    assert (row['tp'], row['fp'], row['fn']) == (1, 1, 0)
    assert row['q'] == pytest.approx(expected)
