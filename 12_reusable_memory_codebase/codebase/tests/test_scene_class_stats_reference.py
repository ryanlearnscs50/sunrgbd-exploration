"""Original LDMR behavior on real SUN annotations that broke strict guards."""
import json
from pathlib import Path

import numpy as np

from mmdet3d.core.bbox import DepthInstance3DBoxes, Box3DMode
from mmdet3d.core.evaluation.scene_class_stats import compute_scene_class_match_stats


def test_real_zero_height_annotations_preserve_reference_gt_denominator():
    fixture = json.loads((Path(__file__).parent /
        'reference_fixtures/sun_zero_height_gt.json').read_text())
    for example in fixture['examples']:
        annotation = dict(gt_num=len(example['boxes']),
            gt_boxes_upright_depth=np.asarray(example['boxes'],np.float32),
            **{'class':np.asarray(example['labels'],np.int64)})
        result = compute_scene_class_match_stats([annotation], [{}],
            seat_keys=[(example['scene'],1)], eval_class_indices=example['classes'],
            iou_thr=.25, box_type_3d=DepthInstance3DBoxes, box_mode_3d=Box3DMode.DEPTH)
        assert json.loads(json.dumps(result)) == example['expected']
