from __future__ import annotations

import pytest

from mmdet3d.utils.pregenerate_pseudo_labels import PseudoLabelPreGenerator


def _make_generator(*, stage_id, stage_definitions=None):
    obj = PseudoLabelPreGenerator.__new__(PseudoLabelPreGenerator)
    obj.stage_id = int(stage_id)
    obj.previous_stage_id = int(stage_id) - 1
    obj.stage_definitions = stage_definitions
    return obj


def test_scannet_pregenerated_prev_classes_uses_passed_stage_definitions():
    stage_defs = [
        {'stage_id': 1, 'class_indices': [0, 1, 2]},
        {'stage_id': 2, 'class_indices': [3, 4]},
        {'stage_id': 3, 'class_indices': [5, 6]},
    ]

    gen_s2 = _make_generator(stage_id=2, stage_definitions=stage_defs)
    assert gen_s2._resolve_previous_stage_classes() == [0, 1, 2]

    gen_s3 = _make_generator(stage_id=3, stage_definitions=stage_defs)
    assert gen_s3._resolve_previous_stage_classes() == [0, 1, 2, 3, 4]


def test_scannet_pregenerated_prev_classes_fallback_and_strict_failure():
    gen_s5 = _make_generator(stage_id=5, stage_definitions=None)
    assert gen_s5._resolve_previous_stage_classes() == list(range(0, 28))

    gen_s8 = _make_generator(stage_id=8, stage_definitions=None)
    with pytest.raises(ValueError, match='Cannot infer previous-stage classes'):
        gen_s8._resolve_previous_stage_classes()
