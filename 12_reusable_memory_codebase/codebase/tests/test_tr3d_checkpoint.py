import pytest
import torch

from mmdet3d.utils.tr3d_checkpoint import convert_openmmlab_tr3d_state


def fixture():
    source = {'head.conv_reg.kernel': torch.arange(12).reshape(3, 4).float(),
              'head.conv_reg.bias': torch.ones(1, 4),
              'head.conv_cls.kernel': torch.arange(6).reshape(3, 2).float(),
              'backbone.weight': torch.ones(2)}
    target = {'head.bbox_conv.kernel': torch.empty(3, 4),
              'head.bbox_conv.bias': torch.empty(1, 4),
              'head.cls_conv.kernel': torch.empty(3, 2),
              'head.cls_conv.bias': torch.empty(1, 2),
              'backbone.weight': torch.empty(2)}
    return source, target


def test_conversion_preserves_classifier_logits_and_all_learned_values():
    source, target = fixture()
    result = convert_openmmlab_tr3d_state(source, target)
    features = torch.randn(17, 3)
    assert torch.equal(features @ source['head.conv_cls.kernel'],
                       features @ result['head.cls_conv.kernel'] + result['head.cls_conv.bias'])
    assert torch.equal(features @ source['head.conv_reg.kernel'] + source['head.conv_reg.bias'],
                       features @ result['head.bbox_conv.kernel'] + result['head.bbox_conv.bias'])
    assert result['backbone.weight'] is source['backbone.weight']
    assert 'head.cls_conv.bias' not in source


@pytest.mark.parametrize('damage', ['bias', 'mixed', 'missing', 'shape', 'dtype', 'nan'])
def test_conversion_rejects_unsupported_or_corrupt_checkpoints(damage):
    source, target = fixture()
    if damage == 'bias':
        source['head.conv_cls.bias'] = torch.ones(1, 2)
    elif damage == 'mixed':
        source['head.cls_conv.kernel'] = source['head.conv_cls.kernel']
    elif damage == 'missing':
        del source['backbone.weight']
    elif damage == 'shape':
        source['backbone.weight'] = torch.ones(3)
    elif damage == 'dtype':
        source['backbone.weight'] = torch.ones(2, dtype=torch.float64)
    else:
        source['backbone.weight'][0] = float('nan')
    with pytest.raises(ValueError):
        convert_openmmlab_tr3d_state(source, target)
