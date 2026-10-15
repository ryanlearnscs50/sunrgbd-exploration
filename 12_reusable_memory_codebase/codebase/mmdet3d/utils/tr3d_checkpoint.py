"""Explicit conversion of the published OpenMMLab TR3D head parameter names."""
import torch


def convert_openmmlab_tr3d_state(source, target):
    """Map the known bias-free-classifier schema and require exact compatibility.

    OpenMMLab's published classifier has no bias. LDMR's classifier does, so a
    zero bias preserves its logits. No learned values are initialized or dropped.
    The target is a model state dictionary, used for key/shape/dtype validation.
    Optimizer state is deliberately outside this inference conversion contract.
    """
    renames = {'head.conv_reg.kernel': 'head.bbox_conv.kernel',
               'head.conv_reg.bias': 'head.bbox_conv.bias',
               'head.conv_cls.kernel': 'head.cls_conv.kernel'}
    if not set(renames).issubset(source) or 'head.conv_cls.bias' in source:
        raise ValueError('expected published TR3D bias-free classifier schema')
    if any(key in source for key in (*renames.values(), 'head.cls_conv.bias')):
        raise ValueError('mixed or already converted TR3D head schema')
    result = {renames.get(key, key): value for key, value in source.items()}
    if 'head.cls_conv.bias' not in target:
        raise ValueError('target must expose the LDMR classifier bias')
    result['head.cls_conv.bias'] = torch.zeros_like(target['head.cls_conv.bias'])
    if set(result) != set(target):
        raise ValueError(f'incompatible model keys: missing={sorted(set(target)-set(result))}, '
                         f'extra={sorted(set(result)-set(target))}')
    for key, value in result.items():
        expected = target[key]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f'incompatible tensor shape/dtype: {key}')
        if not torch.isfinite(value).all():
            raise ValueError(f'nonfinite checkpoint tensor: {key}')
    return result
