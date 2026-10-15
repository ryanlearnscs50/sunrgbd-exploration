"""Validate and expand a UniDet3D model state at an LDMR stage boundary."""
from collections.abc import Mapping
from collections import OrderedDict
from copy import deepcopy
from .protocol import remap_classifier


def _checkpoint_state(checkpoint, protocol, stage):
    import torch
    if not isinstance(checkpoint, Mapping):
        raise ValueError('checkpoint must be a mapping')
    meta = checkpoint.get('meta')
    if not isinstance(meta, Mapping):
        raise ValueError('checkpoint protocol metadata must be a mapping')
    if meta.get('ldmr_protocol_id') != protocol.identity or meta.get('ldmr_stage') != stage:
        raise ValueError('checkpoint protocol or stage mismatch')
    # Equality alone would accept True or 1.0 as stage 1.
    protocol.seen_count(meta.get('ldmr_stage'))
    state = checkpoint.get('state_dict')
    if not isinstance(state, Mapping) or not state:
        raise ValueError('checkpoint state_dict must be a nonempty mapping')
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, torch.Tensor):
            raise ValueError('checkpoint state_dict must contain named tensors')
        if not torch.isfinite(value).all():
            raise ValueError(f'nonfinite checkpoint tensor: {name}')
    return state


def validate_checkpoint_state(checkpoint, expected_state, protocol, stage):
    """Require exact stage, tensor schema and finite weights before host evaluation."""
    protocol.seen_count(stage)
    state = _checkpoint_state(checkpoint, protocol, stage)
    validate_state_metadata(state, expected_state)
    if set(state) != set(expected_state):
        raise ValueError('checkpoint model parameter names differ')
    for name, expected in expected_state.items():
        value = state[name]
        if value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f'checkpoint tensor shape/dtype mismatch: {name}')


def expand_checkpoint(previous, initialized_state, protocol, stage):
    """Return full weights for a new-stage model, resetting optimizer/RNG state.

    initialized_state comes from a freshly initialized host model for `stage`.
    All non-classifier tensors must match exactly by name and shape. A native
    ScanNet-18 checkpoint or a checkpoint with unknown protocol is rejected.
    """
    protocol.seen_count(stage)
    if stage == 1:
        raise ValueError('stage 1 has no preceding incremental checkpoint')
    import torch
    old = _checkpoint_state(previous, protocol, stage - 1)
    if set(old) != set(initialized_state):
        raise ValueError('checkpoint model parameter names differ')
    weight, bias = 'decoder.outs_cls.2.weight', 'decoder.outs_cls.2.bias'
    if weight not in old or bias not in old:
        raise ValueError('expected pinned UniDet3D classifier layout')
    validate_state_metadata(old, initialized_state)
    expanded = OrderedDict()
    if hasattr(initialized_state, '_metadata'):
        expanded._metadata = deepcopy(initialized_state._metadata)
    for name, value in initialized_state.items():
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f'invalid initialized tensor: {name}')
        if old[name].dtype != value.dtype:
            raise ValueError(f'checkpoint tensor dtype mismatch: {name}')
        if name not in (weight, bias):
            if old[name].shape != value.shape:
                raise ValueError(f'non-classifier shape mismatch: {name}')
            expanded[name] = old[name].detach().clone()
    expanded[weight], expanded[bias] = remap_classifier(
        old[weight], old[bias], protocol.decoder_classes(stage - 1),
        initialized_state[weight], initialized_state[bias], protocol.decoder_classes(stage))
    return dict(state_dict=expanded, meta=dict(ldmr_protocol_id=protocol.identity,
        ldmr_stage=stage, ldmr_initialized_from_stage=stage - 1))


def validate_state_metadata(state, expected):
    """Native sparse layer versions determine kernel layout at load time."""
    metadata = getattr(expected, '_metadata', None)
    if metadata is not None and getattr(state, '_metadata', None) != metadata:
        raise ValueError('checkpoint module metadata missing or incompatible with native model')


def assert_loaded_checkpoint(model, checkpoint, protocol, stage):
    """Check actual runner tensors after loading, before any training update."""
    import torch
    actual = model.state_dict()
    validate_checkpoint_state(checkpoint, actual, protocol, stage)
    for name, expected in checkpoint['state_dict'].items():
        if not torch.equal(actual[name].detach().cpu(), expected.detach().cpu()):
            raise ValueError(f'runner did not load initialization tensor exactly: {name}')
    return len(actual)
