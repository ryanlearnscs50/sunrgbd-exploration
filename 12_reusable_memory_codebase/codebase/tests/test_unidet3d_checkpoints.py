"""Reject corrupt stage transfers before they can contaminate a training run."""
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'adapters'))
from ldmr_unidet3d import ScanNet35Protocol
from ldmr_unidet3d.checkpoints import expand_checkpoint, validate_checkpoint_state


def states():
    protocol = ScanNet35Protocol()
    old = {'decoder.outs_cls.2.weight': torch.arange(24).reshape(8, 3).float(),
           'decoder.outs_cls.2.bias': torch.arange(8).float(),
           'backbone.weight': torch.ones(2),
           'backbone.num_batches_tracked': torch.tensor(4)}
    new = {**copy.deepcopy(old), 'decoder.outs_cls.2.weight': torch.zeros(15, 3),
           'decoder.outs_cls.2.bias': torch.zeros(15)}
    checkpoint = dict(state_dict=old, meta=dict(ldmr_protocol_id=protocol.identity,
                                               ldmr_stage=1))
    return protocol, checkpoint, new


@pytest.mark.parametrize('name', ['decoder.outs_cls.2.weight',
                                  'decoder.outs_cls.2.bias', 'backbone.weight'])
@pytest.mark.parametrize('damage', ['nan', 'inf', 'dtype', 'not_tensor'])
def test_expansion_rejects_corrupt_tensors(name, damage):
    protocol, checkpoint, new = states()
    old = checkpoint['state_dict']
    if damage in ('nan', 'inf'):
        old[name].view(-1)[0] = float(damage)
    elif damage == 'dtype':
        old[name] = old[name].double()
    else:
        old[name] = old[name].tolist()
    with pytest.raises(ValueError):
        expand_checkpoint(checkpoint, new, protocol, 2)


@pytest.mark.parametrize('stage', [True, 1.0, None, 0, 2])
@pytest.mark.parametrize('operation', ['expand', 'evaluate'])
def test_checkpoint_stage_requires_exact_integer(stage, operation):
    protocol, checkpoint, new = states()
    checkpoint['meta']['ldmr_stage'] = stage
    with pytest.raises(ValueError):
        if operation == 'expand':
            expand_checkpoint(checkpoint, new, protocol, 2)
        else:
            validate_checkpoint_state(checkpoint, checkpoint['state_dict'], protocol, 1)


@pytest.mark.parametrize('damage', ['root', 'meta', 'state', 'empty', 'missing_stage'])
def test_checkpoint_schema_errors_are_explicit(damage):
    protocol, checkpoint, new = states()
    if damage == 'root':
        checkpoint = None
    elif damage == 'meta':
        checkpoint['meta'] = None
    elif damage == 'state':
        checkpoint['state_dict'] = []
    elif damage == 'empty':
        checkpoint['state_dict'] = {}
    else:
        del checkpoint['meta']['ldmr_stage']
    with pytest.raises(ValueError):
        expand_checkpoint(checkpoint, new, protocol, 2)


def test_expansion_preserves_buffers_and_does_not_alias_inputs():
    protocol, checkpoint, new = states()
    before = copy.deepcopy(checkpoint)
    expanded = expand_checkpoint(checkpoint, new, protocol, 2)
    validate_checkpoint_state(expanded, new, protocol, 2)
    for name, value in expanded['state_dict'].items():
        assert value.data_ptr() != checkpoint['state_dict'][name].data_ptr()
        assert value.data_ptr() != new[name].data_ptr()
        assert torch.equal(checkpoint['state_dict'][name], before['state_dict'][name])
    assert expanded['state_dict']['backbone.num_batches_tracked'].item() == 4


def test_expansion_rejects_nonfinite_new_class_initialization():
    protocol, checkpoint, new = states()
    new['decoder.outs_cls.2.weight'][-2, 0] = float('nan')
    with pytest.raises(ValueError, match='initialized'):
        expand_checkpoint(checkpoint, new, protocol, 2)


def test_expansion_preserves_native_module_metadata_without_aliasing():
    from collections import OrderedDict
    protocol, checkpoint, new = states()
    checkpoint['state_dict'] = OrderedDict(checkpoint['state_dict'])
    new = OrderedDict(new)
    metadata = {'backbone': {'version': 2}}
    checkpoint['state_dict']._metadata = copy.deepcopy(metadata)
    new._metadata = copy.deepcopy(metadata)
    expanded = expand_checkpoint(checkpoint, new, protocol, 2)
    assert expanded['state_dict']._metadata == metadata
    expanded['state_dict']._metadata['backbone']['version'] = 99
    assert checkpoint['state_dict']._metadata == new._metadata == metadata


def test_native_model_rejects_checkpoint_with_lost_module_metadata():
    from collections import OrderedDict
    protocol, checkpoint, _ = states()
    expected = OrderedDict(checkpoint['state_dict'])
    expected._metadata = {'backbone': {'version': 2}}
    with pytest.raises(ValueError, match='metadata'):
        validate_checkpoint_state(checkpoint, expected, protocol, 1)
