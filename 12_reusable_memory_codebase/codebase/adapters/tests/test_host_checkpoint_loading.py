"""Exercise actual sparse-module loading, not just saved tensor dictionaries."""
from copy import deepcopy
import torch
import pytest
import spconv.pytorch as spconv
from ldmr_unidet3d import host, ScanNet35Protocol
from ldmr_unidet3d.checkpoints import expand_checkpoint, assert_loaded_checkpoint


class SmallSparseDetector(torch.nn.Module):
    def __init__(self, classes):
        super().__init__()
        self.backbone = spconv.SubMConv3d(2, 4, 3, bias=False)
        self.decoder = torch.nn.Module()
        self.decoder.outs_cls = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity(),
                                                   torch.nn.Linear(4, classes + 1))


def test_expanded_checkpoint_really_loads_sparse_weights_and_guards_runner(tmp_path):
    protocol = ScanNet35Protocol()
    old, new = SmallSparseDetector(7), SmallSparseDetector(14)
    with torch.no_grad(): old.backbone.weight.fill_(.375)
    checkpoint = dict(state_dict=old.state_dict(), meta=dict(ldmr_protocol_id=protocol.identity, ldmr_stage=1))
    expanded = expand_checkpoint(checkpoint, new.state_dict(), protocol, 2)
    path = tmp_path / 'initialized_stage_2.pth'
    torch.save(expanded, path)
    expanded = torch.load(path, map_location='cpu')
    new.load_state_dict(expanded['state_dict'], strict=True)
    assert assert_loaded_checkpoint(new, expanded, protocol, 2) == len(new.state_dict())
    assert torch.equal(new.backbone.weight, old.backbone.weight)
    from types import SimpleNamespace
    runner = SimpleNamespace(work_dir=str(tmp_path), model=new)
    hook = host.LDMRProtocolCheckpointHook(2)
    hook.before_train(runner)
    assert (tmp_path / 'loaded_initialization.json').is_file()
    with torch.no_grad(): new.backbone.weight.zero_()
    with pytest.raises(ValueError, match='did not load'):
        hook.before_train(runner)


def test_old_expansion_without_metadata_fails_real_sparse_loading():
    model = SmallSparseDetector(7)
    lost = dict(deepcopy(model.state_dict()))
    with pytest.raises(RuntimeError, match='size mismatch'):
        model.load_state_dict(lost, strict=True)
