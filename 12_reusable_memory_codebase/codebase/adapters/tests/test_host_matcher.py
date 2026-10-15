"""Run in the isolated host runtime alongside test_host_runtime.py."""
from types import SimpleNamespace
import torch
import pytest
from ldmr_unidet3d.host import LDMRUniMatcher
from unidet3d.criterion import UniMatcher


def match(cost, masks, topk=6, cls=LDMRUniMatcher):
    matcher = cls([])
    matcher.costs = [lambda *_: cost]
    gt = SimpleNamespace(labels=torch.arange(masks.shape[0]), query_masks=masks)
    return matcher(SimpleNamespace(), gt, topk)


@pytest.mark.parametrize('queries', [0, 1, 3, 6, 7, 10])
def test_tied_positive_queries_are_bounded_not_dropped(queries):
    q, g = match(torch.zeros(queries, 1), torch.ones(1, queries, dtype=torch.bool))
    assert q.tolist() == list(range(min(6, queries)))
    assert g.tolist() == [0] * min(6, queries)


def test_unsupported_queries_never_become_positives():
    masks = torch.tensor([[False, True, False], [False, False, False]])
    q, g = match(torch.zeros(3, 2), masks)
    assert q.tolist() == [1] and g.tolist() == [0]


def test_distinct_costs_preserve_upstream_assignments():
    cost = torch.arange(60).reshape(20, 3).float().flip(0)
    masks = torch.zeros(3, 20, dtype=torch.bool)
    masks[0, :9] = True
    masks[1, 9:17] = True
    masks[2, 17:] = True
    expected = match(cost, masks, cls=UniMatcher)
    actual = match(cost, masks)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


def test_no_targets_returns_empty_assignment():
    assert all(x.numel() == 0 for x in match(torch.empty(3, 0), torch.empty(0, 3, dtype=torch.bool)))


@pytest.mark.parametrize('topk', [0, -1, True, 1.5])
def test_invalid_topk_fails(topk):
    with pytest.raises(ValueError, match='positive integer'):
        match(torch.zeros(3, 1), torch.ones(1, 3, dtype=torch.bool), topk)


@pytest.mark.parametrize('invalid', [float('nan'), float('inf')])
def test_nonfinite_costs_fail(invalid):
    with pytest.raises(ValueError, match='finite'):
        match(torch.full((3, 1), invalid), torch.ones(1, 3, dtype=torch.bool))
