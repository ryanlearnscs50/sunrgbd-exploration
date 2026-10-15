from __future__ import annotations

from mmdet3d.datasets.incremental_scannet import IncrementalScanNetDataset


class _DummySceneMemoryBank:
    def __init__(self):
        self.calls = []

    def add_stage_scenes(self, **kwargs):
        self.calls.append(kwargs)

    def save_state(self, *_args, **_kwargs):
        return None

    def print_summary(self):
        return None


def test_scannet_memory_bank_update_forwards_ld_review_payloads():
    bank = _DummySceneMemoryBank()

    ds = IncrementalScanNetDataset.__new__(IncrementalScanNetDataset)
    ds.scene_memory_bank = bank
    ds.stage_idx = 1  # stage_id = 2
    ds.all_stage_definitions = [
        {'stage_id': 1, 'class_indices': [0, 1]},
        {'stage_id': 2, 'class_indices': [2]},
    ]
    ds.data_infos = [
        {'sample_idx': 'natural_a', 'is_replay': False, 'annos': {'gt_num': 1}},
        {'sample_idx': 'replay_b', 'is_replay': True, 'annos': {'gt_num': 1}},
    ]
    ds.mappings = {'model_idx_to_name': {0: 'a', 1: 'b', 2: 'c'}}
    ds.evaluation_mode = False
    ds.work_dir = None

    forgetness = {0: 0.3}
    underlearning_ap = {2: 0.2}
    underlearning_new_classes = [2]
    ld_forgetness = {'natural_a': {1: 0.1}}
    ld_replay_priority = {'natural_a': {1: 1.2}}
    ld_design1_payload = {
        'stage_id': 2,
        'q_metric': 'f1',
        'class_need': {'0': 0.5},
        'seat_class_terms': {'natural_a': {'1': {'0': {'q': 0.5, 'need': 0.5, 'supply': 1.0}}}},
    }
    dataset_ref = object()

    ds.update_scene_memory_bank_from_stage(
        model=None,
        forgetness_class_drops=forgetness,
        underlearning_class_ap=underlearning_ap,
        underlearning_new_classes=underlearning_new_classes,
        learning_dynamics_forgetness_by_seat=ld_forgetness,
        learning_dynamics_replay_priority_by_seat=ld_replay_priority,
        learning_dynamics_design1_payload=ld_design1_payload,
        dataset_ref=dataset_ref,
    )

    assert len(bank.calls) == 1
    call = bank.calls[0]
    assert int(call['stage_id']) == 2
    assert call['scene_infos'] == [ds.data_infos[0]]
    assert call['seen_classes'] == [0, 1, 2]
    assert call['dataset_ref'] is dataset_ref
    assert call['forgetness_class_drops'] == forgetness
    assert call['underlearning_class_ap'] == underlearning_ap
    assert call['underlearning_new_classes'] == underlearning_new_classes
    assert call['learning_dynamics_forgetness_by_seat'] == ld_forgetness
    assert call['learning_dynamics_replay_priority_by_seat'] == ld_replay_priority
    assert call['learning_dynamics_design1_payload'] == ld_design1_payload
