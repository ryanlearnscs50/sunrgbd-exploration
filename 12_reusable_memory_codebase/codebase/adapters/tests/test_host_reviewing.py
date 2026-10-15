"""Exercise the real DataLoader across an online sampler update."""
import torch
from torch.utils.data import DataLoader
from ldmr_unidet3d.reviewing_host import LDMRReviewSampler


class ReviewDataset:
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return index

    def get_data_info(self, index):
        return dict(sample_idx=str(index), is_replay=index >= 10,
                    ldmr_replay_from_stage=1, ldmr_include_current=index == 11)


def test_next_epoch_uses_reviewed_indices_with_persistent_workers():
    dataset = ReviewDataset()
    sampler = LDMRReviewSampler(dataset, seed=200, shuffle=True)
    loader = DataLoader(dataset, sampler=sampler, batch_size=1,
                        num_workers=2, persistent_workers=True)
    initial = [int(batch[0]) for batch in loader]
    assert sorted(initial) == list(range(12))
    before_rng = torch.get_rng_state().clone()
    report = sampler.apply_review({'10_stage1': 6., '11_stage1': 1.},
                                 stage=2, review_k=1, seed=200)
    assert torch.equal(before_rng, torch.get_rng_state())
    sampler.set_epoch(1)
    actual = [int(batch[0]) for batch in loader]
    assert actual == list(iter(sampler))
    assert sorted(actual) == sorted(report['indices'])
    assert actual.count(10) > 1 and 11 in actual
    assert report['memory_never_seen_count'] == 0


class ObjectReviewDataset:
    def __init__(self, insertion, length=160):
        self.insertion, self.length = insertion, length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        import numpy as np
        from mmdet3d.structures import DepthPoints, DepthInstance3DBoxes
        p = self.insertion.protocol
        result = self.insertion(dict(ldmr_protocol_id=p.identity,
            points=DepthPoints(np.array([[-10,-10,0,127,127,127], [10,10,4,127,127,127]], np.float32),points_dim=6),
            gt_bboxes_3d=DepthInstance3DBoxes(np.empty((0,6),np.float32),box_dim=6,with_yaw=False),
            gt_labels_3d=np.empty(0,np.int64),pts_semantic_mask=np.zeros(2,np.int64),
            pts_instance_mask=np.full(2,-1,np.int64),sp_pts_mask=np.arange(2)))
        return dict(version=result['ldmr_object_review_k'],draws=result['ldmr_object_draws'],
                    inserted=result['ldmr_inserted_count'])


def object_insertion(tmp_path, max_exemplars=1):
    import itertools
    import numpy as np
    from ldmr_unidet3d import ScanNet35Protocol
    from ldmr_unidet3d.objects import extract_crop, save_crops
    from ldmr_unidet3d.transforms import LDMRInsertObjects
    p = ScanNet35Protocol()
    xyz = np.array(list(itertools.product([-.5,.5],repeat=3)),np.float32)
    pts = np.column_stack([xyz,np.full((8,3),127,np.float32)])
    crops = [extract_crop(pts,np.full(8,p.raw_ids[0]),np.zeros(8,np.int64),np.arange(8),
        bbox=[0,0,0,1,1,1],instance_id=0,class_id=0,scene_id=s,object_idx=0,stage=1,
        protocol=p,min_points=1) for s in ('a','b')]
    path = tmp_path/'crops.pkl'
    save_crops(path,crops,p,1)
    insertion = LDMRInsertObjects(path,2,max_exemplars=max_exemplars)
    insertion.enable_reviewing()
    return insertion


def identity_collate(batch):
    return batch


def seed_object_worker(worker_id):
    import numpy as np
    np.random.seed(200 + worker_id)


import pytest


@pytest.mark.parametrize('context', ['fork', 'spawn'])
def test_object_reviews_reach_actual_persistent_worker_draws(tmp_path, context):
    insertion = object_insertion(tmp_path)
    loader = DataLoader(ObjectReviewDataset(insertion),batch_size=8,num_workers=2,
        persistent_workers=True,multiprocessing_context=context,collate_fn=identity_collate,
        worker_init_fn=seed_object_worker)
    try:
        initial = [row for batch in loader for row in batch]
        assert all(r['version']==0 and r['draws'][0]['weight']==1. for r in initial)
        rng = torch.get_rng_state().clone()
        insertion.apply_review({'a_stage1':10.,'b_stage1':1.},review_k=1)
        assert torch.equal(rng,torch.get_rng_state())
        updated = [row for batch in loader for row in batch]
        assert all(r['version']==1 and r['inserted']==1 for r in updated)
        assert all(r['draws'][0]['weight']==(10. if r['draws'][0]['crop_index']==0 else 1.)
                   for r in updated)
        before = sum(r['draws'][0]['crop_index']==0 for r in initial)
        after = sum(r['draws'][0]['crop_index']==0 for r in updated)
        assert after > before and after > .75 * len(updated)
        insertion.apply_review({'a_stage1':1.,'b_stage1':10.},review_k=2)
        reversed_rows = [row for batch in loader for row in batch]
        assert all(r['version']==2 for r in reversed_rows)
        assert sum(r['draws'][0]['crop_index']==1 for r in reversed_rows) > .75 * len(reversed_rows)
    finally:
        if loader._iterator is not None:
            loader._iterator._shutdown_workers()


def test_object_draws_without_replacement_and_rejects_partial_update(tmp_path):
    insertion = object_insertion(tmp_path,2)
    before = insertion.review_state.clone()
    with pytest.raises(ValueError,match='exactly'):
        insertion.apply_review({'a_stage1':10.},review_k=1)
    assert torch.equal(insertion.review_state,before)
    insertion.apply_review({'a_stage1':10.,'b_stage1':1.},review_k=1)
    row = ObjectReviewDataset(insertion,1)[0]
    assert sorted(d['crop_index'] for d in row['draws']) == [0,1]
    assert all(d['weight']==(10. if d['crop_index']==0 else 1.) for d in row['draws'])
