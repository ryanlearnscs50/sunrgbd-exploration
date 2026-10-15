"""Single-process online scene/object reviewing for the isolated modern host."""
import json
from pathlib import Path
import subprocess
import sys

import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import Hook
from mmdet3d.registry import DATA_SAMPLERS, HOOKS

from .protocol import ScanNet35Protocol
from .reviewing import review_boundaries, review_indices, review_weights
from .sequence import sha256, verify_hashes


@DATA_SAMPLERS.register_module()
class LDMRReviewSampler(DefaultSampler):
    """Keep original rows immutable; change only the sampled row indices."""
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        if self.world_size != 1:
            raise ValueError('online reviewing currently requires a single process')
        self.indices = list(range(len(dataset)))
        self.review_k = 0

    def __iter__(self):
        # Use DefaultSampler's local generator, preserving training RNG state.
        return iter([self.indices[i] for i in super().__iter__()])

    def apply_review(self, weights, *, stage, review_k, seed):
        records = [self.dataset.get_data_info(i) for i in range(len(self.dataset))]
        self.indices, debug = review_indices(records, weights, seed=seed,
                                              stage=stage, review_k=review_k)
        if debug['memory_candidate_count'] == 0:
            debug['memory_coverage_ratio'] = None  # JSON has no undefined ratio
        self.review_k = review_k
        return dict(indices=self.indices, **debug)


@HOOKS.register_module()
class LDMROnlineReviewHook(Hook):
    # Capture hook runs first. Collection is an isolated subprocess so it
    # cannot alter training RNG, model mode, BN buffers, optimizer or scheduler.
    priority = 'VERY_LOW'

    def __init__(self, stage, total_epochs, upstream_root, train_info, seed,
                 stage_setting='scannet35_s5_freqorder', memory_mode='scene',
                 object_source_info=None):
        self.stage = stage
        self.protocol = ScanNet35Protocol(stage_setting)
        self.boundaries = review_boundaries(total_epochs, stage_setting)
        self.upstream_root, self.train_info, self.seed = upstream_root, train_info, seed
        if memory_mode not in ('scene', 'object') or (memory_mode == 'object') != bool(object_source_info):
            raise ValueError('object reviewing requires full source metadata only in object mode')
        self.memory_mode, self.object_source_info = memory_mode, object_source_info
        self.insertion = None
        self.review_k = 0

    def before_train(self, runner):
        from mmengine.dist import get_world_size
        if (get_world_size() != 1 or runner.epoch or runner.iter
                or not isinstance(runner.train_dataloader.sampler,
                                  LDMRReviewSampler if self.memory_mode == 'scene' else DefaultSampler)
                or runner.train_dataloader.drop_last):
            raise ValueError('reviewing requires fresh single-process training and full sample coverage')
        self.reports = []
        if self.memory_mode == 'object':
            from .transforms import LDMRInsertObjects
            def leaves(dataset):
                if hasattr(dataset, 'datasets'):
                    return [leaf for child in dataset.datasets for leaf in leaves(child)]
                return [dataset]
            datasets = leaves(runner.train_dataloader.dataset)
            insertions = []
            for dataset in datasets:
                for transform in dataset.pipeline.transforms:
                    if isinstance(transform, LDMRInsertObjects):
                        insertions.append(transform)
                    if hasattr(transform, 'meta_keys'):
                        transform.meta_keys = tuple(dict.fromkeys((*transform.meta_keys,
                            'ldmr_scene_id', 'ldmr_object_review_k', 'ldmr_object_draws')))
            if len(datasets) != 1 or len(insertions) != int(self.stage > 1):
                raise ValueError('object reviewing requires one natural dataset and its insertion transform')
            if insertions:
                self.insertion = insertions[0]
                self.insertion.enable_reviewing()

    def after_train_epoch(self, runner):
        epoch = runner.epoch + 1
        if self.memory_mode == 'object':
            self.epoch_sampling['samples'] = self.epoch_samples
            if len(self.epoch_samples) != len(runner.train_dataloader.dataset):
                raise ValueError('object review receipt does not cover the complete training epoch')
            with (Path(runner.work_dir) / 'reviewing' / f'epoch_{epoch}_sampling.json').open('x') as stream:
                json.dump(self.epoch_sampling, stream, indent=2, allow_nan=False)
        if self.stage == 1 or epoch not in self.boundaries[1:-1]:
            return
        root = Path(runner.work_dir)
        review_k = self.boundaries.index(epoch)
        output = root / 'reviewing' / f'review_{review_k}'
        output.parent.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'tools/collect_ld.py'),
            '--config', str(root / 'resolved_stage.py'), '--train-info', self.train_info,
            '--upstream-root', self.upstream_root, '--output', str(output),
            '--seed', str(self.seed), '--q-metric', 'f1']
        if self.object_source_info:
            command += ['--object-source-info', self.object_source_info]
        for boundary in self.boundaries[:review_k + 1]:
            command += ['--checkpoint', f'{boundary}={root / "ld_checkpoints" / f"epoch_{boundary}.pth"}']
        if review_k > 1:
            command += ['--reuse-collection', str(output.parent / f'review_{review_k - 1}')]
        torch.cuda.empty_cache()
        with (output.parent / f'review_{review_k}.log').open('x') as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        complete = json.loads((output / 'complete.json').read_text())
        verify_hashes({str(output / name): digest for name, digest in complete['artifacts'].items()}, 'review collection')
        verify_hashes(complete['checkpoints'], 'review checkpoint')
        snapshots = [json.loads((output / f'snapshot_{b}.json').read_text())
                     for b in self.boundaries[:review_k + 1]]
        weights = review_weights(snapshots, self.protocol)
        if self.memory_mode == 'object':
            sampling = self.insertion.apply_review(weights['weights_by_uid'], review_k=review_k)
            self.review_k = review_k
        else:
            sampling = runner.train_dataloader.sampler.apply_review(weights['weights_by_uid'],
                stage=self.stage, review_k=review_k, seed=self.seed)
        report = dict(epoch=epoch, next_epoch=epoch + 1, review_k=review_k,
            collection_sha256=sha256(output / 'complete.json'), weights=weights, sampling=sampling)
        with (output.parent / f'review_{review_k}.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        self.reports.append(report)

    def before_train_epoch(self, runner):
        root = Path(runner.work_dir) / 'reviewing'
        root.mkdir(exist_ok=True)
        sampler = runner.train_dataloader.sampler
        if self.memory_mode == 'object':
            self.epoch_samples = []
            self.epoch_sampling = dict(epoch=runner.epoch + 1, review_k=self.review_k,
                indices=list(iter(sampler)), crop_weights=(self.insertion.review_state[1:].tolist()
                                                         if self.insertion is not None else []))
            return
        with (root / f'epoch_{runner.epoch + 1}_sampling.json').open('x') as stream:
            json.dump(dict(epoch=runner.epoch + 1, review_k=sampler.review_k,
                           indices=list(iter(sampler))), stream, indent=2)

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        if self.memory_mode != 'object':
            return
        for sample in data_batch['data_samples']:
            meta = sample.metainfo
            version = meta.get('ldmr_object_review_k', 0 if self.stage == 1 else None)
            if version != self.review_k:
                raise ValueError('training worker used stale object review weights')
            draws = meta.get('ldmr_object_draws', [])
            for draw in draws:
                if draw['weight'] != self.epoch_sampling['crop_weights'][draw['crop_index']]:
                    raise ValueError('training crop draw differs from reviewed weight')
            self.epoch_samples.append(dict(scene_id=meta['ldmr_scene_id'],
                review_k=version, draws=draws))
