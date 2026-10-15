#!/usr/bin/env python3
"""Populate existing LDMR banks and export UniDet3D replay payloads.

Run with the LDMR Python, never the host Python: the original bank imports the
old framework. Input is trusted, converted full-protocol TRAIN metadata with
real point/mask/superpoint files. Every output goes into a new directory.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO/'adapters')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--stage-setting', default='scannet35_s5_freqorder')
    parser.add_argument('--memory-mode', choices=['scene', 'object'], required=True)
    parser.add_argument('--previous-bank', type=Path)
    parser.add_argument('--previous-crops', type=Path)
    parser.add_argument('--seed', type=int, default=200)
    parser.add_argument('--scene-budget', type=int, default=120)
    parser.add_argument('--objects-per-class', type=int, default=20)
    parser.add_argument('--max-objects', type=int, default=1000)
    parser.add_argument('--min-points', type=int, default=5)
    parser.add_argument('--object-selector', choices=['random', 'largest_point_count', 'tr3d_obj_replace'], default='random')
    parser.add_argument('--object-policy', choices=['ldmr_weighted', 'tr3d_obj'], default='ldmr_weighted')
    parser.add_argument('--ld-collection', type=Path, help='sealed complete F1 trajectory; enables ScanNet Design-2 selection')
    parser.add_argument('--reviewing', action='store_true', help='persist last online object weights into the next stage')
    args = parser.parse_args()
    if args.reviewing and (args.memory_mode != 'object' or not args.ld_collection):
        parser.error('--reviewing export requires object memory and a sealed LD collection')
    if args.ld_collection and args.stage_setting != 'scannet35_s5_freqorder':
        parser.error('--ld-collection requires the ScanNet S5 policy')
    if args.output_dir.exists():
        parser.error('output exists; refusing to overwrite bank artifacts')
    if (args.stage > 1) != bool(args.previous_bank):
        parser.error('later stages require a preceding bank; stage 1 cannot load one')
    needs_crops = args.memory_mode == 'object' and args.stage > 1
    if needs_crops != bool(args.previous_crops):
        parser.error('preceding rich crops are required only for later object stages')

    import numpy as np
    from ldmr_unidet3d import ScanNet35Protocol
    from ldmr_unidet3d.data import stage_records, verify_point_files
    from ldmr_unidet3d.scene import replay_records
    from ldmr_unidet3d.objects import extract_crop, selected_payloads, save_crops, load_crops
    protocol = ScanNet35Protocol(args.stage_setting)
    protocol.seen_count(args.stage)
    source_bytes = args.input.read_bytes()
    records = pickle.loads(source_bytes)['data_list']
    by_id = {record['sample_idx']:record for record in records}
    if len(by_id) != len(records):
        raise ValueError('full metadata has duplicate scene IDs')
    # Validate protocol/labels using the final stage's all-seen view, and files.
    stage_records(records, protocol, len(protocol.stages), evaluation=True)
    for record in records:
        if record.get('is_replay', False):
            raise ValueError('memory population requires original natural records')
        verify_point_files(record)
    context = dict(dataset='unidet3d_scannet35', protocol_id=protocol.identity,
                   source_sha256=hashlib.sha256(source_bytes).hexdigest())
    if args.previous_bank:
        previous_meta = json.loads((args.previous_bank.parent/'provenance.json').read_text())
        expected = dict(context, stage=args.stage-1, memory_mode=args.memory_mode)
        if any(previous_meta.get(k) != v for k,v in expected.items()):
            raise ValueError('preceding memory export provenance mismatch')
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.memory-', dir=args.output_dir.parent))
    try:
        if args.memory_mode == 'scene':
            from mmdet3d.datasets.scene_memory_bank import SceneMemoryBank
            config = dict(max_memory_scenes=args.scene_budget, total_training_scenes=len(records),
                random_seed=args.seed, selection_strategy='random', quota_strategy='stage_ratio',
                min_objects_per_scene=1, debug_mode=False,
                stage_scene_counts={s:len(stage_records(records, protocol, s)) for s in range(1,len(protocol.stages)+1)})
            if args.ld_collection:
                from ldmr_unidet3d.selection import reference_scene_policy, collection_payload
                config.update(reference_scene_policy())
            bank = SceneMemoryBank(**config)
            if args.previous_bank:
                bank = SceneMemoryBank.load_checkpoint(args.previous_bank,
                    expected_stage=args.stage-1, expected_context=context,
                    expected_config=bank._checkpoint_config)
            eligible = {r['sample_idx'] for r in stage_records(records, protocol, args.stage)}
            natural = []
            for record in records:
                if record['sample_idx'] not in eligible:
                    continue
                inst = record['instances']
                natural.append(dict(sample_idx=record['sample_idx'], annos=dict(
                    gt_num=len(inst), **{'class':np.array([protocol.raw_ids[i['bbox_label_3d']] for i in inst], dtype=np.int64)},
                    gt_boxes_upright_depth=np.asarray([i['bbox_3d'] for i in inst], dtype=np.float32))))
            mapping = dict(nyu40_to_model_idx=dict(zip(protocol.raw_ids, range(35))),
                           valid_nyu40_ids=list(protocol.raw_ids))
            payload = None
            if args.ld_collection:
                payload, evidence = collection_payload(args.ld_collection, protocol, args.stage, records, bank)
                (temporary/'selection.json').write_text(json.dumps(dict(evidence, policy=config), indent=2)+'\n')
            bank.add_stage_scenes(args.stage, natural, list(range(protocol.seen_count(args.stage))), mapping,
                learning_dynamics_design2_payload=payload)
            bank.save_checkpoint(temporary/'bank.pkl', args.stage, context)
            if args.stage < len(protocol.stages):
                replay = replay_records(bank, by_id, protocol, args.stage+1)
                with (temporary/'replay_records.pkl').open('wb') as f:
                    pickle.dump(replay, f, protocol=pickle.HIGHEST_PROTOCOL)
            count = len(bank.list_memory_entries())
        else:
            from mmdet3d.datasets.object_memory_bank import ObjectMemoryBank
            from mmdet3d.datasets.memory_bank.source_cache import SourcePointCache
            config = dict(exemplars_per_class=args.objects_per_class, max_total_exemplars=args.max_objects,
                          min_points=args.min_points, selection_strategy=args.object_selector, random_seed=args.seed)
            if args.object_policy == 'tr3d_obj':
                from ldmr_unidet3d.recipes import tr3d_obj_settings
                profile, _ = tr3d_obj_settings()
                config.update({key: profile[key] for key in
                               ('crop_margin', 'crop_tolerance', 'floor_percentile')})
            if args.ld_collection:
                from ldmr_unidet3d.selection import reference_object_policy, collection_payload
                config.update(reference_object_policy())
            bank = ObjectMemoryBank(**config)
            old_crops = []
            if args.previous_bank:
                bank = ObjectMemoryBank.load_state(args.previous_bank, expected_config=config)
                # Bank state is paired with the validated rich export. Geometry
                # join below rejects a mismatched pair before publishing output.
                old_crops = load_crops(args.previous_crops, protocol, args.stage-1)
                selected_payloads(bank, old_crops, protocol)
            payload = None
            if args.ld_collection:
                payload, evidence = collection_payload(args.ld_collection, protocol, args.stage,
                    records, bank, memory_mode='object', reviewing=args.reviewing)
                if evidence.get('reviewing_weights') is not None:
                    bank.apply_source_seat_replay_weights(evidence['reviewing_weights'], strict=True)
                (temporary/'selection.json').write_text(json.dumps(dict(evidence, policy=config), indent=2)+'\n')
            def load_points(scene_id):
                record = by_id[scene_id]
                pts = np.fromfile(record['lidar_points']['lidar_path'], np.float32).reshape(-1,6)
                align = np.asarray(record['axis_align_matrix'])
                pts[:,:3] = pts[:,:3] @ align[:3,:3].T + align[:3,3]
                return pts
            cache = SourcePointCache(load_points, bank.source_cache_max_bytes)
            for cid in protocol.visible_classes(args.stage):
                candidates = []
                for record in records:
                    for idx, inst in enumerate(record['instances']):
                        if inst['bbox_label_3d'] == cid:
                            candidates.append(dict(scene_id=record['sample_idx'], object_idx=idx,
                                bbox=inst['bbox_3d'], class_id=cid, nyu40_id=protocol.raw_ids[cid]))
                bank.add_exemplars(cid, candidates, scene_points_dict=cache, stage_id=args.stage,
                    learning_dynamics_design2_payload=payload)
            rich = list(old_crops)
            selected_by_scene = defaultdict(list)
            for obj in bank.get_all_exemplars():
                if obj['stage_id'] == args.stage:
                    selected_by_scene[obj['scene_id']].append(obj)
            for scene_id, selected in selected_by_scene.items():
                record = by_id[scene_id]
                pts = load_points(scene_id)
                sem = np.fromfile(record['pts_semantic_mask_path'], np.int64)
                ins = np.fromfile(record['pts_instance_mask_path'], np.int64)
                sp = np.fromfile(record['super_pts_path'], np.int64)
                bounds = {}
                for iid in np.unique(ins[ins >= 0]):
                    member = ins == iid
                    labels = np.unique(sem[member])
                    xyz = pts[member,:3]
                    if len(labels)==1:
                        bounds[int(iid)] = (int(labels[0]), np.r_[(xyz.min(0)+xyz.max(0))/2, xyz.max(0)-xyz.min(0)])
                for obj in selected:
                    matches = [iid for iid,(raw, box) in bounds.items()
                               if raw==protocol.raw_ids[obj['class_id']] and np.allclose(box, obj['bbox'][:6], atol=1e-4)]
                    if len(matches)!=1:
                        raise ValueError(f'cannot uniquely resolve source instance for {scene_id}:{obj["object_idx"]}')
                    crop = extract_crop(pts, sem, ins, sp, bbox=obj['bbox'], instance_id=matches[0],
                        class_id=obj['class_id'], scene_id=scene_id, object_idx=obj['object_idx'],
                        stage=args.stage, protocol=protocol, min_points=bank.min_points,
                        crop_margin=bank.crop_margin, floor_percentile=bank.floor_percentile,
                        crop_tolerance=bank.crop_tolerance)
                    if crop is None:
                        raise ValueError('selected crop has no usable point-instance membership')
                    rich.append(crop)
            retained = selected_payloads(bank, rich, protocol)
            bank.save_state(str(temporary/'bank.pkl'), stage_id=args.stage)
            save_crops(temporary/'crops.pkl', retained, protocol, args.stage)
            count = len(retained)
            context['population_io'] = cache.statistics()
        (temporary/'provenance.json').write_text(json.dumps(dict(context, stage=args.stage,
            memory_mode=args.memory_mode, retained_entries=count), indent=2)+'\n')
        os.rename(temporary, args.output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(args.output_dir)


if __name__ == '__main__':
    main()
