"""Transfer LDMR-selected scene seats without importing its training runtime."""
from copy import deepcopy


def replay_records(bank, host_records, protocol, stage):
    """Join selected LDMR seats to already converted UniDet3D scene records.

    Run this bridge in the LDMR environment, then serialize the returned records
    for the host. host_records is keyed by original scene ID; each record must
    declare ldmr_protocol_id and store bbox_label_3d in LDMR frequency order.
    The existing bank remains responsible for budget/selection/persistence.
    Missing records or protocol mismatches fail instead of dropping replay.
    """
    protocol.seen_count(stage)
    if stage == 1:
        return []
    old_classes = list(range(protocol.seen_count(stage - 1)))
    seats, _ = bank.get_replay_scenes(old_classes, stage)
    output = []
    for seat in seats:
        scene_id = str(seat['original_scene_id'])
        if scene_id not in host_records:
            raise ValueError(f'missing converted host scene: {scene_id}')
        record = deepcopy(host_records[scene_id])
        if record.get('ldmr_protocol_id') != protocol.identity:
            raise ValueError('host scene class/stage protocol mismatch')
        saved_stage = seat['replay_from_stage']
        visible = set(protocol.visible_classes(stage, saved_stage))
        if 'instances' not in record:
            raise ValueError('host scene must contain converted detection instances')
        for instance in record['instances']:
            label = instance['bbox_label_3d']
            if type(label) is not int or not 0 <= label < len(protocol.names):
                raise ValueError('host detection labels must be LDMR class indices')
        record['instances'] = [i for i in record['instances']
                               if i['bbox_label_3d'] in visible]
        record.update(is_replay=True, ldmr_replay_from_stage=saved_stage,
                      ldmr_replay_unique_id=seat['replay_unique_id'])
        output.append(record)
    return output


def merge_replay_records(natural, replay, strategy='merge_labels'):
    """Match LDMR's default unique-scene merge, or explicit keep_both control.

    A merged record exposes current classes plus its saved snapshot classes;
    it never restores intermediate or future classes automatically.
    """
    if strategy == 'keep_both':
        return deepcopy(natural) + deepcopy(replay)
    if strategy != 'merge_labels':
        raise ValueError('unsupported scene dedup strategy')
    by_id = {r['sample_idx']: r for r in natural}
    if len(by_id) != len(natural):
        raise ValueError('duplicate natural scene IDs')
    seen = set()
    merged, memory_only = [], []
    for record in replay:
        sid = record['sample_idx']
        if sid in seen:
            raise ValueError('merge_labels requires the default unique-scene bank policy')
        seen.add(sid)
        if sid not in by_id:
            memory_only.append(deepcopy(record))
            continue
        current = by_id[sid]
        for key in ('lidar_points', 'pts_instance_mask_path', 'pts_semantic_mask_path',
                    'super_pts_path', 'axis_align_matrix', 'ldmr_protocol_id'):
            if record[key] != current[key]:
                raise ValueError('natural/replay scene source or alignment mismatch')
        result = deepcopy(record)
        result['instances'] = deepcopy(current['instances']) + deepcopy(record['instances'])
        result['ldmr_include_current'] = True
        merged.append(result)
    return [deepcopy(r) for r in natural if r['sample_idx'] not in seen] + merged + memory_only
