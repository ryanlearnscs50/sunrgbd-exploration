"""Point-aligned targets for UniDet3D's ScanNet bbox_by_mask training path."""
import numpy as np


def stage_targets(raw_semantic, instance_ids, superpoint_ids, protocol, stage,
                  replay_from_stage=None, inserted_mask=None, include_current=False):
    """Filter supervision, compact IDs and pool with the upstream >0.5 rule.

    Input semantic IDs MUST be raw NYU40 (before PointSegClassMapping).
    Instance IDs may be sparse nonnegative integers; -1 is ignored. Background
    raw classes are ignored regardless of their instance IDs. Points are never
    removed, including future/old objects absent from the active supervision.
    Natural scenes expose only current classes; replay exposes saved classes.
    """
    semantic = protocol.map_raw(raw_semantic, stage, replay_from_stage)
    if include_current:
        if replay_from_stage is None:
            raise ValueError('merged visibility requires a saved replay stage')
        current = protocol.map_raw(raw_semantic, stage)
        semantic[current >= 0] = current[current >= 0]
    instances = np.asarray(instance_ids)
    superpoints = np.asarray(superpoint_ids)
    for name, values in [('instance', instances), ('superpoint', superpoints)]:
        if (values.ndim != 1 or values.dtype.kind not in 'iu'
                or values.shape != semantic.shape):
            raise ValueError(f'{name} IDs must be point-aligned integer arrays')
    if np.any(instances < -1) or np.any(superpoints < 0):
        raise ValueError('invalid negative instance/superpoint IDs')
    pasted = np.zeros(semantic.shape, dtype=bool)
    if inserted_mask is not None:
        pasted = np.asarray(inserted_mask)
        if pasted.dtype != np.bool_ or pasted.shape != semantic.shape:
            raise ValueError('inserted mask must be a point-aligned boolean array')
        if np.any(pasted & (instances < 0)):
            raise ValueError('inserted target points require an instance ID')
        if pasted.any():
            if replay_from_stage is not None or stage == 1:
                raise ValueError('object replay requires a later-stage natural scene')
            seen = protocol.map_raw(raw_semantic, stage, evaluation=True)
            old_count = protocol.seen_count(stage - 1)
            if np.any((seen[pasted] < 0) | (seen[pasted] >= old_count)):
                raise ValueError('inserted targets must belong to previous classes')
            semantic[pasted] = seen[pasted]
    # An instance spanning two semantic classes is corrupt input, even if one
    # of those classes would be hidden at this stage.
    raw = np.asarray(raw_semantic)
    for instance in np.unique(instances[instances >= 0]):
        if np.unique(pasted[instances == instance]).size != 1:
            raise ValueError('natural and pasted points share an instance ID')
        if np.unique(raw[instances == instance]).size != 1:
            raise ValueError('one instance contains multiple raw semantic classes')
    _, sp = np.unique(superpoints, return_inverse=True)
    sp = sp.astype(np.int64)
    visible = (semantic >= 0) & (instances >= 0)
    kept_ids = np.unique(instances[visible])
    inst = np.full(semantic.shape, -1, dtype=np.int64)
    labels = np.empty(len(kept_ids), dtype=np.int64)
    masks = np.zeros((len(kept_ids), len(np.unique(sp))), dtype=bool)
    sizes = np.bincount(sp, minlength=masks.shape[1])
    for compact, original in enumerate(kept_ids):
        member = visible & (instances == original)
        inst[member] = compact
        labels[compact] = semantic[member][0]
        counts = np.bincount(sp[member], minlength=masks.shape[1])
        masks[compact] = counts > sizes * 0.5
    semantic = semantic.copy()
    semantic[inst < 0] = -1
    return dict(pts_semantic_mask=semantic, pts_instance_mask=inst,
                sp_pts_mask=sp, gt_labels_3d=labels, gt_sp_masks=masks)
