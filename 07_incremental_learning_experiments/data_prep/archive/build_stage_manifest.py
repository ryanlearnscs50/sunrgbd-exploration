#!/usr/bin/env python
"""Chunk 1 of Step B: build a scene->stage manifest.

For each ordering (3-stage, 6-stage) and each scene in the split CSV, resolve:
  CSV Windows path  ->  normalized key  ->  mmdet3d index (via SUNRGBDMeta3DBB_v2.mat)
                    ->  train/val membership (via canonical pkls)
Emit per-ordering manifests + a coverage report. NO pkl carving yet (that's chunk 2/3).
"""
import os, csv, json, pickle
import scipy.io as sio

ROOT = '/data3/ryan/incremental_learning'
INC  = '/data3/ryan/tr3d/data/sunrgbd_incremental'
MAT  = '/data3/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat'
OUT  = os.path.join(INC, 'stage_manifest')
os.makedirs(OUT, exist_ok=True)

# 5 classes that have NO 3D boxes in sunrgbd_40 -> dropped from detection stage class-lists
NO_BOX = {'window', 'picture', 'bag', 'light', 'counter'}

def mat_key(seq):
    # 'SUNRGBD/kv2/kinect2data/xxx' -> 'kv2/kinect2data/xxx'
    return seq.split('SUNRGBD/', 1)[1].rstrip('/')

def csv_key(winpath):
    # 'C:\\...\\SUNRGBD\\SUNRGBD\\kv2\\align_kv2\\xxx' -> 'kv2/align_kv2/xxx'
    p = winpath.replace('\\', '/')
    marker = 'SUNRGBD/SUNRGBD/'
    assert marker in p, p
    return p.split(marker, 1)[1].rstrip('/')

# --- 1. sequence key -> mmdet3d index (row i, 0-based -> i+1) ---
S = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)['SUNRGBDMeta']
key2idx = {}
for i in range(len(S)):
    key2idx[mat_key(str(S[i].sequenceName))] = i + 1
assert len(key2idx) == len(S) == 10335, (len(key2idx), len(S))

# --- 2. mmdet3d index -> train/val membership ---
def idxset(pkl):
    d = pickle.load(open(os.path.join(INC, pkl), 'rb'))
    return {int(e['point_cloud']['lidar_idx']): e for e in d}
train = idxset('sunrgbd_infos_train.pkl')
val   = idxset('sunrgbd_infos_val.pkl')
print('train scenes', len(train), 'val scenes', len(val))

def membership(idx):
    if idx in train: return 'train'
    if idx in val:   return 'val'
    return 'none'

# --- 3. read split CSV, resolve every row ---
rows = list(csv.DictReader(open(os.path.join(ROOT, 'incremental_splits_v2_all_bestorder.csv'))))
SPLIT_3 = 'SceneType-3stage-bestorder'
SPLIT_6 = 'SceneType-6stage-bestorder'

manifest = {SPLIT_3: [], SPLIT_6: []}
unmapped = []
for r in rows:
    k = csv_key(r['scene'])
    idx = key2idx.get(k)
    if idx is None:
        unmapped.append(k); continue
    classes = [c for c in r['classes_present'].split('|') if c]
    box_classes = [c for c in classes if c not in NO_BOX]
    rec = {
        'scene_key': k,
        'mmdet3d_idx': idx,
        'pts_path': 'points/%06d.bin' % idx,
        'stage': r['stage'],
        'scene_type': r['scene_type'],
        'split_membership': membership(idx),
        'classes_present': classes,
        'box_classes_present': box_classes,
    }
    manifest[r['split']].append(rec)

# --- 4. coverage report ---
report = {'unmapped_count': len(unmapped), 'unmapped_examples': unmapped[:10],
          'train_scenes_total': len(train), 'val_scenes_total': len(val), 'orderings': {}}
for split, recs in manifest.items():
    by_stage = {}
    for rec in recs:
        s = rec['stage']
        d = by_stage.setdefault(s, {'train': 0, 'val': 0, 'none': 0, 'total': 0})
        d[rec['split_membership']] += 1
        d['total'] += 1
    report['orderings'][split] = {'rows': len(recs), 'by_stage': by_stage}

with open(os.path.join(OUT, 'stage_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=1)
with open(os.path.join(OUT, 'coverage_report.json'), 'w') as f:
    json.dump(report, f, indent=1)

# flat CSV for human inspection
with open(os.path.join(OUT, 'stage_manifest_flat.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['ordering', 'stage', 'scene_type', 'mmdet3d_idx', 'split_membership',
                'box_classes_present', 'classes_present'])
    for split, recs in manifest.items():
        for rec in recs:
            w.writerow([split, rec['stage'], rec['scene_type'], rec['mmdet3d_idx'],
                        rec['split_membership'], '|'.join(rec['box_classes_present']),
                        '|'.join(rec['classes_present'])])

print(json.dumps(report, indent=1))
