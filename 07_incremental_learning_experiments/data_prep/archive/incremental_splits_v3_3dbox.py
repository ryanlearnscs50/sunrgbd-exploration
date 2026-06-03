#!/usr/bin/env python
"""
incremental_splits_v3_3dbox.py
------------------------------
3D-BOX-DRIVEN replacement for incremental_splits_v2.py.

WHY V3 (mentor's direction, 2026-05-31):
  V2 built its presence matrix / lift / class->stage assignment / strict filter from
  2D annotations (annotation2Dfinal/index.json) -- i.e. what classes are *drawn* in a
  scene's 2D labels. But the incremental experiment trains a 3D detector on 3D boxes.
  These disagree massively. Verified known-answer check (this server, 2026-05-31):
    - 'door' in classroom scenes: 157 have a 2D door annotation, only 1 has a 3D door box.
    - 'window','picture','bag','light','counter': present in hundreds of 2D scenes, 0 3D boxes.
    - 'book'/'paper'/'bottle'/'cup'/'cabinet': <15% of 2D-present scenes have a 3D box.
  => V2's lift and stage assignment are driven by a signal that does not exist in the
     training labels, which is exactly why ~1/3 of the assigned classes were near-empty
     in 3D per stage. V3 recomputes EVERYTHING from 3D-box presence.

WHAT IS IDENTICAL TO V2 (faithful port):
  - lift = P(class | scene_type) / P(class)
  - balanced greedy assignment by decisiveness (best_lift - second_best_lift)
  - strict filter: (a) scene_type label == stage scene type, (b) no future-stage classes
  - ordering sweep: maximise the minimum surviving stage count (tiebreak total survived)
  - stage scene types: bedroom/office/classroom (3) ; +living_room/library/dining_area (6)
  - TOP_N = 24, CLASSES_PER_STAGE = 24 // n_stages

WHAT CHANGED:
  - presence[scene, class] = 1 iff the class has a 3D box in that scene (from the 40-class
    mmdet3d pkls), NOT a 2D annotation.
  - class vocabulary = TOP-24 by 3D-BOX scene-frequency (exclude wall/floor/ceiling, which
    have no 3D boxes anyway). This drops V2's 5 zero-box classes and adds genuinely common
    3D classes (garbage_bin, night_stand, sink, dresser, toilet, whiteboard, tv, ...).
  - EXACT 3D pkl names, NO synonym merging (the 3D vocab is already canonical/clean;
    bookshelf, sofa_chair, coffee_table stay distinct from shelf/sofa/table). Matches the
    prior verified carve decision.
  - scene universe = the 10,335 trainval scenes in SUNRGBDMeta3DBB_v2.mat (= the 40-class
    pkl coverage), labelled by their official scene.txt.

DATA SOURCES (all local to this server):
  - /data3/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat  (row i -> mmdet3d idx i+1, sequenceName)
  - /data3/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBD/<key>/scene.txt  (ground-truth scene type)
  - /data3/sunrgbd/sunrgbd_40/sunrgbd_infos_{train,val}_40class.pkl  (3D boxes per lidar_idx)

OUTPUTS (under /data3/ryan/incremental_learning/v3_3dbox/):
  scene_labels.csv                         scene idx/key -> scene_type (all 10,335)
  scene_type_class_lift.csv                3D lift for every (class, stage scene_type)
  class_stage_assignments_v3.csv           class -> stage, with per-stage lift
  incremental_splits_v3_3dbox_all_bestorder.csv   surviving scenes (both orderings, best order)
  split_stats_v3_{3,6}stage_bestorder.csv  per-stage survival
  split_analysis_report_v3_3dbox_bestorder.md
  stage_def_v3.json                        machine-readable per-stage class lists (best order)
  stage_manifest.json                      scene->stage manifest (same schema as build_stage_manifest.py)
"""
import json, pathlib, pickle, csv
from itertools import permutations as _permutations
from collections import Counter, defaultdict

import scipy.io as sio
import pandas as pd

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
MAT      = "/data3/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat"
RAW_ROOT = pathlib.Path("/data3/sunrgbd/OFFICIAL_SUNRGBD/SUNRGBD")
PKL = {
    "train": "/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_train_40class.pkl",
    "val":   "/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_val_40class.pkl",
}
OUT = pathlib.Path("/data3/ryan/incremental_learning/v3_3dbox")
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"wall", "floor", "ceiling"}
TOP_N   = 24
UNVIABLE_THRESHOLD = 50

STAGE_SCENE_TYPES_3 = ["bedroom", "office", "classroom"]
STAGE_SCENE_TYPES_6 = ["bedroom", "living_room", "office", "classroom", "library", "dining_area"]
CLASSES_PER_STAGE_3 = TOP_N // len(STAGE_SCENE_TYPES_3)   # 8
CLASSES_PER_STAGE_6 = TOP_N // len(STAGE_SCENE_TYPES_6)   # 4

# ---------------------------------------------------------------------------
# 1. scene index -> (key, scene_type)   and   index -> set of 3D-box class names
# ---------------------------------------------------------------------------
print("Loading SUNRGBDMeta3DBB_v2.mat ...")
arr = sio.loadmat(MAT, struct_as_record=False, squeeze_me=True)["SUNRGBDMeta"]
idx2key, idx2type = {}, {}
for i, e in enumerate(arr):
    key = str(e.sequenceName).split("SUNRGBD/", 1)[1].rstrip("/")
    idx = i + 1
    idx2key[idx] = key
    p = RAW_ROOT / key / "scene.txt"
    idx2type[idx] = p.read_text(errors="ignore").strip().lower() if p.exists() else "unknown"
print(f"  scenes: {len(idx2key)}  | scene types: {len(set(idx2type.values()))}")

idx2boxcls = defaultdict(set)   # EXACT names
for split, path in PKL.items():
    for it in pickle.load(open(path, "rb")):
        li = int(it["point_cloud"]["lidar_idx"])
        an = it.get("annos", {})
        for nm in (an.get("name", []) if an else []):
            nm = str(nm)
            if nm not in EXCLUDE:
                idx2boxcls[li].add(nm)

# ---------------------------------------------------------------------------
# 2. select TOP_N classes by 3D-box scene-frequency
# ---------------------------------------------------------------------------
freq = Counter()
for clss in idx2boxcls.values():
    for c in clss:
        freq[c] += 1
selected_classes = [c for c, _ in freq.most_common(TOP_N)]
selected_set = set(selected_classes)
print(f"\nTOP {TOP_N} classes by 3D-box scene-frequency:")
for i, c in enumerate(selected_classes, 1):
    print(f"  {i:2d}. {c:<14} {freq[c]:>5} scenes ({100*freq[c]/len(idx2key):.1f}%)")

# ---------------------------------------------------------------------------
# 3. presence matrix (rows = scenes, cols = selected classes) + scene_type
# ---------------------------------------------------------------------------
records = []
for idx in sorted(idx2key):
    box = idx2boxcls.get(idx, set())
    row = {"idx": idx, "scene": idx2key[idx], "scene_type": idx2type[idx]}
    for c in selected_classes:
        row[c] = 1 if c in box else 0
    records.append(row)
presence_df = pd.DataFrame(records)
presence_df[["idx", "scene", "scene_type"]].to_csv(OUT / "scene_labels.csv", index=False)

all_stage_types = sorted(set(STAGE_SCENE_TYPES_3 + STAGE_SCENE_TYPES_6))
print("\nStage scene-type counts:")
for s in all_stage_types:
    print(f"  {s:<14}{int((presence_df['scene_type']==s).sum())}")

# ---------------------------------------------------------------------------
# 4. lift matrix (3D)
# ---------------------------------------------------------------------------
overall_freq = presence_df[selected_classes].mean()
lift_records = []
for stype in all_stage_types:
    mask = presence_df["scene_type"] == stype
    n = int(mask.sum())
    if n == 0:
        continue
    freq_in_type = presence_df.loc[mask, selected_classes].mean()
    for cls in selected_classes:
        p_cls = float(overall_freq[cls]); p_giv = float(freq_in_type[cls])
        lift_records.append({"class": cls, "scene_type": stype, "n_scenes": n,
                             "freq_in_type": round(p_giv, 4), "overall_freq": round(p_cls, 4),
                             "lift": round(p_giv / p_cls if p_cls > 0 else 0.0, 4)})
lift_df = pd.DataFrame(lift_records)
lift_df.to_csv(OUT / "scene_type_class_lift.csv", index=False)

# ---------------------------------------------------------------------------
# 5. balanced greedy class -> stage assignment  (identical algorithm to v2)
# ---------------------------------------------------------------------------
def assign_classes_balanced(selected_classes, lift_df, stage_scene_types, classes_per_stage):
    lookup = {(r["class"], r["scene_type"]): r["lift"]
              for _, r in lift_df[lift_df["scene_type"].isin(stage_scene_types)].iterrows()}
    capacity = {s: classes_per_stage for s in stage_scene_types}

    def decisiveness(cls):
        lifts = sorted([lookup.get((cls, s), 0.0) for s in stage_scene_types], reverse=True)
        return (lifts[0] - lifts[1]) if len(lifts) >= 2 else (lifts[0] if lifts else 0.0)

    ordered = sorted(selected_classes, key=decisiveness, reverse=True)
    class_stage, stage_classes = {}, {s: [] for s in stage_scene_types}
    for cls in ordered:
        for s in sorted(stage_scene_types, key=lambda s: -lookup.get((cls, s), 0.0)):
            if capacity[s] > 0:
                class_stage[cls] = s; stage_classes[s].append(cls); capacity[s] -= 1; break
    return stage_classes, class_stage

stage_classes_3, class_stage_3 = assign_classes_balanced(selected_classes, lift_df, STAGE_SCENE_TYPES_3, CLASSES_PER_STAGE_3)
stage_classes_6, class_stage_6 = assign_classes_balanced(selected_classes, lift_df, STAGE_SCENE_TYPES_6, CLASSES_PER_STAGE_6)

# assignment CSV
assign_rows = []
for cls in selected_classes:
    r = {"class": cls, "freq_3d_scenes": freq[cls],
         "assigned_3stage": class_stage_3.get(cls, ""), "assigned_6stage": class_stage_6.get(cls, "")}
    for stype in all_stage_types:
        sub = lift_df[(lift_df["class"] == cls) & (lift_df["scene_type"] == stype)]
        r[f"lift_{stype}"] = round(float(sub["lift"].values[0]), 4) if len(sub) else 0.0
    assign_rows.append(r)
pd.DataFrame(assign_rows).to_csv(OUT / "class_stage_assignments_v3.csv", index=False)

# ---------------------------------------------------------------------------
# 6. strict filter (3D presence)  +  ordering sweep  (identical to v2)
# ---------------------------------------------------------------------------
def build_split_def(stage_scene_types, stage_classes_dict):
    return [(f"Stage {i} - {s.replace('_',' ').title()}", s, stage_classes_dict[s])
            for i, s in enumerate(stage_scene_types, 1)]

def apply_strict_filter(presence_df, split_def):
    n_stages = len(split_def)
    stage_class_lists = [cl for _, _, cl in split_def]
    stats, csv_rows = [], []
    for i, (label, stype, stage_cls) in enumerate(split_def):
        has_stage = presence_df["scene_type"] == stype
        future = set()
        for j in range(i + 1, n_stages):
            future |= set(stage_class_lists[j])
        no_future = (presence_df[sorted(future)].sum(axis=1) == 0) if future else pd.Series(True, index=presence_df.index)
        total = int(has_stage.sum()); mask = has_stage & no_future; survived = int(mask.sum())
        stats.append({"stage": label, "scene_type": stype, "classes": ", ".join(stage_cls),
                      "future_cls": ", ".join(sorted(future)) if future else "(none)",
                      "total": total, "survived": survived,
                      "pct_lost": round(100.0 * (1 - survived / total), 1) if total else float("nan")})
        for _, row in presence_df[mask].iterrows():
            csv_rows.append({"stage": label, "scene_type": stype, "idx": int(row["idx"]),
                             "scene": row["scene"],
                             "classes_present": "|".join(c for c in selected_classes if row[c] == 1)})
    return stats, csv_rows

def sweep_orderings(stage_scene_types, stage_classes_dict):
    rows = []
    for perm in _permutations(stage_scene_types):
        stats, csvr = apply_strict_filter(presence_df, build_split_def(list(perm), stage_classes_dict))
        rows.append({"ordering": list(perm), "min_survived": min(s["survived"] for s in stats),
                     "total_survived": sum(s["survived"] for s in stats), "stats": stats, "csv_rows": csvr})
    rows.sort(key=lambda r: (-r["min_survived"], -r["total_survived"]))
    return rows

best_3 = sweep_orderings(STAGE_SCENE_TYPES_3, stage_classes_3)[0]
best_6 = sweep_orderings(STAGE_SCENE_TYPES_6, stage_classes_6)[0]

# default-ordering stats (for report comparison)
stats_3, _ = apply_strict_filter(presence_df, build_split_def(STAGE_SCENE_TYPES_3, stage_classes_3))
stats_6, _ = apply_strict_filter(presence_df, build_split_def(STAGE_SCENE_TYPES_6, stage_classes_6))

# ---------------------------------------------------------------------------
# 7. write split CSVs + stats
# ---------------------------------------------------------------------------
cols = ["stage", "scene_type", "idx", "scene", "classes_present"]
df3 = pd.DataFrame(best_3["csv_rows"], columns=cols).assign(split="SceneType-3stage-bestorder")
df6 = pd.DataFrame(best_6["csv_rows"], columns=cols).assign(split="SceneType-6stage-bestorder")
pd.concat([df3, df6], ignore_index=True).to_csv(OUT / "incremental_splits_v3_3dbox_all_bestorder.csv", index=False)
pd.DataFrame(best_3["stats"]).to_csv(OUT / "split_stats_v3_3stage_bestorder.csv", index=False)
pd.DataFrame(best_6["stats"]).to_csv(OUT / "split_stats_v3_6stage_bestorder.csv", index=False)

# ---------------------------------------------------------------------------
# 8. machine-readable stage_def + scene->stage manifest (for the downstream pipeline)
# ---------------------------------------------------------------------------
# global class order = introduction order across the BEST ordering's stages
def stage_def_for(best, stage_classes_dict):
    out = []
    for i, stype in enumerate(best["ordering"], 1):
        out.append({"stage": f"Stage {i} - {stype.replace('_',' ').title()}",
                    "scene_type": stype, "novel_classes": stage_classes_dict[stype]})
    return out

stage_def = {"3stage": stage_def_for(best_3, stage_classes_3),
             "6stage": stage_def_for(best_6, stage_classes_6)}
json.dump(stage_def, open(OUT / "stage_def_v3.json", "w"), indent=1)

# scene->stage manifest, same schema as build_stage_manifest.py output
train_idx = {int(e["point_cloud"]["lidar_idx"]) for e in pickle.load(open(PKL["train"], "rb"))}
val_idx   = {int(e["point_cloud"]["lidar_idx"]) for e in pickle.load(open(PKL["val"], "rb"))}
def membership(idx):
    return "train" if idx in train_idx else ("val" if idx in val_idx else "none")

manifest = {}
for which, best in [("SceneType-3stage-bestorder", best_3), ("SceneType-6stage-bestorder", best_6)]:
    recs = []
    for r in best["csv_rows"]:
        idx = r["idx"]
        classes = [c for c in r["classes_present"].split("|") if c]
        recs.append({"scene_key": r["scene"], "mmdet3d_idx": idx,
                     "pts_path": "points/%06d.bin" % idx, "stage": r["stage"],
                     "scene_type": r["scene_type"], "split_membership": membership(idx),
                     "classes_present": classes, "box_classes_present": classes})
    manifest[which] = recs
json.dump(manifest, open(OUT / "stage_manifest.json", "w"), indent=1)

# ---------------------------------------------------------------------------
# 9. console summary
# ---------------------------------------------------------------------------
def show(best, stage_classes_dict, name):
    print(f"\n{'='*72}\n{name}  best ordering: {' -> '.join(best['ordering'])}\n{'='*72}")
    for s in best["stats"]:
        flag = "  *** UNVIABLE ***" if s["survived"] < UNVIABLE_THRESHOLD else ""
        print(f"  {s['stage']:<26} type={s['scene_type']:<12} survived={s['survived']:>5}/{s['total']:<5} "
              f"lost={s['pct_lost']:.1f}%{flag}")
        print(f"      classes: {s['classes']}")
    print(f"  min_survived={best['min_survived']}  total_survived={best['total_survived']}")

show(best_3, stage_classes_3, "3-STAGE (bedroom/office/classroom)")
show(best_6, stage_classes_6, "6-STAGE (bedroom/living_room/office/classroom/library/dining_area)")
print("\nWrote outputs to", OUT)
