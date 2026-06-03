"""
incremental_splits_v4_3dbox.py
------------------------------
Final 3D-box-driven rebuild of the scene-type incremental split.

Faithful to ryan's original `incremental_splits_v2.py` SKELETON:
    lift  ->  class-to-stage assignment  ->  strict filter  ->  ordering sweep
The ordering objective is UNCHANGED: maximise the MINIMUM surviving scene
count per stage (so no stage collapses).

WHAT CHANGED vs v2 (the fixes decided with the mentor):
  (1) SIGNAL = 3D bounding boxes, not 2D `annotation2Dfinal` presence.
      v2's lift/strict-filter were driven by what is DRAWN in a scene (2D),
      which wildly overstates what has a trainable 3D box (e.g. classroom
      doors: 157 2D annotations, 1 actual 3D box). Everything here is keyed
      on the presence of a real 3D box from the 40-class pkl.
  (2) MERGES: cpu->computer, sofa_chair->sofa (near-duplicate names).
  (3) NO equal-capacity rule. Each class goes to its highest-3D-lift candidate
      scene type; a SUPPORT FLOOR (>=20 train scenes with a box of that class
      in its assigned stage) drops classes that can't be learned. Homeless
      classes (true home not among the candidate types) fall below the floor
      and are dropped -- "let the data decide".

Two INDEPENDENT schedules (mentor's call -- different vocabularies):
  3-stage : bedroom / office / classroom                          (ryan's original types)
  6-stage : office / bedroom / living_room / bathroom / classroom / kitchen  (data-driven)

Outputs (to v4_3dbox/, v2 and v3 left untouched):
  scene_type_class_lift_<sched>.csv
  class_stage_assignments_<sched>.csv
  stage_def_<sched>.json          (ordered stages: type, classes, global ids)
  stage_manifest_<sched>.json     (per-stage train lidar_idx after strict filter; per-stage val lidar_idx)
  post_filter_box_counts_<sched>.csv
  split_report_v4.md

NOT done here (downstream, Step C prep -- carve_stage_pkls/build_val_pkls):
  carving the actual per-stage train/val .pkl files.
"""

import json
import pathlib
import collections
from itertools import permutations

import scipy.io as sio
import pickle

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------
ROOT = "/data3/sunrgbd/OFFICIAL_SUNRGBD"
MAT = f"{ROOT}/SUNRGBDMeta3DBB_v2.mat"
TRAIN_PKL = "/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_train_40class.pkl"
VAL_PKL = "/data3/sunrgbd/sunrgbd_40/sunrgbd_infos_val_40class.pkl"
OUT = pathlib.Path("/data3/ryan/incremental_learning/v4_3dbox")
OUT.mkdir(parents=True, exist_ok=True)

MERGE = {"cpu": "computer", "sofa_chair": "sofa"}
FLOOR = 20

STAGE_SETS = {
    "3stage": ["bedroom", "office", "classroom"],
    "6stage": ["office", "bedroom", "living_room", "bathroom", "classroom", "kitchen"],
}

norm = lambda n: MERGE.get(n, n)

# --------------------------------------------------------------------------
# 1. lidar_idx (1-based) -> scene_type, via the .mat sequenceName + scene.txt
# --------------------------------------------------------------------------
def load_idx2type():
    m = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
    arr = m["SUNRGBDMeta"]
    idx2type = {}
    miss = 0
    for i, e in enumerate(arr):
        p = pathlib.Path(ROOT) / str(e.sequenceName) / "scene.txt"
        if p.exists():
            idx2type[i + 1] = p.read_text(errors="ignore").strip()
        else:
            idx2type[i + 1] = "__missing__"
            miss += 1
    return idx2type, len(arr), miss


def scan(pkl_path, idx2type):
    """Return list of {idx, type, classes:set(merged names with a 3D box)}."""
    d = pickle.load(open(pkl_path, "rb"))
    out = []
    unmapped = 0
    for it in d:
        idx = int(it["point_cloud"]["lidar_idx"])
        st = idx2type.get(idx, "__unmapped__")
        if st == "__unmapped__":
            unmapped += 1
        a = it.get("annos")
        names = a.get("name") if a else None
        cls = {norm(str(n)) for n in names} if names is not None else set()
        out.append({"idx": idx, "type": st, "classes": cls})
    return out, len(d), unmapped


# --------------------------------------------------------------------------
# 2. lift + assignment (no capacity) + floor
# --------------------------------------------------------------------------
def assign(train, cands, floor):
    N = len(train)
    type_scene = collections.Counter(r["type"] for r in train)
    tc = collections.defaultdict(collections.Counter)   # type -> class -> n_scenes
    cls_total = collections.Counter()                    # class -> n_scenes (any type)
    allcls = set()
    for r in train:
        for c in r["classes"]:
            tc[r["type"]][c] += 1
            allcls.add(c)
        for c in set(r["classes"]):
            cls_total[c] += 1

    def lift(c, t):
        nt = type_scene[t]
        if nt == 0 or cls_total[c] == 0:
            return 0.0
        return (tc[t][c] / nt) / (cls_total[c] / N)

    stage_classes = {t: [] for t in cands}              # type -> [class]
    class_stage = {}
    dropped = []
    lift_rows = []
    for c in sorted(allcls):
        best = max(cands, key=lambda t: lift(c, t))
        n_in = tc[best][c]
        lift_rows.append({"class": c, **{f"lift_{t}": round(lift(c, t), 4) for t in cands},
                          **{f"nsc_{t}": tc[t][c] for t in cands},
                          "best": best, "n_best": n_in})
        if n_in >= floor:
            stage_classes[best].append(c)
            class_stage[c] = best
        else:
            dropped.append((c, best, n_in))
    # order classes within a type by descending support (deterministic)
    for t in cands:
        stage_classes[t].sort(key=lambda c: -tc[t][c])
    return stage_classes, class_stage, dropped, lift_rows, type_scene, tc


# --------------------------------------------------------------------------
# 3. strict filter + ordering sweep (max min surviving train scenes)
# --------------------------------------------------------------------------
def strict_filter(train, ordering, stage_classes):
    """ordering: list of types. Returns per-stage list of surviving train idx."""
    n = len(ordering)
    survivors = []
    for i, t in enumerate(ordering):
        future = set()
        for j in range(i + 1, n):
            future |= set(stage_classes[ordering[j]])
        keep = [r["idx"] for r in train
                if r["type"] == t and not (r["classes"] & future)]
        survivors.append(keep)
    return survivors


def sweep(train, cands, stage_classes, train_by_idx):
    """Ordering objective (ryan's true intent): MAXIMIN over INDIVIDUAL CLASSES of the
    per-class post-strict-filter scene count -- i.e. maximize the smallest class's scene
    count, then the next-smallest, ... (lexicographic). NOT the per-stage total min.
    Final tiebreak: total surviving scenes."""
    best = None
    for perm in permutations(cands):
        surv = strict_filter(train, list(perm), stage_classes)
        per_class = []
        for t, s in zip(perm, surv):
            ss = set(s)
            for c in stage_classes[t]:
                per_class.append(sum(1 for idx in ss if c in train_by_idx.get(idx, set())))
        counts = [len(s) for s in surv]                       # per-stage totals (reporting only)
        key = (tuple(sorted(per_class)), sum(counts))         # lexicographic maximin over classes
        if best is None or key > best["key"]:
            best = {"ordering": list(perm), "survivors": surv, "counts": counts,
                    "per_class_min": (min(per_class) if per_class else 0), "key": key}
    return best


# --------------------------------------------------------------------------
# 4. build everything for one schedule
# --------------------------------------------------------------------------
def build(sched, cands, train, val):
    stage_classes, class_stage, dropped, lift_rows, type_scene, tc = assign(train, cands, FLOOR)
    train_by_idx0 = {r["idx"]: r["classes"] for r in train}

    # iterative POST-filter floor: drop the single worst sub-floor class, re-sweep,
    # repeat. Dropping a class also removes it as a "future" constraint, which can
    # lift other classes back above the floor -> drops the minimal set.
    post_dropped = []
    while True:
        best = sweep(train, cands, stage_classes, train_by_idx0)
        post_counts = {}
        for t, surv in zip(best["ordering"], best["survivors"]):
            ss = set(surv)
            for c in stage_classes[t]:
                post_counts[c] = sum(1 for idx in ss if c in train_by_idx0.get(idx, set()))
        below = [c for c, nb in post_counts.items() if nb < FLOOR]
        if not below:
            break
        worst = min(below, key=lambda c: post_counts[c])
        stage_classes[class_stage[worst]].remove(worst)
        post_dropped.append((worst, class_stage[worst], post_counts[worst]))
        del class_stage[worst]
    dropped = dropped + post_dropped
    ordering = best["ordering"]

    # global class ids in introduction order (stage order, then within-stage by support)
    gid = {}
    stage_def = []
    for si, t in enumerate(ordering, 1):
        classes = stage_classes[t]
        ids = []
        for c in classes:
            gid[c] = len(gid)
            ids.append(gid[c])
        stage_def.append({"stage": si, "scene_type": t,
                          "classes": classes, "global_ids": ids})

    # train manifest (post strict-filter survivors, best ordering)
    train_manifest = {str(si): surv for si, surv in
                      zip([s["stage"] for s in stage_def], best["survivors"])}

    # val: per-stage = val scenes whose scene_type == that stage's type
    val_by_type = collections.defaultdict(list)
    for r in val:
        val_by_type[r["type"]].append(r["idx"])
    val_manifest = {str(s["stage"]): val_by_type.get(s["scene_type"], [])
                    for s in stage_def}

    # post-filter per-class box counts (verification): count survivors with a box
    train_by_idx = {r["idx"]: r["classes"] for r in train}
    post = []
    for s, surv in zip(stage_def, best["survivors"]):
        survset = surv
        for c in s["classes"]:
            nb = sum(1 for idx in survset if c in train_by_idx.get(idx, set()))
            post.append({"stage": s["stage"], "scene_type": s["scene_type"],
                         "class": c, "global_id": gid[c],
                         "assign_nsc": tc[s["scene_type"]][c],
                         "postfilter_nsc": nb,
                         "below_floor": nb < FLOOR})

    # ---- write outputs ----
    with open(OUT / f"stage_def_{sched}.json", "w") as f:
        json.dump({"schedule": sched, "ordering": ordering,
                   "n_classes": len(gid), "floor": FLOOR,
                   "merges": MERGE, "stages": stage_def,
                   "min_surviving": best["counts"], "dropped": dropped}, f, indent=2)
    with open(OUT / f"stage_manifest_{sched}.json", "w") as f:
        json.dump({"schedule": sched, "ordering": ordering,
                   "train": train_manifest, "val": val_manifest}, f, indent=2)

    import csv
    with open(OUT / f"scene_type_class_lift_{sched}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lift_rows[0].keys()))
        w.writeheader(); w.writerows(lift_rows)
    with open(OUT / f"post_filter_box_counts_{sched}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(post[0].keys()))
        w.writeheader(); w.writerows(post)
    with open(OUT / f"class_stage_assignments_{sched}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["class", "assigned_stage_type", "global_id"])
        for c in sorted(class_stage, key=lambda c: gid[c]):
            w.writerow([c, class_stage[c], gid[c]])

    return {"sched": sched, "ordering": ordering, "stage_def": stage_def,
            "counts": best["counts"], "per_class_min": best["per_class_min"], "n_classes": len(gid),
            "dropped": dropped, "post": post,
            "train_manifest": {k: len(v) for k, v in train_manifest.items()},
            "val_manifest": {k: len(v) for k, v in val_manifest.items()}}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
idx2type, n_mat, miss = load_idx2type()
print(f"ANCHOR mat entries={n_mat} (expect 10335), scene.txt missing={miss} (expect 0)")
train, n_tr, un_tr = scan(TRAIN_PKL, idx2type)
val, n_va, un_va = scan(VAL_PKL, idx2type)
print(f"ANCHOR train={n_tr} (expect 5285) unmapped={un_tr}; val={n_va} unmapped={un_va}")

results = {}
for sched, cands in STAGE_SETS.items():
    results[sched] = build(sched, cands, train, val)

# concise summary + markdown report
lines = ["# v4 3D-box scene-type incremental split — summary\n",
         f"Floor: >= {FLOOR} train scenes-with-box in assigned stage. "
         f"Merges: {MERGE}. Signal: 3D boxes.\n"]
for sched, r in results.items():
    print(f"\n================ {sched} : order = {' -> '.join(r['ordering'])} ================")
    print(f"  vocab classes: {r['n_classes']}   MIN PER-CLASS scenes (maximin objective): "
          f"{r['per_class_min']}   (per-stage totals: {r['counts']})")
    lines.append(f"\n## {sched} — order: {' -> '.join(r['ordering'])} "
                 f"({r['n_classes']} classes; min per-class scenes = {r['per_class_min']})\n")
    lines.append("| Stage | Scene type | Train (post-filter) | Val | Classes (post-filter box scenes) |")
    lines.append("|---|---|---|---|---|")
    post_by_stage = collections.defaultdict(list)
    for p in r["post"]:
        post_by_stage[p["stage"]].append(p)
    for s in r["stage_def"]:
        st = s["stage"]
        tr_n = r["train_manifest"][str(st)]
        va_n = r["val_manifest"][str(st)]
        cls_str = ", ".join(f"{p['class']}({p['postfilter_nsc']})"
                            + ("⚠" if p["below_floor"] else "")
                            for p in sorted(post_by_stage[st], key=lambda x: -x["postfilter_nsc"]))
        print(f"  stage {st} [{s['scene_type']}] train={tr_n} val={va_n}: {cls_str}")
        lines.append(f"| {st} | {s['scene_type']} | {tr_n} | {va_n} | {cls_str} |")
    below = [p for p in r["post"] if p["below_floor"]]
    if below:
        print(f"  ⚠ classes dipping BELOW floor after strict filter: "
              + ", ".join(f"{p['class']}({p['postfilter_nsc']})" for p in below))
        lines.append(f"\n**⚠ below floor after strict filter:** "
                     + ", ".join(f"{p['class']}({p['postfilter_nsc']})" for p in below))
    print(f"  dropped (<floor at assignment): "
          + ", ".join(f"{c}:{n}" for c, _, n in r["dropped"]))
    lines.append(f"\n_Dropped at assignment (<{FLOOR}):_ "
                 + ", ".join(f"{c}({n})" for c, _, n in r["dropped"]))

(OUT / "split_report_v4.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\nWrote outputs to {OUT}")
