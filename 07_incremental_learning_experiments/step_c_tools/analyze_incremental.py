#!/usr/bin/env python
"""Step C — incremental eval analysis: old-vs-new mAP + forgetting.

Parses the per-stage `eval.log` files produced by run_naive_v4.sh (mmdet3d's
indoor_eval prints a per-class AP_0.25/AP_0.50 table) and assembles, for one
schedule, a class x stage AP matrix. From it we report what matters for a
class-incremental experiment:

  * per stage: overall mAP@0.25 / mAP@0.50, plus the mean over OLD classes
    (introduced in an earlier stage) and NEW classes (introduced this stage);
  * forgetting: for each old class, AP at the stage it was introduced minus AP
    at the final stage; reported per class and averaged.

"Old" / "new" are derived purely from the cumulative class_names of the configs,
so this works for any schedule (3stage / 6stage) and any variant whose stage
work_dirs follow work_dirs/<root>/<sched>/stage{t}/eval.log.

Usage:
  python tools_incremental/analyze_incremental.py \
      --root work_dirs/incremental_v4_naive --sched 3stage \
      --cfg-dir configs/tr3d_incremental_v4 [--metric 0.25]
"""
import argparse
import os
import re

from mmcv import Config

ROW_RE = re.compile(r'^\|\s*([A-Za-z0-9_ ]+?)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|')


def parse_eval_log(path):
    """Return {class_name: (AP25, AP50)} from one indoor_eval table (last table in file)."""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            m = ROW_RE.match(line.strip())
            if not m:
                continue
            name = m.group(1).strip()
            if name.lower() in ('classes', 'class'):
                continue
            ap25, ar25, ap50, ar50 = (float(m.group(i)) for i in range(2, 6))
            out[name] = (ap25, ap50)  # last occurrence wins (final table)
    return out


def stage_class_names(cfg_dir, sched, t):
    return list(Config.fromfile(f'{cfg_dir}/{sched}/stage{t}.py').class_names)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, help='work_dirs root (e.g. work_dirs/incremental_v4_naive)')
    p.add_argument('--sched', required=True, choices=['3stage', '6stage'])
    p.add_argument('--cfg-dir', default='configs/tr3d_incremental_v4')
    p.add_argument('--metric', default='0.25', choices=['0.25', '0.50'], help='AP threshold to summarise')
    args = parse_args = p.parse_args()
    mi = 0 if args.metric == '0.25' else 1

    nstage = 3 if args.sched == '3stage' else 6
    # introduction stage of each class (1-based)
    intro = {}
    cum = []
    for t in range(1, nstage + 1):
        for c in stage_class_names(args.cfg_dir, args.sched, t):
            if c not in intro:
                intro[c] = t
                cum.append(c)

    # class x stage AP matrix
    ap = {t: parse_eval_log(f'{args.root}/{args.sched}/stage{t}/eval.log') for t in range(1, nstage + 1)}
    have = {t: bool(ap[t]) for t in ap}

    print(f'\n===== {args.sched}  (AP@{args.metric}) =====')
    print(f'logs present: ' + ', '.join(f'stage{t}={"Y" if have[t] else "-"}' for t in range(1, nstage + 1)))

    # per-stage summary
    print(f'\n{"stage":6} {"#cls":>4} {"mAP":>7} {"old_mAP":>8} {"new_mAP":>8}')
    for t in range(1, nstage + 1):
        if not have[t]:
            continue
        names = stage_class_names(args.cfg_dir, args.sched, t)
        vals = [ap[t][c][mi] for c in names if c in ap[t]]
        new = [ap[t][c][mi] for c in names if c in ap[t] and intro[c] == t]
        old = [ap[t][c][mi] for c in names if c in ap[t] and intro[c] < t]
        mean = lambda xs: sum(xs) / len(xs) if xs else float('nan')
        print(f'stage{t:<1} {len(names):>4} {mean(vals):>7.4f} '
              f'{mean(old):>8.4f} {mean(new):>8.4f}')

    # forgetting: AP at introduction stage vs at final stage
    final_t = max(t for t in range(1, nstage + 1) if have[t]) if any(have.values()) else None
    if final_t:
        print(f'\nforgetting (AP@{args.metric} at intro stage - at final stage{final_t}):')
        drops = []
        for c in cum:
            it = intro[c]
            if it >= final_t or not have[it] or c not in ap[it] or c not in ap[final_t]:
                continue
            a0, a1 = ap[it][c][mi], ap[final_t][c][mi]
            drops.append(a0 - a1)
            print(f'  {c:14} intro_s{it} {a0:.4f} -> final {a1:.4f}   drop {a0 - a1:+.4f}')
        if drops:
            print(f'  mean forgetting over {len(drops)} old classes: {sum(drops) / len(drops):+.4f}')


if __name__ == '__main__':
    main()
