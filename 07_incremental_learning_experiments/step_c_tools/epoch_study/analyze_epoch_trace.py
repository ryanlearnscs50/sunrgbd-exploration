#!/usr/bin/env python
"""Epoch-count trace analysis (2026-06-10 task).

For one fine-tuning stage that was re-trained keeping all 12 epoch checkpoints
(via run_epoch_trace.sh), read the cumulative-val eval at every epoch and report
old-class vs new-class mAP as a function of epoch. The hypothesis under test:
old-class mAP peaks early and then decays as the model over-fits the novel-only
data, so an early-stop epoch < 12 (the TR3D paper schedule) retains old classes
better at little new-class cost.

Usage:
  python tools_incremental/analyze_epoch_trace.py --sched 3stage --stage 2 [--metric 0.25]
"""
import argparse
import os

from analyze_incremental import parse_eval_log, stage_class_names  # reuse validated parser

CFG_DIR = 'configs/tr3d_incremental_v4'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sched', required=True, choices=['3stage', '6stage'])
    p.add_argument('--stage', type=int, required=True)
    p.add_argument('--metric', default='0.25', choices=['0.25', '0.50'])
    p.add_argument('--root', default='work_dirs/incremental_v4_epochtrace')
    args = p.parse_args()
    mi = 0 if args.metric == '0.25' else 1

    new = stage_class_names(CFG_DIR, args.sched, args.stage)             # introduced this stage
    cum = stage_class_names(CFG_DIR, args.sched, args.stage)            # cumulative after this stage
    # old = cumulative-after-this-stage minus new == cumulative through stage-1
    prev = [] if args.stage == 1 else stage_class_names(CFG_DIR, args.sched, args.stage - 1)
    old = prev
    new_only = [c for c in cum if c not in set(prev)]

    wd = f'{args.root}/{args.sched}/stage{args.stage}'
    print(f'\n===== epoch trace  {args.sched} stage{args.stage}  (AP@{args.metric}) =====')
    print(f'old classes ({len(old)}): {old}')
    print(f'new classes ({len(new_only)}): {new_only}')
    print(f'\n{"epoch":>5} | {"overall":>8} | {"old_mAP":>8} | {"new_mAP":>8} | {"chair":>7}')
    print('-' * 52)

    rows = []
    for e in range(1, 13):
        ap = parse_eval_log(f'{wd}/eval_epoch_{e}.log')
        if not ap:
            continue
        def m(names):
            vals = [ap[c][mi] for c in names if c in ap]
            return sum(vals) / len(vals) if vals else float('nan')
        overall = m(cum)
        old_m = m(old) if old else float('nan')
        new_m = m(new_only)
        chair = ap.get('chair', (float('nan'),))[mi] if 'chair' in ap else float('nan')
        rows.append((e, overall, old_m, new_m, chair))
        print(f'{e:>5} | {overall:>8.4f} | {old_m:>8.4f} | {new_m:>8.4f} | {chair:>7.3f}')

    if rows and old:
        best_old = max(rows, key=lambda r: (r[2] if r[2] == r[2] else -1))
        last = rows[-1]
        print('-' * 52)
        print(f'best old_mAP @ epoch {best_old[0]}: old={best_old[2]:.4f} new={best_old[3]:.4f} overall={best_old[1]:.4f}')
        print(f'final (ep{last[0]})          : old={last[2]:.4f} new={last[3]:.4f} overall={last[1]:.4f}')
        if best_old[2] == best_old[2] and last[2] == last[2]:
            print(f'old-class gain from stopping at ep{best_old[0]} vs ep12: '
                  f'{best_old[2] - last[2]:+.4f}  (new-class cost {best_old[3] - last[3]:+.4f})')


if __name__ == '__main__':
    main()
