#!/usr/bin/env python
"""Optimal-epoch study — final comparison across epoch budgets (2026-06-10 task).

For the 3stage schedule, reads the FINAL-stage (stage3) cumulative-val eval of
each full-chain run trained at epoch budget E in {2,4,6,8,12}, for both the naive
and pseudo-label variants, and reports old / new / overall mAP@0.25 and @0.50 as
a function of E. Picks the E that best retains OLD classes. E=12 reuses the
original runs (work_dirs/incremental_v4_{naive,pseudo}); E<12 use the *_ep{E}
sweep dirs.

Usage: python tools_incremental/analyze_epoch_sweep.py [--sched 3stage]
"""
import argparse

from analyze_incremental import parse_eval_log, stage_class_names

CFG_DIR = 'configs/tr3d_incremental_v4'
EPOCHS = [2, 4, 6, 8, 12, 16, 24, 36]


def root_for(variant, E):
    base = {'naive': 'work_dirs/incremental_v4_naive',
            'pseudo': 'work_dirs/incremental_v4_pseudo'}[variant]
    return base if E == 12 else f'{base}_ep{E}'


def summarize(sched, variant, E, old, new, cum):
    nstage = 3 if sched == '3stage' else 6
    path = f'{root_for(variant, E)}/{sched}/stage{nstage}/eval.log'
    ap = parse_eval_log(path)
    if not ap:
        return None

    def mean(names, mi):
        v = [ap[c][mi] for c in names if c in ap]
        return sum(v) / len(v) if v else float('nan')
    return {
        'old25': mean(old, 0), 'new25': mean(new, 0), 'all25': mean(cum, 0),
        'old50': mean(old, 1), 'new50': mean(new, 1), 'all50': mean(cum, 1),
        'chair': ap.get('chair', (float('nan'), float('nan')))[0],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sched', default='3stage', choices=['3stage', '6stage'])
    args = p.parse_args()
    nstage = 3 if args.sched == '3stage' else 6

    cum = stage_class_names(CFG_DIR, args.sched, nstage)
    prev = stage_class_names(CFG_DIR, args.sched, nstage - 1)
    old = prev
    new = [c for c in cum if c not in set(prev)]

    print(f'\n############ OPTIMAL-EPOCH STUDY — {args.sched} (final stage{nstage} cumulative val) ############')
    print(f'old classes (introduced before final stage): {len(old)} | new (final stage): {len(new)}')

    for variant in ('naive', 'pseudo'):
        print(f'\n===== {variant.upper()} =====')
        print(f'{"E":>3} | {"old@.25":>8} {"new@.25":>8} {"all@.25":>8} | '
              f'{"old@.50":>8} {"all@.50":>8} | {"chair@.25":>9}')
        print('-' * 70)
        rows = []
        for E in EPOCHS:
            s = summarize(args.sched, variant, E, old, new, cum)
            if s is None:
                print(f'{E:>3} | {"(no eval yet)":>8}')
                continue
            rows.append((E, s))
            print(f'{E:>3} | {s["old25"]:>8.4f} {s["new25"]:>8.4f} {s["all25"]:>8.4f} | '
                  f'{s["old50"]:>8.4f} {s["all50"]:>8.4f} | {s["chair"]:>9.3f}')
        if rows:
            best_old = max(rows, key=lambda r: r[1]['old25'] if r[1]['old25'] == r[1]['old25'] else -1)
            best_all = max(rows, key=lambda r: r[1]['all25'] if r[1]['all25'] == r[1]['all25'] else -1)
            e12 = dict(rows).get(12)
            print('-' * 70)
            print(f'best OLD-class @.25 : E={best_old[0]}  old={best_old[1]["old25"]:.4f}  '
                  f'new={best_old[1]["new25"]:.4f}  all={best_old[1]["all25"]:.4f}')
            print(f'best OVERALL  @.25 : E={best_all[0]}  all={best_all[1]["all25"]:.4f}  '
                  f'old={best_all[1]["old25"]:.4f}')
            if e12 and best_old[0] != 12:
                d_old = best_old[1]['old25'] - e12['old25']
                d_new = best_old[1]['new25'] - e12['new25']
                print(f'vs paper E=12      : stopping at E={best_old[0]} gives old {d_old:+.4f}, '
                      f'new {d_new:+.4f} (old@.25 {e12["old25"]:.4f} -> {best_old[1]["old25"]:.4f})')


if __name__ == '__main__':
    main()
