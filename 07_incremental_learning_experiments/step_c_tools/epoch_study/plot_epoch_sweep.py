#!/usr/bin/env python
"""Plot the optimal-epoch sweep (2026-06-11 extension: epochs BEYOND 12).

Reads the FINAL-stage cumulative-val eval of each full-chain run at epoch budget
E for the pseudo (and naive, if present) variant, then:
  * writes a PNG (old / new / overall mAP@0.25 vs E) to figures/, and
  * prints an ASCII version of the same curve to stdout (so it shows in the
    console as well as a file the user can open).

The whole point of going past E=12 is to see whether pseudo keeps climbing,
plateaus, or OVERFITS (old-class mAP turning back down). The PNG marks the best
old-class E and the paper E=12.

Usage:
  python tools_incremental/plot_epoch_sweep.py --sched 3stage \
      --epochs 2,4,6,8,12,16,24,36 --out figures/epoch_sweep_3stage.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from analyze_incremental import parse_eval_log, stage_class_names

CFG_DIR = 'configs/tr3d_incremental_v4'


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
    return {'old25': mean(old, 0), 'new25': mean(new, 0), 'all25': mean(cum, 0),
            'old50': mean(old, 1), 'all50': mean(cum, 1)}


def collect(sched, variant, epochs, old, new, cum):
    out = []
    for E in epochs:
        s = summarize(sched, variant, E, old, new, cum)
        if s is not None:
            out.append((E, s))
    return out


def ascii_plot(label, points, key, width=56, height=15):
    """Tiny ASCII scatter/line of metric `key` vs E."""
    if not points:
        print(f'  (no data for {label})')
        return
    xs = [E for E, _ in points]
    ys = [s[key] for _, s in points]
    ymax = max(ys) * 1.10 or 1.0
    ymin = 0.0
    xmin, xmax = min(xs), max(xs)
    print(f'\n  {label}: {key} vs epoch budget E  '
          f'(y 0..{ymax:.3f}, x {xmin}..{xmax})')
    grid = [[' '] * width for _ in range(height)]
    def col(E):
        if xmax == xmin:
            return 0
        return int(round((E - xmin) / (xmax - xmin) * (width - 1)))
    def row(y):
        return height - 1 - int(round((y - ymin) / (ymax - ymin) * (height - 1)))
    for E, s in points:
        r, c = row(s[key]), col(E)
        grid[max(0, min(height-1, r))][max(0, min(width-1, c))] = '*'
    for r in range(height):
        ylab = ymax - (ymax - ymin) * r / (height - 1)
        print(f'  {ylab:6.3f} |' + ''.join(grid[r]))
    print('         +' + '-' * width)
    # x axis labels
    lab = [' '] * width
    for E in xs:
        c = col(E)
        s = str(E)
        for i, ch in enumerate(s):
            if 0 <= c + i < width:
                lab[c + i] = ch
    print('          ' + ''.join(lab))
    # numeric table
    print('   E     : ' + ' '.join(f'{E:>6}' for E in xs))
    print(f'   {key:<6}: ' + ' '.join(f'{s[key]:>6.4f}' for _, s in points))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sched', default='3stage', choices=['3stage', '6stage'])
    p.add_argument('--epochs', default='2,4,6,8,12,16,24,36')
    p.add_argument('--out', default=None)
    args = p.parse_args()
    epochs = [int(x) for x in args.epochs.split(',')]
    nstage = 3 if args.sched == '3stage' else 6
    out = args.out or f'figures/epoch_sweep_{args.sched}.png'

    cum = stage_class_names(CFG_DIR, args.sched, nstage)
    prev = stage_class_names(CFG_DIR, args.sched, nstage - 1)
    old, new = prev, [c for c in cum if c not in set(prev)]

    print(f'\n############ EPOCH SWEEP PLOT — {args.sched} '
          f'(final stage{nstage} cumulative val) ############')
    print(f'old classes: {len(old)} | new classes: {len(new)} | epochs probed: {epochs}')

    series = {}
    for variant in ('pseudo', 'naive'):
        pts = collect(args.sched, variant, epochs, old, new, cum)
        if pts:
            series[variant] = pts

    # ---- console (ASCII) ----
    for variant, pts in series.items():
        print(f'\n===== {variant.upper()} =====')
        print(f'{"E":>4} | {"old@.25":>8} {"new@.25":>8} {"all@.25":>8} | '
              f'{"old@.50":>8} {"all@.50":>8}')
        print('-' * 56)
        for E, s in pts:
            print(f'{E:>4} | {s["old25"]:>8.4f} {s["new25"]:>8.4f} {s["all25"]:>8.4f} | '
                  f'{s["old50"]:>8.4f} {s["all50"]:>8.4f}')
        best_old = max(pts, key=lambda r: r[1]['old25'])
        print(f'  best OLD@.25: E={best_old[0]} ({best_old[1]["old25"]:.4f})')
        if variant == 'pseudo':
            ascii_plot('pseudo', pts, 'old25')

    # ---- PNG ----
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {'old25': 'tab:red', 'new25': 'tab:green', 'all25': 'tab:blue'}
    titles = {'pseudo': 'Pseudo-label', 'naive': 'Naive fine-tune'}
    for ax, variant in zip(axes, ('pseudo', 'naive')):
        pts = series.get(variant)
        if not pts:
            ax.set_title(f'{titles.get(variant, variant)} (no data)')
            continue
        xs = [E for E, _ in pts]
        for key, lab in [('old25', 'old mAP@.25'),
                         ('new25', 'new mAP@.25'),
                         ('all25', 'overall mAP@.25')]:
            ax.plot(xs, [s[key] for _, s in pts], 'o-', color=colors[key], label=lab)
        ax.axvline(12, color='gray', ls='--', lw=1, label='paper E=12')
        best_old = max(pts, key=lambda r: r[1]['old25'])
        ax.scatter([best_old[0]], [best_old[1]['old25']], s=140,
                   facecolors='none', edgecolors='black', linewidths=1.8,
                   zorder=5, label=f'best old (E={best_old[0]})')
        ax.set_xlabel('epoch budget E  (×5 real passes via RepeatDataset)')
        ax.set_ylabel('mAP@0.25')
        ax.set_title(f'{titles.get(variant, variant)} — {args.sched}')
        ax.set_xticks(xs)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f'Incremental {args.sched}: final-stage mAP vs epoch budget '
                 f'(does pseudo overfit past E=12?)')
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f'\n[saved PNG] {os.path.abspath(out)}')


if __name__ == '__main__':
    main()
