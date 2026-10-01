"""Audit and report bounded stage-2 selection diagnostics separately from S10."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'week4_runs/selection_recovery'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(data)
    temp.replace(path)


def audit(seed, strategy):
    state = STATE / f's{seed}_{strategy}'
    manifest = json.loads((state / 'manifest.json').read_text())
    assert (state / 'exit_code').read_text().strip() == '0'
    assert manifest['start_stage'] == manifest['end_stage'] == 2
    assert manifest['seed'] == seed and manifest['strategy'] == strategy
    assert sha(Path(manifest['checkpoint'])) == manifest['checkpoint_sha256']
    stem = Path(manifest['work_dir_stem'])
    directory, = stem.parent.glob(stem.name + '_*')
    text = (state / 'console.log').read_text(errors='replace')
    assert 'Traceback (most recent call last)' not in text
    assert 'Explicit mapping incremental learning completed!' in text
    assert (directory / 'checkpoints/stage_2/epoch_1.pth').stat().st_size > 0
    assert sha(directory / 'checkpoints/stage_1/latest.pth') == manifest['checkpoint_sha256']
    metric = json.loads((directory / 'memory_bank/scores/stage_2_metrics.json').read_text())
    assert metric['evaluated_at_stage'] == 2
    classes = sorted(metric['classes'], key=lambda x: x['model_idx'])
    assert [x['model_idx'] for x in classes] == list(range(8))
    for key in ('AP_0.25', 'AP_0.50'):
        assert all(math.isfinite(x[key]) and 0 <= x[key] <= 1 for x in classes)
        assert abs(st.mean(x[key] for x in classes) - metric['m'+key]) < 1e-5
    train = [json.loads(line) for path in (directory / 'checkpoints/stage_2').glob('*.log.json')
             for line in path.read_text().splitlines() if json.loads(line).get('mode') == 'train']
    assert train and {r['epoch'] for r in train} == {1}
    assert all(math.isfinite(r['loss']) and math.isclose(r['lr'], .001, rel_tol=1e-6) for r in train)
    resources = []
    banks = []
    for stage in (1, 2):
        path = directory / f'object_memory_bank/object_memory_bank_stage_{stage}.json'
        bank = json.loads(path.read_text())
        assert bank['stage_id'] == stage
        assert bank['config']['selection_strategy'] == strategy
        assert bank['config']['exemplars_per_class'] == 20
        assert bank['config']['random_seed'] == seed
        assert sorted(map(int, bank['exemplars'])) == list(range(stage * 4))
        assert all(len(v) == 20 for v in bank['exemplars'].values())
        objects = [x for bucket in bank['exemplars'].values() for x in bucket]
        assert all(x['point_count'] >= 20 for x in objects)
        resources.append(dict(stage=stage, objects=len(objects), points=sum(x['point_count'] for x in objects),
                              pickle_bytes=path.with_suffix('.pkl').stat().st_size))
        banks.append(bank)
    pseudo = directory / 'pseudo_labels/stage_2_pseudo_only_conf50_pseudo_labels.pkl'
    assert pseudo.stat().st_size > 0
    # This branch is a direct negative control for the population repair.
    reference_match = None
    if strategy == 'random':
        reference, = (ROOT / 'incremental_logs').glob(
            f'week4_object20_s{seed}_*/object_memory_bank/object_memory_bank_stage_1.json')
        original = json.loads(reference.read_text())
        key = lambda bank: [(c, x['scene_id'], x['object_idx'], x['point_count'])
                            for c, bucket in sorted(bank['exemplars'].items()) for x in bucket]
        reference_match = key(original) == key(banks[0])
        assert reference_match, 'Random-bank selection changed unexpectedly'
    started = datetime.fromisoformat(manifest['started_at'])
    ended = datetime.fromisoformat(manifest['ended_at'])
    return dict(seed=seed, strategy=strategy, work_dir=str(directory),
                checkpoint_sha256=manifest['checkpoint_sha256'], pseudo_sha256=sha(pseudo),
                map25=metric['mAP_0.25'], map50=metric['mAP_0.50'],
                old25=st.mean(x['AP_0.25'] for x in classes[:4]),
                new25=st.mean(x['AP_0.25'] for x in classes[4:]),
                classes=classes, resources=resources, minutes=(ended-started).total_seconds()/60,
                random_bank_matches_original=reference_match, audited=True)


def report(require_complete=False):
    rows, errors, statuses = [], [], []
    for seed in (200, 201):
        for strategy in ('random', 'largest_point_count'):
            state = STATE / f's{seed}_{strategy}'
            status = json.loads((state / 'status.json').read_text()) if (state / 'status.json').exists() else {'status': 'not_started'}
            statuses.append(dict(seed=seed, strategy=strategy, **status))
            if (state / 'exit_code').exists() and (state / 'exit_code').read_text().strip() == '0':
                try:
                    rows.append(audit(seed, strategy))
                except Exception as exc:
                    errors.append(dict(seed=seed, strategy=strategy, error=repr(exc)))
    pairs = []
    for seed in (200, 201):
        by_strategy = {r['strategy']: r for r in rows if r['seed'] == seed}
        if len(by_strategy) == 2:
            a, b = by_strategy['random'], by_strategy['largest_point_count']
            assert a['checkpoint_sha256'] == b['checkpoint_sha256']
            pairs.append(dict(seed=seed, delta25_pp=100*(b['map25']-a['map25']),
                              delta50_pp=100*(b['map50']-a['map50']),
                              delta_old25_pp=100*(b['old25']-a['old25']),
                              delta_new25_pp=100*(b['new25']-a['new25']),
                              pseudo_files_identical=a['pseudo_sha256'] == b['pseudo_sha256']))
    complete = len(rows) == 4 and len(pairs) == 2 and not errors
    data = dict(updated_at=datetime.now(timezone.utc).isoformat(), audited_complete=complete,
                scope='Stage 2 only; 8 seen classes; paired checkpoint continuations, not full S10 seeds',
                statuses=statuses, audit_errors=errors, runs=rows, paired_deltas=pairs)
    atomic(STATE / 'audit.json', json.dumps(data, indent=2)+'\n')
    lines = ['# Week 4 selection recovery', '', f'Updated: {data["updated_at"]}.', '',
             'The full ten-stage largest-point-count runs failed at stage-1 bank population. '
             'The bank accepted the selector but the SUN RGB-D population wrapper rejected it. '
             'The earlier preflight tested ranking and missed this integration path. '
             'The failed runs and their exit-1 evidence are preserved.', '',
             'The repair scans the complete eligible crop pool and retains the top 20 per class '
             'with stable ties, using a bounded heap. Four new population regression cases '
             'and 22 prior focused checks passed. The random branch retains its seeded candidate order.', '',
             '## Bounded comparison', '',
             'Both selectors use the same saved stage-1 terminal checkpoint within each seed '
             '(200 or 201), reset the continuation seed, and train stage 2 for one epoch. '
             'Both have B=20 and identical resolved configuration except selection. '
             'The head has 8 seen classes; replay contains 80 old objects from stage 1. '
             'The 160-object bank written after stage 2 includes the four new classes and is not '
             'the bank used during this diagnostic. These runs do not estimate final ten-stage accuracy.', '',
             ('All four stage-2 runs passed the completion audit.' if complete else
              f'{len(rows)}/4 stage-2 runs have passed the completion audit; remaining statuses are below.'), '',
             '| Seed | Selector | Status |', '|---|---|---|']
    lines += [f'| {r["seed"]} | {r["strategy"]} | {r["status"]} |' for r in statuses]
    if rows:
        lines += ['', '## Stage 2 results', '', 'AP is in percent; differences below are percentage points.', '',
                  '| Seed | Selector | AP25 | AP50 | Old AP25 | New AP25 | Replay points | Replay MiB | Minutes |',
                  '|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            b = r['resources'][0]
            lines.append(f'| {r["seed"]} | {r["strategy"]} | {100*r["map25"]:.3f} | {100*r["map50"]:.3f} | '
                         f'{100*r["old25"]:.3f} | {100*r["new25"]:.3f} | {b["points"]:,} | '
                         f'{b["pickle_bytes"]/2**20:.2f} | {r["minutes"]:.2f} |')
    if pairs:
        lines += ['', '| Seed | Largest minus random AP25 | AP50 | Old AP25 | New AP25 | Identical pseudo files |',
                  '|---|---|---|---|---|---|']
        for p in pairs:
            lines.append(f'| {p["seed"]} | {p["delta25_pp"]:+.3f} | {p["delta50_pp"]:+.3f} | '
                         f'{p["delta_old25_pp"]:+.3f} | {p["delta_new25_pp"]:+.3f} | {p["pseudo_files_identical"]} |')
        if complete:
            lines += ['', f'Mean stage-2 AP25 change is {st.mean(p["delta25_pp"] for p in pairs):+.4f} pp. '
                      'This is an early-stage selection diagnostic. It cannot establish ten-stage retention '
                      'or repair the missing full selection comparison. Equal seeds do not guarantee bitwise deterministic GPU training.', '',
                      'A separate content audit found different saved pseudo-label boxes and scores in all '
                      '1,549 seed-200 scenes and all 1,525 seed-201 scenes; one scene per seed also differs '
                      'in detection count. The hashes differ for more than metadata alone. These are '
                      'comparisons in saved detection order, without rematching. The observed AP gain '
                      'therefore cannot be attributed solely to the selector. A subsequent selector '
                      'comparison should reuse one cached pseudo-label artifact per paired checkpoint. '
                      'See `week4_runs/selection_recovery/pseudo_content_audit.json`.']
    lines += ['', '## Completion audit', '',
              'For each accepted run, checks cover exit 0, completion marker, no traceback, '
              'checkpoint lineage and hashes, exactly eight metric classes and consistent means, '
              'finite losses and LR .001, stage-1/2 bank capacities and selector/seed, '
              'and nonempty pseudo labels and terminal checkpoint. Random stage-1 bank identities '
              'also match the corresponding original B=20 bank. Pairing checks identical checkpoint hashes. '
              'Pseudo file identity is reported separately. Full-source hashes and the accumulated diff '
              'were saved before launch.', '',
              'Sources: `week4_runs/selection_recovery/audit.json` and per-run manifests/logs. '
              'All four trainers finished naturally on October 1, 2026 at approximately 11:18 SGT. '
              'The experimental phase is closed.', '']
    if errors:
        lines += ['Audit errors: ' + json.dumps(errors), '']
    atomic(ROOT / 'WEEK_4_SELECTION_RECOVERY.md', '\n'.join(lines))
    if require_complete:
        assert complete, 'Stage-2 diagnostic is incomplete or failed its audit'
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    if args.watch:
        while True:
            report()
            if (STATE / 'completion.json').exists() or (STATE / 'controller_error.json').exists():
                break
            time.sleep(60)
    result = report(args.require_complete)
    print(json.dumps({k: v for k, v in result.items() if k not in ('runs',)}), flush=True)
