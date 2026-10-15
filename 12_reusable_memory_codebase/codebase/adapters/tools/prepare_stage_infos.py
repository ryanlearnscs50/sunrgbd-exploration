#!/usr/bin/env python3
"""Filter converted ScanNet infos and optional selected replay records by stage."""
import argparse
from pathlib import Path
import pickle
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'adapters'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='converted full-protocol info pickle')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--stage', required=True, type=int)
    parser.add_argument('--stage-setting', default='scannet35_s5_freqorder')
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--memory-mode', choices=['none', 'scene', 'object'], required=True)
    parser.add_argument('--scene-dedup', choices=['merge_labels', 'keep_both'], default='merge_labels')
    parser.add_argument('--replay-records', type=Path, help='trusted pickle of scene.replay_records return value')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; refusing to overwrite')
    from ldmr_unidet3d import ScanNet35Protocol
    from ldmr_unidet3d.data import stage_records, validate_stage_manifest
    protocol = ScanNet35Protocol(args.stage_setting)
    with args.input.open('rb') as f:
        full = pickle.load(f)
    evaluation = args.split == 'val'
    records = stage_records(full['data_list'], protocol, args.stage, evaluation=evaluation)
    needs_replay = args.split == 'train' and args.memory_mode == 'scene' and args.stage > 1
    if needs_replay != bool(args.replay_records):
        parser.error('selected replay records are required only for later-stage scene training')
    if args.replay_records:
        with args.replay_records.open('rb') as f:
            replay = pickle.load(f)
        if not all(r.get('is_replay', False) for r in replay):
            parser.error('replay export contains unmarked records')
        from ldmr_unidet3d.scene import merge_replay_records
        records = merge_replay_records(records, replay, args.scene_dedup)
    payload = dict(metainfo=dict(classes=protocol.decoder_classes(args.stage),
        ldmr_protocol_id=protocol.identity, ldmr_stage=args.stage,
        ldmr_split=args.split, ldmr_memory_mode=args.memory_mode), data_list=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import tempfile
    import os
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=args.output.parent, delete=False) as f:
            temporary = Path(f.name)
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        validate_stage_manifest(temporary, protocol, args.stage, args.memory_mode, evaluation=evaluation)
        # Link publishes without overwriting an existing destination, even on a race.
        os.link(temporary, args.output)
    finally:
        if temporary and temporary.exists():
            temporary.unlink()
    print(f'{args.output}: {len(records)} records')


if __name__ == '__main__':
    main()
