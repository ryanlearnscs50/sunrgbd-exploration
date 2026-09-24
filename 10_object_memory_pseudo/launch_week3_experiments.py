"""Launch selected Week 3 experiments with persistent logs and sessions."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--seed', type=int, default=201)
parser.add_argument('--modes', nargs='+', default=['pseudo_only', 'object_pseudo'],
                    choices=['pseudo_only', 'object_pseudo', 'dose25', 'dose50', 'pseudo_cosine', 'dose25_cosine', 'pseudo_cosine_v2', 'dose25_cosine_v2'])
args = parser.parse_args()
if len(args.modes) > 2 or len(set(args.modes)) != len(args.modes):
    parser.error('Select one or two distinct modes (one per GPU).')
state_root = ROOT / 'week3_runs'
state_root.mkdir(exist_ok=True)
checkpoint = ROOT / 'checkpoints/sunrgbd_5stage/stage_01.pth'
if not checkpoint.is_file():
    raise SystemExit('Missing starting checkpoint')
configs = {
    'pseudo_cosine_v2': 'tr3d_dynamic_head_8x5_pseudo_only_cosine_52211.py',
    'dose25_cosine_v2': 'tr3d_dynamic_head_8x5_object_memory_pseudo_dose25_cosine_52211.py',
    'pseudo_cosine': 'tr3d_dynamic_head_8x5_pseudo_only_cosine_52211.py',
    'dose25_cosine': 'tr3d_dynamic_head_8x5_object_memory_pseudo_dose25_cosine_52211.py',
    'pseudo_only': 'tr3d_dynamic_head_8x5_pseudo_only_matched_52211.py',
    'object_pseudo': 'tr3d_dynamic_head_8x5_object_memory_random_pseudo_52211.py',
    'dose25': 'tr3d_dynamic_head_8x5_object_memory_pseudo_dose25_52211.py',
    'dose50': 'tr3d_dynamic_head_8x5_object_memory_pseudo_dose50_52211.py',
}
configs = {mode: configs[mode] for mode in args.modes}
def run_id(mode):
    return mode if args.seed == 201 else f'{mode}_s{args.seed}'

with (state_root / 'launch.lock').open('w') as launch_lock:
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Check both destinations before launching either experiment.
    for mode in configs:
        state = state_root / run_id(mode)
        if any((state / name).exists() for name in
               ('manifest.json', 'exit_code', 'console.log', 'supervisor.pid', 'trainer.pid')):
            raise SystemExit(f'{mode} already launched; inspect its state before resuming')
        config_path = ROOT / 'repo/configs/incremental/sunrgbd' / configs[mode]
        if not config_path.is_file():
            raise SystemExit(f'Missing config: {config_path}')
    if not args.dry_run:
        subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.used',
                        '--format=csv,noheader'], check=True)
        jobs = subprocess.check_output([
            'nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'
        ], text=True).strip()
        if jobs:
            raise SystemExit('Compute processes already active; inspect GPUs first')
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    for gpu, (mode, config) in enumerate(configs.items()):
        state = state_root / run_id(mode)
        config_path = ROOT / 'repo/configs/incremental/sunrgbd' / config
        command = ['nohup', 'bash', str(ROOT / 'run_week3_experiment.sh'), mode, str(args.seed)]
        manifest = dict(mode=mode, run_id=run_id(mode), gpu=gpu, seed=args.seed, start_stage=2, end_stage=5,
                        checkpoint=str(checkpoint), checkpoint_sha256=digest,
                        config=config, command=command,
                        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                        work_dir_stem=str(ROOT / 'incremental_logs' / f'week3_{mode}_s{args.seed}'),
                        launched_at=datetime.now(timezone.utc).isoformat())
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
            continue
        state.mkdir(exist_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                   MPLCONFIGDIR='/tmp/ldmr-week3-mpl', PYTHONUNBUFFERED='1')
        with (state / 'console.log').open('xb') as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True)
        manifest['supervisor_pid'] = proc.pid
        (state / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'{mode}: detached supervisor PID {proc.pid}, GPU {gpu}, log {state / "console.log"}')
