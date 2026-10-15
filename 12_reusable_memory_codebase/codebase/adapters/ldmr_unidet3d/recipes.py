"""Named reference methods; host optimization stays native to UniDet3D.

Epochs in the host are single dataset passes. LDMR's 12 nominal ScanNet epochs
with RepeatDataset(15) become 180 host epochs, including review boundaries at
36/72/108/144 passes. Never label the historical 12-pass experiment a reference
training recipe. The SUN TR3D_OBJ schedule belongs to the legacy SUN config;
its ScanNet transfer uses LDMR's ScanNet exposure, not SUN's 5/2/2/1/1 schedule.
"""
from pathlib import Path
import runpy


TR3D_OBJ_PROFILE = (Path(__file__).resolve().parents[2]
                   / 'configs/memory/scannet_s5_tr3d_obj.py')


def tr3d_obj_settings():
    """Read the same method settings as legacy TR3D, without importing MMCV.

    This profile declares its bank, insertion and pseudo sections completely;
    its inherited detector config belongs to the legacy runtime only.
    """
    config = runpy.run_path(str(TR3D_OBJ_PROFILE))
    return config['object_memory_config'], config['pseudo_label_config']


def tr3d_obj_insertion():
    options = tr3d_obj_settings()[0]['insertion']
    return dict(max_exemplars=options['max_exemplars_per_scene'],
                probability=options['insertion_probability'],
                sampling_policy=options['sampling_policy'],
                max_attempts=options['max_placement_attempts'],
                floor_percentile=options['floor_percentile'],
                preserve_source_height=options['preserve_source_height'],
                collision_iou=options['collision_threshold'],
                placement_jitter=options['placement_jitter'])


def apply_reference_recipe(args):
    if args.recipe == 'custom':
        if getattr(args, 'validation_interval', None) is None:
            args.validation_interval = 1
        return
    mode = 'scene' if args.recipe == 'ldmr' else 'object'
    if args.memory_mode != mode or args.stage_setting != 'scannet35_s5_freqorder':
        raise ValueError(f'{args.recipe} recipe requires {mode} memory and ScanNet35 S5')
    if args.ld_epochs is not None or args.reviewing:
        raise ValueError('reference recipe owns selection/reviewing; omit manual LD/review flags')
    args.nominal_epochs = args.epochs
    args.reference_repeat_times = 15
    args.epochs *= args.reference_repeat_times
    # Host epochs are single passes, while the reference validates/checkpoints
    # in nominal epoch units. Do not multiply full validation cost by 15.
    if getattr(args, 'validation_interval', None) is None:
        args.validation_interval = args.reference_repeat_times
    args.target_mode = 'boxes'
    args.pseudo = True
    if mode == 'scene':
        args.reviewing = True
    else:
        bank, _ = tr3d_obj_settings()
        args.object_policy = 'tr3d_obj'
        args.object_selector = bank['selection_strategy']
        args.objects_per_class = bank['exemplars_per_class']
        args.max_objects = bank['max_total_exemplars']
        args.min_points = bank['min_points']
    print(f'Reference {args.recipe}: {args.nominal_epochs} nominal epochs x '
          f'{args.reference_repeat_times} repeats = {args.epochs} full dataset passes/stage; '
          f'batch {args.batch_size}. UniDet3D optimizer and detector remain native.', flush=True)
