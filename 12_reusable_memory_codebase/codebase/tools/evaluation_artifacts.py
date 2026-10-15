"""Machine-readable evaluation evidence for the legacy detector entry points."""
import hashlib
import json
import math
from numbers import Real
import os
from pathlib import Path
import tempfile


def file_identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest())


def write_evaluation_report(path, metrics, *, dataset, config, checkpoint,
                            seed, stage=None, trained_classes=None):
    """Publish finite metrics with the actual class denominator and input hashes.

    The report is created exclusively. Failure leaves no partial report and
    never overwrites an earlier result. Config text includes inherited configs
    as captured by MMCV; it does not replace a source-code/environment archive.
    """
    values = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'evaluation metric must be a finite scalar: {key}')
        values[str(key)] = float(value)
    if not values or not any(key.startswith('mAP_') for key in values):
        raise ValueError('evaluation report requires aggregate mAP metrics')
    classes = list(dataset.CLASSES)
    seen = getattr(dataset, 'seen_classes_for_eval', None)
    ids = list(range(len(classes))) if seen is None else list(seen)
    ann_files = dataset.ann_file
    if isinstance(ann_files, (str, Path)):
        ann_files = [ann_files]
    report = dict(format_version=1, metrics=values, seed=seed, stage=stage,
        trained_class_count=trained_classes, evaluated_class_ids=ids,
        evaluated_classes=[classes[i] for i in ids], dataset_type=type(dataset).__name__,
        scene_count=len(dataset), checkpoint=file_identity(checkpoint),
        annotations=[file_identity(p) for p in ann_files],
        config=file_identity(config.filename), config_source_text=config.text,
        resolved_config_repr=repr(config._cfg_dict.to_dict()))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.link(temporary, path)  # exclusive publication, including on a race
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return report
