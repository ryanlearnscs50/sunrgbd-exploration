from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.incremental_scene.pseudo_pipeline import resolve_previous_stage_checkpoint


def test_pseudo_teacher_uses_latest_with_multiple_retained_epochs(tmp_path):
    for epoch in (1, 2, 5):
        (tmp_path / f'epoch_{epoch}.pth').write_bytes(str(epoch).encode())
    latest = tmp_path / 'latest.pth'
    latest.symlink_to('epoch_5.pth')
    assert resolve_previous_stage_checkpoint(tmp_path).read_bytes() == b'5'


@pytest.mark.parametrize('latest', ['absent', 'broken', 'copy'])
def test_pseudo_teacher_numeric_fallback_and_regular_latest(tmp_path, latest):
    for epoch in (1, 9, 10):
        (tmp_path / f'epoch_{epoch}.pth').write_bytes(str(epoch).encode())
    (tmp_path / 'epoch_invalid.pth').write_bytes(b'invalid')
    (tmp_path / 'epoch_100.pth').mkdir()
    if latest == 'broken':
        (tmp_path / 'latest.pth').symlink_to('missing.pth')
    elif latest == 'copy':
        (tmp_path / 'latest.pth').write_bytes(b'10')
    assert resolve_previous_stage_checkpoint(tmp_path).read_bytes() == b'10'


def test_pseudo_teacher_missing_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError, match='previous-stage checkpoint'):
        resolve_previous_stage_checkpoint(tmp_path)
