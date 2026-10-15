"""Run in the isolated host environment, without a detector import requirement."""
import sys

from ldmr_unidet3d.paired_audit import read_config


def test_audit_config_does_not_import_training_plugins(tmp_path):
    config = tmp_path / 'resolved.py'
    config.write_text("custom_imports = dict(imports=['unavailable_training_plugin'], "
                      "allow_failed_imports=False)\nmodel = dict(type='Detector')\n")
    assert read_config(config)['model']['type'] == 'Detector'
    assert 'unavailable_training_plugin' not in sys.modules
