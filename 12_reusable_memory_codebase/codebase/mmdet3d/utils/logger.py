# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
from functools import lru_cache

from mmcv.utils import get_logger


@lru_cache(maxsize=1)
def _maybe_suppress_mmcv_init_info_dump() -> None:
    """Suppress MMCV BaseModule init_weights parameter dumps.

    MMCV's BaseModule writes per-parameter initialization info into the log
    file (via FileHandler). For large models this adds thousands of lines and
    drowns out incremental-learning stage logs.

    Set `TR3D_DUMP_INIT_INFO=1` to re-enable the original behavior.
    """
    enabled = os.getenv('TR3D_DUMP_INIT_INFO', '').strip().lower() in {
        '1', 'true', 'yes', 'y', 'on'
    }
    if enabled:
        return

    try:
        from mmcv.runner.base_module import BaseModule
    except Exception:
        return

    if getattr(BaseModule, '_tr3d_init_info_dump_suppressed', False):
        return

    def _dump_init_info_noop(self, logger_name: str) -> None:  # noqa: ARG001
        return

    BaseModule._dump_init_info = _dump_init_info_noop  # type: ignore[assignment]
    BaseModule._tr3d_init_info_dump_suppressed = True


@lru_cache(maxsize=1)
def _maybe_guard_mmcv_text_logger_hook_dirs() -> None:
    """Make MMCV TextLoggerHook resilient to missing work_dir.

    MMCV's TextLoggerHook writes `*.log.json` by opening `self.json_log_path`.
    If the work_dir is deleted mid-run (common on NFS when users clean folders),
    MMCV crashes with FileNotFoundError. Guard by recreating the parent dir.
    """
    try:
        from mmcv.runner.hooks.logger.text import TextLoggerHook
    except Exception:
        return

    if getattr(TextLoggerHook, '_tr3d_dir_guard_installed', False):
        return

    orig_dump_log = TextLoggerHook._dump_log

    def _dump_log_with_dir_guard(self, log_dict, runner) -> None:  # type: ignore[no-untyped-def]
        if getattr(runner, 'rank', 0) == 0:
            json_log_path = getattr(self, 'json_log_path', None)
            if isinstance(json_log_path, str) and json_log_path:
                try:
                    os.makedirs(os.path.dirname(json_log_path), exist_ok=True)
                except Exception:
                    pass
        return orig_dump_log(self, log_dict, runner)

    TextLoggerHook._dump_log = _dump_log_with_dir_guard  # type: ignore[assignment]
    TextLoggerHook._tr3d_dir_guard_installed = True


def get_root_logger(log_file=None, log_level=logging.INFO, name='mmdet3d'):
    """Get root logger and add a keyword filter to it.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmdet3d".

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str, optional): The name of the root logger, also used as a
            filter keyword. Defaults to 'mmdet3d'.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    _maybe_suppress_mmcv_init_info_dump()
    _maybe_guard_mmcv_text_logger_hook_dirs()
    logger = get_logger(name=name, log_file=log_file, log_level=log_level)

    # add a logging filter
    logging_filter = logging.Filter(name)
    logging_filter.filter = lambda record: record.find(name) != -1

    return logger
