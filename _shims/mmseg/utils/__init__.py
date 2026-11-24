"""mmseg.utils stand-in: only get_root_logger is referenced by MPViT."""
import logging

from mmcv.utils import get_logger


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Return the root logger used by mmseg, backed by mmcv's get_logger."""
    return get_logger("mmseg", log_file, log_level)
