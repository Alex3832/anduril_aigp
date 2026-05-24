"""Centralised logging configuration for the whole project."""

import logging
import sys


_FMT = "%(asctime)s [%(levelname)s] %(name)s — %(message)s"
_DATE_FMT = "%H:%M:%S"

_configured = False


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """Return a named logger, configuring the root handler on first call.

    Args:
        name:  Logger name, typically __name__ of the calling module.
        level: Logging level (default DEBUG).

    Returns:
        Configured Logger instance.
    """
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE_FMT))
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.DEBUG)
        _configured = True

    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
