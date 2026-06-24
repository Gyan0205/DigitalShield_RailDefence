"""
logger.py — Structured logging for Digital Shield 2
Logs to both console (INFO+) and file (DEBUG+).
"""

import logging
import sys
from pathlib import Path

_LOGGER_NAME = "digital_shield"
_LOG_DIR = Path("logs")
_LOG_FILE = _LOG_DIR / "pipeline.log"


def get_logger(name: str | None = None) -> logging.Logger:
    """
    Get (or create) the application logger.
    First call initialises handlers; subsequent calls return the cached logger.
    """
    logger = logging.getLogger(name or _LOGGER_NAME)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(module)-14s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — DEBUG and above (full trace)
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
