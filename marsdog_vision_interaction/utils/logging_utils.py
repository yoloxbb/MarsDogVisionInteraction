"""Unified logging for MarsDog perception nodes.

Provides structured logging with optional module tags and file output.
Uses Python's standard logging with a custom logger that supports key=value kwargs.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


_log_initialized: bool = False
_log_dir: str = "log"


# ── Set custom logger class at import time ─────────────────────────
# This MUST happen before any module-level `logger = getLogger(...)` call,
# otherwise those loggers will be plain logging.Logger instances and
# fail when called with key=value kwargs like logger.info("msg", key=val).


class StructuredLogger(logging.Logger):
    """Logger subclass that supports key=value structured logging.

    Usage:
        logger.info("camera_init", device="/dev/video0", width=640)
        # → "camera_init  device='/dev/video0'  width=640"
    """

    def _log_with_kwargs(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        if kwargs:
            parts = [f"{k}={v!r}" for k, v in kwargs.items()]
            msg = f"{msg}  " + "  ".join(parts)
        self._log(level, msg, args)

    def debug(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(logging.DEBUG):
            self._log_with_kwargs(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(logging.INFO):
            self._log_with_kwargs(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(logging.WARNING):
            self._log_with_kwargs(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Any) -> None:
        if self.isEnabledFor(logging.ERROR):
            self._log_with_kwargs(logging.ERROR, msg, *args, **kwargs)


# Register the custom logger class globally at import time.
# This ensures ALL loggers (including module-level ones created before
# setup_logging() is called) support key=value structured logging.
logging.setLoggerClass(StructuredLogger)


def setup_logging(
    log_dir: str = "log",
    level: str = "INFO",
    node: str = "marsdog",
    console: bool = True,
    file: bool = True,
) -> None:
    """Initialize logging for a node.

    Sets StructuredLogger as the default logger class so all loggers
    created via getLogger() support key=value structured logging.

    Args:
        log_dir: Directory for log files.
        level: Log level name (DEBUG, INFO, WARNING, ERROR).
        node: Node name for log file prefix.
        console: Enable console output.
        file: Enable file output.
    """
    global _log_initialized, _log_dir
    _log_dir = log_dir

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers
    if _log_initialized:
        return

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(fmt)
        root.addHandler(ch)

    if file:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        date_str = datetime.now().strftime("%Y%m%d")
        fh = logging.FileHandler(
            str(Path(log_dir) / f"{node}_{date_str}.log"),
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    _log_initialized = True


def set_log_level(level: str) -> None:
    """Change the root logger level at runtime.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR).
    """
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str, module: str = "") -> logging.Logger:
    """Get a logger with optional module tag.

    Args:
        name: Logger name (usually __name__).
        module: Optional module tag for filtering.

    Returns:
        Configured StructuredLogger instance.
    """
    if module:
        return logging.getLogger(f"{module}.{name}")
    return logging.getLogger(name)
