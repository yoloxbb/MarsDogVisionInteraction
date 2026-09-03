"""Unified logging for MarsDog perception nodes.

Provides structured logging with optional module tags and file output.
Uses Python's standard logging with a custom logger that supports key=value kwargs.
"""

from __future__ import annotations

import logging
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import threading
from typing import Any


_log_initialized: bool = False
_log_dir: str = "log"
_trace_enabled: bool = False
_trace_context: dict[str, str] = {"run_id": "", "case_id": ""}
_trace_logger = logging.getLogger("vision.trace")
_trace_handler: logging.Handler | None = None
_trace_lock = threading.Lock()
_timing_trace_interval_ms: float = 5000.0
_timing_trace_last_ms: dict[tuple[str, str, str], float] = {}
_timing_trace_lock = threading.Lock()


# ── Set custom logger class at import time ─────────────────────────
# This MUST happen before any module-level `logger = getLogger(...)` call,
# otherwise those loggers will be plain logging.Logger instances and
# fail when called with key=value kwargs like logger.info("msg", key=val).


class StructuredLogger(logging.Logger):
    """Logger subclass that supports key=value structured logging.

    Usage:
        logger.info("camera_init", device="0", width=640)
        # → "camera_init  device='0'  width=640"
    """

    def _log_with_kwargs(self, level: int, msg: str, *args: Any, **kwargs: Any) -> None:
        # ``extra``, ``exc_info``, ``stack_info`` and ``stacklevel`` belong to
        # Python's logging API.  Uvicorn uses ``extra.color_message`` containing
        # its own %-placeholders, so serialising it into ``msg`` corrupts the
        # original format string and produces "not enough arguments" errors.
        standard_keys = {"exc_info", "extra", "stack_info", "stacklevel"}
        standard = {
            key: kwargs.pop(key)
            for key in tuple(kwargs)
            if key in standard_keys
        }
        if kwargs:
            parts = [f"{k}={v!r}" for k, v in kwargs.items()]
            msg = f"{msg}  " + "  ".join(parts)
        self._log(level, msg, args, **standard)

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


def configure_event_trace(
    *,
    enabled: bool = True,
    log_dir: str = "log",
    run_id: str = "",
    case_id: str = "",
    timing_interval_sec: float = 5.0,
) -> None:
    """Configure machine-readable ``VISION_TRACE`` JSONL records.

    Trace records also propagate to the normal runtime log.  The dedicated
    file contains only trace lines so QA tooling does not need to parse ROS or
    Python log prefixes.
    """
    global _trace_enabled, _trace_context, _trace_handler
    global _timing_trace_interval_ms, _timing_trace_last_ms
    _trace_enabled = bool(enabled)
    _timing_trace_interval_ms = max(0.0, float(timing_interval_sec)) * 1000.0
    with _timing_trace_lock:
        _timing_trace_last_ms = {}
    _trace_context = {
        "run_id": str(run_id or os.environ.get("MARSDOG_TEST_RUN_ID", "")),
        "case_id": str(case_id or os.environ.get("MARSDOG_TEST_CASE_ID", "")),
    }
    if _trace_handler is not None:
        _trace_logger.removeHandler(_trace_handler)
        _trace_handler.close()
        _trace_handler = None
    if not _trace_enabled:
        return
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _trace_handler = logging.FileHandler(
        Path(log_dir) / f"vision_trace_{timestamp}_{os.getpid()}.jsonl",
        encoding="utf-8",
    )
    _trace_handler.setFormatter(logging.Formatter("%(message)s"))
    _trace_logger.addHandler(_trace_handler)
    _trace_logger.setLevel(logging.INFO)
    _trace_logger.propagate = True


def vision_trace(record: str, **fields: Any) -> None:
    """Emit one correlated, single-line JSON record for test evidence."""
    if not _trace_enabled:
        return
    payload: dict[str, Any] = {
        "schema_version": 1,
        "record": str(record),
        "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
        "monotonic_ms": round(time_monotonic_ms(), 3),
        **_trace_context,
    }
    payload.update(fields)
    line = "VISION_TRACE " + json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), default=str
    )
    with _trace_lock:
        _trace_logger.info(line)


def vision_timing_trace(
    *,
    node: str,
    module: str,
    stage: str,
    latency_ms: float,
    result: str = "success",
    force: bool = False,
    **fields: Any,
) -> bool:
    """Emit a rate-limited ``stage_complete`` timing record.

    Continuous vision stages execute several times per second.  Rate limiting
    keeps QA timing evidence available without making file logging part of the
    measured workload.  Set ``timing_interval_sec`` to zero to trace every run;
    on-demand stages may pass ``force=True``.
    """
    if not _trace_enabled:
        return False
    force = bool(force or str(result) not in {"success", "skipped"})
    now_ms = time_monotonic_ms()
    key = (str(node), str(module), str(stage))
    with _timing_trace_lock:
        previous_ms = _timing_trace_last_ms.get(key)
        if (
            not force
            and _timing_trace_interval_ms > 0.0
            and previous_ms is not None
            and now_ms - previous_ms < _timing_trace_interval_ms
        ):
            return False
        _timing_trace_last_ms[key] = now_ms
    vision_trace(
        "stage_complete",
        result=str(result),
        node=str(node),
        module=str(module),
        stage=str(stage),
        latency_ms=round(max(0.0, float(latency_ms)), 3),
        sampled=not force,
        **fields,
    )
    return True


def time_monotonic_ms() -> float:
    """Small wrapper kept separate for deterministic trace tests."""
    import time

    return time.monotonic_ns() / 1_000_000.0
