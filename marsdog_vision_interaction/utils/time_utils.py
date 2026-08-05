"""Time utilities for MarsDog perception."""

from __future__ import annotations

import time


def now_stamp() -> float:
    """Return the current Unix timestamp as a float.

    Returns:
        time.time() value — seconds since Unix epoch.
    """
    return time.time()


def now_ms() -> int:
    """Return the current time in milliseconds.

    Returns:
        int(now_stamp() * 1000).
    """
    return int(now_stamp() * 1000)
