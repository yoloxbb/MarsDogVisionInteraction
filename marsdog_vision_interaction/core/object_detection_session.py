"""Thread-safe lifecycle for an on-demand object-detection stream."""

from __future__ import annotations

import math
import threading
import time
from typing import Any


class ObjectDetectionSessionManager:
    """Own one detection stream lease without owning inference or motion."""

    def __init__(
        self,
        *,
        startup_rate_hz: float = 0.0,
        default_rate_hz: float = 2.0,
        max_rate_hz: float = 5.0,
        default_confidence: float = 0.2,
        default_lease_sec: float = 3.0,
        max_lease_sec: float = 30.0,
    ) -> None:
        self._lock = threading.RLock()
        self._max_rate_hz = max(0.1, float(max_rate_hz))
        self._default_rate_hz = min(
            self._max_rate_hz,
            max(0.1, float(default_rate_hz)),
        )
        self._default_confidence = min(
            1.0, max(0.0, float(default_confidence))
        )
        self._max_lease_sec = max(0.5, float(max_lease_sec))
        self._default_lease_sec = min(
            self._max_lease_sec,
            max(0.5, float(default_lease_sec)),
        )
        self._active = False
        self._session_id = ""
        self._rate_hz = self._default_rate_hz
        self._confidence = self._default_confidence
        self._target_labels: tuple[str, ...] = ()
        self._lease_deadline: float | None = None
        self._next_inference_at = 0.0

        startup_rate_hz = float(startup_rate_hz)
        if startup_rate_hz > 0.0:
            self._active = True
            self._session_id = "configured"
            self._rate_hz = min(startup_rate_hz, self._max_rate_hz)
            # A configured stream is intended for the dedicated debug launch
            # and has no lease. Production configuration starts disabled.
            self._lease_deadline = None

    @staticmethod
    def normalize_target_labels(value: Any) -> list[str]:
        if value is None:
            return []
        values = [value] if isinstance(value, str) else value
        if not isinstance(values, list):
            raise ValueError("target_labels must be a string or string array")
        labels: list[str] = []
        seen: set[str] = set()
        for item in values:
            if not isinstance(item, str):
                raise ValueError("target_labels must contain only strings")
            label = item.strip()
            if not label:
                continue
            if len(label) > 128:
                raise ValueError("target label is too long")
            normalized = label.casefold()
            if normalized not in seen:
                seen.add(normalized)
                labels.append(label)
        if len(labels) > 32:
            raise ValueError("at most 32 target_labels are allowed")
        return labels

    def configure(
        self,
        params: dict[str, Any],
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Start, renew, update or stop the session named in ``params``."""
        timestamp = time.monotonic() if now is None else float(now)
        enabled = params.get("enabled")
        if not isinstance(enabled, bool):
            return {"ok": False, "error": "enabled must be a JSON boolean"}
        session_id = str(params.get("session_id", "")).strip()
        if not session_id:
            return {"ok": False, "error": "session_id is required"}
        if len(session_id) > 128:
            return {"ok": False, "error": "session_id is too long"}

        with self._lock:
            self._expire_locked(timestamp)
            if not enabled:
                if self._active and self._session_id != session_id:
                    return self._ownership_error_locked(timestamp)
                stopped_session_id = self._session_id or session_id
                self._clear_locked()
                return {
                    "ok": True,
                    "stopped_session_id": stopped_session_id,
                    "stream": self._snapshot_locked(timestamp),
                }

            if self._active and self._session_id != session_id:
                return self._ownership_error_locked(timestamp)

            try:
                rate_hz = float(
                    params.get(
                        "rate_hz",
                        self._rate_hz if self._active else self._default_rate_hz,
                    )
                )
                confidence = float(
                    params.get(
                        "confidence",
                        (
                            self._confidence
                            if self._active
                            else self._default_confidence
                        ),
                    )
                )
                lease_sec = float(
                    params.get("lease_sec", self._default_lease_sec)
                )
                target_labels = (
                    self.normalize_target_labels(params.get("target_labels"))
                    if "target_labels" in params
                    else list(self._target_labels)
                )
            except (TypeError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}
            if not math.isfinite(rate_hz) or not 0.0 < rate_hz <= self._max_rate_hz:
                return {
                    "ok": False,
                    "error": f"rate_hz must be within (0, {self._max_rate_hz}]",
                }
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                return {"ok": False, "error": "confidence must be within [0, 1]"}
            if (
                not math.isfinite(lease_sec)
                or not 0.5 <= lease_sec <= self._max_lease_sec
            ):
                return {
                    "ok": False,
                    "error": (
                        "lease_sec must be within "
                        f"[0.5, {self._max_lease_sec}]"
                    ),
                }

            was_active = self._active
            self._active = True
            self._session_id = session_id
            self._rate_hz = rate_hz
            self._confidence = confidence
            self._target_labels = tuple(target_labels)
            self._lease_deadline = timestamp + lease_sec
            if not was_active:
                self._next_inference_at = timestamp
            return {"ok": True, "stream": self._snapshot_locked(timestamp)}

    def poll(self, *, now: float | None = None) -> dict[str, Any]:
        """Return whether inference is due or the current lease just expired."""
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            if self._lease_is_expired_locked(timestamp):
                expired = self._snapshot_locked(timestamp)
                expired["active"] = False
                expired["lease_remaining_sec"] = 0.0
                self._clear_locked()
                return {
                    "state": "expired",
                    "stream": expired,
                    "stop_reason": "lease_expired",
                }
            if not self._active:
                return {"state": "inactive"}
            if timestamp < self._next_inference_at:
                return {"state": "waiting"}
            self._next_inference_at = timestamp + 1.0 / self._rate_hz
            stream = self._snapshot_locked(timestamp)
            return {
                "state": "due",
                "stream": stream,
                "params": {
                    "confidence": self._confidence,
                    "target_labels": list(self._target_labels),
                },
            }

    def snapshot(self, *, now: float | None = None) -> dict[str, Any]:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            self._expire_locked(timestamp)
            return self._snapshot_locked(timestamp)

    def _ownership_error_locked(self, now: float) -> dict[str, Any]:
        return {
            "ok": False,
            "error": (
                "object detection stream is owned by session "
                f"{self._session_id!r}"
            ),
            "stream": self._snapshot_locked(now),
        }

    def _lease_is_expired_locked(self, now: float) -> bool:
        return (
            self._active
            and self._lease_deadline is not None
            and now >= self._lease_deadline
        )

    def _expire_locked(self, now: float) -> None:
        if self._lease_is_expired_locked(now):
            self._clear_locked()

    def _clear_locked(self) -> None:
        self._active = False
        self._session_id = ""
        self._target_labels = ()
        self._lease_deadline = None
        self._next_inference_at = 0.0

    def _snapshot_locked(self, now: float) -> dict[str, Any]:
        remaining = (
            max(0.0, self._lease_deadline - now)
            if self._lease_deadline is not None
            else None
        )
        return {
            "active": self._active,
            "session_id": self._session_id,
            "rate_hz": round(self._rate_hz, 3) if self._active else 0.0,
            "confidence": (
                round(self._confidence, 4) if self._active else None
            ),
            "target_labels": (
                list(self._target_labels) if self._active else []
            ),
            "lease_remaining_sec": (
                round(remaining, 3) if remaining is not None else None
            ),
        }
