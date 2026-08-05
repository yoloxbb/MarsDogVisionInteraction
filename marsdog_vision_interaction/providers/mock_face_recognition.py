"""Mock face recognition provider — in-memory match + enrollment.

Matches the FaceRecognitionProvider interface so downstream consumers
can develop against mock and later switch to real SFace ONNX.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from marsdog_vision_interaction.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class MockFaceRecognitionProvider(BaseProvider):
    """Mock face recognition — in-memory embedding store.

    Provides the same API as FaceRecognitionProvider:
      - enroll(face_roi, user_id) → {"success": bool, "user_id": str}
      - recognize(face_roi)       → {"user_id": str, "confidence": float, "matched": bool}

    Mock stores dummy embeddings by user_id. recognize() matches against
    enrolled users (for testing multiple identities).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._match_threshold = float(config.get("match_threshold", 0.5))
        self._enrolled: dict[str, int] = {}  # user_id → dummy hash
        self._lock = threading.Lock()

    @property
    def enrolled_count(self) -> int:
        with self._lock:
            return len(self._enrolled)

    def start(self) -> None:
        try:
            logger.info("MockFaceRecognitionProvider starting")
            self.available = True
            logger.info("MockFaceRecognitionProvider started (mock)")
        except Exception as exc:
            self.available = False
            logger.warning("MockFaceRecognitionProvider start failed: %s", exc)

    def stop(self) -> None:
        self.available = False
        with self._lock:
            self._enrolled.clear()
        logger.info("MockFaceRecognitionProvider stopped")

    # ── Public API ─────────────────────────────────────────────────

    def enroll(self, face_roi: np.ndarray | None = None,
               user_id: str | None = None) -> dict[str, Any]:
        """Enroll a face with a user ID.

        Args:
            face_roi: BGR face image (unused in mock).
            user_id: User label to associate.

        Returns:
            Dict with success and user_id.
        """
        sid = user_id or "user_{:03d}".format(len(self._enrolled) + 1)

        with self._lock:
            self._enrolled[sid] = hash(sid)  # dummy "embedding"

        logger.info("MockFaceRecognition enrolled: id=%s, total=%d", sid, len(self._enrolled))
        return {"success": True, "user_id": sid}

    def recognize(self, face_roi: np.ndarray | None = None) -> dict[str, Any]:
        """Recognize a face against enrolled users.

        Args:
            face_roi: BGR face image (unused in mock).

        Returns:
            Dict with user_id, confidence, and matched flag.
        """
        _ = face_roi

        if not self._enrolled:
            return {"user_id": "unknown", "confidence": 0.0, "matched": False}

        # Mock: always return the first enrolled user
        with self._lock:
            user_id = next(iter(self._enrolled), "unknown")

        if user_id == "unknown":
            return {"user_id": "unknown", "confidence": 0.0, "matched": False}

        logger.debug("MockFaceRecognition recognized: id=%s", user_id)
        return {"user_id": user_id, "confidence": 0.90, "matched": True}
