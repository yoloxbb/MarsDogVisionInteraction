"""Face recognition provider using SFace ONNX model.

Lazy-loaded — model is only initialized on first enroll/recognize call.
Extracts 128-dim embeddings and matches via cosine similarity.

Triggered by:
  /perception/perception_task enroll_face    → extract + store embedding
  /perception/perception_task recognize_face → extract + search enrolled
  wakeup auto-recognition             → recognize largest face in observation
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from marsdog_vision_interaction.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# SFace input size
_FACE_SIZE = (112, 112)
# Normalization: (pixel - 127.5) / 128.0
_FACE_SCALE = 1.0 / 128.0
_FACE_MEAN = (127.5, 127.5, 127.5)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class FaceRecognitionProvider(BaseProvider):
    """Face recognition — SFace embedding extraction + matching.

    Lazy-loads model on first call. Thread-safe for embedding store.

    Attributes:
        _model_path: Path to SFace ONNX model.
        _match_threshold: Cosine similarity threshold for match.
        _net: cv2.dnn.Net (SFace ONNX).
        _enrolled: Dict of user_id → embedding (np.ndarray 128-dim).
        _lock: Protects _enrolled.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self._model_path = config.get("face_recogn_model", "")
        self._match_threshold = float(config.get("match_threshold", 0.5))

        self._net: Any = None  # cv2.dnn.Net
        self._loaded = False
        self._enrolled: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    @property
    def enrolled_count(self) -> int:
        with self._lock:
            return len(self._enrolled)

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        try:
            if not self._model_path:
                logger.info("FaceRecognitionProvider — no model path, lazy-load disabled")
                self.available = True  # Still available for enroll (in-memory)
            else:
                logger.info("FaceRecognitionProvider — lazy-load, model=%s", self._model_path)
                self.available = True
            logger.info("FaceRecognitionProvider started (lazy)")
        except Exception as exc:
            self.available = False
            logger.warning("FaceRecognitionProvider start failed: %s", exc, exc_info=True)

    def stop(self) -> None:
        self._net = None
        self._loaded = False
        self._enrolled.clear()
        self.available = False
        logger.info("FaceRecognitionProvider stopped")

    # ── Lazy model loading ─────────────────────────────────────

    def _ensure_loaded(self) -> bool:
        """Load SFace model if not already loaded. Returns True on success."""
        if self._loaded:
            return self._net is not None

        if not self._model_path:
            logger.warning("FaceRecognitionProvider — no model path configured")
            self._loaded = True
            return False

        try:
            import cv2

            self._net = cv2.dnn.readNetFromONNX(self._model_path)
            self._loaded = True
            logger.info("FaceRecognitionProvider — SFace model loaded: %s", self._model_path)
            return True
        except Exception as exc:
            logger.warning("FaceRecognitionProvider — SFace load failed: %s", exc)
            self._loaded = True
            return False

    # ── Embedding extraction ───────────────────────────────────

    def _extract_embedding(self, face_roi: np.ndarray) -> np.ndarray | None:
        """Extract 128-dim embedding from a face ROI (BGR).

        Args:
            face_roi: BGR face image (any size, will be resized to 112x112).

        Returns:
            128-dim float32 numpy array, or None on failure.
        """
        if not self._ensure_loaded() or self._net is None:
            return None

        import cv2

        try:
            blob = cv2.dnn.blobFromImage(
                face_roi,
                scalefactor=_FACE_SCALE,
                size=_FACE_SIZE,
                mean=_FACE_MEAN,
                swapRB=True,
                crop=False,
            )
            self._net.setInput(blob)
            embedding = self._net.forward()  # [1, 128]
            return embedding.flatten().astype(np.float32)
        except Exception as exc:
            logger.error("Face embedding extraction error: %s", exc, exc_info=True)
            return None

    # ── Public API ─────────────────────────────────────────────

    def enroll(self, face_roi: np.ndarray | None = None,
               user_id: str | None = None) -> dict[str, Any]:
        """Enroll a face with a user ID.

        Args:
            face_roi: BGR face image (cropped from observation).
            user_id: User label to associate with the embedding.

        Returns:
            Dict with success and user_id.
        """
        if face_roi is None:
            logger.warning("FaceRecognition enroll — no face ROI provided")
            return {"success": False, "user_id": user_id or "unknown"}

        sid = user_id or f"user_{len(self._enrolled) + 1:03d}"

        embedding = self._extract_embedding(face_roi)
        if embedding is None:
            return {"success": False, "user_id": sid}

        with self._lock:
            self._enrolled[sid] = embedding

        logger.info("FaceRecognition enrolled: id=%s, total=%d", sid, len(self._enrolled))
        return {"success": True, "user_id": sid}

    def recognize(self, face_roi: np.ndarray | None = None) -> dict[str, Any]:
        """Recognize a face against enrolled users.

        Args:
            face_roi: BGR face image (cropped from observation).

        Returns:
            Dict with user_id, confidence, and matched flag.
        """
        if face_roi is None:
            return {"user_id": "unknown", "confidence": 0.0, "matched": False}

        if not self._enrolled:
            return {"user_id": "unknown", "confidence": 0.0, "matched": False}

        embedding = self._extract_embedding(face_roi)
        if embedding is None:
            return {"user_id": "unknown", "confidence": 0.0, "matched": False}

        best_id = "unknown"
        best_score = 0.0

        with self._lock:
            for uid, stored_emb in self._enrolled.items():
                score = _cosine(embedding, stored_emb)
                if score > best_score:
                    best_score = score
                    best_id = uid

        matched = best_score >= self._match_threshold

        logger.info(
            "FaceRecognition recognized: id=%s score=%.3f matched=%s",
            best_id, best_score, matched,
        )

        return {
            "user_id": best_id if matched else "unknown",
            "confidence": round(float(best_score), 4),
            "matched": matched,
        }
