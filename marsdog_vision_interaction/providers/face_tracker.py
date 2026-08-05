"""Face tracking (ByteTrack) + throttled SFace recognition.

FaceByteTracker:
  Wraps supervision ByteTrack for face-specific tracking.
  Handles version compatibility, empty detections, tracker_id=-1.

FaceRecognitionThrottle:
  Per-track_id SFace recognition with quality gating and throttling.
  Maintains identity state machine (unverified→confirmed_known/confirmed_unknown).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ── FaceByteTracker ──────────────────────────────────────────────

class FaceByteTracker:
    """ByteTrack wrapper for face tracking.

    Converts YuNet detections to supervision Detections,
    runs ByteTrack, returns tracker_id per detection.

    Handles:
      - Empty detections (no faces)
      - tracker_id is None or -1
      - Different supervision ByteTrack API versions
    """

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 2,
    ) -> None:
        import supervision as sv

        self._lost_track_buffer = lost_track_buffer
        self._min_consecutive_frames = minimum_consecutive_frames

        # supervision 0.29.x ByteTrack API
        self._tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=10,  # matched to observation rate
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def update(
        self, detections_xyxy: np.ndarray, scores: np.ndarray,
    ) -> np.ndarray:
        """Run ByteTrack on face detections.

        Args:
            detections_xyxy: (N, 4) float32 array of [x1, y1, x2, y2] in pixel coords.
            scores: (N,) float32 array of detection scores.

        Returns:
            (N,) int array of tracker_id. -1 means no track assigned.
        """
        import supervision as sv

        N = len(detections_xyxy)
        if N == 0:
            # Feed empty detections to keep tracker state consistent
            empty_xyxy = np.empty((0, 4), dtype=np.float32)
            empty_scores = np.empty((0,), dtype=np.float32)
            empty_class = np.empty((0,), dtype=np.int32)
            dets = sv.Detections(
                xyxy=empty_xyxy,
                confidence=empty_scores,
                class_id=empty_class,
            )
            _ = self._tracker.update_with_detections(dets)
            return np.array([], dtype=np.int32)

        dets = sv.Detections(
            xyxy=detections_xyxy.astype(np.float32),
            confidence=scores.astype(np.float32),
            class_id=np.zeros(N, dtype=np.int32),  # all face
        )

        tracked = self._tracker.update_with_detections(dets)

        ids = tracked.tracker_id
        if ids is None:
            return np.full(N, -1, dtype=np.int32)

        # Replace None with -1
        result = np.array([tid if tid is not None else -1 for tid in ids], dtype=np.int32)
        return result

    def reset(self) -> None:
        """Reset tracker state."""
        import supervision as sv
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=self._lost_track_buffer,
            minimum_matching_threshold=0.8,
            frame_rate=10,
            minimum_consecutive_frames=self._min_consecutive_frames,
        )


# ── Face recognition throttle ───────────────────────────────────

@dataclass
class TrackState:
    """Per-track identity state for throttled recognition."""
    track_id: int
    identity: str = "unknown"
    identity_confidence: float = 0.0
    identity_state: str = "unverified"

    last_recognition_ts: float = 0.0
    last_verified_ts: float = 0.0
    recognition_attempts: int = 0
    consecutive_same_identity: int = 0
    consecutive_unknown: int = 0
    last_seen_ts: float = 0.0

    bbox_xyxy: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    face_score: float = 0.0
    face_quality: float = 0.0

    @property
    def is_known(self) -> bool:
        return self.identity_state in ("candidate_known", "confirmed_known")

    @property
    def is_confirmed(self) -> bool:
        return self.identity_state in ("confirmed_known", "confirmed_unknown")


class FaceRecognitionThrottle:
    """Throttled SFace recognition per track.

    Strategies (configurable intervals):
      - New track: recognize immediately if quality ok
      - Unknown active candidate: retry every ~0.5s
      - Unknown inactive candidate: retry every ~1.5s
      - Known active: reverify every ~3s
      - Known inactive: reverify every ~8s

    Quality gating:
      - face_score >= min_face_score (default 0.85)
      - bbox size >= min_face_size_px (default 40)
      - SFace alignCrop must succeed

    Identity state machine:
      unverified → candidate_known (1st match) → confirmed_known (2nd match)
      unverified → unknown_candidate (1st miss) → confirmed_unknown (4th miss)
    """

    def __init__(
        self,
        face_recognizer: cv2.FaceRecognizerSF,
        min_face_score: float = 0.85,
        min_face_size_px: int = 40,
        unknown_active_retry_sec: float = 0.5,
        unknown_inactive_retry_sec: float = 1.5,
        known_active_reverify_sec: float = 3.0,
        known_inactive_reverify_sec: float = 8.0,
        confirm_known_count: int = 2,
        confirm_unknown_count: int = 4,
        sface_cosine_threshold: float = 0.36,
    ) -> None:
        self._recognizer = face_recognizer
        self._min_face_score = min_face_score
        self._min_face_size_px = min_face_size_px
        self._unknown_active_retry = unknown_active_retry_sec
        self._unknown_inactive_retry = unknown_inactive_retry_sec
        self._known_active_reverify = known_active_reverify_sec
        self._known_inactive_reverify = known_inactive_reverify_sec
        self._confirm_known = confirm_known_count
        self._confirm_unknown = confirm_unknown_count
        self._cosine_threshold = sface_cosine_threshold

        self._tracks: dict[int, TrackState] = {}
        self._enrolled_embeddings: dict[str, np.ndarray] = {}  # name → embedding

    # ── Public API ──────────────────────────────────────────────

    def should_recognize(
        self, track_id: int, face_score: float, bbox_w: int, bbox_h: int,
        is_active: bool, now: float,
    ) -> bool:
        """Check if this face should be recognized now."""
        if track_id < 0:
            return False

        # Quality gate
        if face_score < self._min_face_score:
            return False
        if bbox_w < self._min_face_size_px or bbox_h < self._min_face_size_px:
            return False

        track = self._get_or_create(track_id, now)
        track.face_score = face_score

        # New track or re-appeared after loss
        if track.recognition_attempts == 0:
            return True
        if now - track.last_seen_ts > 3.0:
            return True  # re-appeared

        # Throttle by identity state
        if track.is_confirmed:
            if track.is_known:
                interval = self._known_active_reverify if is_active else self._known_inactive_reverify
            else:
                interval = self._unknown_inactive_retry  # confirmed_unknown, rarely retry
        else:
            if is_active:
                interval = self._unknown_active_retry
            else:
                interval = self._unknown_inactive_retry

        if now - track.last_recognition_ts >= interval:
            return True

        return False

    def update_identity(
        self, track_id: int, identity: str, confidence: float, now: float,
    ) -> None:
        """Update track with recognition result."""
        if track_id < 0:
            return
        track = self._get_or_create(track_id, now)
        track.recognition_attempts += 1
        track.last_recognition_ts = now

        if identity and identity != "unknown":
            # Matched
            if track.identity == identity:
                track.consecutive_same_identity += 1
            else:
                track.consecutive_same_identity = 1
                track.identity = identity
            track.consecutive_unknown = 0
            track.identity_confidence = confidence

            if track.consecutive_same_identity >= self._confirm_known:
                track.identity_state = "confirmed_known"
                track.last_verified_ts = now
            elif track.consecutive_same_identity >= 1:
                track.identity_state = "candidate_known"
            track.identity_confidence = confidence
        else:
            # Not matched
            track.consecutive_unknown += 1
            track.consecutive_same_identity = 0
            if track.consecutive_unknown >= self._confirm_unknown:
                track.identity_state = "confirmed_unknown"
            elif track.consecutive_unknown >= 1:
                if track.identity_state not in ("confirmed_known", "candidate_known"):
                    track.identity_state = "unknown_candidate"

    def mark_seen(self, track_id: int, bbox_xyxy: np.ndarray, now: float) -> None:
        """Update track last-seen timestamp and bbox."""
        if track_id < 0:
            return
        track = self._get_or_create(track_id, now)
        track.last_seen_ts = now
        track.bbox_xyxy = bbox_xyxy.tolist() if hasattr(bbox_xyxy, 'tolist') else list(bbox_xyxy)

    def get_track_state(self, track_id: int) -> TrackState | None:
        return self._tracks.get(track_id)

    def set_enrolled_embeddings(self, enrolled: dict[str, np.ndarray]) -> None:
        self._enrolled_embeddings = enrolled

    @property
    def enrolled_count(self) -> int:
        return len(self._enrolled_embeddings)

    # ── Internal ────────────────────────────────────────────────

    def _get_or_create(self, track_id: int, now: float) -> TrackState:
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackState(track_id=track_id)
            self._tracks[track_id].last_seen_ts = now
        return self._tracks[track_id]

    def cleanup_stale(self, max_age_sec: float = 30.0, now: float = 0.0) -> None:
        """Remove tracks not seen recently."""
        if now == 0.0:
            now = time.time()
        stale = [
            tid for tid, t in self._tracks.items()
            if now - t.last_seen_ts > max_age_sec
        ]
        for tid in stale:
            del self._tracks[tid]
