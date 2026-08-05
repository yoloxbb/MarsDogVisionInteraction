"""Visual-only active target selection.

This module intentionally knows nothing about VAD, speakers, ASR or the
interaction state machine. Cross-modal target selection belongs downstream.
"""

from __future__ import annotations

import copy
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActiveVisualTarget:
    track_id: int = 0
    face_track_id: int = -1
    identity: str = "unknown"
    identity_confidence: float = 0.0
    identity_state: str = "unverified"
    is_registered: bool = False
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    face_bbox: tuple[float, float, float, float] = (0, 0, 0, 0)
    face_center: tuple[float, float] = (0, 0)
    body_center: tuple[float, float] = (0, 0)
    pose_state: str = "unknown"
    pose_action: str = ""
    pose_action_label: str = ""
    keypoints: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    face_confidence: float = 0.0
    selection_reason: str = ""
    created_at: float = 0.0
    last_seen_at: float = 0.0
    frame_count: int = 0

    def tracking_state(self, current_timeout: float = 0.35) -> str:
        if not self.last_seen_at or self.confidence <= 0:
            return "lost"
        age = max(0.0, time.time() - self.last_seen_at)
        return "tracking" if age <= current_timeout else "temporarily_lost"

    def to_dict(self) -> dict[str, Any]:
        # Compatibility placeholders remain empty until old consumers migrate
        # cross-modal fields to /perception/target_event.
        return {
            "track_id": self.track_id,
            "face_track_id": self.face_track_id,
            "identity": self.identity,
            "identity_confidence": round(self.identity_confidence, 4),
            "identity_state": self.identity_state,
            "speaker_id": "unknown",
            "is_speaking": False,
            "is_registered": self.is_registered,
            "bbox": list(self.bbox),
            "face_bbox": list(self.face_bbox),
            "face_center": list(self.face_center),
            "body_center": list(self.body_center),
            "pose_state": self.pose_state,
            "pose_action": self.pose_action,
            "pose_action_label": self.pose_action_label,
            "keypoints": copy.deepcopy(self.keypoints),
            "confidence": round(self.confidence, 4),
            "face_confidence": round(self.face_confidence, 4),
            "speaker_confidence": 0.0,
            "selection_reason": self.selection_reason,
            "tracking_state": self.tracking_state(),
            "last_seen_age_ms": (
                round(max(0.0, time.time() - self.last_seen_at) * 1000.0, 1)
                if self.last_seen_at else -1.0
            ),
        }


class VisualTargetManager:
    """Select a stable target using identity, size and persistence."""

    def __init__(
        self,
        persistence_timeout: float = 3.0,
        switch_hysteresis: float = 1.0,
    ) -> None:
        self._lock = threading.Lock()
        self._active = ActiveVisualTarget()
        self._next_track_id = 1
        self._persistence_timeout = persistence_timeout
        self._switch_hysteresis = switch_hysteresis
        self._last_switch_time = 0.0

    def update_vision(
        self,
        humans: list[dict[str, Any]],
        faces: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            now = time.time()
            candidates: list[dict[str, Any]] = []
            for human in humans:
                bbox = (
                    float(human.get("x", 0)),
                    float(human.get("y", 0)),
                    float(human.get("w", 0)),
                    float(human.get("h", 0)),
                )
                area = bbox[2] * bbox[3]
                if area < 0.005:
                    continue
                face = self._match_face(bbox, faces)
                identity = face.get("recognized_user", "") if face else ""
                known = identity not in ("", "unknown")
                face_bbox = (
                    float(face.get("x", 0)),
                    float(face.get("y", 0)),
                    float(face.get("w", 0)),
                    float(face.get("h", 0)),
                ) if face else (0.0, 0.0, 0.0, 0.0)
                body_center = self._torso_center(
                    human.get("keypoints", []), bbox
                )
                candidates.append({
                    "bbox": bbox,
                    "area": area,
                    # A torso centroid is much less sensitive to arm/leg
                    # motion than the centre of the full landmark bbox.
                    "body_center": body_center,
                    "confidence": float(human.get("confidence", 0)),
                    "pose_state": human.get("pose_state", "unknown"),
                    "pose_action": human.get("pose_action", ""),
                    "pose_action_label": human.get("pose_action_label", ""),
                    "keypoints": human.get("keypoints", []),
                    "identity": identity if known else "unknown",
                    "identity_confidence": float(
                        face.get("identity_confidence", 0) if face else 0
                    ),
                    "identity_state": (
                        face.get("identity_state", "unverified")
                        if face else "unverified"
                    ),
                    "is_registered": known,
                    "face_track_id": int(face.get("track_id", -1) if face else -1),
                    "face_bbox": face_bbox,
                    "face_center": (
                        face_bbox[0] + face_bbox[2] / 2,
                        face_bbox[1] + face_bbox[3] / 2,
                    ) if face else (0.0, 0.0),
                    "face_confidence": float(
                        face.get("confidence", 0) if face else 0
                    ),
                    "priority": 2 if known else 1,
                    "reason": (
                        f"identity ({identity})" if known
                        else "largest visual target"
                    ),
                })

            if not candidates:
                if (
                    self._active.last_seen_at
                    and now - self._active.last_seen_at > self._persistence_timeout
                ):
                    self._active = ActiveVisualTarget()
                return

            candidates.sort(
                key=lambda item: (
                    item["priority"],
                    item["area"] * item["confidence"],
                ),
                reverse=True,
            )
            best = candidates[0]
            same = self._same_target(best)
            if (
                same
                or not self._active.last_seen_at
                or now - self._last_switch_time >= self._switch_hysteresis
                or best["priority"] > (2 if self._active.is_registered else 1)
            ):
                self._apply(best, now, same)

    def get_active_target(self) -> ActiveVisualTarget:
        with self._lock:
            return copy.deepcopy(self._active)

    def get_active_dict(self) -> dict[str, Any]:
        return self.get_active_target().to_dict()

    def _apply(self, candidate: dict[str, Any], now: float, same: bool) -> None:
        if not same:
            self._active.track_id = self._next_track_id
            self._next_track_id += 1
            self._active.created_at = now
            self._last_switch_time = now
        self._active.bbox = candidate["bbox"]
        self._active.body_center = candidate["body_center"]
        self._active.confidence = candidate["confidence"]
        self._active.pose_state = candidate["pose_state"]
        self._active.pose_action = candidate["pose_action"]
        self._active.pose_action_label = candidate["pose_action_label"]
        self._active.keypoints = copy.deepcopy(candidate["keypoints"])
        self._active.identity = candidate["identity"]
        self._active.identity_confidence = candidate["identity_confidence"]
        self._active.identity_state = candidate["identity_state"]
        self._active.is_registered = candidate["is_registered"]
        self._active.face_track_id = candidate["face_track_id"]
        self._active.face_bbox = candidate["face_bbox"]
        self._active.face_center = candidate["face_center"]
        self._active.face_confidence = candidate["face_confidence"]
        self._active.selection_reason = candidate["reason"]
        self._active.last_seen_at = now
        self._active.frame_count += 1

    def _same_target(self, candidate: dict[str, Any]) -> bool:
        """Associate pose detections despite limb-driven bbox changes."""
        active_face = self._active.face_track_id
        candidate_face = int(candidate.get("face_track_id", -1))
        if active_face >= 0 and candidate_face == active_face:
            return True
        if self._iou(self._active.bbox, candidate["bbox"]) >= 0.15:
            return True
        if self._active.last_seen_at:
            ax, ay = self._active.body_center
            bx, by = candidate["body_center"]
            return math.hypot(ax - bx, ay - by) <= 0.15
        return False

    @staticmethod
    def _torso_center(
        keypoints: Any,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float]:
        """Return the visible shoulder/hip centroid, or bbox centre."""
        torso: list[tuple[float, float]] = []
        if isinstance(keypoints, list):
            for point in keypoints:
                if not isinstance(point, dict):
                    continue
                try:
                    point_id = int(point.get("id", -1))
                    confidence = float(point.get("confidence", 0.0))
                    x = float(point.get("x", 0.0))
                    y = float(point.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                if (
                    point_id in (11, 12, 23, 24)
                    and confidence >= 0.3
                    and 0.0 <= x <= 1.0
                    and 0.0 <= y <= 1.0
                ):
                    torso.append((x, y))
        if len(torso) >= 2:
            return (
                sum(point[0] for point in torso) / len(torso),
                sum(point[1] for point in torso) / len(torso),
            )
        return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)

    @staticmethod
    def _match_face(
        human_bbox: tuple[float, float, float, float],
        faces: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        best: dict[str, Any] | None = None
        best_score = 0.5
        for face in faces:
            face_bbox = (
                float(face.get("x", 0)),
                float(face.get("y", 0)),
                float(face.get("w", 0)),
                float(face.get("h", 0)),
            )
            coverage = VisualTargetManager._intersection_over_second(
                human_bbox, face_bbox
            )
            score = coverage + VisualTargetManager._iou(human_bbox, face_bbox)
            if coverage >= 0.5 and score > best_score:
                best = face
                best_score = score
        return best

    @staticmethod
    def _intersection_over_second(
        container: tuple[float, float, float, float],
        item: tuple[float, float, float, float],
    ) -> float:
        if min(container[2], container[3], item[2], item[3]) <= 0:
            return 0.0
        ax1, ay1 = container[0], container[1]
        ax2, ay2 = ax1 + container[2], ay1 + container[3]
        bx1, by1 = item[0], item[1]
        bx2, by2 = bx1 + item[2], by1 + item[3]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return ((ix2 - ix1) * (iy2 - iy1)) / (item[2] * item[3])

    @staticmethod
    def _iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        if min(first[2], first[3], second[2], second[3]) <= 0:
            return 0.0
        ax1, ay1, ax2, ay2 = (
            first[0], first[1], first[0] + first[2], first[1] + first[3]
        )
        bx1, by1, bx2, by2 = (
            second[0], second[1],
            second[0] + second[2], second[1] + second[3],
        )
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = first[2] * first[3] + second[2] * second[3] - intersection
        return intersection / union if union > 0 else 0.0


# Compatibility alias used by the migrated stereo fusion provider.
TargetManager = VisualTargetManager
