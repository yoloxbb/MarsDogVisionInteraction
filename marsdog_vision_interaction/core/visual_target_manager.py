"""Visual-only active target selection.

This module intentionally knows nothing about VAD, speakers, ASR or the
interaction state machine. Cross-modal target selection belongs downstream.
"""

from __future__ import annotations

import copy
import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ActiveVisualTarget:
    vision_epoch: str = ""
    target_id: str = ""
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
    # Freshness and expiry must not depend on adjustments to the system clock.
    # ``last_seen_at`` remains the public wall-clock observation time, while
    # this private process-local value drives all safety decisions.
    last_seen_monotonic: float = 0.0
    frame_count: int = 0

    def tracking_state(self, current_timeout: float = 0.35) -> str:
        if not self.last_seen_at or self.confidence <= 0:
            return "lost"
        monotonic_age = (
            max(0.0, time.monotonic() - self.last_seen_monotonic)
            if self.last_seen_monotonic > 0.0 else 0.0
        )
        # Retain a wall-clock fallback for old serialized/test targets which
        # predate ``last_seen_monotonic``.  Taking the maximum is fail-closed
        # if the wall clock jumps forward, while a backward jump cannot make a
        # genuinely old target current again.
        wall_age = max(0.0, time.time() - self.last_seen_at)
        age = max(monotonic_age, wall_age)
        return "tracking" if age <= current_timeout else "temporarily_lost"

    def to_dict(self) -> dict[str, Any]:
        # Compatibility placeholders remain empty until old consumers migrate
        # cross-modal fields to /perception/target_event.
        return {
            "vision_epoch": self.vision_epoch,
            "target_id": self.target_id,
            "target_type": "human",
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
            "detection_confidence": round(self.confidence, 4),
            "face_confidence": round(self.face_confidence, 4),
            "speaker_confidence": 0.0,
            "selection_reason": self.selection_reason,
            "tracking_state": self.tracking_state(),
            "last_seen_age_ms": (
                round(self._age_sec() * 1000.0, 1)
                if self.last_seen_at else -1.0
            ),
            "range_valid": False,
            "distance_m": None,
            "pose_3d": {
                "valid": False,
                "frame_id": "",
                "x": None,
                "y": None,
                "z": None,
            },
        }

    def _age_sec(self) -> float:
        monotonic_age = (
            max(0.0, time.monotonic() - self.last_seen_monotonic)
            if self.last_seen_monotonic > 0.0 else 0.0
        )
        wall_age = (
            max(0.0, time.time() - self.last_seen_at)
            if self.last_seen_at > 0.0 else 0.0
        )
        return max(monotonic_age, wall_age)


class VisualTargetManager:
    """Track every visible human and retain one legacy active target.

    ``human_candidates`` is deliberately policy-neutral: this class associates
    face/body observations and gives every physical track a process-epoch
    identifier, while consumers decide which person is appropriate for a
    social, wake or safety behavior.  ``active_target`` keeps the historical
    known-first/large-target selection for existing consumers.
    """

    def __init__(
        self,
        persistence_timeout: float = 3.0,
        switch_hysteresis: float = 1.0,
        horizontal_fov_deg: float = 69.0,
        vision_epoch: str | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._active = ActiveVisualTarget()
        self._tracks: dict[int, ActiveVisualTarget] = {}
        self._next_track_id = 1
        self._persistence_timeout = persistence_timeout
        self._switch_hysteresis = switch_hysteresis
        self._horizontal_fov_deg = max(1.0, float(horizontal_fov_deg))
        self._vision_epoch = str(vision_epoch or uuid.uuid4().hex)
        self._last_switch_time = 0.0

    @property
    def vision_epoch(self) -> str:
        return self._vision_epoch

    def configure(self, *, horizontal_fov_deg: float | None = None) -> None:
        """Apply non-identity runtime settings without resetting live tracks."""
        if horizontal_fov_deg is None:
            return
        value = float(horizontal_fov_deg)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("horizontal_fov_deg must be finite and > 0")
        with self._lock:
            self._horizontal_fov_deg = value

    def update_vision(
        self,
        humans: list[dict[str, Any]],
        faces: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            now = time.time()
            now_monotonic = time.monotonic()
            self._discard_expired_tracks(now, now_monotonic)
            candidates = self._build_candidates(humans, faces)

            if not candidates:
                if (
                    self._active.last_seen_at
                    and now - self._active.last_seen_at > self._persistence_timeout
                ):
                    self._active = ActiveVisualTarget()
                return

            assignments = self._associate_tracks(candidates, now, now_monotonic)
            visible_tracks: list[tuple[ActiveVisualTarget, dict[str, Any]]] = []
            for index, candidate in enumerate(candidates):
                track_id = assignments.get(index)
                if track_id is None:
                    track_id = self._next_track_id
                    self._next_track_id += 1
                    track = ActiveVisualTarget(
                        vision_epoch=self._vision_epoch,
                        target_id=self._target_id(track_id),
                        track_id=track_id,
                        created_at=now,
                    )
                    self._tracks[track_id] = track
                else:
                    track = self._tracks[track_id]
                self._apply_to_track(track, candidate, now, now_monotonic)
                visible_tracks.append((track, candidate))

            ranked = sorted(
                visible_tracks,
                key=lambda item: (
                    item[1]["priority"],
                    item[1]["area"] * item[1]["confidence"],
                ),
                reverse=True,
            )
            best_track, _ = ranked[0]
            active_track = self._tracks.get(self._active.track_id)
            active_visible = any(
                track.track_id == self._active.track_id
                for track, _ in visible_tracks
            )
            should_switch = (
                self._active.track_id <= 0
                or best_track.track_id == self._active.track_id
                or now - self._last_switch_time >= self._switch_hysteresis
                or (
                    best_track.is_registered
                    and not self._active.is_registered
                )
            )
            if should_switch:
                if best_track.track_id != self._active.track_id:
                    self._last_switch_time = now
                self._active = copy.deepcopy(best_track)
            elif active_visible and active_track is not None:
                self._active = copy.deepcopy(active_track)

    def _build_candidates(
        self,
        humans: list[dict[str, Any]],
        faces: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        human_values: list[dict[str, Any]] = []
        for human in humans:
            if not isinstance(human, dict):
                continue
            try:
                bbox = (
                    float(human.get("x", 0)),
                    float(human.get("y", 0)),
                    float(human.get("w", 0)),
                    float(human.get("h", 0)),
                )
            except (TypeError, ValueError):
                continue
            area = bbox[2] * bbox[3]
            if area < 0.005:
                continue
            human_values.append({
                "human": human,
                "bbox": bbox,
                "area": area,
                "body_center": self._torso_center(
                    human.get("keypoints", []), bbox
                ),
            })

        face_matches = self._associate_faces(human_values, faces)
        candidates: list[dict[str, Any]] = []
        for index, value in enumerate(human_values):
            human = value["human"]
            bbox = value["bbox"]
            area = value["area"]
            face = face_matches.get(index)
            identity = face.get("recognized_user", "") if face else ""
            known = identity not in ("", "unknown")
            face_bbox = (
                float(face.get("x", 0)),
                float(face.get("y", 0)),
                float(face.get("w", 0)),
                float(face.get("h", 0)),
            ) if face else (0.0, 0.0, 0.0, 0.0)
            body_center = value["body_center"]
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
                "face_track_id": int(
                    face.get("track_id", -1) if face else -1
                ),
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
        return candidates

    def get_active_target(self) -> ActiveVisualTarget:
        with self._lock:
            return copy.deepcopy(self._active)

    def get_active_dict(self) -> dict[str, Any]:
        return self.get_active_target().to_dict()

    def get_human_candidates(self) -> list[dict[str, Any]]:
        """Return every non-expired track with explicit freshness metadata."""
        with self._lock:
            self._discard_expired_tracks(time.time(), time.monotonic())
            values = [
                self._candidate_dict(track)
                for track in sorted(
                    self._tracks.values(), key=lambda item: item.track_id
                )
            ]
        return values

    def get_snapshot(self) -> dict[str, Any]:
        """Atomically copy the active target and complete candidate set."""
        with self._lock:
            self._discard_expired_tracks(time.time(), time.monotonic())
            return {
                "vision_epoch": self._vision_epoch,
                "active_target": copy.deepcopy(self._active),
                "human_candidates": [
                    self._candidate_dict(track)
                    for track in sorted(
                        self._tracks.values(), key=lambda item: item.track_id
                    )
                ],
            }

    def _target_id(self, track_id: int) -> str:
        return f"{self._vision_epoch}:human:{track_id}"

    def _discard_expired_tracks(
        self,
        now: float,
        now_monotonic: float | None = None,
    ) -> None:
        current_monotonic = (
            time.monotonic() if now_monotonic is None else now_monotonic
        )
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if (
                not track.last_seen_at
                or (
                    track.last_seen_monotonic > 0.0
                    and current_monotonic - track.last_seen_monotonic
                    > self._persistence_timeout
                )
                or (
                    track.last_seen_monotonic <= 0.0
                    and now - track.last_seen_at > self._persistence_timeout
                )
            )
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def _associate_tracks(
        self,
        candidates: list[dict[str, Any]],
        now: float,
        now_monotonic: float | None = None,
    ) -> dict[int, int]:
        current_monotonic = (
            time.monotonic() if now_monotonic is None else now_monotonic
        )
        assignments: dict[int, int] = {}
        available = {
            track_id
            for track_id, track in self._tracks.items()
            if (
                (
                    track.last_seen_monotonic > 0.0
                    and current_monotonic - track.last_seen_monotonic
                    <= self._persistence_timeout
                )
                or (
                    track.last_seen_monotonic <= 0.0
                    and now - track.last_seen_at <= self._persistence_timeout
                )
            )
        }

        # A stable face-track ID is stronger than any body-box geometry.
        for index, candidate in enumerate(candidates):
            face_track_id = int(candidate.get("face_track_id", -1))
            if face_track_id < 0:
                continue
            matches = [
                track_id
                for track_id in available
                if self._tracks[track_id].face_track_id == face_track_id
            ]
            if matches:
                assignments[index] = matches[0]
                available.remove(matches[0])

        pairs: list[tuple[float, int, int]] = []
        for index, candidate in enumerate(candidates):
            if index in assignments:
                continue
            for track_id in available:
                score = self._track_match_score(
                    self._tracks[track_id], candidate
                )
                if score is not None:
                    pairs.append((score, index, track_id))
        for _, index, track_id in sorted(pairs, reverse=True):
            if index in assignments or track_id not in available:
                continue
            assignments[index] = track_id
            available.remove(track_id)
        return assignments

    def _track_match_score(
        self,
        track: ActiveVisualTarget,
        candidate: dict[str, Any],
    ) -> float | None:
        candidate_face = int(candidate.get("face_track_id", -1))
        face_track_changed = (
            track.face_track_id >= 0
            and candidate_face >= 0
            and track.face_track_id != candidate_face
        )
        candidate_identity = str(candidate.get("identity", "unknown"))
        if (
            track.identity not in ("", "unknown")
            and candidate_identity not in ("", "unknown")
            and track.identity != candidate_identity
        ):
            return None
        iou = self._iou(track.bbox, candidate["bbox"])
        distance = math.hypot(
            track.body_center[0] - candidate["body_center"][0],
            track.body_center[1] - candidate["body_center"][1],
        )
        if iou < 0.1 and distance > 0.15:
            return None
        score = iou * 2.0 + max(0.0, 1.0 - distance / 0.15)
        # YuNet/ByteTrack face IDs are useful positive evidence, but they are
        # not physical-person identities. A face can disappear or get a new ID
        # during a fall while the body geometry still clearly matches. Keeping
        # a small penalty preserves exact-face preference without splitting one
        # person into a fresh visual target and losing the armed fall history.
        if face_track_changed:
            score -= 0.25
        return score

    @staticmethod
    def _apply_to_track(
        track: ActiveVisualTarget,
        candidate: dict[str, Any],
        now: float,
        now_monotonic: float | None = None,
    ) -> None:
        track.bbox = candidate["bbox"]
        track.body_center = candidate["body_center"]
        track.confidence = candidate["confidence"]
        track.pose_state = candidate["pose_state"]
        track.pose_action = candidate["pose_action"]
        track.pose_action_label = candidate["pose_action_label"]
        track.keypoints = copy.deepcopy(candidate["keypoints"])
        track.identity = candidate["identity"]
        track.identity_confidence = candidate["identity_confidence"]
        track.identity_state = candidate["identity_state"]
        track.is_registered = candidate["is_registered"]
        track.face_track_id = candidate["face_track_id"]
        track.face_bbox = candidate["face_bbox"]
        track.face_center = candidate["face_center"]
        track.face_confidence = candidate["face_confidence"]
        track.selection_reason = candidate["reason"]
        track.last_seen_at = now
        track.last_seen_monotonic = (
            time.monotonic() if now_monotonic is None else now_monotonic
        )
        track.frame_count += 1

    def _candidate_dict(self, track: ActiveVisualTarget) -> dict[str, Any]:
        value = track.to_dict()
        center = [
            round(float(track.body_center[0]), 4),
            round(float(track.body_center[1]), 4),
        ]
        value.update({
            "bbox": [round(float(item), 4) for item in track.bbox],
            "center": center,
            "body_center": center,
            "bearing_deg": round(
                (center[0] - 0.5) * self._horizontal_fov_deg, 3
            ),
            "bearing_valid": True,
            "bearing_source": "configured_hfov",
        })
        return value

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

    @classmethod
    def _associate_faces(
        cls,
        humans: list[dict[str, Any]],
        faces: list[dict[str, Any]],
    ) -> dict[int, dict[str, Any]]:
        """Greedily bind each face and body at most once in one frame."""
        pairs: list[tuple[float, int, int]] = []
        for human_index, human in enumerate(humans):
            human_bbox = human["bbox"]
            for face_index, face in enumerate(faces):
                if not isinstance(face, dict):
                    continue
                try:
                    face_bbox = (
                        float(face.get("x", 0)),
                        float(face.get("y", 0)),
                        float(face.get("w", 0)),
                        float(face.get("h", 0)),
                    )
                except (TypeError, ValueError):
                    continue
                coverage = cls._intersection_over_second(
                    human_bbox, face_bbox
                )
                score = coverage + cls._iou(human_bbox, face_bbox)
                if coverage >= 0.5 and score > 0.5:
                    pairs.append((score, human_index, face_index))

        result: dict[int, dict[str, Any]] = {}
        used_faces: set[int] = set()
        for _, human_index, face_index in sorted(pairs, reverse=True):
            if human_index in result or face_index in used_faces:
                continue
            result[human_index] = faces[face_index]
            used_faces.add(face_index)
        return result

    @classmethod
    def _match_face(
        cls,
        human_bbox: tuple[float, float, float, float],
        faces: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Compatibility helper for older callers and focused unit tests."""
        matches = cls._associate_faces(
            [{"bbox": human_bbox}], faces
        )
        return matches.get(0)

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
