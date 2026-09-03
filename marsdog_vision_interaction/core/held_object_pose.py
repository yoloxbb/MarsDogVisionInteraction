"""Temporal person-hand-object association for held toy/food poses."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
from typing import Any


TOY_LABELS = frozenset({
    "dog toy ball",
    "dog frisbee toy",
    "dog tug ring toy",
})
FOOD_LABELS = frozenset({
    "dog bowl",
    "dog food can",
    "dog treat bag",
})
HELD_OBJECT_LABELS = tuple(sorted(TOY_LABELS | FOOD_LABELS))

HOLDING_TOY = "holding_toy"
HOLDING_DOG_FOOD = "holding_dog_food"
HOLDING_ACTION_LABELS = {
    HOLDING_TOY: "手持玩具",
    HOLDING_DOG_FOOD: "手持狗粮",
}


@dataclass(frozen=True, slots=True)
class HeldObjectPoseStatus:
    """Current candidate/confirmed result exposed in the visual snapshot."""

    state: str = "inactive"
    action: str = ""
    action_label: str = ""
    candidate_action: str = ""
    object_label: str = ""
    object_track_id: int = -1
    hand_source: str = ""
    association_score: float = 0.0
    wrist_distance_ratio: float | None = None
    evidence_hits: int = 0
    required_hits: int = 2
    last_positive_age_ms: float | None = None
    object_result_sequence: int = 0
    rejection_reason: str = ""
    evaluated_object_label: str = ""
    evaluated_object_confidence: float = 0.0
    pose_object_sync_delta_ms: float | None = None
    valid_wrist_count: int = 0
    evaluated_wrist_distance_ratio: float | None = None
    wrist_distance_threshold_ratio: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Association:
    action: str
    object_label: str
    object_track_id: int
    hand_source: str
    score: float
    wrist_distance_ratio: float


class HeldObjectPoseManager:
    """Confirm that a tracked person is holding a configured object class.

    Object inference normally runs more slowly than the 10 Hz visual state
    publisher. Evidence is therefore counted by object-result sequence rather
    than by repeated visual snapshots.
    """

    def __init__(
        self,
        *,
        min_object_confidence: float = 0.35,
        wrist_distance_ratio: float = 0.16,
        human_bbox_expansion_ratio: float = 0.15,
        required_hits: int = 2,
        confirmation_window_s: float = 1.5,
        hold_s: float = 1.25,
        max_pose_object_sync_delta_s: float = 0.75,
    ) -> None:
        self._min_object_confidence = min(
            1.0, max(0.0, float(min_object_confidence))
        )
        self._wrist_distance_ratio = max(
            0.01, float(wrist_distance_ratio)
        )
        self._human_bbox_expansion_ratio = max(
            0.0, float(human_bbox_expansion_ratio)
        )
        self._required_hits = max(1, int(required_hits))
        self._confirmation_window_s = max(
            0.1, float(confirmation_window_s)
        )
        self._hold_s = max(0.1, float(hold_s))
        self._max_pose_object_sync_delta_s = max(
            0.0, float(max_pose_object_sync_delta_s)
        )
        self.reset()

    def update(
        self,
        *,
        now: float,
        active_target: dict[str, Any],
        hands: list[dict[str, Any]],
        objects: list[dict[str, Any]],
        object_result_sequence: int,
        pose_observation_stamp: float | None = None,
    ) -> HeldObjectPoseStatus:
        target_id = str(active_target.get("target_id", ""))
        if (
            not target_id
            or str(active_target.get("tracking_state", "")) != "tracking"
        ):
            reason = "missing_target" if not target_id else "target_not_tracking"
            self.reset()
            self._last_evaluation["rejection_reason"] = reason
            return self._status(now)
        if target_id != self._target_id:
            target_changed = bool(self._target_id)
            self.reset()
            self._target_id = target_id
            if target_changed:
                # A cached object result that belonged to the previous selected
                # person cannot become evidence for the new target.
                self._last_result_sequence = int(
                    object_result_sequence or 0
                )

        self._prune(now)
        sequence = int(object_result_sequence or 0)
        if sequence > 0 and sequence != self._last_result_sequence:
            if sequence < self._last_result_sequence:
                self._evidence.clear()
                self._intermediate_misses = 0
            self._last_result_sequence = sequence
            association = self._best_association(
                active_target,
                hands,
                objects,
                object_result_sequence=sequence,
                pose_observation_stamp=pose_observation_stamp,
            )
            if association is None:
                # A single 2 Hz detector miss must not erase a recent positive
                # wrist association.  Only distinct positive result sequences
                # count, and the bounded confirmation window still expires old
                # evidence.  Repeated 10 Hz visual snapshots never add hits.
                if self._evidence:
                    self._intermediate_misses += 1
                    if self._intermediate_misses > 1:
                        self._evidence.clear()
                        self._candidate = None
                        self._intermediate_misses = 0
            else:
                if (
                    self._candidate is None
                    or association.action != self._candidate.action
                ):
                    self._evidence.clear()
                self._candidate = association
                self._evidence.append((now, association.action))
                self._intermediate_misses = 0
                self._last_positive_at = now
                if len(self._evidence) >= self._required_hits:
                    self._active_action = association.action
                    self._active_association = association
                    self._active_until = now + self._hold_s
                    self._evidence.clear()

        self._prune(now)
        return self._status(now)

    def reset(self) -> None:
        self._target_id = ""
        self._last_result_sequence = 0
        self._evidence: deque[tuple[float, str]] = deque()
        self._intermediate_misses = 0
        self._candidate: _Association | None = None
        self._active_action = ""
        self._active_association: _Association | None = None
        self._active_until = 0.0
        self._last_positive_at: float | None = None
        self._last_evaluation: dict[str, Any] = {
            "rejection_reason": "not_evaluated",
            "object_result_sequence": 0,
            "evaluated_object_label": "",
            "evaluated_object_confidence": 0.0,
            "pose_object_sync_delta_ms": None,
            "valid_wrist_count": 0,
            "wrist_distance_ratio": None,
        }

    def _prune(self, now: float) -> None:
        while (
            self._evidence
            and now - self._evidence[0][0] > self._confirmation_window_s
        ):
            self._evidence.popleft()
        if not self._evidence and not self._active_action:
            self._candidate = None
            self._intermediate_misses = 0
        if self._active_action and now >= self._active_until:
            self._active_action = ""
            self._active_association = None
            self._candidate = None

    def _status(self, now: float) -> HeldObjectPoseStatus:
        active = bool(self._active_action and now < self._active_until)
        association = self._active_association if active else self._candidate
        candidate_action = association.action if association else ""
        action = self._active_action if active else ""
        age_ms = (
            max(0.0, (now - self._last_positive_at) * 1000.0)
            if self._last_positive_at is not None
            else None
        )
        return HeldObjectPoseStatus(
            state=("confirmed" if active else "candidate" if association else "inactive"),
            action=action,
            action_label=HOLDING_ACTION_LABELS.get(action, ""),
            candidate_action=candidate_action,
            object_label=association.object_label if association else "",
            object_track_id=(association.object_track_id if association else -1),
            hand_source=association.hand_source if association else "",
            association_score=round(association.score, 4) if association else 0.0,
            wrist_distance_ratio=(
                round(association.wrist_distance_ratio, 4)
                if association else None
            ),
            evidence_hits=(
                self._required_hits if active else len(self._evidence)
            ),
            required_hits=self._required_hits,
            last_positive_age_ms=round(age_ms, 1) if age_ms is not None else None,
            object_result_sequence=int(
                self._last_evaluation.get("object_result_sequence", 0)
            ),
            rejection_reason=str(self._last_evaluation.get("rejection_reason", "")),
            evaluated_object_label=str(
                self._last_evaluation.get("evaluated_object_label", "")
            ),
            evaluated_object_confidence=round(float(
                self._last_evaluation.get("evaluated_object_confidence", 0.0)
            ), 4),
            pose_object_sync_delta_ms=(
                round(float(self._last_evaluation["pose_object_sync_delta_ms"]), 1)
                if self._last_evaluation.get("pose_object_sync_delta_ms") is not None
                else None
            ),
            valid_wrist_count=int(
                self._last_evaluation.get("valid_wrist_count", 0)
            ),
            evaluated_wrist_distance_ratio=(
                round(float(self._last_evaluation["wrist_distance_ratio"]), 4)
                if self._last_evaluation.get("wrist_distance_ratio") is not None
                else None
            ),
            wrist_distance_threshold_ratio=round(self._wrist_distance_ratio, 4),
        )

    def _best_association(
        self,
        active_target: dict[str, Any],
        hands: list[dict[str, Any]],
        objects: list[dict[str, Any]],
        *,
        object_result_sequence: int,
        pose_observation_stamp: float | None,
    ) -> _Association | None:
        self._last_evaluation = {
            "rejection_reason": "object_not_detected",
            "object_result_sequence": int(object_result_sequence),
            "evaluated_object_label": "",
            "evaluated_object_confidence": 0.0,
            "pose_object_sync_delta_ms": None,
            "valid_wrist_count": 0,
            "wrist_distance_ratio": None,
        }
        human_bbox = _bbox(active_target)
        if human_bbox is None:
            self._last_evaluation["rejection_reason"] = "invalid_human_bbox"
            return None
        human_diagonal = math.hypot(human_bbox[2], human_bbox[3])
        maximum_distance = max(
            0.025,
            human_diagonal * self._wrist_distance_ratio,
        )
        expanded_human = _expand_bbox(
            human_bbox,
            self._human_bbox_expansion_ratio,
        )
        wrists = [
            (source, point)
            for source, point in _wrist_points(active_target, hands)
            if _point_in_bbox(point, expanded_human)
        ]
        self._last_evaluation["valid_wrist_count"] = len(wrists)
        if not wrists:
            self._last_evaluation["rejection_reason"] = "no_valid_wrist"
            return None

        best: _Association | None = None
        supported_object_seen = False
        for item in objects:
            label = " ".join(
                str(item.get("label", "")).strip().lower().split()
            )
            action = (
                HOLDING_TOY
                if label in TOY_LABELS
                else HOLDING_DOG_FOOD
                if label in FOOD_LABELS
                else ""
            )
            if not action:
                continue
            supported_object_seen = True
            try:
                confidence = float(item.get("confidence", 0.0))
                source_sequence = int(item.get("source_sequence", 0))
                object_track_id = int(
                    item.get("object_track_id", item.get("track_id", -1))
                )
            except (TypeError, ValueError):
                self._last_evaluation["rejection_reason"] = "invalid_object_fields"
                continue
            self._last_evaluation.update({
                "evaluated_object_label": label,
                "evaluated_object_confidence": confidence,
            })
            object_stamp = _positive_float(
                item.get("header", {}).get("stamp")
                if isinstance(item.get("header"), dict)
                else None
            )
            pose_stamp = _positive_float(pose_observation_stamp)
            sync_delta_s = (
                abs(pose_stamp - object_stamp)
                if pose_stamp is not None and object_stamp is not None
                else None
            )
            self._last_evaluation["pose_object_sync_delta_ms"] = (
                sync_delta_s * 1000.0 if sync_delta_s is not None else None
            )
            object_bbox = _bbox(item)
            if object_bbox is None:
                self._last_evaluation["rejection_reason"] = "invalid_object_bbox"
                continue
            if confidence < self._min_object_confidence:
                self._last_evaluation["rejection_reason"] = "low_confidence"
                continue
            if source_sequence != object_result_sequence:
                self._last_evaluation["rejection_reason"] = "stale_object_sequence"
                continue
            if str(item.get("tracking_state", "tracking")) != "tracking":
                self._last_evaluation["rejection_reason"] = "object_track_not_current"
                continue
            if (
                sync_delta_s is not None
                and sync_delta_s > self._max_pose_object_sync_delta_s
            ):
                self._last_evaluation["rejection_reason"] = "timestamp_mismatch"
                continue

            closest_distance_ratio: float | None = None
            for source, wrist in wrists:
                distance = _point_to_bbox_distance(wrist, object_bbox)
                distance_ratio = distance / max(human_diagonal, 1e-6)
                closest_distance_ratio = (
                    distance_ratio
                    if closest_distance_ratio is None
                    else min(closest_distance_ratio, distance_ratio)
                )
                if distance > maximum_distance:
                    continue
                proximity = 1.0 - min(1.0, distance / maximum_distance)
                score = 0.70 * proximity + 0.30 * min(1.0, confidence)
                candidate = _Association(
                    action=action,
                    object_label=label,
                    object_track_id=object_track_id,
                    hand_source=source,
                    score=score,
                    wrist_distance_ratio=distance_ratio,
                )
                if best is None or candidate.score > best.score:
                    best = candidate
            if best is None:
                self._last_evaluation.update({
                    "rejection_reason": "wrist_too_far",
                    "wrist_distance_ratio": closest_distance_ratio,
                })
        if best is not None:
            self._last_evaluation.update({
                "rejection_reason": "",
                "evaluated_object_label": best.object_label,
                "wrist_distance_ratio": best.wrist_distance_ratio,
            })
        elif not supported_object_seen and objects:
            self._last_evaluation["rejection_reason"] = "unsupported_label"
        return best


def _bbox(value: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = value.get("bbox")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raw = [
            value.get("x"),
            value.get("y"),
            value.get("w"),
            value.get("h"),
        ]
    try:
        result = tuple(float(item) for item in raw)
    except (TypeError, ValueError):
        return None
    if len(result) != 4 or not all(math.isfinite(item) for item in result):
        return None
    if result[2] <= 0.0 or result[3] <= 0.0:
        return None
    return result  # type: ignore[return-value]


def _positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) and result > 0.0 else None


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    x, y, width, height = bbox
    dx, dy = width * ratio, height * ratio
    return (x - dx, y - dy, width + 2.0 * dx, height + 2.0 * dy)


def _point_in_bbox(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> bool:
    return (
        bbox[0] <= point[0] <= bbox[0] + bbox[2]
        and bbox[1] <= point[1] <= bbox[1] + bbox[3]
    )


def _point_to_bbox_distance(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> float:
    x, y = point
    left, top, width, height = bbox
    dx = max(left - x, 0.0, x - (left + width))
    dy = max(top - y, 0.0, y - (top + height))
    return math.hypot(dx, dy)


def _wrist_points(
    target: dict[str, Any],
    hands: list[dict[str, Any]],
) -> list[tuple[str, tuple[float, float]]]:
    points: list[tuple[str, tuple[float, float]]] = []
    for keypoint in target.get("keypoints", []):
        if not isinstance(keypoint, dict):
            continue
        try:
            point_id = int(keypoint.get("id", -1))
            confidence = min(
                float(keypoint.get("confidence", 0.0)),
                float(keypoint.get("presence", 1.0)),
            )
            x, y = float(keypoint.get("x")), float(keypoint.get("y"))
        except (TypeError, ValueError):
            continue
        if point_id in (15, 16) and confidence >= 0.35:
            points.append((
                "pose_left_wrist" if point_id == 15 else "pose_right_wrist",
                (x, y),
            ))
    for index, hand in enumerate(hands):
        landmarks = hand.get("landmarks", []) if isinstance(hand, dict) else []
        wrist = None
        for point in landmarks:
            if not isinstance(point, dict):
                continue
            try:
                point_id = int(point.get("id", -1))
            except (TypeError, ValueError):
                continue
            if point_id == 0:
                wrist = point
                break
        if wrist is None:
            continue
        try:
            point = (float(wrist.get("x")), float(wrist.get("y")))
        except (TypeError, ValueError):
            continue
        handedness = str(hand.get("handedness", "")).strip().lower()
        points.append((
            f"hand_{handedness or index}",
            point,
        ))
    return points
