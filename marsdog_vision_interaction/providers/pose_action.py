"""Deterministic pose and hand action recognition for one stable visual target.

The public action keys intentionally preserve the existing MarsDog visual-event
contract.  Recognition itself is provided by the standalone, temporal
``GesturePose`` engine vendored in :mod:`gesture_pose_engine`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from marsdog_vision_interaction.providers.gesture_pose_engine import (
    ACTION_GROUPS,
    ACTION_TRIGGER_PRIORITIES,
    ActionName,
    BehaviorEngine,
    FallDetectionConfig,
    HandLandmarkSet,
    LandmarkFrame,
    PoseLandmarkSet,
)


logger = logging.getLogger(__name__)


_ACTION_LABEL: dict[str, str] = {
    "arm_raise_wave": "手臂高举/挥舞",
    "jump": "跳跃",
    "lean_forward_arms_open": "身体前倾/张开双臂",
    "nodding": "快速点头",
    "clapping": "双手鼓掌",
    "thumbs_up": "点赞手势",
    "hands_on_hips": "双手叉腰",
    "rapid_wave_slap": "快速挥手/拍打",
    "finger_pointing": "用手指点",
    "stomping": "急促跺脚",
    "arms_crossed": "双臂交叉于胸前",
    "head_down_slumped": "低头/肩膀下垂",
    "hands_covering_face": "双手掩面或抱头",
    "body_curled_up": "身体蜷缩",
    "hunched_back": "驼背/弓着背",
    "neutral_stand_sit": "自然站立/端坐",
    "stop_gesture": "停止/别动手势",
    "fallen_down": "跌倒事件",
}

_POSE_ACTIONS: dict[ActionName, str] = {
    ActionName.FALL: "fallen_down",
    ActionName.JUMPING: "jump",
    ActionName.STOMPING: "stomping",
    ActionName.FAST_NOD: "nodding",
    ActionName.LARGE_ARM_SWING: "rapid_wave_slap",
    ActionName.HANDS_ON_HIPS: "hands_on_hips",
    ActionName.ARMS_CROSSED: "arms_crossed",
    ActionName.HEAD_DOWN: "head_down_slumped",
    ActionName.SHOULDERS_SLUMPED: "head_down_slumped",
    ActionName.CURLED_UP: "body_curled_up",
    ActionName.HUNCHED: "hunched_back",
    ActionName.ARMS_RAISED: "arm_raise_wave",
    ActionName.WAVING: "arm_raise_wave",
    ActionName.ARMS_OPEN: "lean_forward_arms_open",
    ActionName.STANDING: "neutral_stand_sit",
    ActionName.SITTING: "neutral_stand_sit",
    ActionName.LOW_MOTION: "neutral_stand_sit",
}

_HAND_ACTIONS: dict[ActionName, str] = {
    ActionName.STOP_GESTURE: "stop_gesture",
    ActionName.CLAPPING: "clapping",
    ActionName.THUMBS_UP: "thumbs_up",
    ActionName.POINTING: "finger_pointing",
    ActionName.FACE_COVERING: "hands_covering_face",
    ActionName.HANDS_ON_HEAD: "hands_covering_face",
}


def get_action_label(key: str) -> str:
    return _ACTION_LABEL.get(key, "")


def get_action_category(key: str) -> str:
    if key in _POSE_ACTIONS.values():
        return "pose"
    if key in _HAND_ACTIONS.values():
        return "hand"
    return ""


def _rounded_optional(value: float | None) -> float | None:
    return round(float(value), 4) if value is not None else None


def _hand_feature_debug(features: Any, motion: float | None) -> dict[str, Any]:
    straightness = [
        _rounded_optional(value)
        for value in features.finger_straightness_degrees
    ]
    return {
        "detected": features.detected,
        "four_fingers_extended": sum(
            bool(features.extended_fingers[index]) for index in (1, 2, 3, 4)
        ),
        "extended_fingers": list(features.extended_fingers),
        "finger_straightness_degrees": straightness,
        "palm_facing_score": _rounded_optional(features.palm_facing_score),
        "palm_scale": _rounded_optional(features.palm_scale),
        "motion_energy": _rounded_optional(motion),
    }


class PoseActionClassifier:
    """Own one temporal behavior engine per visual target track ID.

    ``update`` must be called once for every processed camera frame.  Missing
    landmarks are also forwarded so fall state and smoothing history reset from
    real elapsed time instead of retaining an old action indefinitely.
    """

    def __init__(self, *, window_size: int = 30, track_timeout_sec: float = 3.0) -> None:
        self._window_size = max(10, int(window_size))
        # Never discard the per-target engine before its fall detector has had
        # a chance to bridge the same brief pose occlusion. Otherwise an armed
        # upright person can return lying with a fresh engine and be classified
        # only as static LYING.
        minimum_fall_handoff_sec = FallDetectionConfig().pose_lost_reset_s + 0.1
        self._track_timeout_sec = max(
            minimum_fall_handoff_sec,
            float(track_timeout_sec),
        )
        self._engines: dict[int, BehaviorEngine] = {}
        self._last_seen: dict[int, float] = {}
        self._pose_action = ""
        self._pose_label = ""
        self._hand_action = ""
        self._hand_label = ""
        self._fall_event_triggered = False
        self._diagnostics: dict[str, Any] = {}

    def update(
        self,
        *,
        track_id: int,
        pose_landmarks: PoseLandmarkSet | None,
        left_hand: HandLandmarkSet | None = None,
        right_hand: HandLandmarkSet | None = None,
        face_observed: bool | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        timestamp = time.monotonic() if now is None else float(now)
        self._clear_output()
        self._cleanup(timestamp)
        observed_track_id = track_id
        if track_id <= 0:
            # The target manager can briefly lose the active ID exactly while a
            # person is falling out of the face/body box. Feed that absence into
            # the most recent engine instead of dropping its armed baseline.
            # No landmarks are borrowed from another person; this only advances
            # the most recent detector through its missing-pose timeline.
            if pose_landmarks is not None or not self._last_seen:
                return self.snapshot()
            track_id = max(
                self._last_seen,
                key=lambda candidate: self._last_seen[candidate],
            )

        engine = self._engines.get(track_id)
        if engine is None:
            engine = BehaviorEngine(window_size=self._window_size)
            self._engines[track_id] = engine
        if observed_track_id > 0:
            self._last_seen[track_id] = timestamp

        try:
            result = engine.update(
                LandmarkFrame(
                    monotonic_s=timestamp,
                    timestamp_s=time.time(),
                    pose_landmarks=pose_landmarks,
                    left_hand=left_hand,
                    right_hand=right_hand,
                    face_observed=face_observed,
                )
            )
        except ValueError as exc:
            # A restarted timestamp stream must not inherit temporal votes.
            logger.warning(
                "GesturePose reset for track_id=%s: %s", track_id, exc
            )
            engine.reset()
            result = engine.update(
                LandmarkFrame(
                    monotonic_s=timestamp,
                    timestamp_s=time.time(),
                    pose_landmarks=pose_landmarks,
                    left_hand=left_hand,
                    right_hand=right_hand,
                    face_observed=face_observed,
                )
            )

        self._fall_event_triggered = result.fall_status.event_triggered
        # Fall alert_active is intentionally held by the reference engine so a
        # BEST_EFFORT subscriber has more than one frame in which to observe it.
        if result.fall_status.alert_active:
            self._pose_action = "fallen_down"
            self._pose_label = get_action_label(self._pose_action)

        for action in result.actions:
            if not self._pose_action and action.name in _POSE_ACTIONS:
                self._pose_action = _POSE_ACTIONS[action.name]
                self._pose_label = get_action_label(self._pose_action)
            if not self._hand_action and action.name in _HAND_ACTIONS:
                self._hand_action = _HAND_ACTIONS[action.name]
                self._hand_label = get_action_label(self._hand_action)
            if self._pose_action and self._hand_action:
                break

        temporal = result.features.temporal
        self._diagnostics = {
            "face_observed": face_observed,
            "primary_action": (
                result.primary_action.name.value
                if result.primary_action is not None
                else ""
            ),
            "primary_priority": (
                f"P{int(result.primary_priority)}"
                if result.primary_priority is not None
                else ""
            ),
            "state_hint": (
                result.state_hint.value if result.state_hint is not None else ""
            ),
            "fall_phase": result.fall_status.phase.value,
            "fall_event_triggered": result.fall_status.event_triggered,
            "fall_alert_active": result.fall_status.alert_active,
            "fall_detector": {
                "phase": result.fall_status.phase.value,
                "armed": result.fall_status.armed,
                "lying_score": round(result.fall_status.lying_score, 4),
                "transition_score": round(
                    result.fall_status.transition_score, 4
                ),
                "event_triggered": result.fall_status.event_triggered,
                "alert_active": result.fall_status.alert_active,
                "cooldown_remaining_s": round(
                    result.fall_status.cooldown_remaining_s, 3
                ),
            },
            "hand_features": {
                "left": _hand_feature_debug(
                    result.features.left_hand,
                    temporal.left_hand_motion_energy,
                ),
                "right": _hand_feature_debug(
                    result.features.right_hand,
                    temporal.right_hand_motion_energy,
                ),
            },
            "recognized_actions": [
                {
                    "name": action.name.value,
                    "priority": f"P{int(ACTION_TRIGGER_PRIORITIES[action.name])}",
                    "group": ACTION_GROUPS[action.name].value,
                    "confidence": round(action.confidence, 4),
                    "support_ratio": round(action.support_ratio, 4),
                    "duration_s": round(action.duration_s, 3),
                }
                for action in result.actions
            ],
            "raw_scores": [
                {
                    "name": name.value,
                    "priority": f"P{int(ACTION_TRIGGER_PRIORITIES[name])}",
                    "group": ACTION_GROUPS[name].value,
                    "score": round(score, 4),
                }
                for name, score in result.raw_scores
            ],
            "feature_ms": round(result.feature_ms, 3),
            "recognition_ms": round(result.recognition_ms, 3),
            "temporal_features": {
                "window_frames": temporal.window_frames,
                "window_duration_s": round(temporal.window_duration_s, 3),
                "pose_motion_energy": _rounded_optional(
                    temporal.pose_motion_energy
                ),
                "head_motion_energy": _rounded_optional(
                    temporal.head_motion_energy
                ),
                "head_vertical_motion_energy": _rounded_optional(
                    temporal.head_vertical_motion_energy
                ),
                "head_vertical_direction_changes": (
                    temporal.head_vertical_direction_changes
                ),
                "head_vertical_range_ratio": _rounded_optional(
                    temporal.head_vertical_range_ratio
                ),
                "wrist_distance_ratio": _rounded_optional(
                    temporal.wrist_distance_ratio
                ),
                "wrist_distance_range_ratio": _rounded_optional(
                    temporal.wrist_distance_range_ratio
                ),
                "wrist_distance_closing_speed": _rounded_optional(
                    temporal.wrist_distance_closing_speed
                ),
                "wrist_distance_opening_speed": _rounded_optional(
                    temporal.wrist_distance_opening_speed
                ),
                "wrist_distance_direction_changes": (
                    temporal.wrist_distance_direction_changes
                ),
            },
        }
        return self.snapshot()

    @property
    def pose_action(self) -> tuple[str, str]:
        return self._pose_action, self._pose_label

    @property
    def hand_action(self) -> tuple[str, str]:
        return self._hand_action, self._hand_label

    @property
    def fall_event_triggered(self) -> bool:
        return self._fall_event_triggered

    def is_active(self) -> bool:
        return bool(self._pose_action or self._hand_action)

    def reset_track(self, track_id: int) -> None:
        self._engines.pop(int(track_id), None)
        self._last_seen.pop(int(track_id), None)

    def reset(self) -> None:
        self._engines.clear()
        self._last_seen.clear()
        self._clear_output()

    def snapshot(self) -> dict[str, Any]:
        return {
            "pose_action": self._pose_action,
            "pose_action_label": self._pose_label,
            "hand_action": self._hand_action,
            "hand_action_label": self._hand_label,
            "fall_event_triggered": self._fall_event_triggered,
            **self._diagnostics,
        }

    def _cleanup(self, now: float) -> None:
        stale = [
            track_id
            for track_id, seen_at in self._last_seen.items()
            if now - seen_at > self._track_timeout_sec
        ]
        for track_id in stale:
            self.reset_track(track_id)

    def _clear_output(self) -> None:
        self._pose_action = ""
        self._pose_label = ""
        self._hand_action = ""
        self._hand_label = ""
        self._fall_event_triggered = False
        self._diagnostics = {}
