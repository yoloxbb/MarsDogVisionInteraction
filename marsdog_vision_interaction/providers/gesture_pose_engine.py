"""Standalone reference implementation of the GesturePose behavior engine.

Copy this file into another robot project when the current repository must not
become a runtime dependency.  The module accepts MediaPipe-compatible landmark
values but does not import MediaPipe, OpenCV, the camera pipeline, or src.*.

One BehaviorEngine instance owns temporal state for one tracked person.  Create
one instance per track ID, call update() from a single worker thread, and call
reset() when that person leaves or the timestamp stream restarts.

Coordinate contract:
- Pose has exactly 33 landmarks; each hand has exactly 21 landmarks.
- x/y are normalized image coordinates.
- MediaPipe-style z is expected: smaller/more negative is nearer the camera.
- monotonic_s must strictly increase.
- Mirroring belongs to display code and must not swap semantic handedness.

This is a reference port of the rule engine.  Keep its default thresholds
unchanged during integration; tune or calibrate only after golden-output tests
match the source project.

Minimal use after constructing PoseLandmark/HandLandmark tuples::

    engine = BehaviorEngine()  # one instance per tracked person
    result = engine.update(
        LandmarkFrame(
            monotonic_s=time.monotonic(),
            pose_landmarks=pose,
            left_hand=left_hand,
            right_hand=right_hand,
        )
    )
    if result.fall_status.event_triggered:
        robot_alert_manager.publish("person_fall")
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)


# ==============================================================================
# LANDMARK AND PIPELINE DATA CONTRACTS
# ==============================================================================

ImageArray = NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class PoseLandmark:
    """One normalized pose landmark returned by MediaPipe."""

    x: float
    y: float
    z: float
    visibility: float
    presence: float


@dataclass(frozen=True, slots=True)
class HandLandmark:
    """One normalized hand landmark returned by MediaPipe."""

    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


PoseLandmarkSet = tuple[PoseLandmark, ...]
HandLandmarkSet = tuple[HandLandmark, ...]


@dataclass(slots=True)
class FrameData:
    """Analysis data retained for every frame that completes inference."""

    timestamp: float
    pose_landmarks: PoseLandmarkSet | None
    left_hand: HandLandmarkSet | None
    right_hand: HandLandmarkSet | None
    fps: float
    # ``False`` means an independent face detector explicitly saw no face.
    # ``None`` keeps the standalone engine usable without a face detector.
    face_observed: bool | None = None


@dataclass(frozen=True, slots=True)
class CapturedFrame:
    """A camera frame with wall-clock and monotonic timestamps."""

    sequence: int
    timestamp: float
    monotonic_ns: int
    image: ImageArray


@dataclass(slots=True)
class InferenceFrame:
    """A frame paired with MediaPipe output and inference timing."""

    captured: CapturedFrame
    data: FrameData
    inference_ms: float


@dataclass(frozen=True, slots=True)
class DisplayFrame:
    """A fully rendered frame ready for the UI event loop."""

    image: ImageArray
    data: FrameData
    inference_ms: float
    feature_ms: float
    recognition_ms: float
    actions: tuple[str, ...]
    candidates: tuple[str, ...]
    end_to_end_ms: float


# ==============================================================================
# FEATURE DATA CONTRACTS
# ==============================================================================


@dataclass(frozen=True, slots=True)
class PoseFeatures:
    visible_ratio: float
    shoulder_width: float | None
    torso_length: float | None
    body_height: float | None
    shoulder_slope_degrees: float | None
    torso_lean_degrees: float | None
    head_drop_ratio: float | None
    nose_ear_vertical_ratio: float | None
    arms_span_ratio: float | None
    elbows_span_ratio: float | None
    left_elbow_angle_degrees: float | None
    right_elbow_angle_degrees: float | None
    left_elbow_angle_3d_degrees: float | None
    right_elbow_angle_3d_degrees: float | None
    left_knee_angle_degrees: float | None
    right_knee_angle_degrees: float | None
    left_knee_angle_3d_degrees: float | None
    right_knee_angle_3d_degrees: float | None
    left_thigh_vertical_degrees: float | None
    right_thigh_vertical_degrees: float | None
    left_thigh_depth_ratio: float | None
    right_thigh_depth_ratio: float | None
    left_upper_arm_elevation_degrees: float | None
    right_upper_arm_elevation_degrees: float | None
    left_arm_elevation_degrees: float | None
    right_arm_elevation_degrees: float | None
    left_wrist_above_shoulder: bool | None
    right_wrist_above_shoulder: bool | None


@dataclass(frozen=True, slots=True)
class HandFeatures:
    detected: bool
    center_x: float | None
    center_y: float | None
    palm_scale: float | None
    openness_ratio: float | None
    pinch_distance_ratio: float | None
    finger_spread_ratio: float | None
    palm_facing_score: float | None
    finger_straightness_degrees: tuple[
        float | None,
        float | None,
        float | None,
        float | None,
        float | None,
    ]
    extended_fingers: tuple[bool, bool, bool, bool, bool]
    extended_finger_count: int


@dataclass(frozen=True, slots=True)
class TemporalFeatures:
    window_frames: int
    window_duration_s: float
    pose_motion_energy: float | None
    pose_acceleration_energy: float | None
    pose_change: float | None
    left_hand_motion_energy: float | None
    right_hand_motion_energy: float | None
    max_wrist_speed: float | None
    wrist_motion_energy: float | None
    wrist_motion_direction_changes: int
    elbow_motion_energy: float | None
    shoulder_sway_energy: float | None
    head_motion_energy: float | None
    head_vertical_motion_energy: float | None
    head_vertical_direction_changes: int
    head_vertical_range_ratio: float | None
    max_ankle_speed: float | None
    wrist_distance_ratio: float | None
    wrist_distance_min_ratio: float | None
    wrist_distance_max_ratio: float | None
    wrist_distance_range_ratio: float | None
    wrist_distance_motion_energy: float | None
    wrist_distance_closing_speed: float | None
    wrist_distance_opening_speed: float | None
    wrist_distance_direction_changes: int
    hip_vertical_velocity: float | None
    stillness_duration_s: float


@dataclass(frozen=True, slots=True)
class FrameFeatures:
    pose: PoseFeatures
    left_hand: HandFeatures
    right_hand: HandFeatures
    temporal: TemporalFeatures


@dataclass(frozen=True, slots=True)
class TemporalLandmarkSample:
    sequence: int
    monotonic_s: float
    data: FrameData


@dataclass(frozen=True, slots=True)
class AnalyzedFrame:
    inference: InferenceFrame
    features: FrameFeatures
    feature_ms: float


# ==============================================================================
# SINGLE-FRAME POSE FEATURES
# ==============================================================================

_EPSILON = 1e-6


def _pose_visible(point: PoseLandmark, threshold: float) -> bool:
    return point.visibility >= threshold and point.presence >= threshold


def _pose_point(
    landmarks: PoseLandmarkSet | None, index: int, threshold: float
) -> PoseLandmark | None:
    if landmarks is None or index >= len(landmarks):
        return None
    point = landmarks[index]
    return point if _pose_visible(point, threshold) else None


def _pose_distance(a: PoseLandmark, b: PoseLandmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _pose_midpoint(a: PoseLandmark, b: PoseLandmark) -> tuple[float, float]:
    return ((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def _pose_angle(
    a: PoseLandmark | None, b: PoseLandmark | None, c: PoseLandmark | None
) -> float | None:
    if a is None or b is None or c is None:
        return None
    first = (a.x - b.x, a.y - b.y)
    second = (c.x - b.x, c.y - b.y)
    denominator = math.hypot(*first) * math.hypot(*second)
    if denominator <= _EPSILON:
        return None
    cosine = max(-1.0, min(1.0, (first[0] * second[0] + first[1] * second[1]) / denominator))
    return math.degrees(math.acos(cosine))


def _pose_angle_3d(
    a: PoseLandmark | None,
    b: PoseLandmark | None,
    c: PoseLandmark | None,
) -> float | None:
    if a is None or b is None or c is None:
        return None
    first = (a.x - b.x, a.y - b.y, a.z - b.z)
    second = (c.x - b.x, c.y - b.y, c.z - b.z)
    denominator = math.sqrt(sum(value * value for value in first)) * math.sqrt(
        sum(value * value for value in second)
    )
    if denominator <= _EPSILON:
        return None
    cosine = max(
        -1.0,
        min(
            1.0,
            sum(
                first_value * second_value
                for first_value, second_value in zip(first, second, strict=True)
            )
            / denominator,
        ),
    )
    return math.degrees(math.acos(cosine))


def _pose_depth_ratio(
    start: PoseLandmark | None,
    end: PoseLandmark | None,
    scale: float | None,
) -> float | None:
    if start is None or end is None or scale is None or scale <= _EPSILON:
        return None
    return abs(end.z - start.z) / scale


def _pose_elevation(shoulder: PoseLandmark | None, wrist: PoseLandmark | None) -> float | None:
    if shoulder is None or wrist is None:
        return None
    dx = wrist.x - shoulder.x
    dy = wrist.y - shoulder.y
    if math.hypot(dx, dy) <= _EPSILON:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, -dy / math.hypot(dx, dy)))))


def _pose_vertical_down_angle(start: PoseLandmark | None, end: PoseLandmark | None) -> float | None:
    if start is None or end is None:
        return None
    dx = end.x - start.x
    dy = end.y - start.y
    length = math.hypot(dx, dy)
    if length <= _EPSILON:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, dy / length))))


def extract_pose_features(
    landmarks: PoseLandmarkSet | None, visibility_threshold: float = 0.5
) -> PoseFeatures:
    """Extract normalized, scale-aware pose geometry from one frame."""

    visible_ratio = (
        sum(_pose_visible(point, visibility_threshold) for point in landmarks) / len(landmarks)
        if landmarks
        else 0.0
    )
    nose = _pose_point(landmarks, 0, visibility_threshold)
    left_ear = _pose_point(landmarks, 7, visibility_threshold)
    right_ear = _pose_point(landmarks, 8, visibility_threshold)
    left_shoulder = _pose_point(landmarks, 11, visibility_threshold)
    right_shoulder = _pose_point(landmarks, 12, visibility_threshold)
    left_elbow = _pose_point(landmarks, 13, visibility_threshold)
    right_elbow = _pose_point(landmarks, 14, visibility_threshold)
    left_wrist = _pose_point(landmarks, 15, visibility_threshold)
    right_wrist = _pose_point(landmarks, 16, visibility_threshold)
    leg_visibility_threshold = min(visibility_threshold, 0.4)
    left_hip = _pose_point(landmarks, 23, leg_visibility_threshold)
    right_hip = _pose_point(landmarks, 24, leg_visibility_threshold)
    left_knee = _pose_point(landmarks, 25, leg_visibility_threshold)
    right_knee = _pose_point(landmarks, 26, leg_visibility_threshold)
    left_ankle = _pose_point(landmarks, 27, leg_visibility_threshold)
    right_ankle = _pose_point(landmarks, 28, leg_visibility_threshold)

    shoulder_width = (
        _pose_distance(left_shoulder, right_shoulder)
        if left_shoulder is not None and right_shoulder is not None
        else None
    )
    shoulder_mid = (
        _pose_midpoint(left_shoulder, right_shoulder)
        if left_shoulder is not None and right_shoulder is not None
        else None
    )
    hip_mid = (
        _pose_midpoint(left_hip, right_hip)
        if left_hip is not None and right_hip is not None
        else None
    )
    torso_length = (
        math.dist(shoulder_mid, hip_mid)
        if shoulder_mid is not None and hip_mid is not None
        else None
    )
    visible_points = [
        point for point in landmarks or () if _pose_visible(point, visibility_threshold)
    ]
    body_height = (
        max(point.y for point in visible_points) - min(point.y for point in visible_points)
        if visible_points
        else None
    )
    shoulder_slope = (
        math.degrees(
            math.atan2(right_shoulder.y - left_shoulder.y, right_shoulder.x - left_shoulder.x)
        )
        if left_shoulder is not None and right_shoulder is not None
        else None
    )
    torso_lean = (
        math.degrees(math.atan2(hip_mid[0] - shoulder_mid[0], hip_mid[1] - shoulder_mid[1]))
        if shoulder_mid is not None and hip_mid is not None
        else None
    )
    head_drop_ratio = (
        (nose.y - shoulder_mid[1]) / torso_length
        if nose is not None
        and shoulder_mid is not None
        and torso_length
        and torso_length > _EPSILON
        else None
    )
    ear_mid = (
        _pose_midpoint(left_ear, right_ear)
        if left_ear is not None and right_ear is not None
        else None
    )
    nose_ear_vertical_ratio = (
        (nose.y - ear_mid[1]) / shoulder_width
        if nose is not None and ear_mid is not None and shoulder_width and shoulder_width > _EPSILON
        else None
    )
    arms_span_ratio = (
        _pose_distance(left_wrist, right_wrist) / shoulder_width
        if left_wrist is not None
        and right_wrist is not None
        and shoulder_width
        and shoulder_width > _EPSILON
        else None
    )
    elbows_span_ratio = (
        _pose_distance(left_elbow, right_elbow) / shoulder_width
        if left_elbow is not None
        and right_elbow is not None
        and shoulder_width
        and shoulder_width > _EPSILON
        else None
    )

    return PoseFeatures(
        visible_ratio=visible_ratio,
        shoulder_width=shoulder_width,
        torso_length=torso_length,
        body_height=body_height,
        shoulder_slope_degrees=shoulder_slope,
        torso_lean_degrees=torso_lean,
        head_drop_ratio=head_drop_ratio,
        nose_ear_vertical_ratio=nose_ear_vertical_ratio,
        arms_span_ratio=arms_span_ratio,
        elbows_span_ratio=elbows_span_ratio,
        left_elbow_angle_degrees=_pose_angle(left_shoulder, left_elbow, left_wrist),
        right_elbow_angle_degrees=_pose_angle(right_shoulder, right_elbow, right_wrist),
        left_elbow_angle_3d_degrees=_pose_angle_3d(left_shoulder, left_elbow, left_wrist),
        right_elbow_angle_3d_degrees=_pose_angle_3d(right_shoulder, right_elbow, right_wrist),
        left_knee_angle_degrees=_pose_angle(left_hip, left_knee, left_ankle),
        right_knee_angle_degrees=_pose_angle(right_hip, right_knee, right_ankle),
        left_knee_angle_3d_degrees=_pose_angle_3d(left_hip, left_knee, left_ankle),
        right_knee_angle_3d_degrees=_pose_angle_3d(right_hip, right_knee, right_ankle),
        left_thigh_vertical_degrees=_pose_vertical_down_angle(left_hip, left_knee),
        right_thigh_vertical_degrees=_pose_vertical_down_angle(right_hip, right_knee),
        left_thigh_depth_ratio=_pose_depth_ratio(left_hip, left_knee, torso_length),
        right_thigh_depth_ratio=_pose_depth_ratio(right_hip, right_knee, torso_length),
        left_upper_arm_elevation_degrees=_pose_elevation(left_shoulder, left_elbow),
        right_upper_arm_elevation_degrees=_pose_elevation(right_shoulder, right_elbow),
        left_arm_elevation_degrees=_pose_elevation(left_shoulder, left_wrist),
        right_arm_elevation_degrees=_pose_elevation(right_shoulder, right_wrist),
        left_wrist_above_shoulder=(left_wrist.y < left_shoulder.y)
        if left_wrist is not None and left_shoulder is not None
        else None,
        right_wrist_above_shoulder=(right_wrist.y < right_shoulder.y)
        if right_wrist is not None and right_shoulder is not None
        else None,
    )


# ==============================================================================
# SINGLE-FRAME HAND FEATURES
# ==============================================================================

_EMPTY_EXTENSIONS = (False, False, False, False, False)
_TIP_INDICES = (4, 8, 12, 16, 20)
_JOINT_INDICES = (3, 6, 10, 14, 18)
_EPSILON = 1e-6
_FINGER_CHAINS = (
    (1, 2, 3, 4),
    (5, 6, 7, 8),
    (9, 10, 11, 12),
    (13, 14, 15, 16),
    (17, 18, 19, 20),
)


def _hand_distance(a: HandLandmark, b: HandLandmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _hand_angle_3d(a: HandLandmark, b: HandLandmark, c: HandLandmark) -> float | None:
    first = (a.x - b.x, a.y - b.y, a.z - b.z)
    second = (c.x - b.x, c.y - b.y, c.z - b.z)
    denominator = math.sqrt(sum(value * value for value in first)) * math.sqrt(
        sum(value * value for value in second)
    )
    if denominator <= _EPSILON:
        return None
    cosine = max(
        -1.0,
        min(
            1.0,
            sum(
                first_value * second_value
                for first_value, second_value in zip(first, second, strict=True)
            )
            / denominator,
        ),
    )
    return math.degrees(math.acos(cosine))


def _hand_finger_straightness(
    landmarks: HandLandmarkSet,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    values: list[float | None] = []
    for mcp, pip, dip, tip in _FINGER_CHAINS:
        joint_angles = (
            _hand_angle_3d(landmarks[mcp], landmarks[pip], landmarks[dip]),
            _hand_angle_3d(landmarks[pip], landmarks[dip], landmarks[tip]),
        )
        present = [angle for angle in joint_angles if angle is not None]
        values.append(min(present) if present else None)
    return tuple(values)  # type: ignore[return-value]


def _hand_palm_facing_score(landmarks: HandLandmarkSet) -> float:
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    first = (
        index_mcp.x - wrist.x,
        index_mcp.y - wrist.y,
        index_mcp.z - wrist.z,
    )
    second = (
        pinky_mcp.x - wrist.x,
        pinky_mcp.y - wrist.y,
        pinky_mcp.z - wrist.z,
    )
    normal = (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )
    magnitude = math.sqrt(sum(value * value for value in normal))
    return abs(normal[2]) / magnitude if magnitude > _EPSILON else 0.0


def extract_hand_features(landmarks: HandLandmarkSet | None) -> HandFeatures:
    """Extract scale-normalized openness, pinch, and finger-extension features."""

    if landmarks is None or len(landmarks) < 21:
        return HandFeatures(
            detected=False,
            center_x=None,
            center_y=None,
            palm_scale=None,
            openness_ratio=None,
            pinch_distance_ratio=None,
            finger_spread_ratio=None,
            palm_facing_score=None,
            finger_straightness_degrees=(None, None, None, None, None),
            extended_fingers=_EMPTY_EXTENSIONS,
            extended_finger_count=0,
        )

    wrist = landmarks[0]
    palm_width = _hand_distance(landmarks[5], landmarks[17])
    palm_height = _hand_distance(wrist, landmarks[9])
    palm_scale = max((palm_width + palm_height) / 2.0, _EPSILON)
    extensions = tuple(
        _hand_distance(wrist, landmarks[tip]) > _hand_distance(wrist, landmarks[joint]) * 1.12
        for tip, joint in zip(_TIP_INDICES, _JOINT_INDICES, strict=True)
    )
    openness = sum(_hand_distance(wrist, landmarks[index]) for index in _TIP_INDICES)
    openness /= len(_TIP_INDICES) * palm_scale
    adjacent_tip_distances = (
        _hand_distance(landmarks[8], landmarks[12]),
        _hand_distance(landmarks[12], landmarks[16]),
        _hand_distance(landmarks[16], landmarks[20]),
    )

    return HandFeatures(
        detected=True,
        center_x=sum(point.x for point in landmarks) / len(landmarks),
        center_y=sum(point.y for point in landmarks) / len(landmarks),
        palm_scale=palm_scale,
        openness_ratio=openness,
        pinch_distance_ratio=_hand_distance(landmarks[4], landmarks[8]) / palm_scale,
        finger_spread_ratio=sum(adjacent_tip_distances) / len(adjacent_tip_distances) / palm_scale,
        palm_facing_score=_hand_palm_facing_score(landmarks),
        finger_straightness_degrees=_hand_finger_straightness(landmarks),
        extended_fingers=extensions,
        extended_finger_count=sum(extensions),
    )


# ==============================================================================
# TEMPORAL FEATURES
# ==============================================================================

_EPSILON = 1e-6
_STILLNESS_THRESHOLD = 0.20
_CLAP_WINDOW_SAMPLES = 18


class _TemporalNormalizedPoint(Protocol):
    x: float
    y: float


def _temporal_distance_xy(
    first: _TemporalNormalizedPoint, second: _TemporalNormalizedPoint
) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _temporal_pose_scale(landmarks: PoseLandmarkSet | None) -> float | None:
    if landmarks is None or len(landmarks) < 25:
        return None
    shoulder_width = _temporal_distance_xy(landmarks[11], landmarks[12])
    shoulder_mid = (
        (landmarks[11].x + landmarks[12].x) / 2.0,
        (landmarks[11].y + landmarks[12].y) / 2.0,
    )
    hip_mid = (
        (landmarks[23].x + landmarks[24].x) / 2.0,
        (landmarks[23].y + landmarks[24].y) / 2.0,
    )
    torso_length = math.dist(shoulder_mid, hip_mid)
    scale = (shoulder_width + torso_length) / 2.0
    return scale if scale > _EPSILON else None


def _temporal_hand_scale(landmarks: HandLandmarkSet | None) -> float | None:
    if landmarks is None or len(landmarks) < 18:
        return None
    scale = (
        _temporal_distance_xy(landmarks[5], landmarks[17])
        + _temporal_distance_xy(landmarks[0], landmarks[9])
    ) / 2.0
    return scale if scale > _EPSILON else None


def _temporal_pose_pair_motion(
    previous: PoseLandmarkSet | None,
    current: PoseLandmarkSet | None,
    delta_s: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    if previous is None or current is None or delta_s <= _EPSILON:
        return None, None, None, None
    scale_values = [
        value for value in (_temporal_pose_scale(previous), _temporal_pose_scale(current)) if value
    ]
    if not scale_values:
        return None, None, None, None
    scale = sum(scale_values) / len(scale_values)
    speeds: list[float] = []
    for before, after in zip(previous, current, strict=False):
        if min(before.visibility, before.presence, after.visibility, after.presence) < 0.5:
            continue
        speed = _temporal_distance_xy(before, after) / scale / delta_s
        speeds.append(speed)
    wrist_speeds = [
        speed
        for speed in (
            _temporal_relative_pose_speed(previous, current, 15, (11,), scale, delta_s),
            _temporal_relative_pose_speed(previous, current, 16, (12,), scale, delta_s),
        )
        if speed is not None
    ]
    head_speed = _temporal_relative_pose_speed(previous, current, 0, (11, 12), scale, delta_s)
    ankle_speeds = [
        speed
        for speed in (
            _temporal_relative_pose_speed(previous, current, 27, (23,), scale, delta_s),
            _temporal_relative_pose_speed(previous, current, 28, (24,), scale, delta_s),
        )
        if speed is not None
    ]
    return (
        sum(speeds) / len(speeds) if speeds else None,
        max(wrist_speeds) if wrist_speeds else None,
        head_speed,
        max(ankle_speeds) if ankle_speeds else None,
    )


def _temporal_relative_pose_speed(
    previous: PoseLandmarkSet,
    current: PoseLandmarkSet,
    point_index: int,
    anchor_indices: tuple[int, ...],
    scale: float,
    delta_s: float,
) -> float | None:
    velocity = _temporal_relative_pose_velocity(
        previous,
        current,
        point_index,
        anchor_indices,
        scale,
        delta_s,
    )
    return math.hypot(*velocity) if velocity is not None else None


def _temporal_relative_pose_velocity(
    previous: PoseLandmarkSet,
    current: PoseLandmarkSet,
    point_index: int,
    anchor_indices: tuple[int, ...],
    scale: float,
    delta_s: float,
) -> tuple[float, float] | None:
    required = (point_index, *anchor_indices)
    if delta_s <= _EPSILON or max(required) >= len(previous) or max(required) >= len(current):
        return None
    if any(
        min(previous[index].visibility, previous[index].presence) < 0.5
        or min(current[index].visibility, current[index].presence) < 0.5
        for index in required
    ):
        return None
    previous_anchor = (
        sum(previous[index].x for index in anchor_indices) / len(anchor_indices),
        sum(previous[index].y for index in anchor_indices) / len(anchor_indices),
    )
    current_anchor = (
        sum(current[index].x for index in anchor_indices) / len(anchor_indices),
        sum(current[index].y for index in anchor_indices) / len(anchor_indices),
    )
    previous_relative = (
        previous[point_index].x - previous_anchor[0],
        previous[point_index].y - previous_anchor[1],
    )
    current_relative = (
        current[point_index].x - current_anchor[0],
        current[point_index].y - current_anchor[1],
    )
    return (
        (current_relative[0] - previous_relative[0]) / scale / delta_s,
        (current_relative[1] - previous_relative[1]) / scale / delta_s,
    )


def _temporal_relative_pose_vertical_velocity(
    previous: PoseLandmarkSet,
    current: PoseLandmarkSet,
    point_index: int,
    anchor_indices: tuple[int, ...],
    scale: float,
    delta_s: float,
) -> float | None:
    velocity = _temporal_relative_pose_velocity(
        previous,
        current,
        point_index,
        anchor_indices,
        scale,
        delta_s,
    )
    return velocity[1] if velocity is not None else None


def _temporal_relative_group_horizontal_velocity(
    previous: PoseLandmarkSet,
    current: PoseLandmarkSet,
    point_indices: tuple[int, ...],
    anchor_indices: tuple[int, ...],
    scale: float,
    delta_s: float,
) -> float | None:
    required = (*point_indices, *anchor_indices)
    if delta_s <= _EPSILON or max(required) >= len(previous) or max(required) >= len(current):
        return None
    if any(
        min(previous[index].visibility, previous[index].presence) < 0.5
        or min(current[index].visibility, current[index].presence) < 0.5
        for index in required
    ):
        return None

    def relative_x(landmarks: PoseLandmarkSet) -> float:
        points_x = sum(landmarks[index].x for index in point_indices) / len(point_indices)
        anchors_x = sum(landmarks[index].x for index in anchor_indices) / len(anchor_indices)
        return points_x - anchors_x

    return (relative_x(current) - relative_x(previous)) / scale / delta_s


def _temporal_hand_center(landmarks: HandLandmarkSet) -> tuple[float, float]:
    palm_indices = (0, 5, 9, 13, 17)
    return (
        sum(landmarks[index].x for index in palm_indices) / len(palm_indices),
        sum(landmarks[index].y for index in palm_indices) / len(palm_indices),
    )


def _temporal_wrist_distance_ratio(data: FrameData) -> float | None:
    landmarks = data.pose_landmarks
    scale = _temporal_pose_scale(landmarks)
    if scale is None:
        return None
    if (
        data.left_hand is not None
        and data.right_hand is not None
        and len(data.left_hand) > 17
        and len(data.right_hand) > 17
    ):
        return (
            math.dist(_temporal_hand_center(data.left_hand), _temporal_hand_center(data.right_hand))
            / scale
        )
    if landmarks is None or len(landmarks) <= 16:
        return None
    if any(min(landmarks[index].visibility, landmarks[index].presence) < 0.5 for index in (15, 16)):
        return None
    return _temporal_distance_xy(landmarks[15], landmarks[16]) / scale


def _temporal_direction_changes(values: Sequence[float], minimum_magnitude: float) -> int:
    previous_sign = 0
    changes = 0
    for value in values:
        if abs(value) < minimum_magnitude:
            continue
        sign = 1 if value > 0 else -1
        if previous_sign and sign != previous_sign:
            changes += 1
        previous_sign = sign
    return changes


def _temporal_hand_pair_motion(
    previous: HandLandmarkSet | None,
    current: HandLandmarkSet | None,
    delta_s: float,
) -> float | None:
    if previous is None or current is None or delta_s <= _EPSILON:
        return None
    scale_values = [
        value for value in (_temporal_hand_scale(previous), _temporal_hand_scale(current)) if value
    ]
    if not scale_values:
        return None
    scale = sum(scale_values) / len(scale_values)
    speeds = [
        _temporal_distance_xy(before, after) / scale / delta_s
        for before, after in zip(previous, current, strict=False)
    ]
    return sum(speeds) / len(speeds) if speeds else None


def _temporal_average(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _temporal_recent_average(values: Sequence[float], max_samples: int) -> float | None:
    return _temporal_average(values[-max_samples:])


def _temporal_pose_change(samples: Sequence[TemporalLandmarkSample]) -> float | None:
    first = samples[0].data.pose_landmarks
    last = samples[-1].data.pose_landmarks
    if first is None or last is None:
        return None
    scale_values = [
        value for value in (_temporal_pose_scale(first), _temporal_pose_scale(last)) if value
    ]
    if not scale_values:
        return None
    scale = sum(scale_values) / len(scale_values)
    distances = [
        _temporal_distance_xy(before, after) / scale
        for before, after in zip(first, last, strict=False)
        if min(before.visibility, before.presence, after.visibility, after.presence) >= 0.5
    ]
    return _temporal_average(distances)


def _temporal_latest_hip_vertical_velocity(
    samples: Sequence[TemporalLandmarkSample],
) -> float | None:
    if len(samples) < 2:
        return None
    previous_sample, current_sample = samples[-2], samples[-1]
    previous = previous_sample.data.pose_landmarks
    current = current_sample.data.pose_landmarks
    delta_s = current_sample.monotonic_s - previous_sample.monotonic_s
    if previous is None or current is None or len(previous) < 25 or len(current) < 25:
        return None
    scale_values = [
        value for value in (_temporal_pose_scale(previous), _temporal_pose_scale(current)) if value
    ]
    if not scale_values or delta_s <= _EPSILON:
        return None
    previous_y = (previous[23].y + previous[24].y) / 2.0
    current_y = (current[23].y + current[24].y) / 2.0
    return (current_y - previous_y) / (sum(scale_values) / len(scale_values)) / delta_s


def extract_temporal_features(
    samples: Sequence[TemporalLandmarkSample],
) -> TemporalFeatures:
    """Extract scale-normalized motion over up to the last 30 frames."""

    if not samples:
        return TemporalFeatures(
            window_frames=0,
            window_duration_s=0.0,
            pose_motion_energy=None,
            pose_acceleration_energy=None,
            pose_change=None,
            left_hand_motion_energy=None,
            right_hand_motion_energy=None,
            max_wrist_speed=None,
            wrist_motion_energy=None,
            wrist_motion_direction_changes=0,
            elbow_motion_energy=None,
            shoulder_sway_energy=None,
            head_motion_energy=None,
            head_vertical_motion_energy=None,
            head_vertical_direction_changes=0,
            head_vertical_range_ratio=None,
            max_ankle_speed=None,
            wrist_distance_ratio=None,
            wrist_distance_min_ratio=None,
            wrist_distance_max_ratio=None,
            wrist_distance_range_ratio=None,
            wrist_distance_motion_energy=None,
            wrist_distance_closing_speed=None,
            wrist_distance_opening_speed=None,
            wrist_distance_direction_changes=0,
            hip_vertical_velocity=None,
            stillness_duration_s=0.0,
        )

    pose_intervals: list[tuple[float, float, float]] = []
    wrist_speeds: list[float] = []
    wrist_axis_velocities: tuple[list[float], ...] = ([], [], [], [])
    elbow_speeds: list[float] = []
    shoulder_sway_speeds: list[float] = []
    head_speeds: list[float] = []
    head_vertical_velocities: list[float] = []
    head_vertical_positions: list[float] = []
    ankle_speeds: list[float] = []
    left_hand_speeds: list[float] = []
    right_hand_speeds: list[float] = []
    for previous, current in zip(samples, samples[1:], strict=False):
        delta_s = current.monotonic_s - previous.monotonic_s
        pose_motion, wrist_speed, head_speed, ankle_speed = _temporal_pose_pair_motion(
            previous.data.pose_landmarks, current.data.pose_landmarks, delta_s
        )
        if pose_motion is not None:
            pose_intervals.append((previous.monotonic_s, current.monotonic_s, pose_motion))
        if wrist_speed is not None:
            wrist_speeds.append(wrist_speed)
        if head_speed is not None:
            head_speeds.append(head_speed)
        previous_pose = previous.data.pose_landmarks
        current_pose = current.data.pose_landmarks
        scale_values = [
            value
            for value in (_temporal_pose_scale(previous_pose), _temporal_pose_scale(current_pose))
            if value
        ]
        head_observation_allowed = (
            previous.data.face_observed is not False
            and current.data.face_observed is not False
        )
        if previous_pose is not None and current_pose is not None and scale_values:
            scale = sum(scale_values) / len(scale_values)
            if head_observation_allowed:
                head_vertical_velocity = (
                    _temporal_relative_pose_vertical_velocity(
                        previous_pose,
                        current_pose,
                        0,
                        (11, 12),
                        scale,
                        delta_s,
                    )
                )
                if head_vertical_velocity is not None:
                    head_vertical_velocities.append(head_vertical_velocity)
            for offset, (point_index, anchors) in enumerate(((15, (11,)), (16, (12,)))):
                velocity = _temporal_relative_pose_velocity(
                    previous_pose,
                    current_pose,
                    point_index,
                    anchors,
                    scale,
                    delta_s,
                )
                if velocity is not None:
                    wrist_axis_velocities[offset * 2].append(velocity[0])
                    wrist_axis_velocities[offset * 2 + 1].append(velocity[1])
            interval_elbow_speeds = [
                speed
                for speed in (
                    _temporal_relative_pose_speed(
                        previous_pose,
                        current_pose,
                        13,
                        (11,),
                        scale,
                        delta_s,
                    ),
                    _temporal_relative_pose_speed(
                        previous_pose,
                        current_pose,
                        14,
                        (12,),
                        scale,
                        delta_s,
                    ),
                )
                if speed is not None
            ]
            if interval_elbow_speeds:
                elbow_speeds.append(max(interval_elbow_speeds))
            shoulder_sway = _temporal_relative_group_horizontal_velocity(
                previous_pose,
                current_pose,
                (11, 12),
                (23, 24),
                scale,
                delta_s,
            )
            if shoulder_sway is not None:
                shoulder_sway_speeds.append(abs(shoulder_sway))
        if ankle_speed is not None:
            ankle_speeds.append(ankle_speed)
        left_speed = _temporal_hand_pair_motion(
            previous.data.left_hand, current.data.left_hand, delta_s
        )
        right_speed = _temporal_hand_pair_motion(
            previous.data.right_hand, current.data.right_hand, delta_s
        )
        if left_speed is not None:
            left_hand_speeds.append(left_speed)
        if right_speed is not None:
            right_hand_speeds.append(right_speed)

    # Velocity and a single sign reversal alone are too sensitive to nose-keypoint
    # jitter.  Keep the recent nose-to-shoulder displacement range as an explicit
    # amplitude gate for fast nodding.  Values are normalized by torso scale.
    for sample in samples[-10:]:
        pose = sample.data.pose_landmarks
        if sample.data.face_observed is False:
            continue
        scale = _temporal_pose_scale(pose)
        if pose is None or len(pose) <= 12 or scale is None:
            continue
        if any(
            min(pose[index].visibility, pose[index].presence) < 0.5
            for index in (0, 11, 12)
        ):
            continue
        shoulder_y = (pose[11].y + pose[12].y) / 2.0
        head_vertical_positions.append((pose[0].y - shoulder_y) / scale)

    accelerations: list[float] = []
    for first, second in zip(pose_intervals, pose_intervals[1:], strict=False):
        delta_s = second[1] - first[1]
        if delta_s > _EPSILON:
            accelerations.append(abs(second[2] - first[2]) / delta_s)

    stillness_duration_s = 0.0
    if pose_intervals:
        window_end = pose_intervals[-1][1]
        stillness_start = window_end
        for start_s, _, motion in reversed(pose_intervals):
            if motion > _STILLNESS_THRESHOLD:
                break
            stillness_start = start_s
        stillness_duration_s = window_end - stillness_start

    wrist_distances = [
        (sample.monotonic_s, ratio)
        for sample in samples
        if (ratio := _temporal_wrist_distance_ratio(sample.data)) is not None
    ]
    recent_wrist_distances = wrist_distances[-_CLAP_WINDOW_SAMPLES:]
    wrist_distance_velocities = [
        (current[1] - previous[1]) / (current[0] - previous[0])
        for previous, current in zip(
            recent_wrist_distances,
            recent_wrist_distances[1:],
            strict=False,
        )
        if current[0] - previous[0] > _EPSILON
    ]
    recent_distance_values = [distance for _, distance in recent_wrist_distances]
    wrist_distance_min = min(recent_distance_values) if recent_distance_values else None
    wrist_distance_max = max(recent_distance_values) if recent_distance_values else None

    return TemporalFeatures(
        window_frames=len(samples),
        window_duration_s=max(0.0, samples[-1].monotonic_s - samples[0].monotonic_s),
        pose_motion_energy=_temporal_average([interval[2] for interval in pose_intervals]),
        pose_acceleration_energy=_temporal_average(accelerations),
        pose_change=_temporal_pose_change(samples),
        left_hand_motion_energy=_temporal_recent_average(left_hand_speeds, 12),
        right_hand_motion_energy=_temporal_recent_average(right_hand_speeds, 12),
        max_wrist_speed=max(wrist_speeds) if wrist_speeds else None,
        wrist_motion_energy=_temporal_recent_average(wrist_speeds, 12),
        wrist_motion_direction_changes=max(
            (
                _temporal_direction_changes(velocities, minimum_magnitude=0.35)
                for velocities in (
                    axis_velocities[-12:] for axis_velocities in wrist_axis_velocities
                )
            ),
            default=0,
        ),
        elbow_motion_energy=_temporal_recent_average(elbow_speeds, 12),
        shoulder_sway_energy=_temporal_recent_average(shoulder_sway_speeds, 12),
        head_motion_energy=_temporal_recent_average(head_speeds, 10),
        head_vertical_motion_energy=_temporal_recent_average(
            [abs(velocity) for velocity in head_vertical_velocities],
            10,
        ),
        head_vertical_direction_changes=_temporal_direction_changes(
            head_vertical_velocities[-10:], minimum_magnitude=0.18
        ),
        head_vertical_range_ratio=(
            max(head_vertical_positions) - min(head_vertical_positions)
            if len(head_vertical_positions) >= 3
            else None
        ),
        max_ankle_speed=max(ankle_speeds) if ankle_speeds else None,
        wrist_distance_ratio=(recent_wrist_distances[-1][1] if recent_wrist_distances else None),
        wrist_distance_min_ratio=wrist_distance_min,
        wrist_distance_max_ratio=wrist_distance_max,
        wrist_distance_range_ratio=(
            wrist_distance_max - wrist_distance_min
            if wrist_distance_max is not None and wrist_distance_min is not None
            else None
        ),
        wrist_distance_motion_energy=_temporal_recent_average(
            [abs(velocity) for velocity in wrist_distance_velocities],
            12,
        ),
        wrist_distance_closing_speed=(
            max(-velocity for velocity in wrist_distance_velocities)
            if wrist_distance_velocities
            else None
        ),
        wrist_distance_opening_speed=(
            max(wrist_distance_velocities) if wrist_distance_velocities else None
        ),
        wrist_distance_direction_changes=_temporal_direction_changes(
            wrist_distance_velocities, minimum_magnitude=0.12
        ),
        hip_vertical_velocity=_temporal_latest_hip_vertical_velocity(samples),
        stillness_duration_s=stillness_duration_s,
    )


# ==============================================================================
# FALL EVENT STATE MACHINE
# ==============================================================================

LOGGER = logging.getLogger(__name__)
_EPSILON = 1e-6
_FAST_NOD_RANGE_START = 0.05
_FAST_NOD_RANGE_FULL = 0.14


def _fall_clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _fall_high(value: float | None, start: float, full: float) -> float:
    if value is None or value <= start + _EPSILON:
        return 0.0
    if value >= full - _EPSILON:
        return 1.0
    return _fall_clamp((value - start) / max(full - start, _EPSILON))


class FallPhase(str, Enum):
    """Human-readable phase of the fall detector state machine."""

    UNKNOWN = "unknown"
    MONITORING = "monitoring"
    FALLING = "falling"
    LYING = "lying"
    RECOVERING = "recovering"


@dataclass(frozen=True, slots=True)
class FallDetectionConfig:
    """Thresholds expressed in real time so behavior is independent of FPS."""

    baseline_hold_s: float = 1.0
    transition_history_s: float = 0.70
    upright_to_lying_max_s: float = 1.50
    transition_timeout_s: float = 2.2
    lying_confirm_s: float = 0.80
    lying_gap_tolerance_s: float = 0.55
    recovery_hold_s: float = 1.50
    alert_hold_s: float = 2.0
    cooldown_s: float = 30.0
    pose_lost_reset_s: float = 1.25
    lying_threshold: float = 0.55
    upright_threshold: float = 0.55
    downward_velocity_start: float = 0.65
    downward_velocity_full: float = 1.60
    motion_start: float = 0.25
    motion_full: float = 0.90
    lying_rise_start: float = 0.30
    lying_rise_full: float = 0.70
    transition_threshold: float = 0.55


@dataclass(frozen=True, slots=True)
class FallEventStatus:
    """Per-frame status for UI, logging, and a later alert transport."""

    phase: FallPhase
    lying_score: float
    transition_score: float
    armed: bool
    event_triggered: bool
    alert_active: bool
    cooldown_remaining_s: float


class FallEventManager:
    """Confirm a fall only after an observed upright-to-lying transition.

    Static lying is deliberately a posture, not an event.  A fall event requires
    an armed upright baseline, rapid body descent or rotation, and a persistent
    lying posture.  The one-shot flag is emitted on the confirmation frame only.
    """

    def __init__(self, config: FallDetectionConfig | None = None) -> None:
        self._config = config or FallDetectionConfig()
        self._phase = FallPhase.UNKNOWN
        self._armed = False
        self._upright_since: float | None = None
        self._last_upright_at: float | None = None
        self._fall_started_at: float | None = None
        self._lying_since: float | None = None
        self._last_lying_at: float | None = None
        self._recovery_since: float | None = None
        self._last_pose_at: float | None = None
        self._last_timestamp: float | None = None
        self._last_alert_at: float | None = None
        self._alert_until: float | None = None
        self._transition_peak = 0.0
        self._lying_history: deque[tuple[float, float]] = deque()

    def update(
        self,
        frame: AnalyzedFrame,
        *,
        lying_score: float,
        upright_score: float,
    ) -> FallEventStatus:
        timestamp_s = frame.inference.captured.monotonic_ns / 1_000_000_000.0
        if self._last_timestamp is not None and timestamp_s < self._last_timestamp:
            LOGGER.info("Fall event clock moved backwards; resetting temporal state")
            self.reset()
        self._last_timestamp = timestamp_s

        pose_detected = frame.inference.data.pose_landmarks is not None
        if pose_detected:
            self._last_pose_at = timestamp_s
        self._update_lying_history(timestamp_s, lying_score)
        transition_score = self._transition_score(frame, lying_score)
        event_triggered = False

        if not pose_detected:
            self._handle_missing_pose(timestamp_s)
            return self._status(timestamp_s, lying_score, transition_score, False)

        is_lying = lying_score >= self._config.lying_threshold
        is_upright = upright_score >= self._config.upright_threshold and not is_lying
        if is_upright:
            self._last_upright_at = timestamp_s
        recent_upright_to_lying = (
            is_lying
            and self._last_upright_at is not None
            and timestamp_s - self._last_upright_at
            <= self._config.upright_to_lying_max_s
        )

        if self._phase is FallPhase.UNKNOWN:
            if is_lying:
                # Entering the camera view while already lying must never fabricate
                # the upright baseline that is required for a fall event.
                self._enter_lying(alerted=False)
            else:
                self._observe_upright(timestamp_s, is_upright)

        elif self._phase is FallPhase.MONITORING:
            self._observe_upright(timestamp_s, is_upright)
            self._arm_if_ready(timestamp_s)
            if self._armed and (
                transition_score >= self._config.transition_threshold
                or recent_upright_to_lying
            ):
                self._phase = FallPhase.FALLING
                self._fall_started_at = timestamp_s
                self._lying_since = timestamp_s if is_lying else None
                self._last_lying_at = timestamp_s if is_lying else None
                self._transition_peak = max(
                    transition_score,
                    self._config.transition_threshold
                    if recent_upright_to_lying
                    else 0.0,
                )
            elif is_lying:
                # A slow intentional lie-down (or an unseen transition) is useful
                # state information, but not enough evidence for an alarm.
                self._enter_lying(alerted=False)

        elif self._phase is FallPhase.FALLING:
            self._transition_peak = max(self._transition_peak, transition_score)
            if is_lying:
                if self._lying_since is None:
                    self._lying_since = timestamp_s
                self._last_lying_at = timestamp_s
                if timestamp_s - self._lying_since >= self._config.lying_confirm_s and self._armed:
                    event_triggered = True
                    self._last_alert_at = timestamp_s
                    self._alert_until = timestamp_s + self._config.alert_hold_s
                    LOGGER.warning(
                        "Fall event confirmed (transition=%.2f, lying=%.2f)",
                        self._transition_peak,
                        lying_score,
                    )
                    self._enter_lying(alerted=True)
            elif (
                self._last_lying_at is None
                or timestamp_s - self._last_lying_at > self._config.lying_gap_tolerance_s
            ):
                # PoseLandmarker often emits one upright-looking frame while a
                # fallen body is self-occluded.  A short gap must not restart the
                # whole confirmation interval, but a sustained contradiction does.
                self._lying_since = None
                self._last_lying_at = None
            if (
                self._phase is FallPhase.FALLING
                and self._fall_started_at is not None
                and timestamp_s - self._fall_started_at > self._config.transition_timeout_s
            ):
                self._cancel_transition(timestamp_s, is_upright)

        elif self._phase is FallPhase.LYING:
            if not is_lying and is_upright:
                self._phase = FallPhase.RECOVERING
                self._recovery_since = timestamp_s

        elif self._phase is FallPhase.RECOVERING:
            if is_lying:
                self._enter_lying(alerted=self._last_alert_at is not None)
            elif is_upright:
                if self._recovery_since is None:
                    self._recovery_since = timestamp_s
                if timestamp_s - self._recovery_since >= self._config.recovery_hold_s:
                    self._phase = FallPhase.MONITORING
                    self._upright_since = self._recovery_since
                    self._lying_since = None
                    self._recovery_since = None
                    self._arm_if_ready(timestamp_s)
            else:
                self._recovery_since = None

        return self._status(timestamp_s, lying_score, transition_score, event_triggered)

    def reset(self) -> None:
        """Clear tracking state without retaining an old event or baseline."""

        self._phase = FallPhase.UNKNOWN
        self._armed = False
        self._upright_since = None
        self._last_upright_at = None
        self._fall_started_at = None
        self._lying_since = None
        self._last_lying_at = None
        self._recovery_since = None
        self._last_pose_at = None
        self._last_timestamp = None
        self._last_alert_at = None
        self._alert_until = None
        self._transition_peak = 0.0
        self._lying_history.clear()

    def _observe_upright(self, timestamp_s: float, is_upright: bool) -> None:
        if not is_upright:
            self._upright_since = None
            return
        if self._upright_since is None:
            self._upright_since = timestamp_s
        if self._phase is FallPhase.UNKNOWN:
            self._phase = FallPhase.MONITORING
        self._arm_if_ready(timestamp_s)

    def _arm_if_ready(self, timestamp_s: float) -> None:
        if self._upright_since is None:
            return
        baseline_ready = timestamp_s - self._upright_since >= self._config.baseline_hold_s
        cooldown_ready = (
            self._last_alert_at is None
            or timestamp_s - self._last_alert_at >= self._config.cooldown_s
        )
        self._armed = baseline_ready and cooldown_ready

    def _enter_lying(self, *, alerted: bool) -> None:
        self._phase = FallPhase.LYING
        self._armed = False
        self._upright_since = None
        self._last_upright_at = None
        self._fall_started_at = None
        self._lying_since = None
        self._last_lying_at = None
        self._recovery_since = None
        self._transition_peak = 0.0
        if not alerted and self._last_alert_at is None:
            self._alert_until = None

    def _cancel_transition(self, timestamp_s: float, is_upright: bool) -> None:
        self._phase = FallPhase.MONITORING if is_upright else FallPhase.UNKNOWN
        self._armed = False
        self._upright_since = timestamp_s if is_upright else None
        self._last_upright_at = timestamp_s if is_upright else None
        self._fall_started_at = None
        self._lying_since = None
        self._last_lying_at = None
        self._transition_peak = 0.0

    def _handle_missing_pose(self, timestamp_s: float) -> None:
        if (
            self._last_pose_at is not None
            and timestamp_s - self._last_pose_at <= self._config.pose_lost_reset_s
        ):
            return
        self._phase = FallPhase.UNKNOWN
        self._armed = False
        self._upright_since = None
        self._last_upright_at = None
        self._fall_started_at = None
        self._lying_since = None
        self._last_lying_at = None
        self._recovery_since = None
        self._transition_peak = 0.0
        self._lying_history.clear()

    def _update_lying_history(self, timestamp_s: float, lying_score: float) -> None:
        self._lying_history.append((timestamp_s, lying_score))
        oldest_allowed = timestamp_s - self._config.transition_history_s
        while self._lying_history and self._lying_history[0][0] < oldest_allowed:
            self._lying_history.popleft()

    def _transition_score(self, frame: AnalyzedFrame, lying_score: float) -> float:
        temporal = frame.features.temporal
        downward = _fall_high(
            temporal.hip_vertical_velocity,
            self._config.downward_velocity_start,
            self._config.downward_velocity_full,
        )
        motion = _fall_high(
            temporal.pose_motion_energy,
            self._config.motion_start,
            self._config.motion_full,
        )
        recent_min_lying = min(
            (score for _, score in self._lying_history),
            default=lying_score,
        )
        rapid_rotation = _fall_high(
            lying_score - recent_min_lying,
            self._config.lying_rise_start,
            self._config.lying_rise_full,
        )
        # Downward screen motion alone can come from robot/camera movement.  It may
        # open a short transition window, but an alert still requires the body to
        # become and remain geometrically horizontal.
        # A fall can disappear from PoseLandmarker for several frames and return
        # already horizontal.  In that case there is no adjacent valid pose pair
        # from which to calculate motion energy, but the recent upright-to-lying
        # rotation remains strong transition evidence.  Static lying still cannot
        # alert because the state machine first requires an armed upright baseline.
        return max(min(downward, motion), rapid_rotation)

    def _status(
        self,
        timestamp_s: float,
        lying_score: float,
        transition_score: float,
        event_triggered: bool,
    ) -> FallEventStatus:
        alert_active = self._alert_until is not None and timestamp_s < self._alert_until
        cooldown_remaining = 0.0
        if self._last_alert_at is not None:
            cooldown_remaining = max(
                0.0,
                self._config.cooldown_s - (timestamp_s - self._last_alert_at),
            )
        return FallEventStatus(
            phase=self._phase,
            lying_score=lying_score,
            transition_score=transition_score,
            armed=self._armed,
            event_triggered=event_triggered,
            alert_active=alert_active,
            cooldown_remaining_s=cooldown_remaining,
        )


# ==============================================================================
# ACTION DATA CONTRACTS AND PRIORITIES
# ==============================================================================


class ActionName(str, Enum):
    ARMS_RAISED = "arms_raised"
    ARMS_OPEN = "arms_open"
    WAVING = "waving"
    JUMPING = "jumping"
    FAST_NOD = "fast_nod"
    CLAPPING = "clapping"
    THUMBS_UP = "thumbs_up"
    VICTORY = "victory"
    HANDS_ON_HIPS = "hands_on_hips"
    LARGE_ARM_SWING = "large_arm_swing"
    POINTING = "pointing"
    STOMPING = "stomping"
    ARMS_CROSSED = "arms_crossed"
    HEAD_DOWN = "head_down"
    SHOULDERS_SLUMPED = "shoulders_slumped"
    FACE_COVERING = "face_covering"
    HANDS_ON_HEAD = "hands_on_head"
    CURLED_UP = "curled_up"
    HUNCHED = "hunched"
    STANDING = "standing"
    SITTING = "sitting"
    LOW_MOTION = "low_motion"
    LYING = "lying"
    STOP_GESTURE = "stop_gesture"
    FALL = "fall"


class ActionGroup(str, Enum):
    """Semantic layers used for output priority and group-local conflicts."""

    EVENT = "event"
    DYNAMIC = "dynamic"
    GESTURE = "gesture"
    POSTURE = "posture"
    ACTIVITY = "activity"


class TriggerPriority(IntEnum):
    """MarsDog trigger tier; lower values take precedence."""

    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3
    P4 = 4


class StateHint(str, Enum):
    """A state candidate only; final state inference must use temporal evidence."""

    SPECIAL = "special_pose"
    AGITATED = "agitated_candidate"
    LOW = "low_candidate"
    POSITIVE = "positive_candidate"
    NEUTRAL = "neutral_candidate"


ACTION_GROUPS: dict[ActionName, ActionGroup] = {
    ActionName.FALL: ActionGroup.EVENT,
    ActionName.STOP_GESTURE: ActionGroup.EVENT,
    ActionName.JUMPING: ActionGroup.DYNAMIC,
    ActionName.STOMPING: ActionGroup.DYNAMIC,
    ActionName.CLAPPING: ActionGroup.DYNAMIC,
    ActionName.FAST_NOD: ActionGroup.DYNAMIC,
    ActionName.WAVING: ActionGroup.DYNAMIC,
    ActionName.LARGE_ARM_SWING: ActionGroup.DYNAMIC,
    ActionName.THUMBS_UP: ActionGroup.GESTURE,
    ActionName.VICTORY: ActionGroup.GESTURE,
    ActionName.POINTING: ActionGroup.GESTURE,
    ActionName.FACE_COVERING: ActionGroup.GESTURE,
    ActionName.HANDS_ON_HEAD: ActionGroup.GESTURE,
    ActionName.ARMS_CROSSED: ActionGroup.GESTURE,
    ActionName.HANDS_ON_HIPS: ActionGroup.GESTURE,
    ActionName.ARMS_RAISED: ActionGroup.GESTURE,
    ActionName.ARMS_OPEN: ActionGroup.GESTURE,
    ActionName.HEAD_DOWN: ActionGroup.POSTURE,
    ActionName.SHOULDERS_SLUMPED: ActionGroup.POSTURE,
    ActionName.CURLED_UP: ActionGroup.POSTURE,
    ActionName.HUNCHED: ActionGroup.POSTURE,
    ActionName.STANDING: ActionGroup.POSTURE,
    ActionName.SITTING: ActionGroup.POSTURE,
    ActionName.LYING: ActionGroup.POSTURE,
    ActionName.LOW_MOTION: ActionGroup.ACTIVITY,
}

ACTION_TRIGGER_PRIORITIES: dict[ActionName, TriggerPriority] = {
    ActionName.FALL: TriggerPriority.P0,
    ActionName.STOP_GESTURE: TriggerPriority.P0,
    ActionName.HANDS_ON_HIPS: TriggerPriority.P1,
    ActionName.LARGE_ARM_SWING: TriggerPriority.P1,
    ActionName.POINTING: TriggerPriority.P1,
    ActionName.STOMPING: TriggerPriority.P1,
    ActionName.ARMS_CROSSED: TriggerPriority.P1,
    ActionName.HEAD_DOWN: TriggerPriority.P2,
    ActionName.SHOULDERS_SLUMPED: TriggerPriority.P2,
    ActionName.FACE_COVERING: TriggerPriority.P2,
    ActionName.HANDS_ON_HEAD: TriggerPriority.P2,
    ActionName.CURLED_UP: TriggerPriority.P2,
    ActionName.HUNCHED: TriggerPriority.P2,
    ActionName.ARMS_RAISED: TriggerPriority.P3,
    ActionName.WAVING: TriggerPriority.P3,
    ActionName.VICTORY: TriggerPriority.P3,
    ActionName.JUMPING: TriggerPriority.P3,
    ActionName.ARMS_OPEN: TriggerPriority.P3,
    ActionName.FAST_NOD: TriggerPriority.P3,
    ActionName.CLAPPING: TriggerPriority.P3,
    ActionName.THUMBS_UP: TriggerPriority.P3,
    ActionName.STANDING: TriggerPriority.P4,
    ActionName.SITTING: TriggerPriority.P4,
    ActionName.LYING: TriggerPriority.P4,
    ActionName.LOW_MOTION: TriggerPriority.P4,
}

PRIORITY_STATE_HINTS: dict[TriggerPriority, StateHint] = {
    TriggerPriority.P0: StateHint.SPECIAL,
    TriggerPriority.P1: StateHint.AGITATED,
    TriggerPriority.P2: StateHint.LOW,
    TriggerPriority.P3: StateHint.POSITIVE,
    TriggerPriority.P4: StateHint.NEUTRAL,
}

_WITHIN_TIER_PRIORITY = {
    ActionName.FALL: 0,
    ActionName.STOP_GESTURE: 1,
    ActionName.HANDS_ON_HIPS: 0,
    ActionName.LARGE_ARM_SWING: 1,
    ActionName.POINTING: 2,
    ActionName.STOMPING: 3,
    ActionName.ARMS_CROSSED: 4,
    ActionName.HEAD_DOWN: 0,
    ActionName.SHOULDERS_SLUMPED: 1,
    ActionName.FACE_COVERING: 2,
    ActionName.HANDS_ON_HEAD: 3,
    ActionName.CURLED_UP: 4,
    ActionName.HUNCHED: 5,
    ActionName.VICTORY: 0,
    ActionName.ARMS_RAISED: 1,
    ActionName.WAVING: 2,
    ActionName.JUMPING: 3,
    ActionName.ARMS_OPEN: 4,
    ActionName.FAST_NOD: 5,
    ActionName.CLAPPING: 6,
    ActionName.THUMBS_UP: 7,
    ActionName.STANDING: 0,
    ActionName.SITTING: 1,
    ActionName.LYING: 2,
    ActionName.LOW_MOTION: 3,
}


def action_priority(name: ActionName) -> tuple[int, int]:
    """Return P0-P4 and within-tier priority; lower values are stronger."""

    tier = ACTION_TRIGGER_PRIORITIES[name]
    return int(tier), _WITHIN_TIER_PRIORITY[name]


@dataclass(frozen=True, slots=True)
class ActionScore:
    name: ActionName
    confidence: float
    support_ratio: float
    duration_s: float


@dataclass(frozen=True, slots=True)
class RecognizedFrame:
    analyzed: AnalyzedFrame
    actions: tuple[ActionScore, ...]
    raw_scores: tuple[tuple[ActionName, float], ...]
    fall_status: FallEventStatus
    recognition_ms: float


# ==============================================================================
# RULE CLASSIFIER AND SMOOTHING
# ==============================================================================

_EPSILON = 1e-6
_HEAD_DOWN_NECK_START = -0.42
_HEAD_DOWN_NECK_FULL = -0.22
_HEAD_DOWN_PITCH_START = 0.10
_HEAD_DOWN_PITCH_FULL = 0.24
_STOP_SCORE_THRESHOLD = 0.65


class _ClassifierNormalizedPoint(Protocol):
    x: float
    y: float


def _classifier_clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _classifier_high(value: float | None, start: float, full: float) -> float:
    if value is None:
        return 0.0
    if value <= start + _EPSILON:
        return 0.0
    if value >= full - _EPSILON:
        return 1.0
    return _classifier_clamp((value - start) / max(full - start, _EPSILON))


def _classifier_low(value: float | None, full: float, end: float) -> float:
    if value is None:
        return 0.0
    if value <= full + _EPSILON:
        return 1.0
    if value >= end - _EPSILON:
        return 0.0
    return _classifier_clamp((end - value) / max(end - full, _EPSILON))


def _classifier_point(
    landmarks: PoseLandmarkSet | None,
    index: int,
    min_confidence: float = 0.5,
) -> PoseLandmark | None:
    if landmarks is None or index >= len(landmarks):
        return None
    point = landmarks[index]
    return (
        point
        if min(point.visibility, point.presence) >= min_confidence
        else None
    )


def _classifier_distance(
    first: _ClassifierNormalizedPoint, second: _ClassifierNormalizedPoint
) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _classifier_centered(value: float | None, center: float, half_width: float) -> float:
    return _classifier_clamp(1.0 - abs(value - center) / half_width) if value is not None else 0.0


def _classifier_undirected_line_tilt(value: float | None) -> float | None:
    """Return a line's 0-90 degree tilt independent of endpoint order."""

    if value is None:
        return None
    normalized = abs(value) % 180.0
    return min(normalized, 180.0 - normalized)


def _classifier_partial_bilateral_score(
    left_value: float | None,
    right_value: float | None,
    scorer: Callable[[float], float],
    single_side_weight: float = 0.82,
) -> float:
    values = [scorer(value) for value in (left_value, right_value) if value is not None]
    if not values:
        return 0.0
    return sum(values) / len(values) * (1.0 if len(values) == 2 else single_side_weight)


def _classifier_weighted_average(*evidence: tuple[float, float]) -> float:
    weight_total = sum(weight for _, weight in evidence)
    if weight_total <= _EPSILON:
        return 0.0
    return sum(score * weight for score, weight in evidence) / weight_total


class RuleActionClassifier:
    """Scores observable actions from geometry and normalized temporal motion."""

    def classify(self, frame: AnalyzedFrame) -> dict[ActionName, float]:
        pose = frame.features.pose
        data = frame.inference.data
        landmarks = data.pose_landmarks
        scores = {name: 0.0 for name in ActionName}

        self._score_basic_pose(scores, pose)
        self._score_temporal_actions(scores, frame)
        self._score_pose_relations(scores, pose, landmarks)
        self._score_hand_actions(scores, frame, data.left_hand, data.right_hand)
        self._resolve_conflicts(scores)
        return scores

    @staticmethod
    def _score_basic_pose(scores: dict[ActionName, float], pose: PoseFeatures) -> None:
        left_raise = (
            _classifier_low(pose.left_arm_elevation_degrees, 20.0, 75.0)
            if pose.left_wrist_above_shoulder
            else 0.0
        )
        right_raise = (
            _classifier_low(pose.right_arm_elevation_degrees, 20.0, 75.0)
            if pose.right_wrist_above_shoulder
            else 0.0
        )
        scores[ActionName.ARMS_RAISED] = max(left_raise, right_raise)

        elbow_extension = _classifier_partial_bilateral_score(
            pose.left_elbow_angle_degrees,
            pose.right_elbow_angle_degrees,
            lambda value: _classifier_high(value, 105.0, 145.0),
            single_side_weight=0.78,
        )
        wrist_horizontal = _classifier_partial_bilateral_score(
            pose.left_arm_elevation_degrees,
            pose.right_arm_elevation_degrees,
            lambda value: _classifier_centered(value, 90.0, 70.0),
            single_side_weight=0.78,
        )
        upper_arm_horizontal = _classifier_partial_bilateral_score(
            pose.left_upper_arm_elevation_degrees,
            pose.right_upper_arm_elevation_degrees,
            lambda value: _classifier_centered(value, 90.0, 65.0),
            single_side_weight=0.78,
        )
        horizontal_arms = max(wrist_horizontal, upper_arm_horizontal * 0.92)
        wrist_span = _classifier_high(pose.arms_span_ratio, 1.50, 2.35)
        elbow_span = _classifier_high(pose.elbows_span_ratio, 1.20, 1.80)
        arm_span = max(wrist_span, elbow_span * 0.92)
        has_elbow_angles = (
            pose.left_elbow_angle_degrees is not None or pose.right_elbow_angle_degrees is not None
        )
        if not has_elbow_angles:
            elbow_extension = elbow_span * 0.78
        if arm_span > 0.0 and elbow_extension > 0.10 and horizontal_arms > 0.0:
            scores[ActionName.ARMS_OPEN] = _classifier_weighted_average(
                (arm_span, 0.45),
                (elbow_extension, 0.30),
                (horizontal_arms, 0.25),
            )
        # A low-mounted robot camera naturally makes people look downward.  Require
        # both a pronounced chin-to-chest compression and a strong face pitch so a
        # normal downward gaze remains neutral.
        head_neck_compression = _classifier_high(
            pose.head_drop_ratio,
            _HEAD_DOWN_NECK_START,
            _HEAD_DOWN_NECK_FULL,
        )
        face_pitch = _classifier_high(
            pose.nose_ear_vertical_ratio,
            _HEAD_DOWN_PITCH_START,
            _HEAD_DOWN_PITCH_FULL,
        )
        scores[ActionName.HEAD_DOWN] = min(head_neck_compression, face_pitch)

        lean = abs(pose.torso_lean_degrees) if pose.torso_lean_degrees is not None else None
        scores[ActionName.HUNCHED] = min(
            _classifier_high(lean, 12.0, 35.0),
            max(0.45, scores[ActionName.HEAD_DOWN]),
        )
        scores[ActionName.SHOULDERS_SLUMPED] = min(
            scores[ActionName.HEAD_DOWN], _classifier_low(lean, 10.0, 35.0)
        )
        shoulder_line_tilt = _classifier_undirected_line_tilt(pose.shoulder_slope_degrees)
        scores[ActionName.LYING] = max(
            _classifier_high(lean, 40.0, 68.0),
            _classifier_high(shoulder_line_tilt, 45.0, 75.0),
        )

        upright = _classifier_low(lean, 8.0, 25.0)
        straight_knees_2d = _classifier_partial_bilateral_score(
            pose.left_knee_angle_degrees,
            pose.right_knee_angle_degrees,
            lambda value: _classifier_high(value, 135.0, 165.0),
        )
        straight_knees_3d = _classifier_partial_bilateral_score(
            pose.left_knee_angle_3d_degrees,
            pose.right_knee_angle_3d_degrees,
            lambda value: _classifier_high(value, 140.0, 170.0),
        )
        straight_knees = (
            straight_knees_3d
            if pose.left_knee_angle_3d_degrees is not None
            or pose.right_knee_angle_3d_degrees is not None
            else straight_knees_2d
        )
        vertical_thighs = _classifier_partial_bilateral_score(
            pose.left_thigh_vertical_degrees,
            pose.right_thigh_vertical_degrees,
            lambda value: _classifier_low(value, 12.0, 45.0),
        )

        bent_knees_2d = _classifier_partial_bilateral_score(
            pose.left_knee_angle_degrees,
            pose.right_knee_angle_degrees,
            lambda value: _classifier_centered(value, 100.0, 70.0),
        )
        bent_knees_3d = _classifier_partial_bilateral_score(
            pose.left_knee_angle_3d_degrees,
            pose.right_knee_angle_3d_degrees,
            lambda value: _classifier_centered(value, 100.0, 75.0),
        )
        horizontal_thighs = _classifier_partial_bilateral_score(
            pose.left_thigh_vertical_degrees,
            pose.right_thigh_vertical_degrees,
            lambda value: _classifier_centered(value, 85.0, 55.0),
        )
        forward_thighs = _classifier_partial_bilateral_score(
            pose.left_thigh_depth_ratio,
            pose.right_thigh_depth_ratio,
            lambda value: _classifier_high(value, 0.20, 0.60),
        )
        sitting_geometry = max(
            bent_knees_2d,
            bent_knees_3d,
            horizontal_thighs * 0.9,
            forward_thighs,
        )
        sitting_contradiction = max(
            bent_knees_2d,
            bent_knees_3d,
            horizontal_thighs * 0.9,
            forward_thighs,
        )
        standing_geometry = max(straight_knees, vertical_thighs * 0.9)
        standing_geometry *= 1.0 - 0.90 * sitting_contradiction

        scores[ActionName.STANDING] = min(upright, standing_geometry)
        scores[ActionName.SITTING] = min(upright, sitting_geometry)

    @staticmethod
    def _score_temporal_actions(scores: dict[ActionName, float], frame: AnalyzedFrame) -> None:
        pose = frame.features.pose
        temporal = frame.features.temporal
        wrist_high = bool(pose.left_wrist_above_shoulder or pose.right_wrist_above_shoulder)
        hand_open = max(
            frame.features.left_hand.extended_finger_count,
            frame.features.right_hand.extended_finger_count,
        )
        wrist_reversal = _classifier_high(float(temporal.wrist_motion_direction_changes), 0.0, 1.0)
        whole_arm_motion = max(
            _classifier_high(temporal.elbow_motion_energy, 0.30, 0.90),
            _classifier_high(temporal.shoulder_sway_energy, 0.12, 0.55),
        )
        open_hand_support = 0.85 + 0.15 * _classifier_high(float(hand_open), 1.0, 4.0)
        small_forearm_wave = min(
            _classifier_high(temporal.wrist_motion_energy, 0.35, 1.10),
            wrist_reversal,
            1.0 if wrist_high else 0.0,
        )
        scores[ActionName.WAVING] = (
            small_forearm_wave * open_hand_support * (1.0 - 0.75 * whole_arm_motion)
        )
        scores[ActionName.LARGE_ARM_SWING] = min(
            _classifier_high(temporal.wrist_motion_energy, 0.65, 1.80),
            whole_arm_motion,
            wrist_reversal,
        )
        scores[ActionName.JUMPING] = min(
            _classifier_high(
                -temporal.hip_vertical_velocity
                if temporal.hip_vertical_velocity is not None
                else None,
                0.8,
                2.0,
            ),
            _classifier_high(temporal.pose_motion_energy, 0.45, 1.2),
        )
        if RuleActionClassifier._reliable_nod_anchor(frame) is not None:
            scores[ActionName.FAST_NOD] = min(
                _classifier_high(
                    temporal.head_vertical_motion_energy, 0.18, 0.65
                ),
                _classifier_high(
                    float(temporal.head_vertical_direction_changes),
                    0.0,
                    1.0,
                ),
                _classifier_high(
                    temporal.head_vertical_range_ratio,
                    _FAST_NOD_RANGE_START,
                    _FAST_NOD_RANGE_FULL,
                ),
            )
        bilateral_hand_motion = min(
            _classifier_high(temporal.left_hand_motion_energy, 0.8, 2.4),
            _classifier_high(temporal.right_hand_motion_energy, 0.8, 2.4),
        )
        # Pose wrists remain available when the two palms overlap and Hand Landmarker
        # temporarily loses one hand.  Treat bilateral hand motion as optional evidence
        # and keep the Pose-wrist path sensitive enough for short, compact claps.
        clap_motion = max(
            bilateral_hand_motion,
            _classifier_high(temporal.wrist_motion_energy, 0.18, 0.75),
        )
        clap_close = _classifier_low(temporal.wrist_distance_min_ratio, 0.25, 1.15)
        clap_range = _classifier_high(temporal.wrist_distance_range_ratio, 0.02, 0.16)
        clap_distance_motion = _classifier_high(
            temporal.wrist_distance_motion_energy,
            0.18,
            0.85,
        )
        clap_closing = _classifier_high(temporal.wrist_distance_closing_speed, 0.35, 0.80)
        clap_opening = _classifier_high(temporal.wrist_distance_opening_speed, 0.35, 0.80)
        clap_zone = RuleActionClassifier._clap_zone_score(frame)
        # Crossing the arms produces one close-to-open reversal as the wrists pass
        # each other.  Clapping is repetitive by contract, so require a second
        # reversal (close/open/close or open/close/open) inside the recent window.
        clap_reversal = _classifier_high(
            float(temporal.wrist_distance_direction_changes),
            1.0,
            2.0,
        )
        clap_impulse = min(clap_closing, clap_opening)
        # A clap must contain an actual close-and-rebound cycle.  Direction changes
        # alone are insufficient because crossing the arms can create noisy wrist
        # distance reversals without the palms separating again.
        clap_cycle = min(
            clap_impulse,
            clap_reversal,
        )
        if (
            clap_cycle >= 0.25
            and clap_close > 0.0
            and clap_range > 0.0
            and clap_motion > 0.0
            and clap_zone >= 0.15
        ):
            scores[ActionName.CLAPPING] = _classifier_weighted_average(
                (clap_close, 0.25),
                (clap_range, 0.10),
                (clap_distance_motion, 0.10),
                (clap_motion, 0.15),
                (clap_cycle, 0.30),
                (clap_zone, 0.10),
            )
        scores[ActionName.STOMPING] = min(
            _classifier_high(temporal.max_ankle_speed, 4.0, 8.0),
            _classifier_high(temporal.pose_motion_energy, 0.45, 1.0),
            _classifier_low(
                abs(temporal.hip_vertical_velocity)
                if temporal.hip_vertical_velocity is not None
                else None,
                0.3,
                1.3,
            ),
        )
        scores[ActionName.LOW_MOTION] = _classifier_low(temporal.pose_motion_energy, 0.12, 0.45)

    @staticmethod
    def _score_pose_relations(
        scores: dict[ActionName, float],
        pose: PoseFeatures,
        landmarks: PoseLandmarkSet | None,
    ) -> None:
        left_shoulder = _classifier_point(landmarks, 11)
        right_shoulder = _classifier_point(landmarks, 12)
        left_elbow = _classifier_point(landmarks, 13)
        right_elbow = _classifier_point(landmarks, 14)
        left_wrist = _classifier_point(landmarks, 15)
        right_wrist = _classifier_point(landmarks, 16)
        # Wrists resting on the torso and hips are commonly self-occluded.  Keep
        # the normal 0.5-confidence points for face/head relations, but permit a
        # conservative 0.3 fallback for the bilateral waist/chest relations below.
        left_relation_wrist = _classifier_point(
            landmarks, 15, min_confidence=0.3
        )
        right_relation_wrist = _classifier_point(
            landmarks, 16, min_confidence=0.3
        )
        left_hip = _classifier_point(landmarks, 23, min_confidence=0.35)
        right_hip = _classifier_point(landmarks, 24, min_confidence=0.35)
        left_knee = _classifier_point(landmarks, 25)
        right_knee = _classifier_point(landmarks, 26)
        nose = _classifier_point(landmarks, 0)
        left_ear = _classifier_point(landmarks, 7)
        right_ear = _classifier_point(landmarks, 8)
        scale = pose.torso_length or pose.shoulder_width
        if scale is None or scale <= _EPSILON:
            return

        if all((left_relation_wrist, right_relation_wrist, left_hip, right_hip)):
            left_hip_proximity = _classifier_low(
                _classifier_distance(left_relation_wrist, left_hip) / scale,
                0.25,
                1.15,
            )
            right_hip_proximity = _classifier_low(
                _classifier_distance(right_relation_wrist, right_hip) / scale,
                0.25,
                1.15,
            )
            bilateral_hip_proximity = min(left_hip_proximity, right_hip_proximity)
            same_side_hips = min(
                _classifier_high(
                    (
                        _classifier_distance(left_relation_wrist, right_hip)
                        - _classifier_distance(left_relation_wrist, left_hip)
                    )
                    / scale,
                    -0.05,
                    0.25,
                ),
                _classifier_high(
                    (
                        _classifier_distance(right_relation_wrist, left_hip)
                        - _classifier_distance(right_relation_wrist, right_hip)
                    )
                    / scale,
                    -0.05,
                    0.25,
                ),
            )
            left_elbow_bent = max(
                _classifier_low(
                    _pose_angle(left_shoulder, left_elbow, left_relation_wrist),
                    100.0,
                    155.0,
                ),
                _classifier_low(
                    _pose_angle_3d(left_shoulder, left_elbow, left_relation_wrist),
                    100.0,
                    155.0,
                ),
            )
            right_elbow_bent = max(
                _classifier_low(
                    _pose_angle(right_shoulder, right_elbow, right_relation_wrist),
                    100.0,
                    155.0,
                ),
                _classifier_low(
                    _pose_angle_3d(right_shoulder, right_elbow, right_relation_wrist),
                    100.0,
                    155.0,
                ),
            )
            elbows_bent = min(left_elbow_bent, right_elbow_bent)
            waist_height = min(
                _classifier_low(
                    abs(left_relation_wrist.y - left_hip.y) / scale,
                    0.20,
                    0.80,
                ),
                _classifier_low(
                    abs(right_relation_wrist.y - right_hip.y) / scale,
                    0.20,
                    0.80,
                ),
            )
            elbow_flare = 0.5
            if all((left_shoulder, right_shoulder, left_elbow, right_elbow)):
                torso_mid_x = (left_shoulder.x + right_shoulder.x + left_hip.x + right_hip.x) / 4.0
                left_flare = (
                    abs(left_elbow.x - torso_mid_x)
                    - abs(left_relation_wrist.x - torso_mid_x)
                ) / scale
                right_flare = (
                    abs(right_elbow.x - torso_mid_x)
                    - abs(right_relation_wrist.x - torso_mid_x)
                ) / scale
                elbow_flare = min(
                    _classifier_high(left_flare, -0.05, 0.35),
                    _classifier_high(right_flare, -0.05, 0.35),
                )
            if (
                bilateral_hip_proximity >= 0.20
                and same_side_hips >= 0.20
                and waist_height >= 0.25
                and elbows_bent >= 0.25
            ):
                scores[ActionName.HANDS_ON_HIPS] = _classifier_weighted_average(
                    (bilateral_hip_proximity, 0.35),
                    (same_side_hips, 0.15),
                    (waist_height, 0.15),
                    (elbows_bent, 0.25),
                    (elbow_flare, 0.10),
                )

        if all((
            left_relation_wrist,
            right_relation_wrist,
            left_elbow,
            right_elbow,
            left_shoulder,
            right_shoulder,
        )):
            cross_distance = (
                _classifier_distance(left_relation_wrist, right_elbow)
                + _classifier_distance(right_relation_wrist, left_elbow)
            ) / (2.0 * scale)
            opposite_arm_proximity = _classifier_low(
                cross_distance,
                0.30,
                0.90,
            )
            elbows_bent = min(
                max(
                    _classifier_low(
                        _pose_angle(left_shoulder, left_elbow, left_relation_wrist),
                        100.0,
                        155.0,
                    ),
                    _classifier_low(
                        _pose_angle_3d(left_shoulder, left_elbow, left_relation_wrist),
                        100.0,
                        155.0,
                    ),
                ),
                max(
                    _classifier_low(
                        _pose_angle(right_shoulder, right_elbow, right_relation_wrist),
                        100.0,
                        155.0,
                    ),
                    _classifier_low(
                        _pose_angle_3d(right_shoulder, right_elbow, right_relation_wrist),
                        100.0,
                        155.0,
                    ),
                ),
            )
            shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
            hip_points = [point for point in (left_hip, right_hip) if point is not None]
            wrist_mid_y = (
                left_relation_wrist.y + right_relation_wrist.y
            ) / 2.0
            below_shoulders = _classifier_high(
                (wrist_mid_y - shoulder_mid_y) / scale,
                0.00,
                0.22,
            )
            above_hips = 1.0
            if hip_points:
                hip_mid_y = sum(point.y for point in hip_points) / len(hip_points)
                above_hips = _classifier_low(
                    (wrist_mid_y - hip_mid_y) / scale,
                    -0.20,
                    0.15,
                )
            shoulder_order = right_shoulder.x - left_shoulder.x
            wrist_order = right_relation_wrist.x - left_relation_wrist.x
            crossed_order = _classifier_high(
                -(shoulder_order * wrist_order) / (scale * scale),
                -0.05,
                0.25,
            )
            chest_zone = min(below_shoulders, above_hips)
            if (
                (
                    crossed_order >= 0.20
                    or opposite_arm_proximity >= 0.75
                )
                and elbows_bent >= 0.25
                and chest_zone >= 0.25
            ):
                scores[ActionName.ARMS_CROSSED] = _classifier_weighted_average(
                    (opposite_arm_proximity, 0.30),
                    (crossed_order, 0.20),
                    (elbows_bent, 0.25),
                    (chest_zone, 0.25),
                )

        if all((left_wrist, right_wrist, left_ear, right_ear)):
            same_side_distance = (
                _classifier_distance(left_wrist, left_ear)
                + _classifier_distance(right_wrist, right_ear)
            ) / (2.0 * scale)
            crossed_distance = (
                _classifier_distance(left_wrist, right_ear)
                + _classifier_distance(right_wrist, left_ear)
            ) / (2.0 * scale)
            scores[ActionName.HANDS_ON_HEAD] = _classifier_low(
                min(same_side_distance, crossed_distance), 0.25, 1.10
            )

        if all((left_knee, right_knee, left_shoulder, right_shoulder)):
            curled_distance = (
                _classifier_distance(left_knee, left_shoulder)
                + _classifier_distance(right_knee, right_shoulder)
            ) / (2.0 * scale)
            scores[ActionName.CURLED_UP] = min(
                _classifier_low(curled_distance, 0.8, 1.8),
                max(scores[ActionName.HEAD_DOWN], scores[ActionName.HUNCHED]),
            )

    @staticmethod
    def _score_hand_actions(
        scores: dict[ActionName, float],
        frame: AnalyzedFrame,
        left: HandLandmarkSet | None,
        right: HandLandmarkSet | None,
    ) -> None:
        hand_inputs = (
            (
                "left",
                left,
                frame.features.left_hand,
                frame.features.temporal.left_hand_motion_energy,
            ),
            (
                "right",
                right,
                frame.features.right_hand,
                frame.features.temporal.right_hand_motion_energy,
            ),
        )
        thumb_scores: list[float] = []
        victory_scores: list[float] = []
        pointing_scores: list[float] = []
        stop_scores: list[float] = []
        for side, landmarks, features, motion in hand_inputs:
            thumb_scores.append(RuleActionClassifier._thumbs_up(landmarks, features))
            victory_scores.append(RuleActionClassifier._victory(landmarks, features))
            pointing_scores.append(RuleActionClassifier._pointing(landmarks, features))
            stop_scores.append(
                RuleActionClassifier._stop_hand(
                    frame,
                    side,
                    landmarks,
                    features,
                    motion,
                )
            )
        scores[ActionName.THUMBS_UP] = max(thumb_scores)
        scores[ActionName.VICTORY] = max(victory_scores)
        scores[ActionName.POINTING] = max(pointing_scores)
        scores[ActionName.STOP_GESTURE] = max(stop_scores)

        RuleActionClassifier._score_hand_to_head_relations(scores, frame)
        bilateral_face_covering = RuleActionClassifier._bilateral_face_covering(frame)
        if bilateral_face_covering >= 0.55:
            # Both palms close to the face can look like two foreshortened Stop
            # hands, while their approach can also create a clap-like close/rebound
            # cycle. Bilateral face contact is the more specific posture.
            scores[ActionName.FACE_COVERING] = max(
                scores[ActionName.FACE_COVERING], bilateral_face_covering
            )
            scores[ActionName.STOP_GESTURE] = 0.0
            scores[ActionName.CLAPPING] = 0.0

    @staticmethod
    def _bilateral_face_covering(frame: AnalyzedFrame) -> float:
        anchor = RuleActionClassifier._reliable_face_anchor(frame)
        if anchor is None:
            return 0.0
        nose, scale = anchor
        pose_landmarks = frame.inference.data.pose_landmarks
        pose = frame.features.pose

        hands = (frame.features.left_hand, frame.features.right_hand)
        # ``face_covering`` means both hands cover the face. Pose wrists alone
        # are not sufficient: a close-up hand can make PoseLandmarker invent a
        # compact pseudo-person and place guessed wrists around a guessed nose.
        if not all(hand.detected for hand in hands):
            return 0.0
        detected_hand_proximity = [
            _classifier_low(
                math.hypot(hand.center_x - nose.x, hand.center_y - nose.y) / scale,
                0.25,
                0.90,
            )
            for hand in hands
            if hand.center_x is not None and hand.center_y is not None
        ]
        if len(detected_hand_proximity) != 2:
            return 0.0
        hand_score = min(detected_hand_proximity)

        left_wrist = _classifier_point(pose_landmarks, 15)
        right_wrist = _classifier_point(pose_landmarks, 16)
        if left_wrist is not None and right_wrist is not None:
            left_elbow = (
                pose.left_elbow_angle_3d_degrees
                if pose.left_elbow_angle_3d_degrees is not None
                else pose.left_elbow_angle_degrees
            )
            right_elbow = (
                pose.right_elbow_angle_3d_degrees
                if pose.right_elbow_angle_3d_degrees is not None
                else pose.right_elbow_angle_degrees
            )
            folded_elbows = min(
                _classifier_low(left_elbow, 85.0, 115.0),
                _classifier_low(right_elbow, 85.0, 115.0),
            )
            hand_score = min(hand_score, folded_elbows)

            # Two truly forward Stop arms can overlap the face in image space. Keep
            # them when both arms are straight and both wrists are clearly in front
            # of the nose according to Pose depth.
            straight_elbows = min(
                _classifier_high(left_elbow, 125.0, 160.0),
                _classifier_high(right_elbow, 125.0, 160.0),
            )
            forward_wrists = min(
                _classifier_high((nose.z - left_wrist.z) / scale, 0.05, 0.30),
                _classifier_high((nose.z - right_wrist.z) / scale, 0.05, 0.30),
            )
            forward_stop_escape = min(straight_elbows, forward_wrists)
            hand_score *= 1.0 - forward_stop_escape

        return hand_score

    @staticmethod
    def _reliable_face_anchor(
        frame: AnalyzedFrame,
    ) -> tuple[PoseLandmark, float] | None:
        """Return a face anchor only for a coherent current upper-body pose."""

        landmarks = frame.inference.data.pose_landmarks
        nose = _classifier_point(landmarks, 0, min_confidence=0.6)
        left_shoulder = _classifier_point(
            landmarks, 11, min_confidence=0.6
        )
        right_shoulder = _classifier_point(
            landmarks, 12, min_confidence=0.6
        )
        pose = frame.features.pose
        scale = pose.torso_length or pose.shoulder_width
        if (
            nose is None
            or left_shoulder is None
            or right_shoulder is None
            or scale is None
            or scale <= _EPSILON
            or pose.shoulder_width is None
            or pose.shoulder_width < 0.04
            or pose.visible_ratio < 0.25
        ):
            return None

        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2.0
        head_above_shoulders = (
            shoulder_mid_y - nose.y
        ) / pose.shoulder_width
        if head_above_shoulders < 0.10 or head_above_shoulders > 1.60:
            return None
        return nose, scale

    @staticmethod
    def _reliable_nod_anchor(
        frame: AnalyzedFrame,
    ) -> tuple[PoseLandmark, float] | None:
        """Require a coherent face/shoulder constellation for head motion."""

        if frame.inference.data.face_observed is False:
            return None
        anchor = RuleActionClassifier._reliable_face_anchor(frame)
        if anchor is None:
            return None

        landmarks = frame.inference.data.pose_landmarks
        left_ear = _classifier_point(landmarks, 7, min_confidence=0.55)
        right_ear = _classifier_point(landmarks, 8, min_confidence=0.55)
        if left_ear is None or right_ear is None:
            return None
        shoulder_width = frame.features.pose.shoulder_width
        if shoulder_width is None or shoulder_width <= _EPSILON:
            return None
        ear_width_ratio = (
            _classifier_distance(left_ear, right_ear) / shoulder_width
        )
        if ear_width_ratio < 0.08 or ear_width_ratio > 0.85:
            return None
        return anchor

    @staticmethod
    def _clap_zone_score(frame: AnalyzedFrame) -> float:
        landmarks = frame.inference.data.pose_landmarks
        left_shoulder = _classifier_point(landmarks, 11)
        right_shoulder = _classifier_point(landmarks, 12)
        left_hip = _classifier_point(landmarks, 23)
        right_hip = _classifier_point(landmarks, 24)
        scale = frame.features.pose.torso_length or frame.features.pose.shoulder_width
        # A desk, seated pose, or half-body camera view commonly hides the hips while
        # both shoulders and wrists remain reliable. Hips refine the lower boundary,
        # but must not be mandatory for a hand interaction such as clapping.
        if left_shoulder is None or right_shoulder is None:
            return 0.0
        if scale is None or scale <= _EPSILON:
            return 0.0

        hand_centers = [
            (features.center_x, features.center_y)
            for features in (frame.features.left_hand, frame.features.right_hand)
            if features.detected and features.center_x is not None and features.center_y is not None
        ]
        if len(hand_centers) < 2:
            left_wrist = _classifier_point(landmarks, 15)
            right_wrist = _classifier_point(landmarks, 16)
            if left_wrist is None or right_wrist is None:
                return 0.0
            hand_centers = [(left_wrist.x, left_wrist.y), (right_wrist.x, right_wrist.y)]

        shoulder_mid = (
            (left_shoulder.x + right_shoulder.x) / 2.0,
            (left_shoulder.y + right_shoulder.y) / 2.0,
        )
        hands_mid = (
            (hand_centers[0][0] + hand_centers[1][0]) / 2.0,
            (hand_centers[0][1] + hand_centers[1][1]) / 2.0,
        )
        horizontal = _classifier_low(abs(hands_mid[0] - shoulder_mid[0]) / scale, 0.35, 0.95)
        same_height = _classifier_low(
            abs(hand_centers[0][1] - hand_centers[1][1]) / scale,
            0.25,
            0.85,
        )
        upper_limit = shoulder_mid[1] - 0.25 * scale
        visible_hips = [point for point in (left_hip, right_hip) if point is not None]
        lower_limit = (
            sum(point.y for point in visible_hips) / len(visible_hips) + 0.30 * scale
            if visible_hips
            else shoulder_mid[1] + 2.40 * scale
        )
        if hands_mid[1] < upper_limit:
            vertical = _classifier_low((upper_limit - hands_mid[1]) / scale, 0.0, 0.45)
        elif hands_mid[1] > lower_limit:
            vertical = _classifier_low((hands_mid[1] - lower_limit) / scale, 0.0, 0.45)
        else:
            vertical = 1.0
        return min(horizontal, same_height, vertical)

    @staticmethod
    def _score_hand_to_head_relations(
        scores: dict[ActionName, float], frame: AnalyzedFrame
    ) -> None:
        landmarks = frame.inference.data.pose_landmarks
        pose = frame.features.pose
        left_ear = _classifier_point(landmarks, 7)
        right_ear = _classifier_point(landmarks, 8)
        scale = pose.torso_length or pose.shoulder_width
        if scale is None or scale <= _EPSILON:
            return
        centers = [
            (features.center_x, features.center_y)
            for features in (frame.features.left_hand, frame.features.right_hand)
            if features.detected and features.center_x is not None and features.center_y is not None
        ]
        if left_ear is not None and right_ear is not None and len(centers) >= 2:
            first, second = centers[:2]
            same_side = (
                math.hypot(first[0] - left_ear.x, first[1] - left_ear.y)
                + math.hypot(second[0] - right_ear.x, second[1] - right_ear.y)
            ) / (2.0 * scale)
            crossed = (
                math.hypot(first[0] - right_ear.x, first[1] - right_ear.y)
                + math.hypot(second[0] - left_ear.x, second[1] - left_ear.y)
            ) / (2.0 * scale)
            scores[ActionName.HANDS_ON_HEAD] = max(
                scores[ActionName.HANDS_ON_HEAD],
                _classifier_low(min(same_side, crossed), 0.25, 1.10),
            )

    @staticmethod
    def _resolve_conflicts(scores: dict[ActionName, float]) -> None:
        RuleActionClassifier._classifier_resolve_dynamic_static_pair(
            scores,
            dynamic=ActionName.CLAPPING,
            static=ActionName.ARMS_OPEN,
        )

        arms_crossed = scores[ActionName.ARMS_CROSSED]
        if arms_crossed >= 0.40:
            scores[ActionName.CLAPPING] = 0.0
            scores[ActionName.HANDS_ON_HIPS] = 0.0
            scores[ActionName.ARMS_OPEN] = 0.0
        elif scores[ActionName.CLAPPING] >= 0.55:
            scores[ActionName.ARMS_CROSSED] = 0.0

        clapping = scores[ActionName.CLAPPING]
        scores[ActionName.LARGE_ARM_SWING] *= 1.0 - clapping

        large_arm_swing = scores[ActionName.LARGE_ARM_SWING]
        waving = scores[ActionName.WAVING]
        if large_arm_swing >= 0.55:
            scores[ActionName.ARMS_OPEN] = 0.0
        dynamic_raised_arm = max(large_arm_swing, waving)
        if dynamic_raised_arm >= 0.55:
            scores[ActionName.ARMS_RAISED] = 0.0

        hands_on_head = scores[ActionName.HANDS_ON_HEAD]
        if hands_on_head >= 0.55:
            scores[ActionName.ARMS_RAISED] *= 1.0 - hands_on_head
            scores[ActionName.WAVING] *= 1.0 - hands_on_head
            scores[ActionName.LARGE_ARM_SWING] *= 1.0 - hands_on_head
            scores[ActionName.HEAD_DOWN] *= 1.0 - hands_on_head
            scores[ActionName.SHOULDERS_SLUMPED] *= 1.0 - hands_on_head

        victory = scores[ActionName.VICTORY]
        if victory >= 0.55:
            scores[ActionName.ARMS_RAISED] *= 1.0 - victory
            scores[ActionName.POINTING] *= 1.0 - victory

        stop_gesture = scores[ActionName.STOP_GESTURE]
        if stop_gesture >= _STOP_SCORE_THRESHOLD:
            for incompatible_action in (
                ActionName.WAVING,
                ActionName.LARGE_ARM_SWING,
                ActionName.ARMS_RAISED,
                ActionName.ARMS_OPEN,
                ActionName.POINTING,
            ):
                scores[incompatible_action] = 0.0

        hands_on_hips = scores[ActionName.HANDS_ON_HIPS]
        if hands_on_hips >= 0.55:
            # A palm resting on the waist can face the camera and appear large, but
            # bent, laterally flared elbows contradict an arm reaching forward.
            scores[ActionName.STOP_GESTURE] = 0.0
            scores[ActionName.ARMS_OPEN] = 0.0
            scores[ActionName.ARMS_RAISED] = 0.0
            scores[ActionName.WAVING] = 0.0
            scores[ActionName.LARGE_ARM_SWING] = 0.0

        arms_crossed = scores[ActionName.ARMS_CROSSED]
        if arms_crossed >= 0.40:
            # A palm resting across the chest may face the camera and satisfy
            # the permissive low-mounted-camera reach rule. The whole-body arm
            # geometry is stronger evidence that this is a folded-arm posture,
            # not an intentional hand held out as a Stop command.
            scores[ActionName.STOP_GESTURE] = 0.0

        lying = scores[ActionName.LYING]
        if lying >= 0.55:
            for arm_action in (
                ActionName.ARMS_OPEN,
                ActionName.ARMS_RAISED,
                ActionName.WAVING,
                ActionName.LARGE_ARM_SWING,
            ):
                scores[arm_action] = 0.0

        disruptive_motion = max(lying, scores[ActionName.JUMPING])
        scores[ActionName.STANDING] *= 1.0 - disruptive_motion
        scores[ActionName.SITTING] *= 1.0 - disruptive_motion
        if scores[ActionName.STANDING] > scores[ActionName.SITTING]:
            scores[ActionName.SITTING] = 0.0
        elif scores[ActionName.SITTING] > scores[ActionName.STANDING]:
            scores[ActionName.STANDING] = 0.0
        elif scores[ActionName.STANDING] > 0.0:
            scores[ActionName.STANDING] *= 0.5
            scores[ActionName.SITTING] *= 0.5

    @staticmethod
    def _classifier_resolve_dynamic_static_pair(
        scores: dict[ActionName, float],
        *,
        dynamic: ActionName,
        static: ActionName,
    ) -> None:
        dynamic_score = scores[dynamic]
        static_score = scores[static]
        if dynamic_score >= 0.55:
            scores[static] *= 1.0 - dynamic_score
        elif static_score >= 0.55:
            scores[dynamic] *= 1.0 - static_score

    @staticmethod
    def _thumbs_up(landmarks: HandLandmarkSet | None, features: HandFeatures) -> float:
        if landmarks is None or len(landmarks) < 21 or features.palm_scale is None:
            return 0.0
        thumb, index, middle, ring, pinky = features.extended_fingers
        thumb_straight = _classifier_high(features.finger_straightness_degrees[0], 140.0, 168.0)
        thumb_extended = max(1.0 if thumb else 0.0, thumb_straight)

        folded_scores: list[float] = []
        for finger_index, extended in enumerate((index, middle, ring, pinky), start=1):
            straightness = features.finger_straightness_degrees[finger_index]
            if straightness is None:
                folded_scores.append(1.0 if not extended else 0.0)
                continue
            angle_curl = _classifier_low(straightness, 140.0, 170.0)
            # The wrist-distance extension flag can flicker when a folded fingertip
            # is viewed obliquely.  A visibly bent joint is stronger curl evidence.
            binary_curl = 0.70 if not extended else 0.0
            folded_scores.append(max(angle_curl, binary_curl))
        folded_others = min(folded_scores)

        upward_ratio = (landmarks[2].y - landmarks[4].y) / max(features.palm_scale, _EPSILON)
        upward = _classifier_high(upward_ratio, 0.08, 0.55)
        return min(thumb_extended, folded_others, upward)

    @staticmethod
    def _victory(landmarks: HandLandmarkSet | None, features: HandFeatures) -> float:
        if landmarks is None or len(landmarks) < 21 or features.palm_scale is None:
            return 0.0
        thumb, index, middle, ring, pinky = features.extended_fingers
        if not index or not middle or ring or pinky:
            return 0.0
        finger_spread = _classifier_distance(landmarks[8], landmarks[12]) / max(
            features.palm_scale, _EPSILON
        )
        thumb_score = 1.0 if not thumb else 0.75
        return min(_classifier_high(finger_spread, 0.20, 0.55), thumb_score)

    @staticmethod
    def _pointing(
        landmarks: HandLandmarkSet | None,
        features: HandFeatures,
    ) -> float:
        if landmarks is None or len(landmarks) < 21:
            return 0.0
        _, index, middle, ring, pinky = features.extended_fingers
        if middle or ring or pinky:
            return 0.0
        index_straightness = features.finger_straightness_degrees[1]
        index_depth_ratio = (landmarks[5].z - landmarks[8].z) / max(
            features.palm_scale or _EPSILON,
            _EPSILON,
        )
        index_score = max(
            1.0 if index and index_straightness is None else 0.0,
            _classifier_high(index_straightness, 145.0, 172.0),
            _classifier_high(index_depth_ratio, 0.08, 0.40),
        )
        curl_scores: list[float] = []
        for finger_index in (2, 3, 4):
            straightness = features.finger_straightness_degrees[finger_index]
            curl_scores.append(_classifier_low(straightness, 125.0, 168.0))
        natural_curl = min(curl_scores)
        return min(index_score, natural_curl)

    @staticmethod
    def _stop_hand(
        frame: AnalyzedFrame,
        side: str,
        landmarks: HandLandmarkSet | None,
        features: HandFeatures,
        motion: float | None,
    ) -> float:
        if landmarks is None or len(landmarks) < 21 or features.palm_scale is None:
            return 0.0
        pose_landmarks = frame.inference.data.pose_landmarks
        pose = frame.features.pose
        if pose_landmarks is None:
            return 0.0
        arm_candidates = (
            (
                "left",
                _classifier_point(pose_landmarks, 11, min_confidence=0.60),
                _classifier_point(pose_landmarks, 13, min_confidence=0.60),
                _classifier_point(pose_landmarks, 15, min_confidence=0.60),
            ),
            (
                "right",
                _classifier_point(pose_landmarks, 12, min_confidence=0.60),
                _classifier_point(pose_landmarks, 14, min_confidence=0.60),
                _classifier_point(pose_landmarks, 16, min_confidence=0.60),
            ),
        )
        hand_wrist = landmarks[0]
        required_hand_points = (0, 5, 8, 9, 12, 13, 16, 17, 20)
        if any(
            not (-0.02 <= landmarks[index].x <= 1.02)
            or not (-0.02 <= landmarks[index].y <= 1.02)
            for index in required_hand_points
        ):
            # A cropped palm cannot prove that all four fingers are extended.
            return 0.0
        available_arms = [
            candidate
            for candidate in arm_candidates
            if candidate[1] is not None
            and candidate[2] is not None
            and candidate[3] is not None
        ]
        if not available_arms:
            return 0.0
        matching_arm = min(
            available_arms,
            key=lambda candidate: (
                (candidate[3].x - hand_wrist.x) ** 2
                + (candidate[3].y - hand_wrist.y) ** 2,
                0 if candidate[0] == side else 1,
            ),
        )
        _, shoulder, elbow, wrist = matching_arm
        elbow_angle_3d = _pose_angle_3d(shoulder, elbow, wrist)
        elbow_angle = (
            elbow_angle_3d
            if elbow_angle_3d is not None
            else _pose_angle(shoulder, elbow, wrist)
        )
        scale = pose.torso_length or pose.shoulder_width
        if shoulder is None or wrist is None or scale is None or scale <= _EPSILON:
            return 0.0

        four_finger_scores = [
            _classifier_high(features.finger_straightness_degrees[index], 135.0, 168.0)
            for index in (1, 2, 3, 4)
        ]
        fingers_straight = sum(four_finger_scores) / len(four_finger_scores)
        all_four_extended = all(features.extended_fingers[index] for index in (1, 2, 3, 4))
        # A Stop palm may have naturally spread or together fingers.  Only reject an
        # extreme spread; hand orientation is determined by the palm normal, not by
        # requiring the fingers to be vertical relative to the wrist.
        finger_spread_allowed = _classifier_low(features.finger_spread_ratio, 1.25, 1.80)
        palm_forward = _classifier_high(features.palm_facing_score, 0.45, 0.80)
        pose_depth_reach = _classifier_high((shoulder.z - wrist.z) / scale, 0.02, 0.30)
        arm_image_length = _classifier_distance(shoulder, wrist) / scale
        foreshortened_arm = _classifier_low(arm_image_length, 0.45, 1.20)
        foreground_hand = _classifier_high(features.palm_scale / scale, 0.25, 0.60)
        perspective_reach = min(foreshortened_arm, foreground_hand)
        forward_reach = max(pose_depth_reach, perspective_reach)
        straight_arm = _classifier_high(elbow_angle, 125.0, 165.0)
        wrist_association = _classifier_low(
            math.hypot(wrist.x - hand_wrist.x, wrist.y - hand_wrist.y) / scale,
            0.08,
            0.50,
        )
        stable_score = _classifier_low(motion if motion is not None else 0.0, 0.45, 1.40)
        if (
            not all_four_extended
            or min(four_finger_scores) < 0.35
            or finger_spread_allowed < 0.20
            or palm_forward < 0.35
            # A dog-head camera looks down from above. In that view either a
            # visibly forward/large palm OR a straight arm is sufficient
            # geometric evidence; requiring both rejects a natural bent-elbow
            # Stop pose. The open-palm and stability gates remain mandatory.
            or max(forward_reach, straight_arm) < 0.20
            or wrist_association < 0.20
            or stable_score < 0.25
            or (
                frame.features.temporal.wrist_motion_direction_changes > 0
                and (motion or 0.0) > 0.45
            )
        ):
            return 0.0
        return _classifier_weighted_average(
            (fingers_straight, 0.28),
            (finger_spread_allowed, 0.05),
            (palm_forward, 0.22),
            (forward_reach, 0.20),
            (straight_arm, 0.10),
            (wrist_association, 0.05),
            (stable_score, 0.10),
        )


class ActionSmoother:
    """Applies per-action majority windows and activation/deactivation hysteresis."""

    _THREE_OF_FIVE_ACTIONS = {
        ActionName.FAST_NOD,
        ActionName.JUMPING,
        ActionName.WAVING,
        ActionName.LARGE_ARM_SWING,
        ActionName.STOP_GESTURE,
        ActionName.FALL,
    }
    _POSTURE_ACTIONS = {ActionName.STANDING, ActionName.SITTING, ActionName.LYING}

    def __init__(self, window_size: int = 10, score_threshold: float = 0.55) -> None:
        if window_size < 5:
            raise ValueError("window_size must be at least 5")
        self._history: deque[dict[ActionName, float]] = deque(maxlen=window_size)
        self._score_threshold = score_threshold
        self._active: set[ActionName] = set()
        self._activated_at: dict[ActionName, float] = {}

    def update(
        self, raw_scores: dict[ActionName, float], timestamp_s: float
    ) -> tuple[ActionScore, ...]:
        if raw_scores.get(ActionName.FACE_COVERING, 0.0) >= 0.55:
            # The bilateral face rule has already zeroed these incompatible raw
            # actions.  Clear votes accumulated while the hands approached the face
            # so their faster 3/5 windows cannot linger over face_covering.
            face_covering_conflicts = tuple(
                action
                for action in (ActionName.STOP_GESTURE, ActionName.CLAPPING)
                if raw_scores.get(action, 0.0) < self._score_threshold
            )
            for historical_scores in self._history:
                for action in face_covering_conflicts:
                    historical_scores[action] = 0.0
            for action in face_covering_conflicts:
                self._active.discard(action)
                self._activated_at.pop(action, None)
        if (
            raw_scores.get(ActionName.HANDS_ON_HIPS, 0.0) >= 0.55
            and raw_scores.get(ActionName.STOP_GESTURE, 0.0) < self._score_threshold
        ):
            # Clear incompatible votes collected while the hands moved down and
            # the elbows flared outward toward the waist.
            hip_conflicts = (
                ActionName.STOP_GESTURE,
                ActionName.ARMS_OPEN,
                ActionName.ARMS_RAISED,
                ActionName.WAVING,
                ActionName.LARGE_ARM_SWING,
            )
            for historical_scores in self._history:
                for action in hip_conflicts:
                    historical_scores[action] = 0.0
            for action in hip_conflicts:
                self._active.discard(action)
                self._activated_at.pop(action, None)
        if raw_scores.get(ActionName.ARMS_CROSSED, 0.0) >= 0.40:
            # A confirmed chest-cross posture invalidates clap-like motion from the
            # transition.  Clear retained clap votes so the old dynamic label cannot
            # linger for several frames after the static posture becomes clear.
            cross_conflicts = (
                ActionName.CLAPPING,
                ActionName.STOP_GESTURE,
                ActionName.HANDS_ON_HIPS,
                ActionName.ARMS_OPEN,
            )
            for historical_scores in self._history:
                for action in cross_conflicts:
                    historical_scores[action] = 0.0
            for action in cross_conflicts:
                self._active.discard(action)
                self._activated_at.pop(action, None)
        if raw_scores.get(ActionName.FALL, 0.0) >= 0.55:
            incompatible_arm_actions = (
                ActionName.ARMS_OPEN,
                ActionName.ARMS_RAISED,
                ActionName.WAVING,
                ActionName.LARGE_ARM_SWING,
            )
            for historical_scores in self._history:
                for action in incompatible_arm_actions:
                    historical_scores[action] = 0.0
            for action in incompatible_arm_actions:
                self._active.discard(action)
                self._activated_at.pop(action, None)
        if raw_scores.get(ActionName.STOP_GESTURE, 0.0) >= _STOP_SCORE_THRESHOLD:
            incompatible_stop_actions = (
                ActionName.WAVING,
                ActionName.LARGE_ARM_SWING,
                ActionName.ARMS_RAISED,
                ActionName.ARMS_OPEN,
                ActionName.POINTING,
            )
            for historical_scores in self._history:
                for action in incompatible_stop_actions:
                    historical_scores[action] = 0.0
            for action in incompatible_stop_actions:
                self._active.discard(action)
                self._activated_at.pop(action, None)
        self._history.append(raw_scores)
        output: list[ActionScore] = []
        for name in ActionName:
            window, activation_ratio, deactivation_ratio, threshold = self._policy(name)
            recent = list(self._history)[-window:]
            if len(recent) < window:
                continue
            supporting = [scores.get(name, 0.0) for scores in recent]
            support_ratio = sum(score >= threshold for score in supporting) / window
            if name in self._active:
                if support_ratio < deactivation_ratio:
                    self._active.remove(name)
                    self._activated_at.pop(name, None)
                    continue
            elif support_ratio >= activation_ratio:
                self._active.add(name)
                self._activated_at[name] = timestamp_s
            else:
                continue

            confidence = sum(supporting) / window
            output.append(
                ActionScore(
                    name=name,
                    confidence=confidence,
                    support_ratio=support_ratio,
                    duration_s=max(0.0, timestamp_s - self._activated_at[name]),
                )
            )
        return tuple(
            sorted(
                output,
                key=lambda action: (*action_priority(action.name), -action.confidence),
            )
        )

    def clear(self, *names: ActionName) -> None:
        """Immediately discard labels whose required observation disappeared."""

        for historical_scores in self._history:
            for name in names:
                historical_scores[name] = 0.0
        for name in names:
            self._active.discard(name)
            self._activated_at.pop(name, None)

    def _policy(self, name: ActionName) -> tuple[int, float, float, float]:
        if name is ActionName.CLAPPING:
            # The raw clap rule already contains an 18-frame close/open cycle.  Keep
            # a short majority window as a second guard against similar arm motion.
            return 5, 3 / 5, 2 / 5, self._score_threshold
        if name in (ActionName.VICTORY, ActionName.THUMBS_UP):
            return 7, 5 / 7, 3 / 7, self._score_threshold
        if name is ActionName.ARMS_OPEN:
            return self._history.maxlen or 10, 0.6, 0.3, min(self._score_threshold, 0.50)
        if name is ActionName.STOP_GESTURE:
            return 5, 3 / 5, 2 / 5, max(self._score_threshold, _STOP_SCORE_THRESHOLD)
        if name in self._THREE_OF_FIVE_ACTIONS:
            return 5, 3 / 5, 2 / 5, self._score_threshold
        if name is ActionName.SITTING:
            return self._history.maxlen or 10, 0.5, 0.3, 0.45
        if name in self._POSTURE_ACTIONS:
            return self._history.maxlen or 10, 0.6, 0.3, min(self._score_threshold, 0.50)
        return self._history.maxlen or 10, 0.8, 0.4, self._score_threshold


class ActionRecognizer:
    """Combines raw explainable rules with temporal label smoothing."""

    def __init__(self) -> None:
        self._classifier = RuleActionClassifier()
        self._smoother = ActionSmoother()
        self._fall_events = FallEventManager()

    def recognize(
        self, frame: AnalyzedFrame
    ) -> tuple[
        tuple[ActionScore, ...],
        tuple[tuple[ActionName, float], ...],
        FallEventStatus,
    ]:
        raw_scores = self._classifier.classify(frame)
        timestamp_s = frame.inference.captured.monotonic_ns / 1_000_000_000.0
        pose_detected = frame.inference.data.pose_landmarks is not None
        upright_from_not_lying = (
            _classifier_low(raw_scores[ActionName.LYING], 0.15, 0.45) if pose_detected else 0.0
        )
        fall_status = self._fall_events.update(
            frame,
            lying_score=raw_scores[ActionName.LYING],
            upright_score=max(
                raw_scores[ActionName.STANDING],
                raw_scores[ActionName.SITTING],
                upright_from_not_lying,
            ),
        )
        # Keep the event score high briefly so the existing 3/5 action smoother can
        # expose a stable P0 label.  event_triggered itself remains a one-frame edge
        # for an alert transport and is never repeated while the person stays down.
        raw_scores[ActionName.FALL] = 1.0 if fall_status.alert_active else 0.0
        if frame.inference.data.left_hand is None and frame.inference.data.right_hand is None:
            # Stop is a current visual safety command, not a posture that may
            # survive after both hands leave the frame.  This also removes stale
            # votes when an upper-body crop no longer contains either hand.
            self._smoother.clear(ActionName.STOP_GESTURE)
        if RuleActionClassifier._bilateral_face_covering(frame) <= 0.0:
            # Face covering is a current bilateral observation. Never attach a
            # retained face-covering vote to a lone hand after the person or the
            # second hand has left the frame.
            self._smoother.clear(ActionName.FACE_COVERING)
        if RuleActionClassifier._reliable_nod_anchor(frame) is None:
            # Nose motion is meaningless without a current, coherent head and
            # shoulders. Clear old votes immediately when only a hand remains.
            self._smoother.clear(ActionName.FAST_NOD)
        if (
            RuleActionClassifier._reliable_face_anchor(frame) is None
            or not (
                frame.features.left_hand.detected
                and frame.features.right_hand.detected
            )
        ):
            self._smoother.clear(ActionName.HANDS_ON_HEAD)
        actions = self._smoother.update(raw_scores, timestamp_s)
        ranked_scores = tuple(
            sorted(
                raw_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        return actions, ranked_scores, fall_status


# ==============================================================================
# PUBLIC STANDALONE API
# ==============================================================================


@dataclass(frozen=True, slots=True)
class LandmarkFrame:
    """One tracked person's framework-independent landmark observation."""

    monotonic_s: float
    pose_landmarks: PoseLandmarkSet | None = None
    left_hand: HandLandmarkSet | None = None
    right_hand: HandLandmarkSet | None = None
    timestamp_s: float | None = None
    fps: float = 0.0
    face_observed: bool | None = None


@dataclass(frozen=True, slots=True)
class BehaviorResult:
    """Complete observable-action result returned for one input frame."""

    frame: LandmarkFrame
    features: FrameFeatures
    actions: tuple[ActionScore, ...]
    raw_scores: tuple[tuple[ActionName, float], ...]
    fall_status: FallEventStatus
    feature_ms: float
    recognition_ms: float

    @property
    def primary_action(self) -> ActionScore | None:
        return self.actions[0] if self.actions else None

    @property
    def primary_priority(self) -> TriggerPriority | None:
        primary = self.primary_action
        return ACTION_TRIGGER_PRIORITIES[primary.name] if primary is not None else None

    @property
    def state_hint(self) -> StateHint | None:
        priority = self.primary_priority
        return PRIORITY_STATE_HINTS[priority] if priority is not None else None

    @property
    def raw_score_map(self) -> dict[ActionName, float]:
        return dict(self.raw_scores)


class BehaviorEngine:
    """Stateful single-person facade around features, rules, smoothing, and fall events."""

    def __init__(self, window_size: int = 30) -> None:
        if window_size < 2:
            raise ValueError("window_size must be at least 2")
        self._window_size = window_size
        self._samples: deque[TemporalLandmarkSample] = deque(maxlen=window_size)
        self._recognizer = ActionRecognizer()
        self._sequence = 0
        self._last_monotonic_s: float | None = None
        self._empty_image: ImageArray = np.empty((0, 0, 3), dtype=np.uint8)

    def update(self, frame: LandmarkFrame) -> BehaviorResult:
        """Consume one observation without camera, UI, network, or robot side effects."""

        self._validate_landmarks(frame)
        if self._last_monotonic_s is not None and frame.monotonic_s <= self._last_monotonic_s:
            raise ValueError("monotonic_s must strictly increase; call reset() for a new stream")
        self._last_monotonic_s = frame.monotonic_s
        self._sequence += 1
        timestamp_s = frame.timestamp_s if frame.timestamp_s is not None else frame.monotonic_s
        data = FrameData(
            timestamp=timestamp_s,
            pose_landmarks=frame.pose_landmarks,
            left_hand=frame.left_hand,
            right_hand=frame.right_hand,
            fps=frame.fps,
            face_observed=frame.face_observed,
        )
        captured = CapturedFrame(
            sequence=self._sequence,
            timestamp=timestamp_s,
            monotonic_ns=int(frame.monotonic_s * 1_000_000_000),
            image=self._empty_image,
        )
        inference = InferenceFrame(captured=captured, data=data, inference_ms=0.0)
        self._samples.append(
            TemporalLandmarkSample(
                sequence=self._sequence,
                monotonic_s=frame.monotonic_s,
                data=data,
            )
        )

        feature_started_ns = time.perf_counter_ns()
        features = FrameFeatures(
            pose=extract_pose_features(frame.pose_landmarks),
            left_hand=extract_hand_features(frame.left_hand),
            right_hand=extract_hand_features(frame.right_hand),
            temporal=extract_temporal_features(tuple(self._samples)),
        )
        feature_ms = (time.perf_counter_ns() - feature_started_ns) / 1_000_000.0
        analyzed = AnalyzedFrame(
            inference=inference,
            features=features,
            feature_ms=feature_ms,
        )

        recognition_started_ns = time.perf_counter_ns()
        actions, raw_scores, fall_status = self._recognizer.recognize(analyzed)
        recognition_ms = (time.perf_counter_ns() - recognition_started_ns) / 1_000_000.0
        return BehaviorResult(
            frame=frame,
            features=features,
            actions=actions,
            raw_scores=raw_scores,
            fall_status=fall_status,
            feature_ms=feature_ms,
            recognition_ms=recognition_ms,
        )

    def reset(self) -> None:
        """Drop landmark history, smoothing votes, and fall-event state."""

        self._samples.clear()
        self._recognizer = ActionRecognizer()
        self._sequence = 0
        self._last_monotonic_s = None

    @staticmethod
    def _validate_landmarks(frame: LandmarkFrame) -> None:
        expected_counts = (
            ("pose_landmarks", frame.pose_landmarks, 33),
            ("left_hand", frame.left_hand, 21),
            ("right_hand", frame.right_hand, 21),
        )
        for name, landmarks, expected in expected_counts:
            if landmarks is not None and len(landmarks) != expected:
                raise ValueError(f"{name} must contain {expected} landmarks")


class _LandmarkLike(Protocol):
    x: float
    y: float
    z: float


def pose_landmarks_from_objects(
    points: Sequence[_LandmarkLike] | None,
) -> PoseLandmarkSet | None:
    """Copy MediaPipe-like objects into the standalone pose contract."""

    if points is None:
        return None
    return tuple(
        PoseLandmark(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z),
            visibility=float(getattr(point, "visibility", 1.0)),
            presence=float(getattr(point, "presence", 1.0)),
        )
        for point in points
    )


def hand_landmarks_from_objects(
    points: Sequence[_LandmarkLike] | None,
) -> HandLandmarkSet | None:
    """Copy MediaPipe-like objects into the standalone hand contract."""

    if points is None:
        return None
    return tuple(
        HandLandmark(
            x=float(point.x),
            y=float(point.y),
            z=float(point.z),
            visibility=float(getattr(point, "visibility", 1.0)),
            presence=float(getattr(point, "presence", 1.0)),
        )
        for point in points
    )


__all__ = [
    "ACTION_GROUPS",
    "ACTION_TRIGGER_PRIORITIES",
    "PRIORITY_STATE_HINTS",
    "ActionGroup",
    "ActionName",
    "ActionScore",
    "BehaviorEngine",
    "BehaviorResult",
    "FallDetectionConfig",
    "FallEventStatus",
    "FallPhase",
    "HandLandmark",
    "HandLandmarkSet",
    "LandmarkFrame",
    "PoseLandmark",
    "PoseLandmarkSet",
    "StateHint",
    "TriggerPriority",
    "hand_landmarks_from_objects",
    "pose_landmarks_from_objects",
]
