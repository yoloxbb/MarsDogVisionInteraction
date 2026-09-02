"""Rendering helpers for the live vision/control diagnostic view."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


_POSE_CONNECTIONS = (
    (0, 7), (0, 8), (7, 11), (8, 12),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
)

_HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)

_MAX_OBJECT_OVERLAYS = 30


def _pixel_box(
    item: dict[str, Any], width: int, height: int
) -> tuple[int, int, int, int]:
    x1 = int(float(item.get("x", 0.0)) * width)
    y1 = int(float(item.get("y", 0.0)) * height)
    x2 = int(
        (float(item.get("x", 0.0)) + float(item.get("w", 0.0))) * width
    )
    y2 = int(
        (float(item.get("y", 0.0)) + float(item.get("h", 0.0))) * height
    )
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(0, min(width - 1, x2)),
        max(0, min(height - 1, y2)),
    )


def _text(
    frame: np.ndarray,
    value: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        frame, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        frame, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
        0.52, color, 1, cv2.LINE_AA,
    )


def _draw_landmarks(
    frame: np.ndarray,
    landmarks: Any,
    connections: tuple[tuple[int, int], ...],
    color: tuple[int, int, int],
    *,
    min_confidence: float = 0.2,
) -> dict[int, tuple[int, int]]:
    """Draw a normalized landmark graph and return its visible pixel points."""
    if not isinstance(landmarks, list):
        return {}
    height, width = frame.shape[:2]
    points: dict[int, tuple[int, int]] = {}
    for fallback_id, landmark in enumerate(landmarks):
        if not isinstance(landmark, dict):
            continue
        try:
            confidence = float(landmark.get("confidence", 1.0))
            if confidence < min_confidence:
                continue
            point_id = int(landmark.get("id", fallback_id))
            x = int(float(landmark.get("x", 0.0)) * width)
            y = int(float(landmark.get("y", 0.0)) * height)
        except (TypeError, ValueError):
            continue
        if 0 <= x < width and 0 <= y < height:
            points[point_id] = (x, y)

    for first, second in connections:
        if first in points and second in points:
            cv2.line(frame, points[first], points[second], color, 2, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(frame, point, 3, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 2, color, -1, cv2.LINE_AA)
    return points


def draw_visual_debug(
    frame: np.ndarray,
    event: dict[str, Any] | None,
    *,
    control: dict[str, Any] | None = None,
    cmd_vel: tuple[float, float] | None = None,
) -> np.ndarray:
    """Draw normalized detections and control state on a BGR frame."""
    output = frame.copy()
    height, width = output.shape[:2]
    event = event or {}
    control = control or {}
    object_only = control.get("mode") == "object_only"

    if not object_only:
        cv2.line(
            output,
            (width // 2, 0),
            (width // 2, height),
            (0, 255, 255),
            1,
        )
        # Solid centre, inner stop band (±0.08), outer activation band (±0.14).
        for ratio in (0.42, 0.58, 0.36, 0.64):
            x = int(width * ratio)
            cv2.line(output, (x, 0), (x, height), (80, 80, 80), 1)

    tracked_objects = [
        item for item in event.get("tracked_objects", [])
        if isinstance(item, dict)
    ]
    tracked_objects.sort(
        key=lambda item: float(item.get("confidence", 0.0) or 0.0),
        reverse=True,
    )
    for detected_object in tracked_objects[:_MAX_OBJECT_OVERLAYS]:
        x1, y1, x2, y2 = _pixel_box(detected_object, width, height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 220), 2)
        _text(
            output,
            f"object {detected_object.get('label', '?')} "
            f"{float(detected_object.get('confidence', 0.0)):.2f}",
            x1,
            max(18, y1 - 6),
            (255, 80, 240),
        )

    for human in event.get("humans", []):
        if not isinstance(human, dict):
            continue
        x1, y1, x2, y2 = _pixel_box(human, width, height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        _text(
            output,
            f"body id={human.get('track_id', -1)} "
            f"{human.get('pose_state', '')} "
            f"{human.get('pose_action', '')}",
            x1,
            max(18, y1 - 6),
            (0, 255, 0),
        )
        _draw_landmarks(
            output,
            human.get("keypoints", []),
            _POSE_CONNECTIONS,
            (0, 255, 0),
        )

    for hand in event.get("hands", []):
        if not isinstance(hand, dict):
            continue
        points = _draw_landmarks(
            output,
            hand.get("landmarks", []),
            _HAND_CONNECTIONS,
            (20, 120, 255),
            min_confidence=0.0,
        )
        if points:
            anchor = points.get(0, next(iter(points.values())))
            _text(
                output,
                f"{hand.get('handedness', 'hand')} "
                f"{hand.get('hand_action', '')}",
                anchor[0],
                max(18, anchor[1] - 8),
                (30, 180, 255),
            )

    for face in event.get("faces", []):
        if not isinstance(face, dict):
            continue
        x1, y1, x2, y2 = _pixel_box(face, width, height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 180, 0), 2)
        name = str(face.get("recognized_user", "") or "unknown")
        _text(
            output,
            f"face {name} {float(face.get('confidence', 0)):.2f}",
            x1,
            min(height - 8, y2 + 18),
            (255, 220, 0),
        )

    active = event.get("active_target", {})
    if (
        isinstance(active, dict)
        and float(active.get("confidence", 0.0) or 0.0) > 0.0
    ):
        values = active.get("bbox", [0, 0, 0, 0])
        if not isinstance(values, (list, tuple)) or len(values) < 4:
            values = [0, 0, 0, 0]
        bbox = {"x": values[0], "y": values[1], "w": values[2], "h": values[3]}
        x1, y1, x2, y2 = _pixel_box(bbox, width, height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 2)
        center = active.get("body_center", [0.0, 0.0])
        if not isinstance(center, (list, tuple)) or len(center) < 2:
            center = [0.0, 0.0]
        cx, cy = int(float(center[0]) * width), int(float(center[1]) * height)
        cv2.drawMarker(
            output, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, 20, 2
        )
        state = str(active.get("tracking_state", "lost"))
        _text(
            output,
            f"ACTIVE id={active.get('track_id', 0)} {state}",
            10, 46, (0, 80, 255),
        )
        _text(
            output,
            f"center_x={float(center[0]):.3f} "
            f"err={float(center[0]) - 0.5:+.3f} "
            f"bbox_h={float(bbox['h']):.3f}",
            10, 68, (0, 80, 255),
        )
        _text(
            output,
            f"pose={active.get('pose_state', '')} "
            f"action={active.get('pose_action', '')}",
            10, 90, (0, 80, 255),
        )

    mode = str(
        control.get(
            "mode", "disabled" if not control.get("enabled") else "centering"
        )
    )
    layout = str(control.get("_input_layout", ""))
    layout_text = f" source={layout}" if layout else ""
    visual_age_ms = control.get("_visual_age_ms")
    freshness = ""
    if isinstance(visual_age_ms, (int, float)):
        freshness = f" visual_age={float(visual_age_ms):.0f}ms"
    _text(
        output,
        f"input={width}x{height}{layout_text} mode={mode}{freshness}",
        10,
        22,
    )
    if event.get("events"):
        _text(
            output,
            "events=" + ",".join(str(item) for item in event["events"][:3]),
            10,
            height - 36,
            (80, 180, 255),
        )
    gesture_debug = control.get("_gesture_debug", {})
    if isinstance(gesture_debug, dict) and gesture_debug:
        recognized = gesture_debug.get("recognized_actions", [])
        names = [
            str(item.get("name", ""))
            for item in recognized
            if isinstance(item, dict) and item.get("name")
        ]
        if names:
            gesture_text = "recognized=" + ",".join(names[:4])
        else:
            candidates = gesture_debug.get("raw_scores", [])
            top = [
                f"{item.get('name')}:{float(item.get('score', 0.0)):.2f}"
                for item in candidates[:3]
                if isinstance(item, dict)
            ]
            gesture_text = "candidates=" + ",".join(top)
        _text(output, gesture_text, 10, 112, (80, 255, 255))
        fall = gesture_debug.get("fall_detector", {})
        if isinstance(fall, dict) and fall:
            _text(
                output,
                f"fall={fall.get('phase', '?')} "
                f"armed={bool(fall.get('armed', False))} "
                f"lying={float(fall.get('lying_score', 0.0)):.2f} "
                f"transition={float(fall.get('transition_score', 0.0)):.2f}",
                10,
                134,
                (80, 255, 255),
            )
    if cmd_vel is not None:
        _text(
            output,
            f"cmd linear.x={cmd_vel[0]:+.3f} angular.z={cmd_vel[1]:+.3f}",
            10,
            height - 14,
            (255, 255, 0),
        )
    return output
