"""Rendering helpers for the live vision/control diagnostic view."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


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

    cv2.line(
        output, (width // 2, 0), (width // 2, height), (0, 255, 255), 1
    )
    # Solid centre, inner stop band (±0.08), outer activation band (±0.14).
    for ratio in (0.42, 0.58, 0.36, 0.64):
        x = int(width * ratio)
        cv2.line(output, (x, 0), (x, height), (80, 80, 80), 1)

    for human in event.get("humans", []):
        if not isinstance(human, dict):
            continue
        x1, y1, x2, y2 = _pixel_box(human, width, height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 220, 0), 2)
        _text(
            output,
            f"body id={human.get('track_id', -1)} "
            f"h={float(human.get('h', 0)):.3f}",
            x1,
            max(18, y1 - 6),
            (0, 255, 0),
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

    mode = str(
        control.get(
            "mode", "disabled" if not control.get("enabled") else "centering"
        )
    )
    layout = str(control.get("_input_layout", ""))
    layout_text = f" source={layout}" if layout else ""
    _text(
        output,
        f"input={width}x{height}{layout_text} mode={mode}",
        10,
        22,
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
