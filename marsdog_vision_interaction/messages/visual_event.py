"""Versioned data contract for ``/perception/visual_event``."""

from __future__ import annotations

import copy
from typing import Any

from marsdog_vision_interaction.utils.time_utils import now_stamp


SCHEMA_VERSION = 1

_ACTIVE_TARGET = {
    "track_id": 0,
    "face_track_id": -1,
    "identity": "unknown",
    "identity_confidence": 0.0,
    "identity_state": "unverified",
    # Kept as compatibility placeholders. Cross-modal values belong to
    # /perception/target_event and are never populated by this project.
    "speaker_id": "unknown",
    "is_speaking": False,
    "is_registered": False,
    "bbox": [0.0, 0.0, 0.0, 0.0],
    "face_bbox": [0.0, 0.0, 0.0, 0.0],
    "face_center": [0.0, 0.0],
    "body_center": [0.0, 0.0],
    "pose_state": "unknown",
    "pose_action": "",
    "pose_action_label": "",
    "keypoints": [],
    "confidence": 0.0,
    "face_confidence": 0.0,
    "speaker_confidence": 0.0,
    "selection_reason": "",
    "tracking_state": "lost",
    "last_seen_age_ms": -1.0,
}

_FACE = {
    "track_id": -1,
    "x": 0.0,
    "y": 0.0,
    "w": 0.0,
    "h": 0.0,
    "confidence": 0.0,
    "recognized_user": "",
    "identity_confidence": 0.0,
    "identity_state": "unverified",
    "quality": 0.0,
}

_HUMAN = {
    "track_id": -1,
    "x": 0.0,
    "y": 0.0,
    "w": 0.0,
    "h": 0.0,
    "confidence": 0.0,
    "pose_state": "",
    "pose_action": "",
    "pose_action_label": "",
    "keypoints": [],
}

_HAND = {
    "handedness": "",
    "hand_action": "",
    "hand_action_label": "",
    "landmarks": [],
}

_OBJECT = {
    "label": "",
    "x": 0.0,
    "y": 0.0,
    "w": 0.0,
    "h": 0.0,
    "confidence": 0.0,
    "center_x": 0.0,
    "center_y": 0.0,
}


def _merge(template: dict[str, Any], value: Any) -> dict[str, Any]:
    result = copy.deepcopy(template)
    if not isinstance(value, dict):
        return result
    for key, default in template.items():
        if key not in value:
            continue
        if isinstance(default, list):
            result[key] = copy.deepcopy(value[key]) if isinstance(value[key], list) else []
        else:
            try:
                result[key] = type(default)(value[key])
            except (TypeError, ValueError):
                pass
    return result


def make_empty_visual_event() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "header": {"stamp": now_stamp(), "frame_id": "camera_link"},
        "active_target": copy.deepcopy(_ACTIVE_TARGET),
        "faces": [],
        "humans": [],
        "hands": [],
        "tracked_objects": [],
        "events": [],
    }


def normalize_visual_event(data: Any) -> dict[str, Any]:
    """Normalize a provider result into the stable public topic payload."""
    event = make_empty_visual_event()
    if not isinstance(data, dict):
        return event

    header = data.get("header")
    if isinstance(header, dict):
        try:
            event["header"]["stamp"] = float(
                header.get("stamp", event["header"]["stamp"])
            )
        except (TypeError, ValueError):
            pass
        event["header"]["frame_id"] = str(
            header.get("frame_id", event["header"]["frame_id"])
        )

    event["active_target"] = _merge(_ACTIVE_TARGET, data.get("active_target"))
    event["faces"] = [
        _merge(_FACE, item) for item in data.get("faces", [])
        if isinstance(item, dict)
    ]
    event["humans"] = [
        _merge(_HUMAN, item) for item in data.get("humans", [])
        if isinstance(item, dict)
    ]
    event["hands"] = [
        _merge(_HAND, item) for item in data.get("hands", [])
        if isinstance(item, dict)
    ]
    event["tracked_objects"] = [
        _merge(_OBJECT, item) for item in data.get("tracked_objects", [])
        if isinstance(item, dict)
    ]
    if isinstance(data.get("events"), list):
        event["events"] = [str(item) for item in data["events"] if item]
    return event
