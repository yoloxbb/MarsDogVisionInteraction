"""Versioned data contract for ``/perception/visual_event``."""

from __future__ import annotations

import copy
from typing import Any

from marsdog_vision_interaction.utils.time_utils import now_stamp


SCHEMA_VERSION = 1

_HELD_OBJECT = {
    "state": "inactive",
    "action": "",
    "action_label": "",
    "candidate_action": "",
    "object_label": "",
    "object_track_id": -1,
    "hand_source": "",
    "association_score": 0.0,
    "wrist_distance_ratio": None,
    "evidence_hits": 0,
    "required_hits": 2,
    "last_positive_age_ms": None,
    "object_result_sequence": 0,
    "rejection_reason": "",
    "evaluated_object_label": "",
    "evaluated_object_confidence": 0.0,
    "pose_object_sync_delta_ms": None,
    "valid_wrist_count": 0,
    "evaluated_wrist_distance_ratio": None,
    "wrist_distance_threshold_ratio": 0.0,
}

_ACTIVE_TARGET = {
    "vision_epoch": "",
    "target_id": "",
    "target_type": "human",
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
    "center": [0.0, 0.0],
    "face_bbox": [0.0, 0.0, 0.0, 0.0],
    "face_center": [0.0, 0.0],
    "body_center": [0.0, 0.0],
    "pose_state": "unknown",
    "pose_action": "",
    "pose_action_label": "",
    "held_object": copy.deepcopy(_HELD_OBJECT),
    "keypoints": [],
    "confidence": 0.0,
    "detection_confidence": 0.0,
    "face_confidence": 0.0,
    "speaker_confidence": 0.0,
    "selection_reason": "",
    "tracking_state": "lost",
    "last_seen_age_ms": -1.0,
    "bearing_deg": 0.0,
    "bearing_valid": False,
    "bearing_source": "",
    "range_valid": False,
    "distance_m": None,
    "range_source": "none",
    "depth_sync_delta_ms": None,
    "pose_3d": {
        "valid": False,
        "frame_id": "",
        "x": None,
        "y": None,
        "z": None,
    },
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
    "held_object": copy.deepcopy(_HELD_OBJECT),
    "keypoints": [],
}

_HUMAN_CANDIDATE = {
    "vision_epoch": "",
    "target_id": "",
    "target_type": "human",
    "track_id": -1,
    "face_track_id": -1,
    "identity": "unknown",
    "identity_confidence": 0.0,
    "identity_state": "unverified",
    "is_registered": False,
    "bbox": [0.0, 0.0, 0.0, 0.0],
    "center": [0.0, 0.0],
    "body_center": [0.0, 0.0],
    "face_bbox": [0.0, 0.0, 0.0, 0.0],
    "face_center": [0.0, 0.0],
    "pose_state": "unknown",
    "pose_action": "",
    "pose_action_label": "",
    "held_object": copy.deepcopy(_HELD_OBJECT),
    "keypoints": [],
    "confidence": 0.0,
    "detection_confidence": 0.0,
    "face_confidence": 0.0,
    "tracking_state": "lost",
    "last_seen_age_ms": -1.0,
    "bearing_deg": 0.0,
    "bearing_valid": False,
    "bearing_source": "",
    "range_valid": False,
    "distance_m": None,
    "range_source": "none",
    "depth_sync_delta_ms": None,
    "pose_3d": {
        "valid": False,
        "frame_id": "",
        "x": None,
        "y": None,
        "z": None,
    },
}

_HAND = {
    "handedness": "",
    "hand_action": "",
    "hand_action_label": "",
    "landmarks": [],
}

_OBJECT = {
    # Preserve the exact detector-result provenance.  Held-object association
    # counts distinct detector results and must not confuse a persisted track
    # from an older result with evidence from the current inference cycle.
    "header": {
        "stamp": 0.0,
        "frame_id": "",
    },
    "source_sequence": 0,
    "source_snapshot_id": "",
    "vision_epoch": "",
    "target_id": "",
    "target_type": "object",
    "track_id": -1,
    "object_track_id": -1,
    "label": "",
    "display_label": "",
    "object_kind": "generic",
    "x": 0.0,
    "y": 0.0,
    "w": 0.0,
    "h": 0.0,
    "confidence": 0.0,
    "center_x": 0.0,
    "center_y": 0.0,
    "bbox": [0.0, 0.0, 0.0, 0.0],
    "center": [0.0, 0.0],
    "detection_confidence": 0.0,
    "tracking_state": "lost",
    "last_seen_age_ms": -1.0,
    "bearing_deg": 0.0,
    "bearing_valid": False,
    "bearing_source": "",
    "range_valid": False,
    "distance_m": None,
    "range_source": "none",
    "depth_sync_delta_ms": None,
    "pose_3d": {
        "valid": False,
        "frame_id": "",
        "x": None,
        "y": None,
        "z": None,
    },
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
        elif isinstance(default, dict):
            result[key] = (
                _merge(default, value[key])
                if isinstance(value[key], dict)
                else copy.deepcopy(default)
            )
        elif default is None:
            incoming = value[key]
            if incoming is None:
                result[key] = None
            else:
                try:
                    result[key] = float(incoming)
                except (TypeError, ValueError):
                    result[key] = None
        elif isinstance(default, bool):
            result[key] = value[key] if isinstance(value[key], bool) else default
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
        "vision_epoch": "",
        "snapshot_id": "",
        "sequence": 0,
        "active_target": copy.deepcopy(_ACTIVE_TARGET),
        "human_candidates": [],
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

    event["vision_epoch"] = str(data.get("vision_epoch", ""))
    event["snapshot_id"] = str(data.get("snapshot_id", ""))
    try:
        event["sequence"] = max(0, int(data.get("sequence", 0)))
    except (TypeError, ValueError):
        event["sequence"] = 0

    event["active_target"] = _merge(_ACTIVE_TARGET, data.get("active_target"))
    event["human_candidates"] = [
        _merge(_HUMAN_CANDIDATE, item)
        for item in data.get("human_candidates", [])
        if isinstance(item, dict)
    ]
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
