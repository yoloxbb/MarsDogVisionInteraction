import copy
import threading
import time
from types import SimpleNamespace

import numpy as np

from marsdog_vision_interaction.core.visual_target_manager import (
    VisualTargetManager,
)
from marsdog_vision_interaction.nodes.vision_interaction_node import (
    VisionInteractionNode,
)


def _human(x: float, confidence: float = 0.9) -> dict:
    return {
        "x": x,
        "y": 0.1,
        "w": 0.25,
        "h": 0.8,
        "confidence": confidence,
        "keypoints": [],
    }


def _snapshot_node(manager: VisualTargetManager) -> SimpleNamespace:
    return SimpleNamespace(
        _target_manager=manager,
        _vision_epoch=manager.vision_epoch,
        _state_lock=threading.Lock(),
        _latest_camera_stamp=123.5,
        _latest_camera_frame_id="camera_color_optical_frame",
        _visual_sequence=0,
        _depth_enabled=False,
    )


def test_multi_human_snapshot_has_stable_ids_bearings_and_sequence() -> None:
    manager = VisualTargetManager(
        vision_epoch="epoch-a",
        horizontal_fov_deg=70.0,
    )
    manager.update_vision([_human(0.05), _human(0.70, 0.8)], [])
    node = _snapshot_node(manager)

    first = VisionInteractionNode._build_visual_snapshot(node, {
        "header": {
            "stamp": 123.5,
            "frame_id": "camera_color_optical_frame",
        },
    })
    second = VisionInteractionNode._build_visual_snapshot(node, {
        "header": {
            "stamp": 123.6,
            "frame_id": "camera_color_optical_frame",
        },
    })

    assert first["vision_epoch"] == "epoch-a"
    assert first["sequence"] == 1
    assert first["snapshot_id"] == "epoch-a:1"
    assert second["sequence"] == 2
    assert second["snapshot_id"] == "epoch-a:2"
    assert len(first["human_candidates"]) == 2
    assert len({
        item["target_id"] for item in first["human_candidates"]
    }) == 2
    assert first["human_candidates"][0]["bearing_deg"] < 0.0
    assert first["human_candidates"][1]["bearing_deg"] > 0.0


def test_query_targets_returns_one_atomic_camera_snapshot() -> None:
    manager = VisualTargetManager(vision_epoch="epoch-query")
    manager.update_vision([_human(0.25), _human(0.60, 0.4)], [])
    node = _snapshot_node(manager)
    snapshot = VisionInteractionNode._build_visual_snapshot(node, {
        "header": {
            "stamp": 456.0,
            "frame_id": "camera_color_optical_frame",
        },
    })
    node._latest_visual_snapshot = copy.deepcopy(snapshot)
    node._latest_visual_snapshot_monotonic = time.monotonic()
    node._target_current_timeout_sec = 0.35

    result = VisionInteractionNode._query_targets(node, {
        "target_types": ["human"],
        "min_confidence": 0.5,
    })

    assert result["ok"] is True
    assert result["header"] == snapshot["header"]
    assert result["header"]["frame_id"] == "camera_color_optical_frame"
    assert result["vision_epoch"] == "epoch-query"
    assert result["sequence"] == snapshot["sequence"]
    assert result["snapshot_id"] == snapshot["snapshot_id"]
    assert len(result["targets"]) == 1
    assert result["targets"] == result["human_candidates"]


def test_query_reages_cached_target_and_invalidates_old_range() -> None:
    node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _target_current_timeout_sec=0.35,
        _latest_visual_snapshot_monotonic=time.monotonic() - 0.5,
        _latest_visual_snapshot={
            "schema_version": 1,
            "header": {"stamp": 10.0, "frame_id": "camera"},
            "vision_epoch": "epoch-stale",
            "sequence": 3,
            "snapshot_id": "epoch-stale:3",
            "active_target": {
                "target_id": "epoch-stale:human:1",
                "last_seen_age_ms": 5.0,
                "tracking_state": "tracking",
                "range_valid": True,
                "distance_m": 2.0,
                "pose_3d": {"valid": True, "x": 0.0, "y": 0.0, "z": 2.0},
            },
            "human_candidates": [{
                "target_id": "epoch-stale:human:1",
                "target_type": "human",
                "detection_confidence": 0.9,
                "last_seen_age_ms": 5.0,
                "tracking_state": "tracking",
                "range_valid": True,
                "distance_m": 2.0,
                "pose_3d": {"valid": True, "x": 0.0, "y": 0.0, "z": 2.0},
            }],
        },
    )

    result = VisionInteractionNode._query_targets(node, {})

    target = result["targets"][0]
    assert target["last_seen_age_ms"] >= 500.0
    assert target["tracking_state"] == "temporarily_lost"
    assert target["range_valid"] is False
    assert target["distance_m"] is None
    assert result["active_target"]["range_valid"] is False


def test_repeated_publish_cannot_refresh_a_frozen_inference_target() -> None:
    manager = VisualTargetManager(vision_epoch="epoch-frozen")
    manager.update_vision([_human(0.3)], [])
    manager._active.last_seen_at -= 0.6
    manager._active.last_seen_monotonic -= 0.6
    for track in manager._tracks.values():
        track.last_seen_at -= 0.6
        track.last_seen_monotonic -= 0.6
    node = _snapshot_node(manager)
    raw = {
        "header": {
            "stamp": 300.0,
            "frame_id": "camera_color_optical_frame",
        },
    }

    first = VisionInteractionNode._build_visual_snapshot(node, raw)
    second = VisionInteractionNode._build_visual_snapshot(node, raw)

    assert second["sequence"] == first["sequence"] + 1
    target = second["human_candidates"][0]
    assert target["last_seen_age_ms"] >= 600.0
    assert target["tracking_state"] == "temporarily_lost"
    assert target["range_valid"] is False


def test_depth_decode_supports_padded_16uc1_rows() -> None:
    rows = np.array([
        [1000, 2000, 9999],
        [3000, 4000, 9999],
    ], dtype="<u2")
    message = SimpleNamespace(
        encoding="16UC1",
        width=2,
        height=2,
        step=6,
        is_bigendian=0,
        data=rows.tobytes(),
    )

    decoded = VisionInteractionNode._decode_depth_image(message)

    assert decoded is not None
    np.testing.assert_allclose(
        decoded,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    malformed = copy.copy(message)
    malformed.step = 2
    assert VisionInteractionNode._decode_depth_image(malformed) is None


def test_depth_fusion_uses_robust_torso_roi_and_valid_intrinsics() -> None:
    stamp = 100.0
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    depth[10, 10] = 7.0  # One centre outlier must not control the range.
    node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _depth_enabled=True,
        _latest_depth_m=depth,
        _latest_depth_stamp=stamp,
        _latest_depth_monotonic=time.monotonic(),
        _latest_depth_frame_id="camera_color_optical_frame",
        _camera_intrinsics={
            "fx": 100.0,
            "fy": 100.0,
            "cx": 9.5,
            "cy": 9.5,
            "width": 20,
            "height": 20,
            "frame_id": "camera_color_optical_frame",
        },
        _depth_stale_timeout_sec=0.5,
        _depth_sync_tolerance_sec=0.1,
        _depth_min_m=0.2,
        _depth_max_m=8.0,
        _depth_sample_radius_px=2,
        _depth_min_valid_samples=5,
        _depth_min_valid_fraction=0.2,
    )
    candidates = [{
        "target_id": "epoch:human:1",
        "bbox": [0.1, 0.1, 0.8, 0.8],
        "body_center": [0.5, 0.5],
        "tracking_state": "tracking",
    }]

    VisionInteractionNode._fuse_human_depth(
        node,
        candidates,
        observation_stamp=stamp,
    )

    target = candidates[0]
    assert target["range_valid"] is True
    assert 1.99 <= target["distance_m"] <= 2.01
    assert target["pose_3d"]["valid"] is True
    assert target["pose_3d"]["frame_id"] == "camera_color_optical_frame"

    node._camera_intrinsics["fx"] = 0.0
    VisionInteractionNode._fuse_human_depth(
        node,
        candidates,
        observation_stamp=stamp,
    )
    assert candidates[0]["range_valid"] is False
    assert candidates[0]["distance_m"] is None


def test_depth_fusion_matches_delayed_observation_to_historical_frame() -> None:
    stamp = 100.0
    now = time.monotonic()
    matching_depth = np.full((20, 20), 2.5, dtype=np.float16)
    newest_depth = np.full((20, 20), 0.8, dtype=np.float16)
    node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _depth_enabled=True,
        _depth_history=[
            (
                stamp - 0.02,
                now - 0.30,
                "camera_color_optical_frame",
                matching_depth,
            ),
            (
                stamp + 0.30,
                now,
                "camera_color_optical_frame",
                newest_depth,
            ),
        ],
        _latest_depth_m=newest_depth,
        _latest_depth_stamp=stamp + 0.30,
        _latest_depth_monotonic=now,
        _latest_depth_frame_id="camera_color_optical_frame",
        _camera_intrinsics={
            "fx": 100.0,
            "fy": 100.0,
            "cx": 9.5,
            "cy": 9.5,
            "width": 20,
            "height": 20,
            "frame_id": "camera_color_optical_frame",
        },
        _depth_stale_timeout_sec=0.5,
        _depth_sync_tolerance_sec=0.1,
        _depth_min_m=0.2,
        _depth_max_m=8.0,
        _depth_sample_radius_px=2,
        _depth_min_valid_samples=5,
        _depth_min_valid_fraction=0.2,
    )
    candidates = [{
        "target_id": "epoch:human:1",
        "bbox": [0.1, 0.1, 0.8, 0.8],
        "body_center": [0.5, 0.5],
        "tracking_state": "tracking",
    }]

    VisionInteractionNode._fuse_human_depth(
        node,
        candidates,
        observation_stamp=stamp,
    )

    target = candidates[0]
    assert target["range_valid"] is True
    assert 2.49 <= target["distance_m"] <= 2.51
    assert target["range_source"] == "aligned_depth"
    assert 19.9 <= target["depth_sync_delta_ms"] <= 20.1


def test_depth_fusion_rejects_history_outside_sync_tolerance() -> None:
    stamp = 200.0
    now = time.monotonic()
    depth = np.full((20, 20), 1.0, dtype=np.float16)
    node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _depth_enabled=True,
        _depth_history=[(
            stamp + 0.15,
            now,
            "camera_color_optical_frame",
            depth,
        )],
        _latest_depth_m=depth,
        _latest_depth_stamp=stamp + 0.15,
        _latest_depth_monotonic=now,
        _latest_depth_frame_id="camera_color_optical_frame",
        _camera_intrinsics={
            "fx": 100.0,
            "fy": 100.0,
            "cx": 9.5,
            "cy": 9.5,
            "width": 20,
            "height": 20,
            "frame_id": "camera_color_optical_frame",
        },
        _depth_stale_timeout_sec=0.5,
        _depth_sync_tolerance_sec=0.1,
        _depth_min_m=0.2,
        _depth_max_m=8.0,
        _depth_sample_radius_px=2,
        _depth_min_valid_samples=5,
        _depth_min_valid_fraction=0.2,
    )
    candidates = [{
        "target_id": "epoch:human:1",
        "bbox": [0.1, 0.1, 0.8, 0.8],
        "body_center": [0.5, 0.5],
        "tracking_state": "tracking",
    }]

    VisionInteractionNode._fuse_human_depth(
        node,
        candidates,
        observation_stamp=stamp,
    )

    assert candidates[0]["range_valid"] is False
    assert candidates[0]["distance_m"] is None
    assert candidates[0]["depth_sync_delta_ms"] is None


def test_query_tracks_animals_and_toys_with_ids_not_labels() -> None:
    now = time.monotonic()
    node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _vision_epoch="epoch-objects",
        _object_tracks={},
        _next_object_track_id=1,
        _object_target_current_timeout_sec=0.75,
        _object_target_persistence_sec=2.0,
        _horizontal_fov_deg=69.0,
        _target_current_timeout_sec=0.35,
        _latest_visual_snapshot_monotonic=now,
        _latest_visual_snapshot={
            "schema_version": 1,
            "header": {
                "stamp": 200.0,
                "frame_id": "camera_color_optical_frame",
            },
            "vision_epoch": "epoch-objects",
            "sequence": 4,
            "snapshot_id": "epoch-objects:4",
            "active_target": {},
            "human_candidates": [],
        },
    )
    first_detections = [
        {
            "label": "cat", "x": 0.1, "y": 0.2,
            "w": 0.2, "h": 0.3, "confidence": 0.91,
        },
        {
            "label": "dog toy ball", "x": 0.6, "y": 0.5,
            "w": 0.15, "h": 0.15, "confidence": 0.88,
        },
    ]
    with node._state_lock:
        VisionInteractionNode._update_object_tracks_locked(
            node,
            first_detections,
            now_monotonic=now,
            observation_stamp=199.9,
            frame_id="camera_color_optical_frame",
            source_sequence=11,
        )

    first = VisionInteractionNode._query_targets(node, {
        "target_types": ["animal", "object"],
        "min_confidence": 0.5,
    })
    first_ids = {item["label"]: item["target_id"] for item in first["targets"]}
    assert first_ids["cat"].startswith("epoch-objects:object:")
    assert first_ids["cat"] != "cat"
    assert first["animal_candidates"][0]["label"] == "cat"
    assert first["object_candidates"][0]["object_kind"] == "toy"
    assert first["targets"][0]["header"]["frame_id"] == (
        "camera_color_optical_frame"
    )

    moved = copy.deepcopy(first_detections)
    moved[0]["x"] += 0.01
    moved[1]["x"] += 0.01
    with node._state_lock:
        VisionInteractionNode._update_object_tracks_locked(
            node,
            moved,
            now_monotonic=time.monotonic(),
            observation_stamp=200.1,
            frame_id="camera_color_optical_frame",
            source_sequence=12,
        )
    second = VisionInteractionNode._query_targets(node, {
        "target_types": ["animal", "object"],
    })
    second_ids = {
        item["label"]: item["target_id"] for item in second["targets"]
    }
    assert second_ids == first_ids
    assert all(item["source_sequence"] == 12 for item in second["targets"])
