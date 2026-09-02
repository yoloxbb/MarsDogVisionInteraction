from pathlib import Path
import time

import pytest

from marsdog_vision_interaction.core.visual_target_manager import (
    VisualTargetManager,
)
from marsdog_vision_interaction.messages.visual_event import (
    normalize_visual_event,
)
from marsdog_vision_interaction.messages import visual_event_types
from marsdog_vision_interaction.providers.gesture_pose_engine import ActionName


def test_visual_contract_preserves_identity_and_pose() -> None:
    value = normalize_visual_event({
        "active_target": {
            "track_id": 7,
            "identity": "alice",
            "pose_action": "arm_raise_wave",
        },
        "faces": [{
            "track_id": 17,
            "recognized_user": "alice",
        }],
        "_gesture_diagnostics": {"primary_action": "victory"},
    })
    assert value["schema_version"] == 1
    assert value["active_target"]["identity"] == "alice"
    assert value["active_target"]["pose_action"] == "arm_raise_wave"
    assert value["active_target"]["range_source"] == "none"
    assert value["active_target"]["depth_sync_delta_ms"] is None
    assert value["faces"][0]["track_id"] == 17
    assert "_gesture_diagnostics" not in value


def test_visual_target_manager_has_no_audio_update_api() -> None:
    manager = VisualTargetManager()
    manager.update_vision(
        [{
            "x": 0.1,
            "y": 0.1,
            "w": 0.4,
            "h": 0.8,
            "confidence": 0.9,
        }],
        [],
    )
    assert manager.get_active_target().track_id == 1
    assert not hasattr(manager, "update_audio")


def test_visual_target_exposes_freshness_and_loss_state() -> None:
    manager = VisualTargetManager()
    manager.update_vision([{
        "x": 0.1, "y": 0.1, "w": 0.4, "h": 0.8,
        "confidence": 0.9,
    }], [])
    current = manager.get_active_dict()
    assert current["tracking_state"] == "tracking"
    assert current["last_seen_age_ms"] >= 0

    manager._active.last_seen_at = time.time() - 1.0
    stale = manager.get_active_dict()
    assert stale["tracking_state"] == "temporarily_lost"


def test_visual_target_uses_torso_center_instead_of_limb_bbox_center() -> None:
    manager = VisualTargetManager()
    manager.update_vision(
        [{
            "x": 0.1, "y": 0.05, "w": 0.8, "h": 0.9,
            "confidence": 0.9,
            "keypoints": [
                {"id": 11, "x": 0.40, "y": 0.30, "confidence": 0.9},
                {"id": 12, "x": 0.50, "y": 0.30, "confidence": 0.9},
                {"id": 23, "x": 0.42, "y": 0.60, "confidence": 0.9},
                {"id": 24, "x": 0.48, "y": 0.60, "confidence": 0.9},
            ],
        }],
        [],
    )
    assert manager.get_active_dict()["body_center"] == pytest.approx(
        [0.45, 0.45]
    )


def test_visual_target_survives_ephemeral_face_track_id_change() -> None:
    manager = VisualTargetManager(vision_epoch="face-churn")
    human = {
        "x": 0.2,
        "y": 0.1,
        "w": 0.4,
        "h": 0.8,
        "confidence": 0.9,
    }
    first_face = {
        "x": 0.34,
        "y": 0.14,
        "w": 0.12,
        "h": 0.16,
        "confidence": 0.9,
        "track_id": 10,
    }
    manager.update_vision([human], [first_face])

    second_face = dict(first_face, track_id=11)
    manager.update_vision([human], [second_face])

    candidates = manager.get_human_candidates()
    assert len(candidates) == 1
    assert candidates[0]["track_id"] == 1
    assert candidates[0]["face_track_id"] == 11
    assert candidates[0]["target_id"] == "face-churn:human:1"


def test_visual_source_does_not_import_voice_project() -> None:
    root = Path(__file__).parents[1] / "marsdog_vision_interaction"
    source = "\n".join(
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
    )
    assert "marsdog_voice_interaction" not in source
    assert "marsdog_perception." not in source


def test_ros2_contract_catalogs_all_visual_events_and_exact_actions() -> None:
    root = Path(__file__).parents[1]
    contract = (root / "docs" / "ROS2_CONTRACT.md").read_text(
        encoding="utf-8"
    )
    manifest = (
        root / "docs" / "integration" / "interface_manifest.yaml"
    ).read_text(encoding="utf-8")

    declared_events = {
        value
        for name, value in vars(visual_event_types).items()
        if name.startswith("EVT_VISION_") and isinstance(value, str)
    }
    for event_name in declared_events:
        assert event_name in contract
        assert event_name in manifest

    for action in ActionName:
        assert f"`{action.value}`" in contract

    assert "当前会实际产生" in contract
    assert "预留事件" in contract
    assert "事件可能在连续消息中重复" in contract
