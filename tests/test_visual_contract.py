from pathlib import Path
import time

import pytest

from marsdog_vision_interaction.core.held_object_pose import (
    HOLDING_DOG_FOOD,
    HeldObjectPoseManager,
)
from marsdog_vision_interaction.core.visual_target_manager import (
    VisualTargetManager,
)
from marsdog_vision_interaction.messages.visual_event import (
    normalize_visual_event,
)
from marsdog_vision_interaction.messages import visual_event_types
from marsdog_vision_interaction.providers.gesture_pose_engine import ActionName
from marsdog_vision_interaction.providers.pose_action import (
    _HAND_ACTIONS,
    get_action_category,
    get_action_label,
)


def test_victory_is_a_public_happy_hand_action() -> None:
    assert _HAND_ACTIONS[ActionName.VICTORY] == "victory"
    assert get_action_category("victory") == "hand"
    assert get_action_label("victory") == "胜利/V字手势"
    assert (
        visual_event_types.pose_action_to_vision_event(
            "victory", identity_confirmed=True
        )
        == visual_event_types.EVT_VISION_MASTER_HAPPY
    )
    assert (
        visual_event_types.pose_action_to_vision_event(
            "victory", identity_confirmed=False
        )
        == ""
    )


def test_visual_contract_preserves_identity_and_pose() -> None:
    value = normalize_visual_event({
        "active_target": {
            "track_id": 7,
            "identity": "alice",
            "pose_action": "arm_raise_wave",
            "held_object": {
                "state": "confirmed",
                "action": "holding_toy",
                "object_label": "dog toy ball",
                "evidence_hits": 2,
            },
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
    assert value["active_target"]["held_object"]["state"] == "confirmed"
    assert value["active_target"]["held_object"]["action"] == "holding_toy"
    assert value["active_target"]["held_object"]["required_hits"] == 2
    assert value["active_target"]["held_object"]["object_result_sequence"] == 0
    assert value["active_target"]["held_object"]["rejection_reason"] == ""
    assert value["active_target"]["held_object"][
        "evaluated_wrist_distance_ratio"
    ] is None
    assert value["active_target"]["range_source"] == "none"
    assert value["active_target"]["depth_sync_delta_ms"] is None
    assert value["faces"][0]["track_id"] == 17


def test_visual_contract_preserves_object_result_provenance() -> None:
    value = normalize_visual_event({
        "tracked_objects": [{
            "label": "dog treat bag",
            "confidence": 0.5621,
            "bbox": [0.72, 0.44, 0.20, 0.42],
            "tracking_state": "tracking",
            "source_sequence": 314,
            "source_snapshot_id": "epoch-1:object-result:314",
            "header": {
                "stamp": 1788424354.5,
                "frame_id": "camera_color_optical_frame",
            },
        }],
    })

    detected = value["tracked_objects"][0]
    assert detected["source_sequence"] == 314
    assert detected["source_snapshot_id"] == "epoch-1:object-result:314"
    assert detected["header"] == {
        "stamp": 1788424354.5,
        "frame_id": "camera_color_optical_frame",
    }
    assert "_gesture_diagnostics" not in value


def test_normalized_object_results_can_confirm_held_food() -> None:
    manager = HeldObjectPoseManager()
    target = {
        "target_id": "epoch-1:human:1",
        "track_id": 1,
        "tracking_state": "tracking",
        "confidence": 0.9,
        "bbox": [0.40, 0.10, 0.48, 0.84],
        "keypoints": [{
            "id": 16,
            "x": 0.82,
            "y": 0.54,
            "confidence": 0.9,
            "presence": 0.9,
        }],
    }

    def normalized_object(sequence: int, stamp: float) -> dict:
        event = normalize_visual_event({
            "tracked_objects": [{
                "label": "dog treat bag",
                "confidence": 0.56,
                "bbox": [0.78, 0.48, 0.18, 0.40],
                "tracking_state": "tracking",
                "source_sequence": sequence,
                "source_snapshot_id": f"epoch-1:object-result:{sequence}",
                "header": {"stamp": stamp, "frame_id": "camera_link"},
            }],
        })
        return event["tracked_objects"][0]

    first = manager.update(
        now=10.0,
        active_target=target,
        hands=[],
        objects=[normalized_object(314, 100.0)],
        object_result_sequence=314,
        pose_observation_stamp=100.1,
    )
    confirmed = manager.update(
        now=10.5,
        active_target=target,
        hands=[],
        objects=[normalized_object(315, 100.5)],
        object_result_sequence=315,
        pose_observation_stamp=100.6,
    )

    assert first.state == "candidate"
    assert first.rejection_reason == ""
    assert confirmed.state == "confirmed"
    assert confirmed.action == HOLDING_DOG_FOOD


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
