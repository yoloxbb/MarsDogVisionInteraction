from marsdog_vision_interaction.nodes.vision_interaction_node import (
    VisionInteractionNode,
)


def _face_observation(identity: str) -> dict:
    return {
        "active_target": {
            "identity": identity,
            "identity_state": (
                "unverified" if identity == "unknown" else "confirmed_known"
            ),
            "tracking_state": "tracking",
            "pose_action": "",
        },
        "faces": [{"recognized_user": "" if identity == "unknown" else identity}],
        "hands": [],
        "tracked_objects": [],
    }


def test_stranger_face_always_emits_only_generic_stranger_event() -> None:
    assert VisionInteractionNode._derive_events(
        _face_observation("unknown")
    ) == ["EVT_VISION_STRANGER"]


def test_known_face_keeps_master_event() -> None:
    assert VisionInteractionNode._derive_events(
        _face_observation("owner")
    ) == ["EVT_VISION_MASTER"]
