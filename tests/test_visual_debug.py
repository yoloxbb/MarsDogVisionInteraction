import numpy as np

from marsdog_vision_interaction.providers.vision_observation import (
    VisionObservationProvider,
)
from marsdog_vision_interaction.utils.visual_debug import draw_visual_debug


def test_draw_visual_debug_keeps_resolution_and_draws_overlay() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    event = {
        "humans": [{
            "track_id": 3, "x": 0.2, "y": 0.1, "w": 0.3, "h": 0.7,
        }],
        "faces": [{
            "x": 0.3, "y": 0.15, "w": 0.1, "h": 0.15,
            "confidence": 0.9,
        }],
        "active_target": {
            "track_id": 3,
            "confidence": 0.9,
            "tracking_state": "tracking",
            "bbox": [0.2, 0.1, 0.3, 0.7],
            "body_center": [0.35, 0.45],
        },
    }
    result = draw_visual_debug(
        frame,
        event,
        control={"enabled": True, "mode": "follow_owner"},
        cmd_vel=(0.1, -0.2),
    )
    assert result.shape == frame.shape
    assert np.any(result != frame)


def test_side_by_side_eye_crop_has_single_view_resolution() -> None:
    frame = np.zeros((240, 640, 3), dtype=np.uint8)
    frame[:, 320:] = 255
    left = VisionObservationProvider._select_stereo_view(frame, "left")
    right = VisionObservationProvider._select_stereo_view(frame, "right")
    assert left.shape == right.shape == (240, 320, 3)
    assert not left.any()
    assert right.all()


def test_normal_widescreen_frame_is_not_mistaken_for_stereo() -> None:
    frame = np.zeros((240, 424, 3), dtype=np.uint8)
    selected = VisionObservationProvider._select_stereo_view(frame, "left")
    assert selected.shape == (240, 424, 3)
