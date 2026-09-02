from collections import deque
from pathlib import Path
import threading
from types import MethodType
from types import SimpleNamespace

import numpy as np

from marsdog_vision_interaction.nodes import (
    vision_debug_viewer_node as viewer_node_module,
)
from marsdog_vision_interaction.providers.vision_observation import (
    VisionObservationProvider,
)
from marsdog_vision_interaction.nodes.vision_debug_viewer_node import (
    VisionDebugViewerNode,
)
from marsdog_vision_interaction.utils.visual_debug import draw_visual_debug
from marsdog_vision_interaction.utils.web_debug_server import (
    VisionDebugWebServer,
)
from marsdog_vision_interaction.utils.config_loader import load_config


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


def test_debug_overlay_draws_pose_hand_and_object_results() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    event = {
        "humans": [{
            "track_id": 9,
            "x": 0.35,
            "y": 0.2,
            "w": 0.3,
            "h": 0.7,
            "pose_state": "standing",
            "pose_action": "arm_raise_wave",
            "keypoints": [
                {"id": 11, "x": 0.4, "y": 0.35, "confidence": 1.0},
                {"id": 12, "x": 0.6, "y": 0.35, "confidence": 1.0},
            ],
        }],
        "hands": [{
            "handedness": "Left",
            "hand_action": "stop_gesture",
            "landmarks": [
                {"id": 0, "x": 0.7, "y": 0.5},
                {"id": 1, "x": 0.75, "y": 0.45},
            ],
        }],
        "tracked_objects": [{
            "label": "ball",
            "x": 0.1,
            "y": 0.1,
            "w": 0.15,
            "h": 0.15,
            "confidence": 0.9,
        }],
    }
    result = draw_visual_debug(frame, event)
    assert tuple(result[30, 20]) == (255, 0, 220)
    assert np.any(result[70, 80:121] != 0)
    assert np.any(result[90:101, 140:151] != 0)


def test_object_only_overlay_omits_follow_alignment_guides() -> None:
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    result = draw_visual_debug(
        frame,
        {},
        control={"mode": "object_only"},
    )

    assert not np.any(result[100, 100])


def test_web_dashboard_asset_and_jpeg_state_are_packaged() -> None:
    server = VisionDebugWebServer(
        "127.0.0.1", 0, lambda: {"ok": True, "camera": {"fps": 30.0}}
    )
    assert "MarsDog 视觉识别调试" in server._html.decode("utf-8")
    assert "GesturePose 原始判定" in server._html.decode("utf-8")
    assert "关键点模型 A/B" in server._html.decode("utf-8")
    assert "MarsDog 物体识别调试" in server._html.decode("utf-8")
    assert "object-only" in server._html.decode("utf-8")
    assert "在线人脸录入" in server._html.decode("utf-8")
    assert "family_member_4" in server._html.decode("utf-8")
    assert "list_face_records" in server._html.decode("utf-8")
    assert "/api/vision/task" in server._html.decode("utf-8")
    assert "Vision 已发布事件记录" in server._html.decode("utf-8")
    assert "visual-panel" in server._html.decode("utf-8")
    assert "event-history-panel" in server._html.decode("utf-8")
    assert "detectObjectsOnce" in server._html.decode("utf-8")
    assert "startObjectStream" in server._html.decode("utf-8")
    assert "stopObjectStream" in server._html.decode("utf-8")
    assert "visionTask('detect_objects'" in server._html.decode("utf-8")
    assert "visionTask('set_object_detection'" in server._html.decode("utf-8")
    assert "/api/debug/event-history/clear" in server._html.decode("utf-8")
    assert "不代表行为树已选中或 Action 已执行" in server._html.decode("utf-8")
    assert server._snapshot_provider()["camera"]["fps"] == 30.0
    server.update_jpeg(b"jpeg-data")
    assert server._jpeg == b"jpeg-data"
    assert server._frame_sequence == 1


def test_web_task_limits_object_stream_control_to_debug_session(
    monkeypatch,
) -> None:
    class FakeRequest:
        pass

    class FakeFuture:
        def add_done_callback(self, callback) -> None:
            callback(self)

        def result(self) -> SimpleNamespace:
            return SimpleNamespace(
                success=True,
                result_json='{"ok":true,"objects":[{"label":"ball"}]}',
                error_message="",
                latency_ms=321.5,
            )

    class FakeClient:
        request = None

        def service_is_ready(self) -> bool:
            return True

        def call_async(self, request):
            self.request = request
            return FakeFuture()

    monkeypatch.setattr(
        viewer_node_module,
        "VisionTask",
        SimpleNamespace(Request=FakeRequest),
    )
    client = FakeClient()
    viewer = SimpleNamespace(
        _vision_task_client=client,
        _lock=threading.Lock(),
        _management_sequence=0,
    )

    result = VisionDebugViewerNode._handle_web_task(
        viewer, "detect_objects", {"confidence": 0.45}
    )
    assert result["ok"] is True
    assert result["objects"] == [{"label": "ball"}]
    assert result["latency_ms"] == 321.5
    assert client.request.task_type == "detect_objects"
    assert client.request.params_json == '{"confidence": 0.45}'
    stream_result = VisionDebugViewerNode._handle_web_task(
        viewer,
        "set_object_detection",
        {"enabled": True, "session_id": "vision-debug-web"},
    )
    assert stream_result["ok"] is True
    assert client.request.task_type == "set_object_detection"
    assert VisionDebugViewerNode._handle_web_task(
        viewer,
        "set_object_detection",
        {"enabled": True, "session_id": "action-owned"},
    ) == {
        "ok": False,
        "error": "web object session_id must be vision-debug-web",
    }


def test_unified_debug_launch_starts_vision_and_viewer() -> None:
    root = Path(__file__).resolve().parents[1]
    launch_text = (root / "launch" / "vision_debug.launch.py").read_text(
        encoding="utf-8"
    )

    assert (
        'DeclareLaunchArgument("start_vision_node", default_value="true")'
        in launch_text
    )
    assert 'executable="vision_interaction"' in launch_text
    assert 'condition=IfCondition(start_vision_node)' in launch_text
    assert 'executable="vision_debug_viewer"' in launch_text
    assert not (root / "launch" / "object_debug.launch.py").exists()


def _event_history_viewer() -> SimpleNamespace:
    viewer = SimpleNamespace(
        _lock=threading.Lock(),
        _published_event_history=deque(maxlen=20),
        _active_published_events={},
        _event_history_epoch="",
        _event_history_sequence=0,
        _event_history_next_id=0,
    )
    for name in (
        "_new_published_event_record",
        "_exit_active_published_events",
        "_record_visual_event_lifecycle",
        "_clear_published_event_history",
    ):
        setattr(
            viewer,
            name,
            MethodType(getattr(VisionDebugViewerNode, name), viewer),
        )
    viewer._published_event_evidence = (
        VisionDebugViewerNode._published_event_evidence
    )
    return viewer


def _visual_packet(
    sequence: int,
    events: list[str],
    *,
    epoch: str = "epoch-a",
) -> dict:
    return {
        "schema_version": 1,
        "header": {"stamp": 1000.0 + sequence},
        "vision_epoch": epoch,
        "sequence": sequence,
        "snapshot_id": f"{epoch}:{sequence}",
        "events": events,
        "active_target": {
            "track_id": 7,
            "tracking_state": "tracking",
            "identity": "owner",
            "identity_state": "confirmed_known",
            "pose_action": "stop_gesture",
        },
        "hands": [{"hand_action": "stop_gesture"}],
    }


def test_published_event_history_compacts_repeating_state_stream() -> None:
    viewer = _event_history_viewer()
    record = viewer._record_visual_event_lifecycle

    assert record(_visual_packet(1, ["EVT_VISION_STOP_GESTURE"]), received_at=10.0)
    assert record(_visual_packet(2, ["EVT_VISION_STOP_GESTURE"]), received_at=10.1)
    assert record(_visual_packet(3, ["EVT_VISION_STOP_GESTURE"]), received_at=10.2)
    assert record(_visual_packet(4, []), received_at=10.3)

    history = list(viewer._published_event_history)
    assert [item["phase"] for item in history] == ["ENTER", "ACTIVE", "EXIT"]
    assert history[1]["repeat_count"] == 3
    assert history[1]["sequence"] == 3
    assert history[2]["repeat_count"] == 3
    assert history[2]["reason"] == "event_cleared"
    assert history[2]["evidence"]["identity_state"] == "confirmed_known"
    assert not viewer._active_published_events

    assert not record(
        _visual_packet(3, ["EVT_VISION_STOP_GESTURE"]), received_at=10.4
    )
    assert len(viewer._published_event_history) == 3


def test_published_event_history_closes_on_epoch_change_and_can_clear() -> None:
    viewer = _event_history_viewer()
    record = viewer._record_visual_event_lifecycle
    assert record(_visual_packet(1, ["EVT_VISION_FALL"]), received_at=20.0)
    assert record(_visual_packet(1, [], epoch="epoch-b"), received_at=20.5)

    history = list(viewer._published_event_history)
    assert [item["phase"] for item in history] == ["ENTER", "EXIT"]
    assert history[-1]["reason"] == "vision_epoch_changed"

    assert record(_visual_packet(2, ["EVT_VISION_FALL"], epoch="epoch-b"), received_at=20.6)
    result = viewer._clear_published_event_history()
    assert result == {
        "ok": True,
        "cleared_records": 3,
        "cleared_active_events": 1,
    }
    assert not viewer._published_event_history
    assert not viewer._active_published_events


def test_debug_viewer_render_limit_drops_early_frames() -> None:
    due = VisionDebugViewerNode._render_is_due
    assert due(10.0, 0.0, 8.0)
    assert not due(10.1, 10.0, 8.0)
    assert due(10.125, 10.0, 8.0)
    assert due(10.001, 10.0, 0.0)


def test_object_only_overlay_hold_does_not_depend_on_service_polling() -> None:
    viewer = SimpleNamespace(_object_overlay_hold=2.5)
    assert VisionDebugViewerNode._object_overlay_hold_sec(viewer) == 2.5


def test_object_only_config_disables_non_object_models() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "vision.object-only.yaml")
    production = load_config(root / "config" / "vision.yaml")
    providers = config["providers"]

    assert providers["vision"]["enabled"] is False
    assert providers["face_recognition"]["enabled"] is False
    assert providers["object"]["enabled"] is True
    assert providers["object"]["config"]["inference_rate_hz"] == 1.0
    assert production["providers"]["object"]["config"][
        "inference_rate_hz"
    ] == 0.0
    for key in (
        "object_model",
        "image_size",
        "det_threshold",
        "nms_threshold",
        "max_detections",
        "on_demand_rate_hz",
        "max_inference_rate_hz",
        "default_lease_sec",
        "max_lease_sec",
    ):
        assert providers["object"]["config"][key] == (
            production["providers"]["object"]["config"][key]
        )
