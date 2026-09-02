import json
import threading
import time
from types import MethodType, SimpleNamespace

import numpy as np

from marsdog_vision_interaction.core.face_enrollment_manager import (
    FaceEnrollmentManager,
    set_storage_root,
)
from marsdog_vision_interaction.core.object_detection_session import (
    ObjectDetectionSessionManager,
)
from marsdog_vision_interaction.core.visual_target_manager import VisualTargetManager
from marsdog_vision_interaction.nodes.camera_driver_node import (
    _video_capture_source,
)
from marsdog_vision_interaction.nodes.vision_interaction_node import VisionInteractionNode
from marsdog_vision_interaction.nodes import vision_interaction_node as vision_node_module
from marsdog_vision_interaction.nodes.vision_debug_viewer_node import (
    VisionDebugViewerNode,
)
from marsdog_vision_interaction.providers.object_detector import ObjectDetectorProvider
from marsdog_vision_interaction.providers.face_recognition import (
    FaceRecognitionProvider,
)
from marsdog_vision_interaction.providers.vision_observation import (
    VisionObservationProvider,
)


class _CachedVision:
    def is_available(self) -> bool:
        return True

    def get_observation(self) -> dict:
        return {
            "humans": [{
                "x": 0.2, "y": 0.1, "w": 0.4, "h": 0.8,
                "confidence": 0.9,
            }],
        }


def test_camera_source_accepts_portable_index_and_explicit_override() -> None:
    assert _video_capture_source("4") == 4
    assert _video_capture_source("custom-device") == "custom-device"


class _Publisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class _FakeLandmarker:
    def __init__(self) -> None:
        self.calls = []

    def detect(self, image):
        self.calls.append(("image", image, None))
        return "image-result"

    def detect_for_video(self, image, timestamp_ms):
        self.calls.append(("video", image, timestamp_ms))
        return "video-result"


def test_stale_camera_does_not_refresh_cached_target() -> None:
    manager = VisualTargetManager()
    manager.update_vision([{
        "x": 0.2, "y": 0.1, "w": 0.4, "h": 0.8,
        "confidence": 0.9,
    }], [])
    manager._active.last_seen_at = time.time() - 1.0
    publisher = _Publisher()
    fake_node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _enrollment_lock=threading.RLock(),
        _enrollment=SimpleNamespace(face_session=None),
        _providers={"vision": _CachedVision()},
        _vision_is_mock=False,
        _latest_camera_monotonic=time.monotonic() - 1.0,
        _camera_stale_timeout_sec=0.5,
        _latest_objects_monotonic=0.0,
        _object_cache_timeout_sec=1.0,
        _latest_objects=[],
        _target_manager=manager,
        _visual_pub=publisher,
        _publish_gesture_debug=lambda raw: None,
        _derive_events=VisionInteractionNode._derive_events,
        _process_enrollment_frame=lambda: None,
    )

    VisionInteractionNode._publish_visual(fake_node)
    payload = json.loads(publisher.messages[-1].data)
    assert payload["humans"] == []
    assert payload["active_target"]["tracking_state"] == "temporarily_lost"
    assert payload["active_target"]["last_seen_age_ms"] >= 900.0


def test_mono_task_frame_is_not_cropped() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _vision_is_mock=False,
        _latest_frame=frame,
        _latest_camera_monotonic=time.monotonic(),
        _camera_stale_timeout_sec=0.5,
        _stereo_enabled=False,
        _stereo_view="left",
        _stereo_min_aspect_ratio=2.2,
    )
    selected = VisionInteractionNode._frame_for_tasks(fake_node)
    assert selected is not None
    assert selected.shape == (480, 640, 3)


def test_real_provider_uses_latest_eligible_frame_without_callback_blocking() -> None:
    provider = VisionObservationProvider({"inference_frame_stride": 2})
    provider.available = True
    inferred_values = []
    first_started = threading.Event()
    release_first = threading.Event()

    def infer(frame):
        value = int(frame[0, 0, 0])
        inferred_values.append(value)
        if value == 0:
            first_started.set()
            assert release_first.wait(timeout=1.0)
        return {"frame_value": value}

    provider._process_frame_impl = infer
    provider._start_inference_worker()
    try:
        provider.process_frame(np.zeros((2, 2, 3), dtype=np.uint8))
        assert first_started.wait(timeout=1.0)
        for value in range(1, 6):
            provider.process_frame(
                np.full((2, 2, 3), value, dtype=np.uint8)
            )
        release_first.set()
        deadline = time.monotonic() + 1.0
        while provider._inferred_frame_count < 2 and time.monotonic() < deadline:
            time.sleep(0.005)

        assert inferred_values == [0, 4]
        assert provider._received_frame_count == 6
        assert provider._inference_candidate_count == 3
        assert provider._inferred_frame_count == 2
        assert provider._replaced_pending_frame_count == 1
        observation = provider.get_observation()
        assert observation["frame_value"] == 4
        assert observation["header"]["frame_id"] == "camera_link"
        assert observation["header"]["stamp"] > 0.0
    finally:
        release_first.set()
        provider.stop()


def test_pose_model_variant_selects_configured_full_model() -> None:
    provider = VisionObservationProvider({
        "pose_model_variant": "full",
        "pose_models": {
            "lite": "models/pose_landmarker_lite.task",
            "full": "models/pose_landmarker_full.task",
        },
    })
    assert provider._pose_model_variant == "full"
    assert provider._mediapipe_model.endswith(
        "pose_landmarker_full.task"
    )


def test_video_landmarker_uses_strict_monotonic_timestamps() -> None:
    provider = VisionObservationProvider({
        "landmarker_running_mode": "video",
    })
    landmarker = _FakeLandmarker()
    first_timestamp = provider._next_landmarker_timestamp()
    second_timestamp = provider._next_landmarker_timestamp()
    assert second_timestamp > first_timestamp
    assert provider._run_landmarker(
        landmarker, "frame", second_timestamp
    ) == "video-result"
    assert landmarker.calls == [("video", "frame", second_timestamp)]


def test_landmarker_ab_metrics_include_pose_quality_and_latency() -> None:
    provider = VisionObservationProvider({
        "pose_model_variant": "lite",
        "mediapipe_model": "models/pose_landmarker_lite.task",
    })
    provider._landmarker_times.extend((10.0, 10.1, 10.2))
    provider._pipeline_latency_ms.extend((20.0, 22.0, 24.0))
    provider._pose_latency_ms.extend((10.0, 12.0, 14.0))
    provider._hand_latency_ms.extend((5.0, 6.0, 7.0))
    provider._hand_detected.extend((1.0, 0.0, 1.0))
    provider._record_pose_quality([{
        "confidence": 0.9,
        "keypoints": [
            {
                "id": index,
                "confidence": 0.9 if index != 16 else 0.2,
                "presence": 0.9,
            }
            for index in range(33)
        ],
    }])
    diagnostics = provider._landmarker_diagnostics()
    assert diagnostics["pose_model_variant"] == "lite"
    assert diagnostics["running_mode"] == "video"
    assert diagnostics["effective_inference_fps"] == 10.0
    assert diagnostics["pose"]["detection_rate"] == 1.0
    assert diagnostics["pose"]["keypoint_valid_ratio"] == round(
        32 / 33, 3
    )
    assert diagnostics["pose"]["critical_keypoint_valid_ratio"] < 1.0


def test_hand_landmarker_probes_slowly_until_a_hand_is_active() -> None:
    provider = VisionObservationProvider({
        "hand_idle_inference_stride": 3,
        "hand_active_hold_inferences": 2,
    })
    assert [provider._hand_inference_is_due() for _ in range(4)] == [
        True,
        False,
        False,
        True,
    ]
    provider._hand_active_remaining = 2
    assert provider._hand_inference_is_due()


def test_gesture_debug_topic_keeps_exact_engine_labels() -> None:
    publisher = _Publisher()
    fake_node = SimpleNamespace(_gesture_debug_pub=publisher)
    VisionInteractionNode._publish_gesture_debug(fake_node, {
        "active_target": {
            "track_id": 8,
            "tracking_state": "tracking",
            "pose_state": "standing",
            "pose_action": "arm_raise_wave",
        },
        "hands": [{"hand_action": "stop_gesture"}],
        "_gesture_diagnostics": {
            "primary_action": "victory",
            "recognized_actions": [{"name": "victory"}],
            "raw_scores": [{"name": "victory", "score": 0.91}],
        },
    })
    payload = json.loads(publisher.messages[-1].data)
    assert payload["track_id"] == 8
    assert payload["pose_state"] == "standing"
    assert payload["primary_action"] == "victory"
    assert payload["legacy_pose_action"] == "arm_raise_wave"
    assert payload["legacy_hand_actions"] == ["stop_gesture"]


def test_object_detector_honors_object_params_without_synthetic_fallback() -> None:
    detector = ObjectDetectorProvider({
        "object_rknn_model": "models/object",
        "det_threshold": 0.6,
    })
    detector.available = True
    detector._loaded = True
    captured = {}

    def inference(frame, threshold):
        captured["threshold"] = threshold
        return []

    detector._run_inference = inference
    assert detector.detect_objects(
        np.zeros((10, 10, 3), dtype=np.uint8),
        {"confidence": 0.25},
    ) == []
    assert captured["threshold"] == 0.25
    assert detector.last_error == ""

    assert detector.detect_objects(None, {}) == []
    assert detector.last_error == "camera frame unavailable or stale"

    explicit_mock = ObjectDetectorProvider({"mock_mode": True})
    explicit_mock.start()
    assert explicit_mock.detect_objects(None, {})[0]["label"] == "dog toy ball"


def test_object_detector_filters_requested_labels_after_inference() -> None:
    detector = ObjectDetectorProvider({"object_model": "models/object"})
    detector.available = True
    detector._loaded = True
    detector._run_inference = lambda frame, threshold: [
        {"label": "Dog Toy Ball", "confidence": 0.8},
        {"label": "dog food can", "confidence": 0.9},
    ]

    objects = detector.detect_objects(
        np.zeros((10, 10, 3), dtype=np.uint8),
        {"target_labels": ["dog toy ball"]},
    )

    assert objects == [{"label": "Dog Toy Ball", "confidence": 0.8}]


def test_object_detection_session_is_leased_owned_and_rate_limited() -> None:
    manager = ObjectDetectionSessionManager(
        default_rate_hz=2.0,
        max_rate_hz=5.0,
        default_lease_sec=3.0,
    )
    assert manager.poll(now=10.0)["state"] == "inactive"

    started = manager.configure({
        "enabled": True,
        "session_id": "find-001",
        "rate_hz": 2.0,
        "confidence": 0.25,
        "target_labels": ["dog toy ball", "Dog Toy Ball"],
        "lease_sec": 3.0,
    }, now=10.0)
    assert started["ok"] is True
    assert started["stream"]["target_labels"] == ["dog toy ball"]
    due = manager.poll(now=10.0)
    assert due["state"] == "due"
    assert due["params"] == {
        "confidence": 0.25,
        "target_labels": ["dog toy ball"],
    }
    assert manager.poll(now=10.49)["state"] == "waiting"
    assert manager.poll(now=10.5)["state"] == "due"

    conflict = manager.configure({
        "enabled": False,
        "session_id": "another-task",
    }, now=10.6)
    assert conflict["ok"] is False
    assert conflict["stream"]["session_id"] == "find-001"

    renewed = manager.configure({
        "enabled": True,
        "session_id": "find-001",
        "lease_sec": 3.0,
    }, now=12.0)
    assert renewed["ok"] is True
    assert renewed["stream"]["target_labels"] == ["dog toy ball"]
    assert renewed["stream"]["confidence"] == 0.25
    assert manager.poll(now=14.99)["state"] in {"due", "waiting"}
    expired = manager.poll(now=15.0)
    assert expired["state"] == "expired"
    assert expired["stream"]["session_id"] == "find-001"
    assert manager.snapshot(now=15.0)["active"] is False


def test_object_detector_uses_ultralytics_predict_and_results_boxes() -> None:
    detector = ObjectDetectorProvider({
        "object_model": "models/object",
        "det_threshold": 0.6,
        "nms_threshold": 0.2,
        "max_detections": 10,
        "image_size": 640,
    })
    detector.available = True
    detector._loaded = True
    captured = {}
    boxes = SimpleNamespace(
        xyxyn=np.array([[0.4, 0.325, 0.6, 0.425]], dtype=np.float32),
        conf=np.array([0.9], dtype=np.float32),
        cls=np.array([7], dtype=np.float32),
    )

    def predict(**kwargs):
        captured.update(kwargs)
        return [SimpleNamespace(boxes=boxes, names={7: "dog food can"})]

    detector._model = SimpleNamespace(predict=predict)
    frame = np.zeros((360, 480, 3), dtype=np.uint8)
    objects = detector.detect_objects(frame, {"confidence": 0.5})

    assert captured.pop("source") is frame
    assert captured == {
        "conf": 0.5,
        "iou": 0.2,
        "max_det": 10,
        "imgsz": 640,
        "save": False,
        "verbose": False,
    }
    assert objects == [{
        "label": "dog food can",
        "x": 0.4,
        "y": 0.325,
        "w": 0.2,
        "h": 0.1,
        "confidence": 0.9,
        "center_x": 0.5,
        "center_y": 0.375,
    }]


def test_object_detector_handles_ultralytics_result_without_boxes() -> None:
    detector = ObjectDetectorProvider({})
    result = SimpleNamespace(boxes=None, names={})

    assert detector._results_to_objects([result]) == []
    assert detector._results_to_objects([]) == []


def test_object_result_is_cached_and_published_with_freshness_metadata() -> None:
    publisher = _Publisher()
    fake_node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _object_sequence=0,
        _vision_epoch="vision-test",
        _latest_camera_frame_id="camera_color_optical_frame",
        _latest_objects=[],
        _latest_objects_monotonic=0.0,
        _object_pub=publisher,
    )
    objects = [{
        "label": "dog toy ball",
        "x": 0.2,
        "y": 0.3,
        "w": 0.1,
        "h": 0.1,
        "confidence": 0.9,
    }]

    tracked = VisionInteractionNode._record_object_result(
        fake_node,
        objects,
        source="stream",
        status="ok",
        latency_ms=42.5,
    )

    payload = json.loads(publisher.messages[-1].data)
    assert payload["schema_version"] == 2
    assert payload["header"]["frame_id"] == "camera_color_optical_frame"
    assert payload["published_at"] >= payload["header"]["stamp"]
    assert payload["sequence"] == 1
    assert payload["source"] == "stream"
    assert payload["status"] == "ok"
    assert payload["stream"]["active"] is False
    assert payload["request"] == {}
    assert payload["stop_reason"] == ""
    assert payload["inference_latency_ms"] == 42.5
    assert payload["objects"] == tracked
    assert tracked[0]["target_id"] == "vision-test:object:1"
    assert tracked[0]["track_id"] == 1
    assert tracked[0]["tracking_state"] == "tracking"
    assert tracked[0]["range_valid"] is False
    assert tracked[0]["distance_m"] is None
    assert fake_node._latest_objects == tracked
    assert fake_node._latest_objects_monotonic > 0.0


def test_object_result_uses_aligned_depth_for_metric_approach() -> None:
    publisher = _Publisher()
    stamp = 100.0
    depth = np.full((20, 20), 2.0, dtype=np.float32)
    fake_node = SimpleNamespace(
        _state_lock=threading.Lock(),
        _object_sequence=0,
        _vision_epoch="vision-depth",
        _latest_camera_frame_id="camera_color_optical_frame",
        _latest_objects=[],
        _latest_objects_monotonic=0.0,
        _object_pub=publisher,
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

    tracked = VisionInteractionNode._record_object_result(
        fake_node,
        [{
            "label": "cat",
            "x": 0.1,
            "y": 0.1,
            "w": 0.8,
            "h": 0.8,
            "confidence": 0.9,
        }],
        source="stream",
        status="ok",
        latency_ms=10.0,
        observation_stamp=stamp,
        frame_id="camera_color_optical_frame",
    )

    assert tracked[0]["target_type"] == "animal"
    assert tracked[0]["range_valid"] is True
    assert 1.99 <= tracked[0]["distance_m"] <= 2.01
    payload = json.loads(publisher.messages[-1].data)
    assert payload["objects"][0]["range_source"] == "aligned_depth"


def test_stopping_object_stream_publishes_terminal_and_clears_cache() -> None:
    publisher = _Publisher()
    manager = ObjectDetectionSessionManager()
    assert manager.configure({
        "enabled": True,
        "session_id": "find-002",
        "lease_sec": 3.0,
    })["ok"]
    fake_node = SimpleNamespace(
        _object_stream=manager,
        _object_inference_lock=threading.Lock(),
        _state_lock=threading.Lock(),
        _object_sequence=0,
        _latest_camera_frame_id="camera_link",
        _latest_objects=[{"label": "dog toy ball"}],
        _latest_objects_monotonic=time.monotonic(),
        _object_tracks={1: {"target_id": "epoch:object:1"}},
        _object_pub=publisher,
    )
    fake_node._record_object_result = MethodType(
        VisionInteractionNode._record_object_result,
        fake_node,
    )

    result = VisionInteractionNode._set_object_detection(fake_node, {
        "enabled": False,
        "session_id": "find-002",
    })

    assert result["ok"] is True
    payload = json.loads(publisher.messages[-1].data)
    assert payload["source"] == "control"
    assert payload["status"] == "stopped"
    assert payload["stop_reason"] == "requested"
    assert payload["stream"]["active"] is False
    assert payload["stream"]["session_id"] == "find-002"
    assert fake_node._latest_objects == []
    assert fake_node._object_tracks == {}


def test_object_stream_skips_an_occupied_inference_slot() -> None:
    inference_lock = threading.Lock()
    inference_lock.acquire()
    fake_node = SimpleNamespace(
        _providers={"object": SimpleNamespace()},
        _object_inference_lock=inference_lock,
    )
    try:
        result = VisionInteractionNode._run_object_detection(
            fake_node,
            {},
            source="stream",
            wait_for_slot=False,
        )
    finally:
        inference_lock.release()

    assert result["ok"] is False
    assert result["busy"] is True


def test_debug_viewer_consumes_periodic_object_topic() -> None:
    fake_viewer = SimpleNamespace(
        _lock=threading.Lock(),
        _debug_objects=[],
        _debug_objects_monotonic=0.0,
        _object_task={},
    )
    message = SimpleNamespace(data=json.dumps({
        "status": "ok",
        "inference_latency_ms": 37.0,
        "objects": [{"label": "dog", "confidence": 0.8}],
        "error": "",
    }))

    VisionDebugViewerNode._on_objects(fake_viewer, message)

    assert fake_viewer._debug_objects == [
        {"label": "dog", "confidence": 0.8}
    ]
    assert fake_viewer._object_task["success"] is True
    assert fake_viewer._object_task["latency_ms"] == 37.0


class _FaceDetector:
    def setInputSize(self, size) -> None:
        self.size = size

    def detect(self, frame):
        detection = np.zeros((1, 15), dtype=np.float32)
        detection[0, :4] = (1.0, 1.0, 10.0, 10.0)
        detection[0, -1] = 0.99
        return None, detection


def test_face_enrollment_completion_is_terminal_and_names_are_safe(tmp_path) -> None:
    set_storage_root(tmp_path)
    manager = FaceEnrollmentManager()
    manager.set_face_detector(_FaceDetector())
    assert not manager.start_face("../outside")["ok"]
    assert not manager.delete_face("../outside")["ok"]
    assert not manager.start_face("测试用户")["ok"]

    assert manager.start_face("owner", required_shots=1)["ok"]
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    manager.process_face_frame(frame)
    manager.process_face_frame(frame)
    result = manager.process_face_frame(frame)
    assert result["done"] is True
    assert manager.face_session is None
    registry = json.loads((tmp_path / "face_registry.json").read_text())
    assert registry["faces"]["owner"]["paths"] == [
        "faces/owner/001.jpg"
    ]
    assert manager.get_face_paths("owner") == [
        str(tmp_path / "faces" / "owner" / "001.jpg")
    ]


def test_face_enrollment_default_collects_three_continuous_shots(tmp_path) -> None:
    set_storage_root(tmp_path)
    manager = FaceEnrollmentManager()
    result = manager.start_face("owner")
    assert result["total_steps"] == 3
    assert result["pose"] == "continuous"
    assert manager.face_session is not None
    assert manager.face_session.poses == ["continuous"] * 3


def test_face_recognition_keeps_and_matches_multiple_templates() -> None:
    provider = FaceRecognitionProvider({"match_threshold": 0.5})
    embeddings = iter([
        np.array([1.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0], dtype=np.float32),
    ])
    provider._extract_embedding = lambda _image: next(embeddings)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    assert provider.enroll(image, "owner")["success"]
    assert provider.enroll(image, "owner")["success"]
    assert provider.enrolled_count == 1
    result = provider.recognize(image)
    assert result["matched"] is True
    assert result["user_id"] == "owner"
    assert result["confidence"] == 1.0


def test_master_pose_event_requires_confirmed_known_identity() -> None:
    base = {
        "faces": [{"recognized_user": "owner"}],
        "hands": [],
        "active_target": {
            "identity": "owner",
            "identity_state": "candidate_known",
            "tracking_state": "tracking",
            "pose_action": "arm_raise_wave",
        },
    }
    assert VisionInteractionNode._derive_events(base) == ["EVT_VISION_MASTER"]

    confirmed = {
        **base,
        "active_target": {
            **base["active_target"],
            "identity_state": "confirmed_known",
        },
    }
    assert VisionInteractionNode._derive_events(confirmed) == [
        "EVT_VISION_MASTER",
        "EVT_VISION_MASTER_HAPPY",
    ]


def test_all_pose_events_require_fixed_confirmed_face_identity() -> None:
    unrecognized = {
        "faces": [],
        "hands": [{"hand_action": "stop_gesture"}],
        "active_target": {
            "identity": "unknown",
            "identity_state": "confirmed_unknown",
            "tracking_state": "tracking",
            "pose_action": "fallen_down",
        },
    }
    assert VisionInteractionNode._derive_events(unrecognized) == []

    candidate_owner = {
        **unrecognized,
        "active_target": {
            **unrecognized["active_target"],
            "identity": "owner",
            "identity_state": "candidate_known",
        },
    }
    assert VisionInteractionNode._derive_events(candidate_owner) == []

    confirmed_owner = {
        **candidate_owner,
        "active_target": {
            **candidate_owner["active_target"],
            "identity_state": "confirmed_known",
        },
    }
    assert VisionInteractionNode._derive_events(confirmed_owner) == [
        "EVT_VISION_FALL",
        "EVT_VISION_STOP_GESTURE",
    ]

    confirmed_family = {
        **confirmed_owner,
        "active_target": {
            **confirmed_owner["active_target"],
            "identity": "family_member_4",
        },
    }
    assert VisionInteractionNode._derive_events(confirmed_family) == [
        "EVT_VISION_FALL",
        "EVT_VISION_STOP_GESTURE",
    ]

    ordinary_family_pose = {
        **confirmed_family,
        "hands": [],
        "active_target": {
            **confirmed_family["active_target"],
            "pose_action": "arm_raise_wave",
        },
    }
    assert VisionInteractionNode._derive_events(ordinary_family_pose) == [
        "EVT_VISION_MASTER_HAPPY",
    ]

    temporarily_lost_owner = {
        **confirmed_owner,
        "active_target": {
            **confirmed_owner["active_target"],
            "tracking_state": "temporarily_lost",
        },
    }
    assert VisionInteractionNode._derive_events(temporarily_lost_owner) == []

    legacy_identity = {
        **confirmed_owner,
        "active_target": {
            **confirmed_owner["active_target"],
            "identity": "legacy_name",
        },
    }
    assert VisionInteractionNode._derive_events(legacy_identity) == []


def test_visual_state_log_reports_gate_and_deduplicates(monkeypatch) -> None:
    messages = []
    monkeypatch.setattr(
        vision_node_module.logger,
        "info",
        lambda message, *args: messages.append(message % args),
    )
    node = SimpleNamespace(_last_visual_log_signature=None)
    event = {
        "active_target": {
            "track_id": 7,
            "tracking_state": "tracking",
            "identity": "owner",
            "identity_state": "candidate_known",
            "pose_action": "arm_raise_wave",
        },
        "hands": [],
        "events": ["EVT_VISION_MASTER"],
    }
    VisionInteractionNode._log_visual_event_state(node, event)
    VisionInteractionNode._log_visual_event_state(node, event)
    assert len(messages) == 1
    assert "identity_state=candidate_known" in messages[0]
    assert "pose_event_gate=blocked" in messages[0]
    assert "events=EVT_VISION_MASTER" in messages[0]

    event["active_target"]["identity_state"] = "confirmed_known"
    event["events"].append("EVT_VISION_MASTER_HAPPY")
    VisionInteractionNode._log_visual_event_state(node, event)
    assert len(messages) == 2
    assert "pose_event_gate=open" in messages[1]
    assert "EVT_VISION_MASTER_HAPPY" in messages[1]
