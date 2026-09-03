"""Independent ROS2 node for the MarsDog visual interaction pipeline."""

from __future__ import annotations

import base64
from collections import deque
import copy
import json
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

try:
    from marsdog_vision_interaction.srv import VisionTask
except ImportError:
    VisionTask = None  # type: ignore[assignment]

from marsdog_vision_interaction.core.face_enrollment_manager import (
    FaceEnrollmentManager,
    set_storage_root,
)
from marsdog_vision_interaction.core.held_object_pose import (
    HELD_OBJECT_LABELS,
    HeldObjectPoseManager,
    HeldObjectPoseStatus,
)
from marsdog_vision_interaction.core.object_detection_session import (
    ObjectDetectionSessionManager,
)
from marsdog_vision_interaction.fusion.stereo_fusion import get_target_manager
from marsdog_vision_interaction.messages.face_identity import (
    ALLOWED_FACE_IDENTITIES,
)
from marsdog_vision_interaction.messages.visual_event import (
    normalize_visual_event,
)
from marsdog_vision_interaction.messages.visual_event_types import (
    face_identity_to_vision_event,
    pose_action_to_vision_event,
)
from marsdog_vision_interaction.providers.base import BaseProvider
from marsdog_vision_interaction.utils.config_loader import load_config
from marsdog_vision_interaction.utils.logging_utils import (
    configure_event_trace,
    get_logger,
    setup_logging,
    vision_timing_trace,
    vision_trace,
)
from marsdog_vision_interaction.utils.stereo_view import select_camera_view


logger = get_logger(__name__, module="vision")

_VISUAL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)
_EVENT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)
_CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

_ANIMAL_LABELS = {"cat", "dog"}
_TOY_LABELS = {
    "dog toy ball",
    "dog frisbee toy",
    "dog tug ring toy",
}


class VisionInteractionNode(Node):
    """Own camera input, visual models, face storage and visual ROS APIs."""

    def __init__(self) -> None:
        super().__init__("vision_interaction")
        self.declare_parameter("config_path", "config/vision.yaml")
        self.declare_parameter("log_level", "INFO")
        self.declare_parameter("log_dir", "log")
        self.declare_parameter("test_run_id", "")
        self.declare_parameter("test_case_id", "")
        self.declare_parameter("pose_model_variant", "")
        self.declare_parameter("landmarker_running_mode", "")
        self.declare_parameter("face_api_host", "")
        self.declare_parameter("face_api_port", 0)
        config_path = self.get_parameter("config_path").value
        setup_logging(
            log_dir=str(self.get_parameter("log_dir").value),
            level=str(self.get_parameter("log_level").value),
            node="vision_interaction",
        )
        try:
            self._config = load_config(str(config_path))
        except Exception as exc:
            logger.error("Cannot load vision config %s: %s", config_path, exc)
            self._config = {}
        logging_config = self._config.get("logging", {})
        if not isinstance(logging_config, dict):
            logging_config = {}
        self._timing_trace_interval_sec = max(
            0.0,
            float(logging_config.get("timing_trace_interval_sec", 5.0)),
        )
        configure_event_trace(
            enabled=bool(logging_config.get("event_trace", True)),
            log_dir=str(self.get_parameter("log_dir").value),
            run_id=str(self.get_parameter("test_run_id").value),
            case_id=str(self.get_parameter("test_case_id").value),
            timing_interval_sec=self._timing_trace_interval_sec,
        )
        self._apply_runtime_model_overrides()
        self._apply_face_api_overrides()

        storage_root = self._config.get("storage", {}).get("root", "data")
        set_storage_root(storage_root)
        self._enrollment = FaceEnrollmentManager()
        self._target_manager = get_target_manager()
        vision_runtime_config = (
            self._config.get("providers", {}).get("vision", {}).get("config", {})
        )
        if not isinstance(vision_runtime_config, dict):
            vision_runtime_config = {}
        self._target_manager.configure(
            horizontal_fov_deg=float(
                vision_runtime_config.get("horizontal_fov_deg", 69.0)
            )
        )
        self._vision_epoch = self._target_manager.vision_epoch
        self._visual_sequence = 0
        self._latest_visual_snapshot: dict[str, Any] = {}
        self._latest_visual_snapshot_monotonic = 0.0
        self._last_visual_log_signature: tuple[Any, ...] | None = None
        self._last_traced_held_signature: tuple[Any, ...] | None = None
        self._last_traced_events: tuple[str, ...] = ()
        self._state_lock = threading.Lock()
        self._enrollment_lock = threading.RLock()
        self._camera_callback_group = MutuallyExclusiveCallbackGroup()
        self._publish_callback_group = MutuallyExclusiveCallbackGroup()
        self._object_callback_group = MutuallyExclusiveCallbackGroup()
        self._service_callback_group = MutuallyExclusiveCallbackGroup()
        self._object_inference_lock = threading.Lock()
        self._providers: dict[str, BaseProvider | None] = {}
        self._latest_frame: np.ndarray | None = None
        self._latest_camera_frame_id = "camera_link"
        self._latest_camera_stamp = 0.0
        self._latest_camera_monotonic = 0.0
        self._latest_depth_m: np.ndarray | None = None
        self._latest_depth_stamp = 0.0
        self._latest_depth_monotonic = 0.0
        self._latest_depth_frame_id = ""
        self._depth_history: deque[
            tuple[float, float, str, np.ndarray]
        ] = deque()
        self._camera_intrinsics: dict[str, Any] | None = None
        self._latest_objects: list[dict[str, Any]] = []
        self._latest_objects_monotonic = 0.0
        self._object_sequence = 0
        self._object_tracks: dict[int, dict[str, Any]] = {}
        self._next_object_track_id = 1
        self._vision_is_mock = (
            self._config.get("providers", {}).get("vision", {}).get("type")
            == "mock"
        )
        self._init_providers()
        self._wire_face_enrollment()
        self._sync_face_registry()
        self._face_api: Any = None
        self._face_api_status: dict[str, Any] = {
            "enabled": False,
            "ready": False,
        }
        self._init_face_api()

        topics = self._config.get("topics", {})
        self._visual_topic = str(
            topics.get("visual_event", "/perception/visual_event")
        )
        self._gesture_debug_topic = str(
            topics.get(
                "gesture_debug", "/perception/vision/gesture_debug"
            )
        )
        self._object_topic = str(
            topics.get(
                "object_detections",
                "/perception/vision/object_detections",
            )
        )
        self._camera_topic = str(
            topics.get("camera_image", "/camera/camera/color/image_raw")
        )
        depth_config = self._config.get("depth_fusion", {})
        if not isinstance(depth_config, dict):
            depth_config = {}
        self._depth_enabled = bool(depth_config.get("enabled", False))
        self._depth_topic = str(
            depth_config.get(
                "aligned_depth_topic",
                "/camera/camera/aligned_depth_to_color/image_raw",
            )
        )
        self._camera_info_topic = str(
            depth_config.get(
                "camera_info_topic", "/camera/camera/color/camera_info"
            )
        )
        self._depth_sync_tolerance_sec = max(
            0.0, float(depth_config.get("max_sync_delta_sec", 0.10))
        )
        self._depth_stale_timeout_sec = max(
            0.05, float(depth_config.get("stale_timeout_sec", 0.5))
        )
        self._depth_history_duration_sec = max(
            self._depth_sync_tolerance_sec,
            float(depth_config.get("history_duration_sec", 1.5)),
        )
        self._depth_history_max_frames = max(
            2, int(depth_config.get("history_max_frames", 30))
        )
        self._depth_history_min_interval_sec = max(
            0.0,
            float(depth_config.get("history_min_interval_sec", 0.04)),
        )
        self._depth_min_m = max(
            0.0, float(depth_config.get("min_depth_m", 0.2))
        )
        self._depth_max_m = max(
            self._depth_min_m,
            float(depth_config.get("max_depth_m", 8.0)),
        )
        self._depth_sample_radius_px = max(
            0, int(depth_config.get("sample_radius_px", 4))
        )
        self._depth_min_valid_samples = max(
            1, int(depth_config.get("min_valid_samples", 5))
        )
        self._depth_min_valid_fraction = min(
            1.0,
            max(0.0, float(depth_config.get("min_valid_fraction", 0.2))),
        )
        self._target_current_timeout_sec = max(
            0.05,
            float(depth_config.get("target_current_timeout_sec", 0.35)),
        )
        self._horizontal_fov_deg = max(
            1.0,
            float(
                depth_config.get(
                    "fallback_horizontal_fov_deg",
                    vision_runtime_config.get("horizontal_fov_deg", 69.0),
                )
            ),
        )
        enrollment_topic = str(
            topics.get(
                "enrollment_event",
                "/perception/vision/enrollment_event",
            )
        )
        self._visual_pub = self.create_publisher(
            String, self._visual_topic, _VISUAL_QOS
        )
        self._gesture_debug_pub = self.create_publisher(
            String, self._gesture_debug_topic, _VISUAL_QOS
        )
        self._object_pub = self.create_publisher(
            String, self._object_topic, _VISUAL_QOS
        )
        self._enrollment_pub = self.create_publisher(
            String, enrollment_topic, _EVENT_QOS
        )
        self._camera_sub = self.create_subscription(
            Image,
            self._camera_topic,
            self._on_camera,
            _CAMERA_QOS,
            callback_group=self._camera_callback_group,
        )
        self._depth_sub = (
            self.create_subscription(
                Image,
                self._depth_topic,
                self._on_depth,
                _CAMERA_QOS,
                callback_group=self._camera_callback_group,
            )
            if self._depth_enabled
            else None
        )
        self._camera_info_sub = (
            self.create_subscription(
                CameraInfo,
                self._camera_info_topic,
                self._on_camera_info,
                _CAMERA_QOS,
                callback_group=self._camera_callback_group,
            )
            if self._depth_enabled
            else None
        )
        rate = float(topics.get("observation_rate_hz", 10.0))
        self._camera_stale_timeout_sec = max(
            0.1, float(topics.get("camera_stale_timeout_sec", 0.5))
        )
        object_runtime_config = (
            self._config.get("providers", {})
            .get("object", {})
            .get("config", {})
        )
        self._object_inference_rate_hz = max(
            0.0, float(object_runtime_config.get("inference_rate_hz", 0.0))
        )
        self._object_stream = ObjectDetectionSessionManager(
            startup_rate_hz=self._object_inference_rate_hz,
            default_rate_hz=float(
                object_runtime_config.get("on_demand_rate_hz", 2.0)
            ),
            max_rate_hz=float(
                object_runtime_config.get("max_inference_rate_hz", 5.0)
            ),
            default_confidence=float(
                object_runtime_config.get("det_threshold", 0.2)
            ),
            default_lease_sec=float(
                object_runtime_config.get("default_lease_sec", 3.0)
            ),
            max_lease_sec=float(
                object_runtime_config.get("max_lease_sec", 30.0)
            ),
        )
        self._held_object_enabled = bool(
            object_runtime_config.get("held_pose_enabled", True)
        )
        self._held_object_rate_hz = min(
            float(object_runtime_config.get("max_inference_rate_hz", 5.0)),
            max(0.1, float(
                object_runtime_config.get("held_pose_rate_hz", 2.0)
            )),
        )
        self._held_object_confidence = min(
            1.0,
            max(0.0, float(
                object_runtime_config.get("held_pose_confidence", 0.35)
            )),
        )
        self._held_object_absence_grace_sec = max(
            0.0,
            float(object_runtime_config.get(
                "held_pose_human_absence_grace_sec", 2.0
            )),
        )
        self._held_object_session_id = "vision-human-holding"
        self._held_object_stream = ObjectDetectionSessionManager(
            default_rate_hz=self._held_object_rate_hz,
            max_rate_hz=float(
                object_runtime_config.get("max_inference_rate_hz", 5.0)
            ),
            default_confidence=self._held_object_confidence,
            default_lease_sec=5.0,
            max_lease_sec=30.0,
        )
        self._held_object_pose = HeldObjectPoseManager(
            min_object_confidence=self._held_object_confidence,
            wrist_distance_ratio=float(object_runtime_config.get(
                "held_pose_wrist_distance_ratio", 0.16
            )),
            human_bbox_expansion_ratio=float(object_runtime_config.get(
                "held_pose_human_bbox_expansion_ratio", 0.15
            )),
            required_hits=int(object_runtime_config.get(
                "held_pose_confirmation_hits", 2
            )),
            confirmation_window_s=float(object_runtime_config.get(
                "held_pose_confirmation_window_sec", 1.5
            )),
            hold_s=float(object_runtime_config.get(
                "held_pose_hold_sec", 1.25
            )),
            max_pose_object_sync_delta_s=float(object_runtime_config.get(
                "held_pose_max_sync_delta_sec", 0.75
            )),
        )
        self._held_object_last_human_at = 0.0
        object_scheduler_rate_hz = max(
            1.0,
            float(object_runtime_config.get("scheduler_rate_hz", 20.0)),
        )
        configured_object_cache_sec = max(
            0.1, float(topics.get("object_cache_timeout_sec", 1.0))
        )
        self._object_cache_timeout_sec = (
            max(
                configured_object_cache_sec,
                2.0 / self._object_inference_rate_hz,
            )
            if self._object_inference_rate_hz > 0.0
            else configured_object_cache_sec
        )
        self._object_target_current_timeout_sec = max(
            0.1,
            float(
                object_runtime_config.get(
                    "target_current_timeout_sec", 0.75
                )
            ),
        )
        self._object_target_persistence_sec = max(
            self._object_target_current_timeout_sec,
            float(
                object_runtime_config.get(
                    "target_persistence_sec",
                    self._object_cache_timeout_sec,
                )
            ),
        )
        self._stereo_enabled = bool(
            vision_runtime_config.get("stereo_enabled", False)
        )
        self._stereo_view = str(
            vision_runtime_config.get("stereo_view", "left")
        ).lower()
        self._stereo_min_aspect_ratio = max(
            1.0,
            float(vision_runtime_config.get("stereo_min_aspect_ratio", 2.2)),
        )
        self._timer = self.create_timer(
            1.0 / max(rate, 0.1),
            self._publish_visual,
            callback_group=self._publish_callback_group,
        )
        self._object_timer = (
            self.create_timer(
                1.0 / object_scheduler_rate_hz,
                self._poll_objects,
                callback_group=self._object_callback_group,
            )
            if self._providers.get("object") is not None
            else None
        )

        service_name = str(
            topics.get("vision_task", "/perception/vision/task")
        )
        self._service = (
            self.create_service(
                VisionTask,
                service_name,
                self._handle_task,
                callback_group=self._service_callback_group,
            )
            if VisionTask is not None else None
        )
        logger.info(
            "Vision node ready: camera=%s visual=%s objects=%s startup=%.2fHz "
            "held_pose=%s@%.2fHz depth=%s service=%s",
            self._camera_topic,
            self._visual_topic,
            self._object_topic,
            self._object_inference_rate_hz,
            self._held_object_enabled,
            self._held_object_rate_hz,
            self._depth_topic if self._depth_enabled else "disabled",
            service_name if self._service is not None else "unavailable",
        )
        vision_trace(
            "runtime_start",
            result="ready",
            node="vision_interaction",
            module="runtime",
            camera_topic=self._camera_topic,
            visual_topic=self._visual_topic,
            object_topic=self._object_topic,
            service=service_name if self._service is not None else "unavailable",
            vision_epoch=self._vision_epoch,
            timing_trace_interval_sec=self._timing_trace_interval_sec,
        )

    def _apply_runtime_model_overrides(self) -> None:
        providers = self._config.get("providers")
        if not isinstance(providers, dict):
            return
        vision = providers.get("vision")
        if not isinstance(vision, dict):
            return
        config = vision.get("config")
        if not isinstance(config, dict):
            return
        overrides = {
            "pose_model_variant": str(
                self.get_parameter("pose_model_variant").value
            ).strip().lower(),
            "landmarker_running_mode": str(
                self.get_parameter("landmarker_running_mode").value
            ).strip().lower(),
        }
        for key, value in overrides.items():
            if value:
                config[key] = value
                logger.info("Runtime vision override: %s=%s", key, value)

    def _apply_face_api_overrides(self) -> None:
        config = self._config.setdefault("face_api", {})
        if not isinstance(config, dict):
            config = {}
            self._config["face_api"] = config
        host = str(self.get_parameter("face_api_host").value).strip()
        port = int(self.get_parameter("face_api_port").value)
        if host:
            config["host"] = host
        if port > 0:
            config["port"] = port

    def _init_providers(self) -> None:
        providers = self._config.get("providers", {})
        vision_config = providers.get("vision", {})
        if vision_config.get("enabled", True):
            if vision_config.get("type", "observation") == "observation":
                from marsdog_vision_interaction.providers.vision_observation import (
                    VisionObservationProvider,
                )
                vision: BaseProvider = VisionObservationProvider(
                    vision_config.get("config", {})
                )
            else:
                from marsdog_vision_interaction.providers.mock_vision import (
                    MockVisionProvider,
                )
                vision = MockVisionProvider(vision_config.get("config", {}))
            vision.start()
            if not vision.is_available():
                logger.error(
                    "Configured vision provider is unavailable; publishing "
                    "empty observations instead of synthetic detections"
                )
            self._providers["vision"] = vision

        object_config = providers.get("object", {})
        if object_config.get("enabled", True):
            from marsdog_vision_interaction.providers.object_detector import (
                ObjectDetectorProvider,
            )
            config = dict(object_config.get("config", {}))
            if object_config.get("type") == "mock":
                config["mock_mode"] = True
            detector = ObjectDetectorProvider(config)
            detector.start()
            self._providers["object"] = detector

        face_config = providers.get("face_recognition", {})
        if face_config.get("enabled", True):
            if face_config.get("type", "sface") == "mock":
                from marsdog_vision_interaction.providers.mock_face_recognition import (
                    MockFaceRecognitionProvider,
                )
                face: BaseProvider = MockFaceRecognitionProvider(
                    face_config.get("config", {})
                )
            else:
                from marsdog_vision_interaction.providers.face_recognition import (
                    FaceRecognitionProvider,
                )
                face = FaceRecognitionProvider(face_config.get("config", {}))
            face.start()
            self._providers["face_recognition"] = face

    def _on_camera(self, message: Image) -> None:
        try:
            frame = self._decode_image(message)
            if frame is None:
                raise ValueError("unsupported or malformed image")
            header = getattr(message, "header", None)
            camera_stamp = self._message_stamp(message)
            frame_id = str(
                getattr(header, "frame_id", "") or "camera_link"
            )
            with self._state_lock:
                self._latest_frame = frame
                self._latest_camera_frame_id = frame_id
                self._latest_camera_stamp = camera_stamp or time.time()
                self._latest_camera_monotonic = time.monotonic()
            vision = self._providers.get("vision")
            if vision is not None and hasattr(vision, "process_frame"):
                vision.process_frame(  # type: ignore[attr-defined]
                    frame,
                    stamp=camera_stamp or time.time(),
                    frame_id=frame_id,
                )
        except Exception as exc:
            logger.debug("Camera frame rejected: %s", exc)

    def _on_depth(self, message: Image) -> None:
        """Cache timestamped aligned depth for delayed inference fusion."""
        if not self._depth_enabled:
            return
        depth_m = self._decode_depth_image(message)
        if depth_m is None:
            with self._state_lock:
                self._latest_depth_m = None
                self._latest_depth_stamp = 0.0
                self._latest_depth_monotonic = 0.0
                self._latest_depth_frame_id = ""
            logger.debug(
                "Depth frame rejected: encoding=%s",
                getattr(message, "encoding", ""),
            )
            return
        header = getattr(message, "header", None)
        depth_stamp = self._message_stamp(message) or time.time()
        received_monotonic = time.monotonic()
        frame_id = str(getattr(header, "frame_id", "") or "")
        # Half precision keeps a 1280x720 history bounded while retaining
        # centimetre-level accuracy throughout the configured 0.2-8.0 m
        # operating range. The array is immutable after entering the cache.
        cached_depth = depth_m.astype(np.float16, copy=False)
        with self._state_lock:
            self._latest_depth_m = cached_depth
            self._latest_depth_stamp = depth_stamp
            self._latest_depth_monotonic = received_monotonic
            self._latest_depth_frame_id = frame_id
            history = self._depth_history
            if history and depth_stamp < history[-1][0]:
                # A camera restart or clock reset creates a new timestamp
                # epoch. Never match frames across that boundary.
                history.clear()
            if (
                not history
                or depth_stamp - history[-1][0]
                >= self._depth_history_min_interval_sec
            ):
                history.append((
                    depth_stamp,
                    received_monotonic,
                    frame_id,
                    cached_depth,
                ))
            cutoff = depth_stamp - self._depth_history_duration_sec
            while history and (
                len(history) > self._depth_history_max_frames
                or history[0][0] < cutoff
            ):
                history.popleft()

    def _on_camera_info(self, message: CameraInfo) -> None:
        """Cache finite pinhole intrinsics for depth deprojection."""
        if not self._depth_enabled:
            return
        try:
            values = list(message.k)
            fx = float(values[0])
            fy = float(values[4])
            cx = float(values[2])
            cy = float(values[5])
            width = int(message.width)
            height = int(message.height)
            if (
                len(values) != 9
                or not all(math.isfinite(item) for item in (fx, fy, cx, cy))
                or fx <= 0.0
                or fy <= 0.0
                or width <= 0
                or height <= 0
            ):
                raise ValueError("invalid camera matrix")
        except (TypeError, ValueError, IndexError) as exc:
            with self._state_lock:
                self._camera_intrinsics = None
            logger.debug("CameraInfo rejected: %s", exc)
            return
        header = getattr(message, "header", None)
        with self._state_lock:
            self._camera_intrinsics = {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "width": width,
                "height": height,
                "frame_id": str(
                    getattr(header, "frame_id", "") or "camera_link"
                ),
            }

    @staticmethod
    def _message_stamp(message: Any) -> float:
        header = getattr(message, "header", None)
        ros_stamp = getattr(header, "stamp", None)
        try:
            value = (
                float(getattr(ros_stamp, "sec", 0))
                + float(getattr(ros_stamp, "nanosec", 0)) * 1e-9
            )
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) and value > 0.0 else 0.0

    def _build_visual_snapshot(
        self,
        raw: dict[str, Any] | Any,
    ) -> dict[str, Any]:
        """Build one versioned target snapshot from one manager read.

        The provider's cached observation carries faces, gestures and source
        image metadata.  Identity, target IDs and target freshness come from a
        single :class:`VisualTargetManager` snapshot so an active target and
        its candidate entry cannot describe different tracker generations.
        """
        raw_dict = dict(raw) if isinstance(raw, dict) else {}
        event = normalize_visual_event(raw_dict)
        target_snapshot = self._target_manager.get_snapshot()
        vision_epoch = str(
            target_snapshot.get("vision_epoch", "")
            or getattr(self, "_vision_epoch", "")
        )
        active_value = target_snapshot.get("active_target")
        active = (
            active_value.to_dict()
            if hasattr(active_value, "to_dict") else {}
        )
        active["vision_epoch"] = vision_epoch
        candidates = [
            dict(item)
            for item in target_snapshot.get("human_candidates", [])
            if isinstance(item, dict)
        ]

        raw_active = raw_dict.get("active_target", {})
        if not isinstance(raw_active, dict):
            raw_active = {}
        raw_candidates = {
            str(item.get("target_id", "")): item
            for item in raw_dict.get("human_candidates", [])
            if isinstance(item, dict) and item.get("target_id")
        }
        active_target_id = str(active.get("target_id", ""))
        for candidate in candidates:
            source = raw_candidates.get(str(candidate.get("target_id", "")))
            if source is not None:
                candidate["pose_action"] = str(
                    source.get("pose_action", "")
                )
                candidate["pose_action_label"] = str(
                    source.get("pose_action_label", "")
                )

        if (
            active_target_id
            and str(raw_active.get("target_id", "")) == active_target_id
            and active.get("tracking_state") == "tracking"
        ):
            active["pose_action"] = str(raw_active.get("pose_action", ""))
            active["pose_action_label"] = str(
                raw_active.get("pose_action_label", "")
            )
        else:
            active["pose_action"] = ""
            active["pose_action_label"] = ""

        observation_stamp = VisionInteractionNode._finite_positive(
            event.get("header", {}).get("stamp")
        )
        with self._state_lock:
            camera_stamp = VisionInteractionNode._finite_positive(
                getattr(self, "_latest_camera_stamp", 0.0)
            )
            camera_frame_id = str(
                getattr(self, "_latest_camera_frame_id", "camera_link")
                or "camera_link"
            )
        if not raw_dict.get("header") and camera_stamp is not None:
            event["header"] = {
                "stamp": camera_stamp,
                "frame_id": camera_frame_id,
            }
            observation_stamp = camera_stamp
        elif observation_stamp is None:
            event["header"]["stamp"] = time.time()
            observation_stamp = float(event["header"]["stamp"])
        if not str(event["header"].get("frame_id", "")).strip():
            event["header"]["frame_id"] = camera_frame_id

        VisionInteractionNode._fuse_human_depth(
            self,
            candidates,
            observation_stamp=observation_stamp,
        )
        matching_active = next(
            (
                item for item in candidates
                if str(item.get("target_id", "")) == active_target_id
            ),
            None,
        )
        if matching_active is not None:
            for key in (
                "center",
                "body_center",
                "bearing_deg",
                "bearing_valid",
                "bearing_source",
                "range_valid",
                "distance_m",
                "range_source",
                "depth_sync_delta_ms",
                "pose_3d",
            ):
                if key in matching_active:
                    active[key] = copy.deepcopy(matching_active[key])

        with self._state_lock:
            sequence = int(getattr(self, "_visual_sequence", 0)) + 1
            self._visual_sequence = sequence
        event["vision_epoch"] = vision_epoch
        event["sequence"] = sequence
        event["snapshot_id"] = f"{vision_epoch}:{sequence}"
        event["active_target"] = active
        event["human_candidates"] = candidates
        return event

    def _query_targets(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return a safely re-aged copy of the latest published snapshot."""
        with self._state_lock:
            snapshot = copy.deepcopy(
                getattr(self, "_latest_visual_snapshot", {})
            )
            cached_monotonic = float(
                getattr(self, "_latest_visual_snapshot_monotonic", 0.0)
            )
            object_candidates = (
                VisionInteractionNode._object_candidates_locked(
                    self,
                    now_monotonic=time.monotonic(),
                )
            )
        if (
            not snapshot
            or int(snapshot.get("sequence", 0) or 0) <= 0
            or not math.isfinite(cached_monotonic)
            or cached_monotonic <= 0.0
        ):
            return {"ok": False, "error": "visual snapshot unavailable"}

        snapshot_age_ms = max(
            0.0, time.monotonic() - cached_monotonic
        ) * 1000.0
        current_timeout_ms = max(
            1.0,
            float(getattr(self, "_target_current_timeout_sec", 0.35))
            * 1000.0,
        )
        for item in [
            snapshot.get("active_target", {}),
            *snapshot.get("human_candidates", []),
        ]:
            if not isinstance(item, dict):
                continue
            age = VisionInteractionNode._finite_nonnegative(
                item.get("last_seen_age_ms")
            )
            if age is None:
                item["tracking_state"] = "lost"
                item["range_valid"] = False
                item["distance_m"] = None
                continue
            age += snapshot_age_ms
            item["last_seen_age_ms"] = round(age, 1)
            if age > current_timeout_ms:
                item["tracking_state"] = "temporarily_lost"
                # Cached range must never survive target loss.
                item["range_valid"] = False
                item["distance_m"] = None
                pose_3d = item.get("pose_3d")
                if isinstance(pose_3d, dict):
                    pose_3d.update({
                        "valid": False,
                        "x": None,
                        "y": None,
                        "z": None,
                    })

        requested_types = params.get("target_types", ["human"])
        if isinstance(requested_types, str):
            requested_types = [requested_types]
        allowed_types = {
            str(value).strip().lower()
            for value in requested_types
            if str(value).strip()
        } if isinstance(requested_types, list) else {"human"}
        minimum_confidence = VisionInteractionNode._finite_nonnegative(
            params.get("min_confidence", 0.0)
        )
        minimum_confidence = min(1.0, minimum_confidence or 0.0)
        maximum_age_ms = VisionInteractionNode._finite_nonnegative(
            params.get("max_age_ms")
        )
        human_targets: list[dict[str, Any]] = []
        if "human" in allowed_types or "person" in allowed_types:
            for item in snapshot.get("human_candidates", []):
                if not isinstance(item, dict):
                    continue
                confidence = VisionInteractionNode._finite_nonnegative(
                    item.get(
                        "detection_confidence",
                        item.get("confidence", 0.0),
                    )
                ) or 0.0
                age = VisionInteractionNode._finite_nonnegative(
                    item.get("last_seen_age_ms")
                )
                if confidence < minimum_confidence:
                    continue
                if maximum_age_ms is not None and (
                    age is None or age > maximum_age_ms
                ):
                    continue
                human_targets.append(copy.deepcopy(item))

        nonhuman_targets: list[dict[str, Any]] = []
        for item in object_candidates:
            target_type = str(item.get("target_type", "object"))
            if target_type not in allowed_types:
                continue
            confidence = VisionInteractionNode._finite_nonnegative(
                item.get(
                    "detection_confidence",
                    item.get("confidence", 0.0),
                )
            ) or 0.0
            age = VisionInteractionNode._finite_nonnegative(
                item.get("last_seen_age_ms")
            )
            if confidence < minimum_confidence:
                continue
            if maximum_age_ms is not None and (
                age is None or age > maximum_age_ms
            ):
                continue
            nonhuman_targets.append(copy.deepcopy(item))
        targets = human_targets + nonhuman_targets

        return {
            "ok": True,
            "schema_version": int(snapshot.get("schema_version", 1)),
            "header": copy.deepcopy(snapshot.get("header", {})),
            "vision_epoch": str(snapshot.get("vision_epoch", "")),
            "sequence": int(snapshot.get("sequence", 0)),
            "snapshot_id": str(snapshot.get("snapshot_id", "")),
            "snapshot_age_ms": round(snapshot_age_ms, 1),
            "targets": targets,
            "human_candidates": copy.deepcopy(human_targets),
            "animal_candidates": [
                copy.deepcopy(item) for item in nonhuman_targets
                if item.get("target_type") == "animal"
            ],
            "object_candidates": [
                copy.deepcopy(item) for item in nonhuman_targets
                if item.get("target_type") == "object"
            ],
            "active_target": copy.deepcopy(
                snapshot.get("active_target", {})
            ),
        }

    @staticmethod
    def _finite_positive(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result > 0.0 else None

    @staticmethod
    def _finite_nonnegative(value: Any) -> float | None:
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        return result if math.isfinite(result) and result >= 0.0 else None

    def _fuse_human_depth(
        self,
        candidates: list[dict[str, Any]],
        *,
        observation_stamp: float,
    ) -> None:
        """Measure one complete aligned-depth fusion attempt."""
        started = time.perf_counter()
        result = (
            "success"
            if bool(getattr(self, "_depth_enabled", False))
            else "skipped"
        )
        try:
            VisionInteractionNode._fuse_human_depth_impl(
                self,
                candidates,
                observation_stamp=observation_stamp,
            )
        except Exception:
            result = "failure"
            raise
        finally:
            vision_timing_trace(
                node="vision_interaction",
                module="depth_fusion",
                stage="aligned_depth_range",
                latency_ms=(time.perf_counter() - started) * 1000.0,
                result=result,
                observation_stamp=round(float(observation_stamp), 6),
                candidate_count=len(candidates),
                fused_count=sum(
                    1 for item in candidates if item.get("range_valid") is True
                ),
            )

    def _fuse_human_depth_impl(
        self,
        candidates: list[dict[str, Any]],
        *,
        observation_stamp: float,
    ) -> None:
        """Fill ranges from an aligned depth image, otherwise fail closed."""
        for candidate in candidates:
            candidate["range_valid"] = False
            candidate["distance_m"] = None
            candidate["range_source"] = "none"
            candidate["depth_sync_delta_ms"] = None
            candidate["pose_3d"] = {
                "valid": False,
                "frame_id": "",
                "x": None,
                "y": None,
                "z": None,
            }
        if not bool(getattr(self, "_depth_enabled", False)):
            return

        now_monotonic = time.monotonic()
        with self._state_lock:
            latest_depth_monotonic = float(
                getattr(self, "_latest_depth_monotonic", 0.0)
            )
            history = list(getattr(self, "_depth_history", ()))
            if history:
                selected = min(
                    history,
                    key=lambda item: abs(float(item[0]) - observation_stamp),
                )
                depth_stamp = float(selected[0])
                depth_frame_id = str(selected[2]).strip()
                depth = selected[3]
            else:
                # Compatibility path for older serialized state and focused
                # unit tests. Production nodes always initialize history.
                depth = getattr(self, "_latest_depth_m", None)
                depth_stamp = float(
                    getattr(self, "_latest_depth_stamp", 0.0)
                )
                depth_frame_id = str(
                    getattr(self, "_latest_depth_frame_id", "")
                ).strip()
            intrinsics = copy.deepcopy(
                getattr(self, "_camera_intrinsics", None)
            )
        sync_delta_sec = abs(depth_stamp - observation_stamp)
        if (
            not isinstance(depth, np.ndarray)
            or depth.ndim != 2
            or depth.size == 0
            or latest_depth_monotonic <= 0.0
            or now_monotonic - latest_depth_monotonic
            > float(getattr(self, "_depth_stale_timeout_sec", 0.5))
            or depth_stamp <= 0.0
            or observation_stamp <= 0.0
            or sync_delta_sec
            > float(getattr(self, "_depth_sync_tolerance_sec", 0.10))
            or not isinstance(intrinsics, dict)
        ):
            return
        for candidate in candidates:
            candidate["depth_sync_delta_ms"] = round(
                sync_delta_sec * 1000.0, 3
            )

        try:
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics["cx"])
            cy = float(intrinsics["cy"])
            width = int(intrinsics["width"])
            height = int(intrinsics["height"])
            intrinsics_frame_id = str(intrinsics["frame_id"]).strip()
        except (KeyError, TypeError, ValueError):
            return
        if (
            not all(math.isfinite(item) for item in (fx, fy, cx, cy))
            or fx <= 0.0
            or fy <= 0.0
            or width <= 0
            or height <= 0
            or depth.shape != (height, width)
            or not depth_frame_id
            or not intrinsics_frame_id
            or depth_frame_id != intrinsics_frame_id
        ):
            return

        for candidate in candidates:
            if candidate.get("tracking_state") != "tracking":
                continue
            sample = VisionInteractionNode._sample_human_range(
                self,
                candidate,
                depth,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                frame_id=depth_frame_id,
            )
            if sample is None:
                continue
            candidate.update(sample)
            candidate["range_source"] = "aligned_depth"

    def _sample_human_range(
        self,
        candidate: dict[str, Any],
        depth_m: np.ndarray,
        *,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        frame_id: str,
    ) -> dict[str, Any] | None:
        """Use a central torso ROI and median depth for one person."""
        try:
            bbox_values = list(candidate.get("bbox", []))
            if len(bbox_values) != 4:
                return None
            bx, by, bw, bh = (float(value) for value in bbox_values)
            center_values = list(
                candidate.get("body_center")
                or candidate.get("center")
                or []
            )
            if len(center_values) != 2:
                center_values = [bx + bw / 2.0, by + bh / 2.0]
            center_x = float(center_values[0])
            center_y = float(center_values[1])
        except (TypeError, ValueError):
            return None
        if (
            not all(
                math.isfinite(value)
                for value in (bx, by, bw, bh, center_x, center_y)
            )
            or bw <= 0.0
            or bh <= 0.0
            or not (0.0 <= center_x <= 1.0)
            or not (0.0 <= center_y <= 1.0)
        ):
            return None

        image_height, image_width = depth_m.shape
        bbox_x1 = max(0, min(image_width - 1, int(math.floor(bx * image_width))))
        bbox_y1 = max(0, min(image_height - 1, int(math.floor(by * image_height))))
        bbox_x2 = max(0, min(image_width, int(math.ceil((bx + bw) * image_width))))
        bbox_y2 = max(0, min(image_height, int(math.ceil((by + bh) * image_height))))
        if bbox_x2 <= bbox_x1 or bbox_y2 <= bbox_y1:
            return None

        center_u = max(
            bbox_x1,
            min(bbox_x2 - 1, int(round(center_x * (image_width - 1)))),
        )
        center_v = max(
            bbox_y1,
            min(bbox_y2 - 1, int(round(center_y * (image_height - 1)))),
        )
        configured_radius = int(
            getattr(self, "_depth_sample_radius_px", 4)
        )
        half_width = max(
            configured_radius,
            int(round((bbox_x2 - bbox_x1) * 0.12)),
        )
        half_height = max(
            configured_radius,
            int(round((bbox_y2 - bbox_y1) * 0.08)),
        )
        roi_x1 = max(bbox_x1, center_u - half_width)
        roi_x2 = min(bbox_x2, center_u + half_width + 1)
        roi_y1 = max(bbox_y1, center_v - half_height)
        roi_y2 = min(bbox_y2, center_v + half_height + 1)
        roi = depth_m[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            return None
        minimum_depth = float(getattr(self, "_depth_min_m", 0.2))
        maximum_depth = float(getattr(self, "_depth_max_m", 8.0))
        valid_mask = (
            np.isfinite(roi)
            & (roi >= minimum_depth)
            & (roi <= maximum_depth)
        )
        valid_count = int(np.count_nonzero(valid_mask))
        if (
            valid_count < int(getattr(self, "_depth_min_valid_samples", 5))
            or valid_count / float(roi.size)
            < float(getattr(self, "_depth_min_valid_fraction", 0.2))
        ):
            return None

        z = float(np.median(roi[valid_mask]))
        if not math.isfinite(z) or not (minimum_depth <= z <= maximum_depth):
            return None
        x = (float(center_u) - cx) * z / fx
        y = (float(center_v) - cy) * z / fy
        # The AGV moves in the camera horizontal x-z plane.  Excluding the
        # vertical camera/person offset prevents a chest-height sample from
        # making the base drive closer than the configured stop distance.
        distance = math.hypot(x, z)
        if not math.isfinite(distance):
            return None
        return {
            "range_valid": True,
            "distance_m": round(distance, 3),
            "bearing_deg": round(math.degrees(math.atan2(x, z)), 3),
            "bearing_valid": True,
            "bearing_source": "camera_intrinsics",
            "pose_3d": {
                "valid": True,
                "frame_id": frame_id,
                "x": round(x, 3),
                "y": round(y, 3),
                "z": round(z, 3),
            },
        }

    def _publish_visual(self) -> None:
        with self._enrollment_lock:
            if self._enrollment.face_session is not None:
                self._process_enrollment_frame()

        vision = self._providers.get("vision")
        now = time.monotonic()
        with self._state_lock:
            latest_camera_monotonic = self._latest_camera_monotonic
            latest_objects_monotonic = self._latest_objects_monotonic
            latest_objects = [dict(item) for item in self._latest_objects]
            object_result_sequence = int(
                getattr(self, "_object_sequence", 0)
            )
        camera_fresh = (
            self._vision_is_mock
            or (
                latest_camera_monotonic > 0.0
                and now - latest_camera_monotonic
                <= self._camera_stale_timeout_sec
            )
        )
        try:
            raw = (
                vision.get_observation()  # type: ignore[attr-defined]
                if camera_fresh and vision is not None and vision.is_available()
                else {}
            )
        except Exception as exc:
            logger.error("Visual observation failed: %s", exc)
            raw = {}
        self._publish_gesture_debug(raw)
        if (
            camera_fresh
            and latest_objects_monotonic > 0.0
            and now - latest_objects_monotonic
            <= self._object_cache_timeout_sec
        ):
            raw = dict(raw)
            raw["tracked_objects"] = latest_objects
        event = VisionInteractionNode._build_visual_snapshot(self, raw)
        held_manager = getattr(self, "_held_object_pose", None)
        if held_manager is not None:
            held_status = held_manager.update(
                now=now,
                active_target=event.get("active_target", {}),
                hands=event.get("hands", []),
                objects=event.get("tracked_objects", []),
                object_result_sequence=object_result_sequence,
                pose_observation_stamp=event.get("header", {}).get("stamp"),
            )
            VisionInteractionNode._apply_held_object_pose(
                event,
                held_status,
            )
        VisionInteractionNode._update_held_object_stream(self, event, now)
        event["events"] = self._derive_events(event)
        # Cache exactly the complete packet that is published.  VisionTask
        # query_targets only ever copies this atomic snapshot; it never
        # rebuilds target IDs from unrelated caches.
        with self._state_lock:
            self._latest_visual_snapshot = copy.deepcopy(event)
            self._latest_visual_snapshot_monotonic = time.monotonic()
        publish_started = time.perf_counter()
        try:
            message = String()
            message.data = json.dumps(event, ensure_ascii=False)
            self._visual_pub.publish(message)
        except Exception:
            vision_timing_trace(
                node="vision_interaction",
                module="visual_event",
                stage="event_publish",
                latency_ms=(time.perf_counter() - publish_started) * 1000.0,
                result="failure",
                force=True,
                sequence=int(event.get("sequence", 0) or 0),
                event_count=len(event.get("events", [])),
            )
            raise
        publish_latency_ms = (time.perf_counter() - publish_started) * 1000.0
        vision_timing_trace(
            node="vision_interaction",
            module="visual_event",
            stage="event_publish",
            latency_ms=publish_latency_ms,
            sequence=int(event.get("sequence", 0) or 0),
            event_count=len(event.get("events", [])),
            events=list(event.get("events", [])),
        )
        VisionInteractionNode._log_visual_event_state(self, event)

    def _log_visual_event_state(self, event: dict[str, Any]) -> None:
        """Log identity, action gating and emitted events on state changes."""
        active = event.get("active_target", {})
        if not isinstance(active, dict):
            active = {}
        hands = event.get("hands", [])
        if not isinstance(hands, list):
            hands = []
        events = tuple(str(value) for value in event.get("events", []))
        hand_actions = tuple(
            str(hand.get("hand_action", ""))
            for hand in hands
            if isinstance(hand, dict) and hand.get("hand_action")
        )
        identity = str(active.get("identity", "unknown") or "unknown")
        identity_state = str(
            active.get("identity_state", "unverified") or "unverified"
        )
        pose_action = str(active.get("pose_action", "") or "")
        held_object = active.get("held_object", {})
        if not isinstance(held_object, dict):
            held_object = {}
        held_signature = (
            str(held_object.get("state", "inactive")),
            str(held_object.get("action", "")),
            str(held_object.get("object_label", "")),
            str(held_object.get("rejection_reason", "")),
            int(held_object.get("object_result_sequence", 0) or 0),
        )
        signature = (
            int(active.get("track_id", 0) or 0),
            str(active.get("tracking_state", "lost")),
            identity,
            identity_state,
            pose_action,
            hand_actions,
            held_signature,
            events,
        )
        if signature == getattr(self, "_last_visual_log_signature", None):
            return
        self._last_visual_log_signature = signature

        gate = (
            "open"
            if VisionInteractionNode._pose_event_identity_confirmed(active)
            else "blocked"
        )
        if not pose_action and not hand_actions:
            gate = "idle"
        logger.info(
            "Visual state changed: track=%s tracking=%s identity=%s "
            "identity_state=%s pose=%s hands=%s held=%s:%s:%s/%s:%.3f:%s "
            "pose_event_gate=%s events=%s",
            signature[0],
            signature[1],
            identity,
            identity_state,
            pose_action or "-",
            ",".join(hand_actions) or "-",
            held_signature[0],
            str(held_object.get("object_label", "")) or "-",
            int(held_object.get("evidence_hits", 0) or 0),
            int(held_object.get("required_hits", 2) or 2),
            float(held_object.get("association_score", 0.0) or 0.0),
            str(held_object.get("rejection_reason", "")) or "-",
            gate,
            ",".join(events) or "-",
        )
        trace_common = {
            "node": "vision_interaction",
            "module": "visual_event",
            "vision_epoch": str(event.get("vision_epoch", "")),
            "sequence": int(event.get("sequence", 0) or 0),
            "snapshot_id": str(event.get("snapshot_id", "")),
            "track_id": signature[0],
            "tracking_state": signature[1],
            "identity": identity,
            "identity_state": identity_state,
            "pose_action": pose_action,
            "hand_actions": list(hand_actions),
            "pose_event_gate": gate,
        }
        if held_signature != getattr(self, "_last_traced_held_signature", None):
            vision_trace(
                "held_object_evaluation",
                result=str(held_object.get("state", "inactive")),
                reason_code=str(held_object.get("rejection_reason", "")),
                candidate_action=str(held_object.get("candidate_action", "")),
                action=str(held_object.get("action", "")),
                object_label=str(held_object.get("evaluated_object_label", "")),
                object_confidence=float(
                    held_object.get("evaluated_object_confidence", 0.0) or 0.0
                ),
                valid_wrist_count=int(
                    held_object.get("valid_wrist_count", 0) or 0
                ),
                pose_object_sync_delta_ms=held_object.get(
                    "pose_object_sync_delta_ms"
                ),
                wrist_distance_ratio=held_object.get(
                    "evaluated_wrist_distance_ratio"
                ),
                wrist_distance_threshold_ratio=float(
                    held_object.get("wrist_distance_threshold_ratio", 0.0) or 0.0
                ),
                evidence_hits=int(held_object.get("evidence_hits", 0) or 0),
                required_hits=int(held_object.get("required_hits", 2) or 2),
                object_result_sequence=int(
                    held_object.get("object_result_sequence", 0) or 0
                ),
                **trace_common,
            )
            self._last_traced_held_signature = held_signature
        previous_events = set(getattr(self, "_last_traced_events", ()))
        current_events = set(events)
        for event_type in sorted(current_events - previous_events):
            vision_trace(
                "event_publish", result="published", event_type=event_type,
                **trace_common,
            )
        for event_type in sorted(previous_events - current_events):
            vision_trace(
                "event_cleared", result="cleared", event_type=event_type,
                **trace_common,
            )
        if gate == "blocked" and (pose_action or hand_actions):
            reason = (
                "target_not_tracking"
                if signature[1] != "tracking"
                else "identity_not_confirmed"
            )
            vision_trace(
                "event_suppressed", result="blocked", reason_code=reason,
                **trace_common,
            )
        self._last_traced_events = events

    def _publish_gesture_debug(self, raw: dict[str, Any]) -> None:
        """Publish exact engine labels without changing the stable visual contract."""
        diagnostics = raw.get("_gesture_diagnostics")
        if not isinstance(diagnostics, dict) or not diagnostics:
            return
        active = raw.get("active_target", {})
        if not isinstance(active, dict):
            active = {}
        hands = raw.get("hands", [])
        payload = {
            "schema_version": 1,
            "stamp": time.time(),
            "track_id": int(active.get("track_id", 0) or 0),
            "tracking_state": str(active.get("tracking_state", "lost")),
            "pose_state": str(active.get("pose_state", "unknown")),
            "legacy_pose_action": str(active.get("pose_action", "")),
            "legacy_hand_actions": [
                str(hand.get("hand_action", ""))
                for hand in hands
                if isinstance(hand, dict) and hand.get("hand_action")
            ],
            **diagnostics,
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self._gesture_debug_pub.publish(message)

    def _poll_objects(self) -> None:
        """Run an explicit stream, or the lower-priority held-pose stream."""
        decision = self._object_stream.poll()
        state = str(decision.get("state", "inactive"))
        if state == "inactive":
            context_stream = getattr(self, "_held_object_stream", None)
            if context_stream is not None:
                decision = context_stream.poll()
                state = str(decision.get("state", "inactive"))
        if state == "expired":
            self._record_object_result(
                [],
                source="control",
                status="stopped",
                latency_ms=0.0,
                stream=decision.get("stream"),
                stop_reason=str(
                    decision.get("stop_reason", "lease_expired")
                ),
            )
            logger.warning(
                "Object detection lease expired: session=%s",
                decision.get("stream", {}).get("session_id", ""),
            )
            return
        if state != "due":
            return
        result = self._run_object_detection(
            dict(decision.get("params", {})),
            source="stream",
            wait_for_slot=False,
            stream=decision.get("stream"),
        )
        if not result.get("ok", False) and not result.get("busy", False):
            logger.debug(
                "Object stream inference skipped: %s",
                result.get("error"),
            )

    def _effective_object_stream_snapshot(self) -> dict[str, Any]:
        """Prefer the externally owned stream, then the automatic context."""
        stream = self._object_stream.snapshot()
        if stream.get("active"):
            return stream
        context_stream = getattr(self, "_held_object_stream", None)
        return (
            context_stream.snapshot()
            if context_stream is not None
            else stream
        )

    def _update_held_object_stream(
        self,
        event: dict[str, Any],
        now: float,
    ) -> None:
        """Enable 2 Hz object sampling only while a human remains visible."""
        context_stream = getattr(self, "_held_object_stream", None)
        if context_stream is None or not getattr(
            self, "_held_object_enabled", False
        ):
            return
        detector = self._providers.get("object")
        if detector is None or not detector.is_available():
            return
        active = event.get("active_target", {})
        human_visible = (
            isinstance(active, dict)
            and str(active.get("tracking_state", "")) == "tracking"
            and float(active.get("confidence", 0.0) or 0.0) > 0.0
        )
        if human_visible:
            self._held_object_last_human_at = now
            snapshot = context_stream.snapshot(now=now)
            remaining = snapshot.get("lease_remaining_sec")
            if (
                not snapshot.get("active")
                or remaining is None
                or float(remaining) < 1.0
            ):
                result = context_stream.configure({
                    "enabled": True,
                    "session_id": self._held_object_session_id,
                    "rate_hz": self._held_object_rate_hz,
                    "confidence": self._held_object_confidence,
                    "target_labels": list(HELD_OBJECT_LABELS),
                    "lease_sec": 5.0,
                }, now=now)
                if result.get("ok") and not snapshot.get("active"):
                    logger.info(
                        "Held-object detection started: session=%s rate=%.2fHz",
                        self._held_object_session_id,
                        self._held_object_rate_hz,
                    )
            return
        last_human_at = float(
            getattr(self, "_held_object_last_human_at", 0.0)
        )
        if (
            (context_snapshot := context_stream.snapshot(now=now)).get("active")
            and (
                last_human_at <= 0.0
                or now - last_human_at
                >= self._held_object_absence_grace_sec
            )
        ):
            result = context_stream.configure({
                "enabled": False,
                "session_id": self._held_object_session_id,
            }, now=now)
            if result.get("ok"):
                logger.info(
                    "Held-object detection stopped: session=%s reason=human_absent",
                    self._held_object_session_id,
                )
                # If no external Action/debug stream owns the detector, publish
                # a terminal packet so the dashboard cannot retain a stale
                # automatic-stream state or stale object facts.
                external_stream = getattr(self, "_object_stream", None)
                external_active = bool(
                    external_stream is not None
                    and external_stream.snapshot(now=now).get("active")
                )
                record_result = getattr(self, "_record_object_result", None)
                if callable(record_result) and not external_active:
                    terminal_stream = dict(context_snapshot)
                    terminal_stream.update({
                        "active": False,
                        "lease_remaining_sec": 0.0,
                    })
                    record_result(
                        [],
                        source="control",
                        status="stopped",
                        latency_ms=0.0,
                        stream=terminal_stream,
                        stop_reason="human_absent",
                    )

    @staticmethod
    def _apply_held_object_pose(
        event: dict[str, Any],
        status: HeldObjectPoseStatus,
    ) -> None:
        """Write the multimodal pose to the selected human only."""
        active = event.get("active_target", {})
        if not isinstance(active, dict):
            return
        details = status.to_dict()
        active["held_object"] = copy.deepcopy(details)
        target_id = str(active.get("target_id", ""))
        track_id = int(active.get("track_id", 0) or 0)
        if status.action and str(active.get("pose_action", "")) != "fallen_down":
            active["pose_action"] = status.action
            active["pose_action_label"] = status.action_label
        for candidate in event.get("human_candidates", []):
            if (
                isinstance(candidate, dict)
                and str(candidate.get("target_id", "")) == target_id
            ):
                candidate["held_object"] = copy.deepcopy(details)
                if status.action and str(
                    candidate.get("pose_action", "")
                ) != "fallen_down":
                    candidate["pose_action"] = status.action
                    candidate["pose_action_label"] = status.action_label
        for human in event.get("humans", []):
            if (
                isinstance(human, dict)
                and int(human.get("track_id", -1) or -1) == track_id
            ):
                human["held_object"] = copy.deepcopy(details)
                if status.action and str(
                    human.get("pose_action", "")
                ) != "fallen_down":
                    human["pose_action"] = status.action
                    human["pose_action_label"] = status.action_label

    def _run_object_detection(
        self,
        params: dict[str, Any],
        *,
        source: str,
        wait_for_slot: bool,
        stream: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        detector = self._providers.get("object")
        if detector is None:
            return {"ok": False, "error": "object detector unavailable"}

        acquired = self._object_inference_lock.acquire(blocking=wait_for_slot)
        if not acquired:
            return {
                "ok": False,
                "busy": True,
                "error": "object detector busy",
            }

        started = time.perf_counter()
        try:
            frame, observation_stamp, frame_id = (
                self._frame_for_tasks_with_metadata()
            )
            objects = detector.detect_objects(  # type: ignore[attr-defined]
                frame, params
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            detector_error = str(getattr(detector, "last_error", ""))
            if detector_error:
                self._record_object_result(
                    [],
                    source=source,
                    status="error",
                    latency_ms=latency_ms,
                    error=detector_error,
                    observation_stamp=observation_stamp,
                    frame_id=frame_id,
                    stream=stream,
                    request=self._object_request_metadata(params),
                )
                return {"ok": False, "error": detector_error}

            normalized_objects = [
                dict(item) for item in objects if isinstance(item, dict)
            ]
            tracked_objects = self._record_object_result(
                normalized_objects,
                source=source,
                status="ok",
                latency_ms=latency_ms,
                observation_stamp=observation_stamp,
                frame_id=frame_id,
                stream=stream,
                request=self._object_request_metadata(params),
            )
            return {"ok": True, "objects": tracked_objects}
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000.0
            self._record_object_result(
                [],
                source=source,
                status="error",
                latency_ms=latency_ms,
                error=str(exc),
                stream=stream,
                request=self._object_request_metadata(params),
            )
            return {"ok": False, "error": str(exc)}
        finally:
            self._object_inference_lock.release()

    def _update_object_tracks_locked(
        self,
        objects: list[dict[str, Any]],
        *,
        now_monotonic: float,
        observation_stamp: float,
        frame_id: str,
        source_sequence: int,
    ) -> None:
        """Associate detection boxes without changing stream ownership.

        This is deliberately a small fact tracker, not an inference scheduler:
        it runs only after an existing service/leased-stream detection succeeds.
        """
        tracks = getattr(self, "_object_tracks", None)
        if not isinstance(tracks, dict):
            tracks = {}
            self._object_tracks = tracks
        persistence_sec = float(
            getattr(self, "_object_target_persistence_sec", 2.0)
        )
        for track_id in [
            key for key, track in tracks.items()
            if now_monotonic - float(track.get("last_seen_monotonic", 0.0))
            > persistence_sec
        ]:
            del tracks[track_id]

        normalized: list[dict[str, Any]] = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            try:
                label = " ".join(
                    str(item.get("label", "")).strip().lower().split()
                )
                bbox = (
                    float(item.get("x", 0.0)),
                    float(item.get("y", 0.0)),
                    float(item.get("w", 0.0)),
                    float(item.get("h", 0.0)),
                )
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                continue
            if (
                not label
                or not all(math.isfinite(value) for value in bbox)
                or not math.isfinite(confidence)
                or bbox[2] <= 0.0
                or bbox[3] <= 0.0
                or confidence < 0.0
            ):
                continue
            normalized.append({
                "label": label,
                "display_label": str(item.get("label", label)),
                "bbox": bbox,
                "confidence": min(1.0, confidence),
            })

        pairs: list[tuple[float, int, int]] = []
        for detection_index, detection in enumerate(normalized):
            for track_id, track in tracks.items():
                if str(track.get("label", "")) != detection["label"]:
                    continue
                score = VisionInteractionNode._bbox_iou(
                    tuple(track.get("bbox", (0.0, 0.0, 0.0, 0.0))),
                    detection["bbox"],
                )
                if score >= 0.15:
                    pairs.append((score, detection_index, track_id))
        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, detection_index, track_id in sorted(pairs, reverse=True):
            if detection_index in assignments or track_id in used_tracks:
                continue
            assignments[detection_index] = track_id
            used_tracks.add(track_id)

        vision_epoch = str(getattr(self, "_vision_epoch", ""))
        for index, detection in enumerate(normalized):
            track_id = assignments.get(index)
            if track_id is None:
                track_id = int(getattr(self, "_next_object_track_id", 1))
                self._next_object_track_id = track_id + 1
            label = detection["label"]
            target_type = "animal" if label in _ANIMAL_LABELS else "object"
            object_kind = "toy" if label in _TOY_LABELS else "generic"
            tracks[track_id] = {
                "vision_epoch": vision_epoch,
                "target_id": f"{vision_epoch}:object:{track_id}",
                "target_type": target_type,
                "track_id": track_id,
                "object_track_id": track_id,
                "label": label,
                "display_label": detection["display_label"],
                "object_kind": object_kind,
                "bbox": detection["bbox"],
                "confidence": detection["confidence"],
                "last_seen_at": float(observation_stamp),
                "last_seen_monotonic": now_monotonic,
                "header": {
                    "stamp": float(observation_stamp),
                    "frame_id": str(frame_id),
                },
                "source_sequence": int(source_sequence),
                "source_snapshot_id": (
                    f"{vision_epoch}:object-result:{source_sequence}"
                ),
            }

    def _object_candidates_locked(
        self,
        *,
        now_monotonic: float,
    ) -> list[dict[str, Any]]:
        tracks = getattr(self, "_object_tracks", {})
        if not isinstance(tracks, dict):
            return []
        persistence_sec = float(
            getattr(self, "_object_target_persistence_sec", 2.0)
        )
        current_timeout_sec = float(
            getattr(self, "_object_target_current_timeout_sec", 0.75)
        )
        expired = [
            track_id for track_id, track in tracks.items()
            if now_monotonic - float(track.get("last_seen_monotonic", 0.0))
            > persistence_sec
        ]
        for track_id in expired:
            del tracks[track_id]
        candidates: list[dict[str, Any]] = []
        horizontal_fov_deg = float(
            getattr(self, "_horizontal_fov_deg", 69.0)
        )
        for track_id in sorted(tracks):
            track = tracks[track_id]
            age_sec = max(
                0.0,
                now_monotonic
                - float(track.get("last_seen_monotonic", now_monotonic)),
            )
            bx, by, bw, bh = track["bbox"]
            center_x = bx + bw / 2.0
            center_y = by + bh / 2.0
            candidates.append({
                "vision_epoch": str(track["vision_epoch"]),
                "target_id": str(track["target_id"]),
                "target_type": str(track["target_type"]),
                "track_id": int(track_id),
                "object_track_id": int(track_id),
                "label": str(track["label"]),
                "display_label": str(track["display_label"]),
                "object_kind": str(track["object_kind"]),
                "bbox": [round(float(value), 4) for value in track["bbox"]],
                "x": round(float(bx), 4),
                "y": round(float(by), 4),
                "w": round(float(bw), 4),
                "h": round(float(bh), 4),
                "center": [round(center_x, 4), round(center_y, 4)],
                "center_x": round(center_x, 4),
                "center_y": round(center_y, 4),
                "confidence": round(float(track["confidence"]), 4),
                "detection_confidence": round(
                    float(track["confidence"]), 4
                ),
                "tracking_state": (
                    "tracking"
                    if age_sec <= current_timeout_sec
                    else "temporarily_lost"
                ),
                "last_seen_age_ms": round(age_sec * 1000.0, 1),
                "bearing_deg": round(
                    (center_x - 0.5) * horizontal_fov_deg, 3
                ),
                "bearing_valid": True,
                "bearing_source": "configured_hfov",
                "range_valid": False,
                "distance_m": None,
                "pose_3d": {
                    "valid": False,
                    "frame_id": "",
                    "x": None,
                    "y": None,
                    "z": None,
                },
                "header": copy.deepcopy(track["header"]),
                "source_sequence": int(track["source_sequence"]),
                "source_snapshot_id": str(track["source_snapshot_id"]),
            })
        return candidates

    @staticmethod
    def _bbox_iou(
        first: tuple[float, float, float, float],
        second: tuple[float, float, float, float],
    ) -> float:
        if len(first) != 4 or len(second) != 4:
            return 0.0
        if min(first[2], first[3], second[2], second[3]) <= 0.0:
            return 0.0
        ax2, ay2 = first[0] + first[2], first[1] + first[3]
        bx2, by2 = second[0] + second[2], second[1] + second[3]
        ix1, iy1 = max(first[0], second[0]), max(first[1], second[1])
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = (
            first[2] * first[3]
            + second[2] * second[3]
            - intersection
        )
        return intersection / union if union > 0.0 else 0.0

    def _record_object_result(
        self,
        objects: list[dict[str, Any]],
        *,
        source: str,
        status: str,
        latency_ms: float,
        error: str = "",
        observation_stamp: float | None = None,
        frame_id: str | None = None,
        stream: dict[str, Any] | None = None,
        request: dict[str, Any] | None = None,
        stop_reason: str = "",
    ) -> list[dict[str, Any]]:
        now_monotonic = time.monotonic()
        published_at = time.time()
        with self._state_lock:
            self._object_sequence += 1
            sequence = self._object_sequence
            resolved_frame_id = frame_id or self._latest_camera_frame_id
            resolved_stamp = (
                float(observation_stamp)
                if observation_stamp is not None and observation_stamp > 0.0
                else published_at
            )
            if status == "ok":
                VisionInteractionNode._update_object_tracks_locked(
                    self,
                    objects,
                    now_monotonic=now_monotonic,
                    observation_stamp=resolved_stamp,
                    frame_id=resolved_frame_id,
                    source_sequence=sequence,
                )
                published_objects = (
                    VisionInteractionNode._object_candidates_locked(
                        self,
                        now_monotonic=now_monotonic,
                    )
                )
            else:
                published_objects = []
                self._latest_objects = []
                self._latest_objects_monotonic = 0.0
                # Error/stopped packets are terminal for the cached facts;
                # no object target may survive them as a movement reference.
                getattr(self, "_object_tracks", {}).clear()

        if status == "ok":
            # Object/animal approach uses only aligned, timestamp-matched depth.
            # Reuse the same fail-closed ROI sampler as human targets; without
            # valid depth every candidate remains range_valid=false.
            VisionInteractionNode._fuse_human_depth(
                self,
                published_objects,
                observation_stamp=resolved_stamp,
            )
            with self._state_lock:
                self._latest_objects = [
                    copy.deepcopy(item) for item in published_objects
                ]
                self._latest_objects_monotonic = now_monotonic

        if stream is None:
            snapshot = getattr(
                self, "_effective_object_stream_snapshot", None
            )
            stream = snapshot() if callable(snapshot) else {
                "active": False,
                "session_id": "",
                "rate_hz": 0.0,
                "confidence": None,
                "target_labels": [],
                "lease_remaining_sec": None,
            }
        payload = {
            "schema_version": 2,
            "header": {
                "stamp": resolved_stamp,
                "frame_id": resolved_frame_id,
            },
            "published_at": published_at,
            "sequence": sequence,
            "source": source,
            "status": status,
            "stream": dict(stream),
            "request": dict(request or {}),
            "stop_reason": stop_reason,
            "inference_latency_ms": round(float(latency_ms), 3),
            "objects": [copy.deepcopy(item) for item in published_objects],
            "error": error,
        }
        message = String()
        message.data = json.dumps(payload, ensure_ascii=False)
        self._object_pub.publish(message)
        vision_trace(
            "stage_complete",
            result=status,
            node="vision_interaction",
            module="object_detection",
            stage="object_inference",
            source=source,
            session_id=str(stream.get("session_id", "")),
            sequence=sequence,
            latency_ms=round(float(latency_ms), 3),
            object_count=len(published_objects),
            reason_code=stop_reason or ("inference_error" if error else ""),
            error=error,
        )
        return [copy.deepcopy(item) for item in published_objects]

    @staticmethod
    def _object_request_metadata(params: dict[str, Any]) -> dict[str, Any]:
        try:
            labels = ObjectDetectionSessionManager.normalize_target_labels(
                params.get("target_labels")
            )
        except ValueError:
            labels = []
        confidence = params.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        return {
            "target_labels": labels,
            "confidence": confidence,
        }

    def _set_object_detection(
        self,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        enabled = params.get("enabled")
        if enabled is True:
            detector = self._providers.get("object")
            if detector is None or not detector.is_available():
                return {
                    "ok": False,
                    "error": "object detector unavailable",
                    "stream": self._object_stream.snapshot(),
                }
        before = self._object_stream.snapshot()
        if enabled is False:
            # Make the stopped status the last packet for this session even if
            # a slow RKNN inference was already running.
            with self._object_inference_lock:
                result = self._object_stream.configure(params)
                if result.get("ok") and before.get("active"):
                    terminal = dict(before)
                    terminal["active"] = False
                    terminal["lease_remaining_sec"] = 0.0
                    self._record_object_result(
                        [],
                        source="control",
                        status="stopped",
                        latency_ms=0.0,
                        stream=terminal,
                        stop_reason="requested",
                    )
                return result
        result = self._object_stream.configure(params)
        if result.get("ok"):
            stream = result.get("stream", {})
            logger.info(
                "Object detection stream active: session=%s rate=%.2fHz "
                "targets=%s",
                stream.get("session_id", ""),
                float(stream.get("rate_hz", 0.0)),
                stream.get("target_labels", []),
            )
        return result

    @staticmethod
    def _pose_event_identity_confirmed(active: dict[str, Any]) -> bool:
        identity = str(active.get("identity", ""))
        return (
            identity in ALLOWED_FACE_IDENTITIES
            and str(active.get("identity_state", "")) == "confirmed_known"
            and str(active.get("tracking_state", "")) == "tracking"
        )

    @staticmethod
    def _derive_events(
        observation: dict[str, Any],
    ) -> list[str]:
        events: list[str] = []
        active = observation.get("active_target", {})
        identity = str(active.get("identity", "unknown"))
        identity_confirmed = (
            VisionInteractionNode._pose_event_identity_confirmed(active)
        )
        if observation.get("faces"):
            events.append(face_identity_to_vision_event(identity))
        action = str(active.get("pose_action", ""))
        action_event = pose_action_to_vision_event(action, identity_confirmed)
        if action_event:
            events.append(action_event)
        for hand in observation.get("hands", []):
            hand_event = pose_action_to_vision_event(
                str(hand.get("hand_action", "")), identity_confirmed
            )
            if hand_event and hand_event not in events:
                events.append(hand_event)
        return events

    def _process_enrollment_frame(self) -> None:
        frame = self._frame_for_tasks()
        if frame is None:
            return
        result = self._run_with_shared_face_models(
            lambda: self._enrollment.process_face_frame(frame)
        )
        if result.get("done"):
            self._sync_face_registry()
        message = String()
        message.data = json.dumps(result, ensure_ascii=False)
        self._enrollment_pub.publish(message)

    def _handle_task(self, request: Any, response: Any) -> Any:
        started = time.perf_counter()
        response.task_id = request.task_id
        response.task_type = request.task_type
        response.success = False
        response.result_json = ""
        response.error_message = ""
        vision_trace(
            "stage_start",
            result="started",
            node="vision_interaction",
            module="vision_task",
            stage="service",
            task_id=str(request.task_id),
            task_type=str(request.task_type),
        )
        try:
            params = json.loads(request.params_json or "{}")
            if isinstance(params, list):
                params = {
                    str(item.get("key", "")): item.get("value")
                    for item in params if isinstance(item, dict)
                }
            if not isinstance(params, dict):
                params = {}
            result = self._run_task(str(request.task_type), params)
            response.success = bool(result.get("ok", True))
            response.result_json = json.dumps(result, ensure_ascii=False)
            if not response.success:
                response.error_message = str(result.get("error", "task failed"))
        except Exception as exc:
            response.error_message = str(exc)
        response.latency_ms = (time.perf_counter() - started) * 1000.0
        vision_trace(
            "stage_complete",
            result="success" if response.success else "failure",
            node="vision_interaction",
            module="vision_task",
            stage="service",
            task_id=str(response.task_id),
            task_type=str(response.task_type),
            latency_ms=round(float(response.latency_ms), 3),
            reason_code="" if response.success else "task_failed",
            error=str(response.error_message),
        )
        return response

    def _run_task(self, task_type: str, params: dict[str, Any]) -> dict[str, Any]:
        vision = self._providers.get("vision")
        if task_type == "check_person":
            if vision is None or not hasattr(vision, "check_person"):
                return {"ok": False, "error": "vision unavailable"}
            return {"ok": True, **vision.check_person()}  # type: ignore[attr-defined]
        if task_type == "query_targets":
            return self._query_targets(params)
        if task_type == "detect_objects":
            return self._run_object_detection(
                params,
                source="service",
                wait_for_slot=True,
            )
        if task_type == "set_object_detection":
            return self._set_object_detection(params)
        if task_type == "get_object_detection_state":
            return {
                "ok": True,
                # `stream` remains the externally owned Action/debug session.
                # The automatic held-pose scheduler is observable separately
                # and must never make a downstream caller think its own session
                # is occupied.
                "stream": self._object_stream.snapshot(),
                "automatic_stream": self._held_object_stream.snapshot(),
            }
        if task_type == "recognize_face":
            recognizer = self._providers.get("face_recognition")
            if recognizer is None:
                return {"ok": False, "error": "face recognizer unavailable"}
            return {
                "ok": True,
                **recognizer.recognize(self._crop_largest_face()),  # type: ignore[attr-defined]
            }
        if task_type == "start_face_enrollment":
            with self._enrollment_lock:
                return self._enrollment.start_face(
                    str(params.get("name", "")),
                    int(params.get("required_shots", 3)),
                )
        if task_type == "cancel_face_enrollment":
            with self._enrollment_lock:
                return self._enrollment.cancel_face()
        if task_type == "upload_face":
            payload = base64.b64decode(
                str(params.get("image_base64", "")), validate=True
            )
            with self._enrollment_lock:
                result = self._run_with_shared_face_models(
                    lambda: self._enrollment.enroll_face_from_image(
                        str(params.get("name", "")), payload
                    )
                )
                if result.get("ok"):
                    self._sync_face_registry()
                return result
        if task_type == "list_face_records":
            with self._enrollment_lock:
                return self._enrollment.list_face_records()
        if task_type == "list_face_samples":
            with self._enrollment_lock:
                return self._enrollment.list_face_samples(
                    str(params.get("name", ""))
                )
        if task_type == "get_face_sample":
            with self._enrollment_lock:
                return self._enrollment.get_face_sample(
                    str(params.get("name", "")),
                    int(params.get("sample_id", 0)),
                )
        if task_type == "replace_face_sample":
            payload = base64.b64decode(
                str(params.get("image_base64", "")), validate=True
            )
            with self._enrollment_lock:
                result = self._run_with_shared_face_models(
                    lambda: self._enrollment.replace_face_sample(
                        str(params.get("name", "")),
                        int(params.get("sample_id", 0)),
                        payload,
                    )
                )
                if result.get("ok"):
                    self._sync_face_registry()
                return result
        if task_type == "delete_face_sample":
            with self._enrollment_lock:
                result = self._enrollment.delete_face_sample(
                    str(params.get("name", "")),
                    int(params.get("sample_id", 0)),
                )
                if result.get("ok"):
                    self._sync_face_registry()
                return result
        if task_type == "list_faces":
            with self._enrollment_lock:
                return {
                    "ok": True,
                    "faces": self._enrollment.list_enrolled_faces(),
                }
        if task_type == "delete_face":
            with self._enrollment_lock:
                result = self._enrollment.delete_face(
                    str(params.get("name", ""))
                )
                if result.get("ok"):
                    self._sync_face_registry()
                return result
        return {"ok": False, "error": f"unsupported task_type: {task_type}"}

    def _crop_largest_face(self) -> np.ndarray | None:
        frame = self._frame_for_tasks()
        vision = self._providers.get("vision")
        if frame is None or vision is None or not hasattr(vision, "get_observation"):
            return None
        faces = vision.get_observation().get("faces", [])  # type: ignore[attr-defined]
        if not faces:
            return None
        face = max(faces, key=lambda item: item.get("w", 0) * item.get("h", 0))
        height, width = frame.shape[:2]
        x1 = max(0, int(face["x"] * width))
        y1 = max(0, int(face["y"] * height))
        x2 = min(width, int((face["x"] + face["w"]) * width))
        y2 = min(height, int((face["y"] + face["h"]) * height))
        return frame[y1:y2, x1:x2] if x2 > x1 and y2 > y1 else None

    def _frame_for_tasks(self) -> np.ndarray | None:
        frame, _, _ = VisionInteractionNode._frame_for_tasks_with_metadata(
            self
        )
        return frame

    def _frame_for_tasks_with_metadata(
        self,
    ) -> tuple[np.ndarray | None, float, str]:
        with self._state_lock:
            frame = self._latest_frame
            latest_camera_monotonic = self._latest_camera_monotonic
            camera_stamp = float(
                getattr(self, "_latest_camera_stamp", 0.0)
            )
            frame_id = str(
                getattr(self, "_latest_camera_frame_id", "camera_link")
            )
        if self._vision_is_mock:
            return frame, camera_stamp, frame_id
        if (
            frame is None
            or latest_camera_monotonic <= 0.0
            or time.monotonic() - latest_camera_monotonic
            > self._camera_stale_timeout_sec
        ):
            return None, camera_stamp, frame_id
        selected, _ = select_camera_view(
            frame,
            stereo_enabled=self._stereo_enabled,
            view=self._stereo_view,
            min_aspect_ratio=self._stereo_min_aspect_ratio,
        )
        return selected, camera_stamp, frame_id

    @staticmethod
    def _decode_image(message: Image) -> np.ndarray | None:
        try:
            import cv2

            encoding = str(getattr(message, "encoding", "bgr8")).lower()
            channels = 1 if encoding in ("mono8", "8uc1") else 3
            width = int(message.width)
            height = int(message.height)
            row_bytes = int(message.step) or width * channels
            raw = np.frombuffer(message.data, dtype=np.uint8).reshape(
                height, row_bytes
            )
            if channels == 1:
                return cv2.cvtColor(
                    raw[:, :width], cv2.COLOR_GRAY2BGR
                )
            frame = raw[:, : width * 3].reshape(height, width, 3)
            if encoding == "rgb8":
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if encoding not in ("bgr8", "8uc3"):
                return None
            return frame.copy()
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _decode_depth_image(message: Image) -> np.ndarray | None:
        """Decode supported ROS depth encodings without assuming tight rows.

        ``16UC1``/``mono16`` are converted from millimetres to metres and
        ``32FC1`` is already in metres.  Invalid, zero and negative samples
        become NaN so downstream range fusion cannot accidentally treat them
        as a nearby obstacle or person.
        """
        try:
            encoding = str(getattr(message, "encoding", "")).strip().lower()
            if encoding in ("16uc1", "mono16"):
                kind = "u2"
                scale = 0.001
            elif encoding == "32fc1":
                kind = "f4"
                scale = 1.0
            else:
                return None
            width = int(message.width)
            height = int(message.height)
            item_size = np.dtype(kind).itemsize
            step = int(message.step) or width * item_size
            if (
                width <= 0
                or height <= 0
                or step < width * item_size
            ):
                return None
            data = memoryview(message.data)
            required_bytes = (height - 1) * step + width * item_size
            if data.nbytes < required_bytes:
                return None
            endian = ">" if bool(getattr(message, "is_bigendian", 0)) else "<"
            dtype = np.dtype(endian + kind)
            values = np.ndarray(
                shape=(height, width),
                dtype=dtype,
                buffer=data,
                strides=(step, item_size),
            ).astype(np.float32, copy=True)
            if scale != 1.0:
                values *= scale
            values[(~np.isfinite(values)) | (values <= 0.0)] = np.nan
            return values
        except (AttributeError, BufferError, TypeError, ValueError):
            return None

    def _wire_face_enrollment(self) -> None:
        vision = self._providers.get("vision")
        detector = getattr(vision, "_face_detector", None)
        if detector is not None:
            self._enrollment.set_face_detector(detector)

    def _init_face_api(self) -> None:
        config = self._config.get("face_api", {})
        if not isinstance(config, dict):
            config = {}
        enabled = bool(config.get("enabled", False))
        self._face_api_status = {"enabled": enabled, "ready": False}
        if not enabled:
            return
        try:
            from marsdog_vision_interaction.api import FaceApiServer

            self._face_api = FaceApiServer(
                config,
                self._enroll_uploaded_face_for_api,
                self._list_faces_for_api,
                self._list_face_samples_for_api,
                self._get_face_sample_for_api,
                self._replace_face_sample_for_api,
                self._delete_face_sample_for_api,
            )
            ready = self._face_api.start()
            self._face_api_status = {
                "enabled": True,
                "ready": ready,
                "address": self._face_api.address,
                "docs": f"{self._face_api.address}/docs",
            }
        except Exception as exc:
            self._face_api = None
            self._face_api_status = {
                "enabled": True,
                "ready": False,
                "error": str(exc),
            }
            logger.error("Face FastAPI unavailable: %s", exc, exc_info=True)

    def _enroll_uploaded_face_for_api(
        self,
        name: str,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        with self._enrollment_lock:
            result = self._run_with_shared_face_models(
                lambda: self._enrollment.enroll_face_from_image(
                    name,
                    image_bytes,
                )
            )
            if result.get("ok"):
                self._sync_face_registry()
            return result

    def _list_faces_for_api(self) -> dict[str, Any]:
        with self._enrollment_lock:
            return self._enrollment.list_face_records()

    def _list_face_samples_for_api(self, name: str) -> dict[str, Any]:
        with self._enrollment_lock:
            return self._enrollment.list_face_samples(name)

    def _get_face_sample_for_api(
        self,
        name: str,
        sample_id: int,
    ) -> dict[str, Any]:
        with self._enrollment_lock:
            return self._enrollment.get_face_sample(name, sample_id)

    def _replace_face_sample_for_api(
        self,
        name: str,
        sample_id: int,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        with self._enrollment_lock:
            result = self._run_with_shared_face_models(
                lambda: self._enrollment.replace_face_sample(
                    name,
                    sample_id,
                    image_bytes,
                )
            )
            if result.get("ok"):
                self._sync_face_registry()
            return result

    def _delete_face_sample_for_api(
        self,
        name: str,
        sample_id: int,
    ) -> dict[str, Any]:
        with self._enrollment_lock:
            result = self._enrollment.delete_face_sample(name, sample_id)
            if result.get("ok"):
                self._sync_face_registry()
            return result

    def _run_with_shared_face_models(self, operation: Any) -> Any:
        vision = self._providers.get("vision")
        runner = getattr(vision, "run_inference_exclusive", None)
        if callable(runner):
            return runner(operation)
        return operation()

    def _sync_face_registry(self) -> None:
        vision = self._providers.get("vision")
        if vision is not None and hasattr(vision, "sync_enrolled_to_throttle"):
            vision.sync_enrolled_to_throttle()  # type: ignore[attr-defined]
        recognizer = self._providers.get("face_recognition")
        if recognizer is None or not hasattr(recognizer, "enroll"):
            return
        try:
            import cv2
            if hasattr(recognizer, "clear_enrolled"):
                recognizer.clear_enrolled()  # type: ignore[attr-defined]
            for name in self._enrollment.list_enrolled_faces():
                for path in self._enrollment.get_face_paths(name):
                    image = cv2.imread(path)
                    if image is None:
                        continue
                    result = recognizer.enroll(image, name)  # type: ignore[attr-defined]
                    # Keep every valid view as a recognition template.
        except Exception as exc:
            logger.warning("Face registry sync failed: %s", exc)

    def destroy_node(self) -> None:
        if self._face_api is not None:
            self._face_api.stop()
            self._face_api = None
        for provider in self._providers.values():
            if provider is not None:
                provider.stop()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VisionInteractionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        executor.shutdown(timeout_sec=2.0)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
