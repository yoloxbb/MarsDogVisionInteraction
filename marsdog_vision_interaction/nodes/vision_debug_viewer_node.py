"""Live annotated viewer for camera, visual target and AGV commands."""

from __future__ import annotations

from collections import deque
import copy
import json
import os
import threading
import time
from typing import Any

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from marsdog_vision_interaction.messages.face_identity import (
    ALLOWED_FACE_IDENTITIES,
)
from marsdog_vision_interaction.utils.visual_debug import draw_visual_debug
from marsdog_vision_interaction.utils.stereo_view import select_camera_view
from marsdog_vision_interaction.utils.web_debug_server import (
    VisionDebugWebServer,
)

try:
    from marsdog_vision_interaction.srv import VisionTask
except ImportError:  # source-only use before the ROSIDL package is built
    VisionTask = None  # type: ignore[assignment,misc]


_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)
_WEB_OBJECT_SESSION_ID = "vision-debug-web"


class VisionDebugViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_debug_viewer")
        self.declare_parameter(
            "camera_topic", "/camera/camera/color/image_raw"
        )
        self.declare_parameter("visual_topic", "/perception/visual_event")
        self.declare_parameter(
            "gesture_debug_topic", "/perception/vision/gesture_debug"
        )
        self.declare_parameter(
            "control_topic", "/behavior/attention_tracking"
        )
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter(
            "debug_image_topic", "/perception/vision/debug_image"
        )
        self.declare_parameter("show_window", True)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("window_scale", 1.0)
        self.declare_parameter("max_render_fps", 8.0)
        self.declare_parameter("render_scale", 0.75)
        self.declare_parameter("stereo_enabled", False)
        self.declare_parameter("stereo_view", "left")
        self.declare_parameter("stereo_min_aspect_ratio", 2.2)
        self.declare_parameter("overlay_stale_sec", 1.0)
        self.declare_parameter("web_enabled", True)
        self.declare_parameter("web_host", "127.0.0.1")
        self.declare_parameter("web_port", 8765)
        self.declare_parameter("jpeg_quality", 75)
        self.declare_parameter("vision_task_service", "/perception/vision/task")
        self.declare_parameter(
            "enrollment_topic", "/perception/vision/enrollment_event"
        )
        self.declare_parameter(
            "object_topic", "/perception/vision/object_detections"
        )
        self.declare_parameter("object_only", False)
        self.declare_parameter("object_overlay_hold_sec", 2.5)
        self.declare_parameter("object_poll_hz", 0.0)
        self.declare_parameter("object_confidence", 0.5)
        self.declare_parameter("event_history_limit", 200)

        self._camera_topic = str(self.get_parameter("camera_topic").value)
        self._visual_topic = str(self.get_parameter("visual_topic").value)
        self._gesture_debug_topic = str(
            self.get_parameter("gesture_debug_topic").value
        )

        self._lock = threading.Lock()
        self._event: dict[str, Any] = {}
        self._gesture_debug: dict[str, Any] = {}
        self._control: dict[str, Any] = {}
        self._cmd_vel = (0.0, 0.0)
        self._started_monotonic = time.monotonic()
        self._camera_last_monotonic = 0.0
        self._visual_last_monotonic = 0.0
        self._gesture_last_monotonic = 0.0
        self._camera_times: deque[float] = deque(maxlen=60)
        self._render_times: deque[float] = deque(maxlen=60)
        self._camera_meta: dict[str, Any] = {}
        self._last_render_monotonic = 0.0
        self._debug_objects: list[dict[str, Any]] = []
        self._debug_objects_monotonic = 0.0
        self._object_future: Any = None
        self._object_sequence = 0
        self._object_task: dict[str, Any] = {
            "enabled": False,
            "pending": False,
            "success": None,
            "status": "waiting for topic",
            "latency_ms": 0.0,
            "error": "",
            "stream": {},
            "stop_reason": "",
        }
        self._enrollment: dict[str, Any] = {"status": "idle", "done": False}
        self._management_sequence = 0
        event_history_limit = max(
            20,
            min(1000, int(self.get_parameter("event_history_limit").value)),
        )
        self._published_event_history: deque[dict[str, Any]] = deque(
            maxlen=event_history_limit
        )
        self._active_published_events: dict[str, dict[str, Any]] = {}
        self._event_history_epoch = ""
        self._event_history_sequence = 0
        self._event_history_next_id = 0
        self._show_window = bool(self.get_parameter("show_window").value)
        self._publish_enabled = bool(
            self.get_parameter("publish_debug_image").value
        )
        self._scale = max(
            0.25, float(self.get_parameter("window_scale").value)
        )
        self._max_render_fps = max(
            0.0, float(self.get_parameter("max_render_fps").value)
        )
        self._render_scale = max(
            0.25,
            min(1.0, float(self.get_parameter("render_scale").value)),
        )
        self._stereo_enabled = bool(
            self.get_parameter("stereo_enabled").value
        )
        self._stereo_view = str(
            self.get_parameter("stereo_view").value
        ).lower()
        self._stereo_min_aspect_ratio = max(
            1.0,
            float(self.get_parameter("stereo_min_aspect_ratio").value),
        )
        self._overlay_stale_sec = max(
            0.1, float(self.get_parameter("overlay_stale_sec").value)
        )
        self._jpeg_quality = max(
            30, min(100, int(self.get_parameter("jpeg_quality").value))
        )
        self._object_poll_hz = max(
            0.0, float(self.get_parameter("object_poll_hz").value)
        )
        self._object_confidence = max(
            0.0,
            min(1.0, float(self.get_parameter("object_confidence").value)),
        )
        self._object_only = bool(
            self.get_parameter("object_only").value
        )
        self._object_overlay_hold = max(
            0.1,
            float(self.get_parameter("object_overlay_hold_sec").value),
        )
        if self._stereo_view not in ("left", "right"):
            self.get_logger().warning(
                "Invalid stereo_view; falling back to left"
            )
            self._stereo_view = "left"
        self._window_name = "MarsDog vision follow debug"
        self._debug_pub = self.create_publisher(
            Image,
            str(self.get_parameter("debug_image_topic").value),
            _BEST_EFFORT,
        )
        self.create_subscription(
            Image,
            self._camera_topic,
            self._on_image,
            _BEST_EFFORT,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("object_topic").value),
            self._on_objects,
            _BEST_EFFORT,
        )
        if not self._object_only:
            self.create_subscription(
                String,
                self._visual_topic,
                self._on_visual,
                _BEST_EFFORT,
            )
            self.create_subscription(
                String,
                self._gesture_debug_topic,
                self._on_gesture_debug,
                _BEST_EFFORT,
            )
            self.create_subscription(
                String,
                str(self.get_parameter("control_topic").value),
                self._on_control,
                10,
            )
            self.create_subscription(
                Twist,
                str(self.get_parameter("cmd_vel_topic").value),
                self._on_cmd_vel,
                10,
            )
            self.create_subscription(
                String,
                str(self.get_parameter("enrollment_topic").value),
                self._on_enrollment,
                10,
            )
        self._object_client = None
        self._vision_task_client = None
        if VisionTask is not None:
            self._vision_task_client = self.create_client(
                VisionTask,
                str(self.get_parameter("vision_task_service").value),
            )
        if self._object_poll_hz > 0.0 and VisionTask is not None:
            service_name = str(
                self.get_parameter("vision_task_service").value
            )
            self._object_client = self.create_client(VisionTask, service_name)
            self._object_task.update(
                enabled=True,
                status="waiting for service",
            )
            self.create_timer(
                1.0 / max(self._object_poll_hz, 0.1),
                self._poll_objects,
            )
        elif self._object_poll_hz > 0.0:
            self._object_task.update(
                status="VisionTask interface unavailable",
                error="Build and source marsdog_vision_interaction first",
            )

        self._web_server: VisionDebugWebServer | None = None
        if bool(self.get_parameter("web_enabled").value):
            host = str(self.get_parameter("web_host").value)
            port = int(self.get_parameter("web_port").value)
            task_handler = self._handle_web_task
            try:
                self._web_server = VisionDebugWebServer(
                    host,
                    port,
                    self._web_snapshot,
                    task_handler,
                    self._clear_published_event_history,
                )
                self._web_server.start()
                display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
                self.get_logger().info(
                    "Web dashboard: http://%s:%d"
                    % (display_host, self._web_server.bound_port)
                )
            except OSError as exc:
                self._web_server = None
                self.get_logger().error("Web dashboard failed to start: %s" % exc)
        if self._show_window and not os.environ.get("DISPLAY"):
            self.get_logger().warning(
                "DISPLAY is not set; disabling OpenCV window and keeping "
                "debug-image publishing"
            )
            self._show_window = False
        self.get_logger().info(
            "Viewer ready; render<=%.1f FPS scale=%.2f publish_debug_image=%s; "
            "q/ESC closes the window, annotated topic=%s"
            % (
                self._max_render_fps,
                self._render_scale,
                self._publish_enabled,
                self.get_parameter("debug_image_topic").value,
            )
        )

    def _on_visual(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(value, dict):
            with self._lock:
                if not self._record_visual_event_lifecycle(
                    value, received_at=time.time()
                ):
                    return
                self._event = value
                self._visual_last_monotonic = time.monotonic()

    @staticmethod
    def _published_event_evidence(value: dict[str, Any]) -> dict[str, Any]:
        active = value.get("active_target", {})
        if not isinstance(active, dict):
            active = {}
        hands = value.get("hands", [])
        if not isinstance(hands, list):
            hands = []
        hand_actions = []
        for hand in hands:
            if not isinstance(hand, dict):
                continue
            action = str(
                hand.get("hand_action_label") or hand.get("hand_action") or ""
            ).strip()
            if action and action not in hand_actions:
                hand_actions.append(action)
        return {
            "track_id": active.get("track_id"),
            "target_id": active.get("target_id"),
            "tracking_state": active.get("tracking_state"),
            "identity": active.get("identity"),
            "identity_state": active.get("identity_state"),
            "identity_confidence": active.get("identity_confidence"),
            "pose_action": active.get("pose_action"),
            "pose_action_label": active.get("pose_action_label"),
            "hand_actions": hand_actions,
            "pose_event_gate": (
                "open"
                if active.get("identity") in ALLOWED_FACE_IDENTITIES
                and active.get("identity_state") == "confirmed_known"
                and active.get("tracking_state") == "tracking"
                else "blocked"
            ),
        }

    def _new_published_event_record(
        self,
        *,
        phase: str,
        event_name: str,
        value: dict[str, Any],
        received_at: float,
        evidence: dict[str, Any],
        repeat_count: int,
        duration_ms: float,
        reason: str = "",
    ) -> dict[str, Any]:
        self._event_history_next_id += 1
        header = value.get("header", {})
        if not isinstance(header, dict):
            header = {}
        record = {
            "id": self._event_history_next_id,
            "phase": phase,
            "event": event_name,
            "received_at": received_at,
            "source_stamp": header.get("stamp"),
            "vision_epoch": str(value.get("vision_epoch", "")),
            "sequence": int(value.get("sequence", 0)),
            "snapshot_id": str(value.get("snapshot_id", "")),
            "repeat_count": repeat_count,
            "duration_ms": round(max(0.0, duration_ms), 1),
            "reason": reason,
            "evidence": copy.deepcopy(evidence),
        }
        self._published_event_history.append(record)
        return record

    def _exit_active_published_events(
        self,
        *,
        value: dict[str, Any],
        received_at: float,
        reason: str,
        keep: set[str] | None = None,
    ) -> None:
        retained = keep or set()
        for event_name in list(self._active_published_events):
            if event_name in retained:
                continue
            state = self._active_published_events.pop(event_name)
            self._new_published_event_record(
                phase="EXIT",
                event_name=event_name,
                value=value,
                received_at=received_at,
                evidence=state["evidence"],
                repeat_count=int(state["repeat_count"]),
                duration_ms=(received_at - float(state["entered_at"])) * 1000.0,
                reason=reason,
            )

    def _record_visual_event_lifecycle(
        self, value: dict[str, Any], *, received_at: float
    ) -> bool:
        try:
            schema_version = int(value.get("schema_version", 0))
            vision_epoch = str(value.get("vision_epoch", "")).strip()
            sequence = int(value.get("sequence", 0))
        except (TypeError, ValueError):
            return False
        if schema_version != 1 or not vision_epoch or sequence <= 0:
            return False
        if vision_epoch == self._event_history_epoch:
            if sequence <= self._event_history_sequence:
                return False
        elif self._event_history_epoch:
            self._exit_active_published_events(
                value=value,
                received_at=received_at,
                reason="vision_epoch_changed",
            )

        self._event_history_epoch = vision_epoch
        self._event_history_sequence = sequence
        raw_events = value.get("events", [])
        if not isinstance(raw_events, list):
            raw_events = []
        current_events: list[str] = []
        for item in raw_events:
            event_name = str(item).strip()
            if event_name and event_name not in current_events:
                current_events.append(event_name)
        current_set = set(current_events)
        self._exit_active_published_events(
            value=value,
            received_at=received_at,
            reason="event_cleared",
            keep=current_set,
        )

        evidence = self._published_event_evidence(value)
        for event_name in current_events:
            state = self._active_published_events.get(event_name)
            if state is None:
                self._new_published_event_record(
                    phase="ENTER",
                    event_name=event_name,
                    value=value,
                    received_at=received_at,
                    evidence=evidence,
                    repeat_count=1,
                    duration_ms=0.0,
                )
                self._active_published_events[event_name] = {
                    "entered_at": received_at,
                    "last_received_at": received_at,
                    "repeat_count": 1,
                    "evidence": copy.deepcopy(evidence),
                    "active_record": None,
                }
                continue
            state["last_received_at"] = received_at
            state["repeat_count"] = int(state["repeat_count"]) + 1
            state["evidence"] = copy.deepcopy(evidence)
            duration_ms = (received_at - float(state["entered_at"])) * 1000.0
            active_record = state.get("active_record")
            if active_record is None:
                active_record = self._new_published_event_record(
                    phase="ACTIVE",
                    event_name=event_name,
                    value=value,
                    received_at=received_at,
                    evidence=evidence,
                    repeat_count=int(state["repeat_count"]),
                    duration_ms=duration_ms,
                )
                state["active_record"] = active_record
            else:
                header = value.get("header", {})
                if not isinstance(header, dict):
                    header = {}
                active_record.update(
                    received_at=received_at,
                    source_stamp=header.get("stamp"),
                    sequence=sequence,
                    snapshot_id=str(value.get("snapshot_id", "")),
                    repeat_count=int(state["repeat_count"]),
                    duration_ms=round(max(0.0, duration_ms), 1),
                    evidence=copy.deepcopy(evidence),
                )
        return True

    def _clear_published_event_history(self) -> dict[str, Any]:
        with self._lock:
            cleared_records = len(self._published_event_history)
            cleared_active = len(self._active_published_events)
            self._published_event_history.clear()
            self._active_published_events.clear()
        return {
            "ok": True,
            "cleared_records": cleared_records,
            "cleared_active_events": cleared_active,
        }

    def _on_enrollment(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(value, dict):
            with self._lock:
                self._enrollment = value

    def _handle_web_task(
        self, task_type: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        allowed = {
            "start_face_enrollment", "cancel_face_enrollment",
            "recognize_face", "list_faces", "list_face_records", "delete_face",
            "detect_objects", "set_object_detection",
        }
        if task_type not in allowed:
            return {"ok": False, "error": "web task is not allowed"}
        if (
            task_type == "set_object_detection"
            and str(params.get("session_id", "")) != _WEB_OBJECT_SESSION_ID
        ):
            return {
                "ok": False,
                "error": "web object session_id must be vision-debug-web",
            }
        client = self._vision_task_client
        if client is None or VisionTask is None:
            return {"ok": False, "error": "VisionTask interface unavailable"}
        if not client.service_is_ready():
            return {"ok": False, "error": "vision task service unavailable"}
        with self._lock:
            self._management_sequence += 1
            sequence = self._management_sequence
        request = VisionTask.Request()
        request.task_id = f"vision-web-{sequence}"
        request.task_type = task_type
        request.params_json = json.dumps(params, ensure_ascii=False)
        future = client.call_async(request)
        completed = threading.Event()
        future.add_done_callback(lambda _: completed.set())
        timeout_sec = 45.0 if task_type == "detect_objects" else 8.0
        if not completed.wait(timeout=timeout_sec):
            return {"ok": False, "error": "vision task timed out"}
        try:
            response = future.result()
            result = json.loads(response.result_json or "{}")
            if not isinstance(result, dict):
                result = {}
            result.setdefault("ok", bool(response.success))
            result.setdefault("latency_ms", float(response.latency_ms))
            if response.error_message:
                result.setdefault("error", str(response.error_message))
            return result
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _on_control(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(value, dict):
            with self._lock:
                self._control = value

    def _on_objects(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if not isinstance(value, dict):
            return
        objects = value.get("objects", [])
        if not isinstance(objects, list):
            objects = []
        status = str(value.get("status", "error"))
        success = status == "ok"
        stream = value.get("stream", {})
        if not isinstance(stream, dict):
            stream = {}
        with self._lock:
            self._debug_objects = (
                [dict(item) for item in objects if isinstance(item, dict)]
                if success else []
            )
            self._debug_objects_monotonic = time.monotonic()
            self._object_task.update(
                enabled=bool(stream.get("active", success)),
                pending=False,
                success=success,
                status=f"topic {status}",
                latency_ms=float(
                    value.get("inference_latency_ms", 0.0) or 0.0
                ),
                error=str(value.get("error", "")),
                stream=dict(stream),
                stop_reason=str(value.get("stop_reason", "")),
            )

    def _on_gesture_debug(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(value, dict):
            with self._lock:
                self._gesture_debug = value
                self._gesture_last_monotonic = time.monotonic()

    def _on_cmd_vel(self, message: Twist) -> None:
        with self._lock:
            self._cmd_vel = (
                float(message.linear.x), float(message.angular.z)
            )

    def _on_image(self, message: Image) -> None:
        received_at = time.monotonic()
        source_width = int(message.width)
        source_height = int(message.height)
        with self._lock:
            self._camera_last_monotonic = received_at
            self._camera_times.append(received_at)
            self._camera_meta.update({
                "topic": self._camera_topic,
                "source_width": source_width,
                "source_height": source_height,
                "encoding": str(message.encoding),
                "frame_id": str(message.header.frame_id),
            })
            if not self._render_is_due(
                received_at,
                self._last_render_monotonic,
                self._max_render_fps,
            ):
                return
            self._last_render_monotonic = received_at

        frame = self._decode_image(message)
        if frame is None:
            return
        source_height, source_width = frame.shape[:2]
        frame, stereo_split = select_camera_view(
            frame,
            stereo_enabled=self._stereo_enabled,
            view=self._stereo_view,
            min_aspect_ratio=self._stereo_min_aspect_ratio,
        )
        if self._render_scale != 1.0:
            frame = cv2.resize(
                frame,
                None,
                fx=self._render_scale,
                fy=self._render_scale,
                interpolation=cv2.INTER_AREA,
            )
        with self._lock:
            self._camera_meta.update({
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            })
            visual_age = (
                received_at - self._visual_last_monotonic
                if self._visual_last_monotonic > 0.0
                else float("inf")
            )
            gesture_age = (
                received_at - self._gesture_last_monotonic
                if self._gesture_last_monotonic > 0.0
                else float("inf")
            )
            event = (
                copy.deepcopy(self._event)
                if visual_age <= self._overlay_stale_sec
                else {}
            )
            object_age = received_at - self._debug_objects_monotonic
            if object_age <= self._object_overlay_hold_sec():
                event["tracked_objects"] = copy.deepcopy(self._debug_objects)
            if self._object_only:
                event = {
                    "schema_version": 1,
                    "tracked_objects": copy.deepcopy(
                        event.get("tracked_objects", [])
                    ),
                }
                control = {"mode": "object_only"}
                cmd_vel = None
            else:
                control = copy.deepcopy(self._control)
                if gesture_age <= self._overlay_stale_sec:
                    control["_gesture_debug"] = copy.deepcopy(
                        self._gesture_debug
                    )
                cmd_vel = self._cmd_vel
        control["_visual_age_ms"] = (
            visual_age * 1000.0 if visual_age != float("inf") else -1.0
        )
        control["_gesture_age_ms"] = (
            gesture_age * 1000.0 if gesture_age != float("inf") else -1.0
        )
        control["_input_layout"] = (
            f"{source_width}x{source_height}->"
            f"{self._stereo_view}" if stereo_split else "mono/full"
        )
        rendered = draw_visual_debug(
            frame, event, control=control, cmd_vel=cmd_vel
        )
        if self._publish_enabled:
            output = Image()
            output.header = message.header
            suffix = self._stereo_view if stereo_split else "full"
            output.header.frame_id = f"{message.header.frame_id}_{suffix}"
            output.height, output.width = rendered.shape[:2]
            output.encoding = "bgr8"
            output.is_bigendian = False
            output.step = output.width * 3
            output.data = rendered.tobytes()
            self._debug_pub.publish(output)
        if self._web_server is not None:
            ok, encoded = cv2.imencode(
                ".jpg",
                rendered,
                [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
            )
            if ok:
                self._web_server.update_jpeg(encoded.tobytes())
        if self._show_window:
            shown = rendered
            if self._scale != 1.0:
                shown = cv2.resize(
                    rendered, None, fx=self._scale, fy=self._scale
                )
            cv2.imshow(self._window_name, shown)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                self._show_window = False
                cv2.destroyWindow(self._window_name)
        with self._lock:
            self._render_times.append(time.monotonic())

    @staticmethod
    def _render_is_due(
        now: float,
        previous: float,
        max_render_fps: float,
    ) -> bool:
        if max_render_fps <= 0.0 or previous <= 0.0:
            return True
        return now - previous >= 1.0 / max_render_fps

    def _poll_objects(self) -> None:
        client = self._object_client
        if client is None or VisionTask is None:
            return
        if self._object_future is not None:
            return
        if not client.service_is_ready():
            with self._lock:
                self._object_task.update(
                    pending=False,
                    success=False,
                    status="service unavailable",
                    error="等待 /perception/vision/task",
                )
            return

        self._object_sequence += 1
        request = VisionTask.Request()
        request.task_id = f"vision-debug-{self._object_sequence}"
        request.task_type = "detect_objects"
        request.params_json = json.dumps(
            {"confidence": self._object_confidence},
            separators=(",", ":"),
        )
        with self._lock:
            self._object_task.update(
                pending=True,
                status="running",
                error="",
            )
        self._object_future = client.call_async(request)
        self._object_future.add_done_callback(self._on_object_result)

    def _on_object_result(self, future: Any) -> None:
        try:
            response = future.result()
            result = json.loads(response.result_json or "{}")
            objects = result.get("objects", []) if isinstance(result, dict) else []
            if not isinstance(objects, list):
                objects = []
            success = bool(response.success)
            error = str(response.error_message or "")
            with self._lock:
                self._object_task.update(
                    pending=False,
                    success=success,
                    status="ok" if success else "failed",
                    latency_ms=float(response.latency_ms),
                    error=error,
                )
                self._debug_objects = (
                    [dict(item) for item in objects if isinstance(item, dict)]
                    if success
                    else []
                )
                self._debug_objects_monotonic = time.monotonic()
        except Exception as exc:
            with self._lock:
                self._object_task.update(
                    pending=False,
                    success=False,
                    status="request error",
                    error=str(exc),
                )
                self._debug_objects = []
                self._debug_objects_monotonic = time.monotonic()
        finally:
            self._object_future = None

    def _object_overlay_hold_sec(self) -> float:
        return self._object_overlay_hold

    def _web_snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        wall_now = time.time()
        with self._lock:
            event = copy.deepcopy(self._event)
            object_age = now - self._debug_objects_monotonic
            if object_age <= self._object_overlay_hold_sec():
                event["tracked_objects"] = copy.deepcopy(self._debug_objects)
            if self._object_only:
                event = {
                    "schema_version": 1,
                    "tracked_objects": copy.deepcopy(
                        event.get("tracked_objects", [])
                    ),
                }
            camera_times = list(self._camera_times)
            render_times = list(self._render_times)
            camera_meta = dict(self._camera_meta)
            camera_age = (
                (now - self._camera_last_monotonic) * 1000.0
                if self._camera_last_monotonic > 0.0
                else None
            )
            visual_age = (
                (now - self._visual_last_monotonic) * 1000.0
                if self._visual_last_monotonic > 0.0
                else None
            )
            gesture_age = (
                (now - self._gesture_last_monotonic) * 1000.0
                if self._gesture_last_monotonic > 0.0
                else None
            )
            gesture_debug = copy.deepcopy(self._gesture_debug)
            control = copy.deepcopy(self._control)
            cmd_vel = self._cmd_vel
            object_task = dict(self._object_task)
            enrollment = copy.deepcopy(self._enrollment)
            published_event_history = copy.deepcopy(
                list(self._published_event_history)
            )
            active_published_events = []
            for event_name, state in self._active_published_events.items():
                active_published_events.append({
                    "event": event_name,
                    "entered_at": state["entered_at"],
                    "last_received_at": state["last_received_at"],
                    "repeat_count": state["repeat_count"],
                    "duration_ms": round(
                        max(0.0, wall_now - float(state["entered_at"]))
                        * 1000.0,
                        1,
                    ),
                    "evidence": copy.deepcopy(state["evidence"]),
                })
        fps = 0.0
        if len(camera_times) >= 2 and camera_times[-1] > camera_times[0]:
            fps = (len(camera_times) - 1) / (camera_times[-1] - camera_times[0])
        render_fps = 0.0
        if len(render_times) >= 2 and render_times[-1] > render_times[0]:
            render_fps = (
                (len(render_times) - 1)
                / (render_times[-1] - render_times[0])
            )
        camera_meta.update(
            age_ms=camera_age,
            fps=round(fps, 2),
            render_fps=round(render_fps, 2),
            render_limit_fps=self._max_render_fps,
        )
        return {
            "ok": True,
            "object_only": self._object_only,
            "uptime_sec": round(now - self._started_monotonic, 1),
            "camera": camera_meta,
            "visual": {
                "topic": self._visual_topic,
                "age_ms": visual_age,
                "schema_version": event.get("schema_version"),
            },
            "visual_event": event,
            "published_event_history": published_event_history,
            "active_published_events": active_published_events,
            "gesture": {
                "topic": self._gesture_debug_topic,
                "age_ms": gesture_age,
            },
            "gesture_debug": gesture_debug,
            "object_task": object_task,
            "enrollment": enrollment,
            "control": control,
            "cmd_vel": {
                "linear_x": cmd_vel[0],
                "angular_z": cmd_vel[1],
            },
        }

    @staticmethod
    def _decode_image(message: Image) -> np.ndarray | None:
        try:
            encoding = str(message.encoding).lower()
            channels = 1 if encoding in ("mono8", "8uc1") else 3
            row_bytes = int(message.step) or int(message.width) * channels
            raw = np.frombuffer(message.data, dtype=np.uint8).reshape(
                int(message.height), row_bytes
            )
            if channels == 1:
                return cv2.cvtColor(
                    raw[:, : int(message.width)], cv2.COLOR_GRAY2BGR
                )
            frame = raw[:, : int(message.width) * 3].reshape(
                int(message.height), int(message.width), 3
            )
            if encoding == "rgb8":
                return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame.copy()
        except (TypeError, ValueError):
            return None

    def destroy_node(self) -> None:
        if self._web_server is not None:
            self._web_server.stop()
            self._web_server = None
        if self._show_window:
            cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionDebugViewerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
