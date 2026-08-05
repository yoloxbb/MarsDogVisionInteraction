"""Live annotated viewer for camera, visual target and AGV commands."""

from __future__ import annotations

import json
import os
import threading
from typing import Any

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from marsdog_vision_interaction.utils.visual_debug import draw_visual_debug
from marsdog_vision_interaction.utils.stereo_view import select_camera_view


_BEST_EFFORT = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class VisionDebugViewerNode(Node):
    def __init__(self) -> None:
        super().__init__("vision_debug_viewer")
        self.declare_parameter(
            "camera_topic", "/camera/camera/color/image_raw"
        )
        self.declare_parameter("visual_topic", "/perception/visual_event")
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
        self.declare_parameter("stereo_enabled", True)
        self.declare_parameter("stereo_view", "left")
        self.declare_parameter("stereo_min_aspect_ratio", 2.2)

        self._lock = threading.Lock()
        self._event: dict[str, Any] = {}
        self._control: dict[str, Any] = {}
        self._cmd_vel = (0.0, 0.0)
        self._show_window = bool(self.get_parameter("show_window").value)
        self._publish_enabled = bool(
            self.get_parameter("publish_debug_image").value
        )
        self._scale = max(
            0.25, float(self.get_parameter("window_scale").value)
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
            str(self.get_parameter("camera_topic").value),
            self._on_image,
            _BEST_EFFORT,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("visual_topic").value),
            self._on_visual,
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
        if self._show_window and not os.environ.get("DISPLAY"):
            self.get_logger().warning(
                "DISPLAY is not set; disabling OpenCV window and keeping "
                "debug-image publishing"
            )
            self._show_window = False
        self.get_logger().info(
            "Viewer ready; q/ESC closes the window, annotated topic=%s"
            % self.get_parameter("debug_image_topic").value
        )

    def _on_visual(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(value, dict):
            with self._lock:
                self._event = value

    def _on_control(self, message: String) -> None:
        try:
            value = json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return
        if isinstance(value, dict):
            with self._lock:
                self._control = value

    def _on_cmd_vel(self, message: Twist) -> None:
        with self._lock:
            self._cmd_vel = (
                float(message.linear.x), float(message.angular.z)
            )

    def _on_image(self, message: Image) -> None:
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
        with self._lock:
            event = dict(self._event)
            control = dict(self._control)
            cmd_vel = self._cmd_vel
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
