"""Camera driver node — sole owner of /dev/video0.

Publishes raw image streams for downstream subscribers (vision, navigation, app).
No AI processing, no image duplication, non-blocking publish loop.

Topics:
  /camera/image_raw   (sensor_msgs/Image)   BEST_EFFORT, depth=1
  /camera/camera_info (sensor_msgs/CameraInfo)

Parameters:
  device:             Video device path (default /dev/video0)
  width:              Capture width (default 640)
  height:             Capture height (default 480)
  fps:                Target FPS (default 30)
  frame_id:           TF frame (default camera_link)
  image_topic:        Image output topic (default /camera/image_raw)
  camera_info_topic:  CameraInfo output topic (default /camera/camera_info)
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from marsdog_vision_interaction.utils.logging_utils import get_logger, setup_logging
from marsdog_vision_interaction.utils.time_utils import now_stamp

logger = get_logger(__name__, module="camera")

# ── QoS: low latency, drop ok, no backlog ──────────────────────
_CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


def _make_camera_info(frame_id: str, width: int, height: int) -> CameraInfo:
    """Build a placeholder CameraInfo message.

    Real intrinsics should come from calibration. This provides
    a valid header so subscribers can match image ↔ info.
    """
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.header.stamp = rclpy.time.Time().to_msg()
    msg.width = width
    msg.height = height
    # Identity rectification, no distortion
    msg.distortion_model = "plumb_bob"
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.k = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


class CameraDriverNode(Node):
    """ROS2 camera driver — single owner of the video device.

    Opens the camera once on startup. Reads frames in a timer-driven
    loop and publishes to /camera/image_raw. Publishes /camera/camera_info
    once on startup.

    Design:
      - One VideoCapture instance (no re-open)
      - No frame buffer queue (read → publish, discard)
      - Timer at configured FPS drives the capture loop
      - Downstream nodes subscribe to /camera/image_raw
    """

    def __init__(self) -> None:
        super().__init__("camera_driver")

        # Init unified logging
        setup_logging(log_dir="log", level="INFO", node="camera_driver")

        # ── Parameters ─────────────────────────────────────────
        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("image_topic", "/camera/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera_info")

        self._device = (
            self.get_parameter("device").get_parameter_value().string_value
        )
        self._width = (
            self.get_parameter("width").get_parameter_value().integer_value
        )
        self._height = (
            self.get_parameter("height").get_parameter_value().integer_value
        )
        self._fps = float(
            self.get_parameter("fps").get_parameter_value().integer_value
        )
        self._frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )
        self._image_topic = (
            self.get_parameter("image_topic").get_parameter_value().string_value
        )
        self._camera_info_topic = (
            self.get_parameter("camera_info_topic")
            .get_parameter_value().string_value
        )

        logger.info(
            "camera_init",
            device=self._device, width=self._width, height=self._height,
            fps=int(self._fps), frame_id=self._frame_id,
            image_topic=self._image_topic,
            camera_info_topic=self._camera_info_topic,
        )

        # ── Open camera (once) ─────────────────────────────────
        self._cap: Optional[cv2.VideoCapture] = None
        self._open_camera()

        # ── Publishers ─────────────────────────────────────────
        self._image_pub = self.create_publisher(
            Image, self._image_topic, qos_profile=_CAMERA_QOS
        )
        self._info_pub = self.create_publisher(
            CameraInfo, self._camera_info_topic, qos_profile=_CAMERA_QOS
        )

        # Publish camera_info once at startup
        self._publish_camera_info()

        # ── Timer: drives the capture loop ─────────────────────
        period = 1.0 / max(self._fps, 1.0)
        self._timer = self.create_timer(period, self._capture_and_publish)

        logger.info(
            "CameraDriverNode ready — publishing %s at %.1f Hz",
            self._image_topic,
            self._fps,
        )

    # ── Camera lifecycle ───────────────────────────────────────

    def _open_camera(self) -> None:
        """Open the camera device with V4L2 backend.

        Tries multiple FourCC codes in order of preference.
        """
        # Use V4L2 CAP_V4L2 for direct kernel access (lower latency)
        self._cap = cv2.VideoCapture(self._device, cv2.CAP_V4L2)

        if not self._cap.isOpened():
            # Fallback: let OpenCV auto-detect backend
            logger.warning("V4L2 open failed, trying auto-detect")
            self._cap = cv2.VideoCapture(self._device)

        if not self._cap.isOpened():
            self.get_logger().error(f"Cannot open camera: {self._device}")
            return

        # Configure capture format
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)
        # Disable internal frame buffer
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Read actual values (driver may adjust)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        fourcc_str = "".join(chr((actual_fourcc >> (8 * i)) & 0xFF) for i in range(4))

        logger.info(
            "Camera opened: %s %dx%d @ %.1f fps codec=%s",
            self._device, actual_w, actual_h, actual_fps, fourcc_str,
        )
        self._width = actual_w
        self._height = actual_h
        # self._fps stays as user-configured — hardware FPS is not writable.
        # The timer uses self._fps to throttle publish rate; actual camera
        # capture runs at hardware speed and we just drop excess frames.

    # ── Publishing ─────────────────────────────────────────────

    def _publish_camera_info(self) -> None:
        """Publish CameraInfo once at startup."""
        msg = _make_camera_info(self._frame_id, self._width, self._height)
        msg.header.stamp = self.get_clock().now().to_msg()
        self._info_pub.publish(msg)
        logger.info("CameraInfo published: %dx%d", self._width, self._height)

    def _capture_and_publish(self) -> None:
        """Read one frame and publish. Non-blocking.

        Called by the ROS2 timer at the configured FPS.
        Skips the frame if camera read fails.
        """
        if self._cap is None or not self._cap.isOpened():
            return

        # Read frame (blocks until next frame is available)
        ret, frame = self._cap.read()

        if not ret or frame is None:
            # Drop frame — camera may have momentary glitch
            return

        # Build Image message
        stamp = now_stamp()
        msg = Image()
        msg.header.stamp = rclpy.time.Time(seconds=int(stamp),
                                           nanoseconds=int((stamp % 1) * 1e9)).to_msg()
        msg.header.frame_id = self._frame_id
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = "bgr8"  # OpenCV default is BGR
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3  # 3 bytes per pixel for bgr8
        msg.data = frame.tobytes()

        self._image_pub.publish(msg)

    # ── Shutdown ───────────────────────────────────────────────

    def destroy_node(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraDriverNode()

    if not node._cap or not node._cap.isOpened():
        node.get_logger().error("Camera failed to open — exiting")
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        return

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except rclpy._rclpy_pybind11.RCLError:
            pass


if __name__ == "__main__":
    main()
