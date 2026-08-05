"""Independent ROS2 node for the MarsDog visual interaction pipeline."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from marsdog_vision_interaction.srv import VisionTask
except ImportError:
    VisionTask = None  # type: ignore[assignment]

from marsdog_vision_interaction.core.face_enrollment_manager import (
    FaceEnrollmentManager,
    set_storage_root,
)
from marsdog_vision_interaction.fusion.stereo_fusion import get_target_manager
from marsdog_vision_interaction.messages.visual_event import (
    normalize_visual_event,
)
from marsdog_vision_interaction.messages.visual_event_types import (
    face_identity_to_vision_event,
    object_to_vision_event,
    pose_action_to_vision_event,
)
from marsdog_vision_interaction.providers.base import BaseProvider
from marsdog_vision_interaction.utils.config_loader import load_config
from marsdog_vision_interaction.utils.logging_utils import (
    get_logger,
    setup_logging,
)


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


class VisionInteractionNode(Node):
    """Own camera input, visual models, face storage and visual ROS APIs."""

    def __init__(self) -> None:
        super().__init__("vision_interaction")
        self.declare_parameter("config_path", "config/vision.yaml")
        self.declare_parameter("log_level", "INFO")
        self.declare_parameter("log_dir", "log")
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

        storage_root = self._config.get("storage", {}).get("root", "data")
        set_storage_root(storage_root)
        self._enrollment = FaceEnrollmentManager()
        self._target_manager = get_target_manager()
        self._providers: dict[str, BaseProvider | None] = {}
        self._latest_frame: np.ndarray | None = None
        self._init_providers()
        self._wire_face_enrollment()
        self._sync_face_registry()

        topics = self._config.get("topics", {})
        self._visual_topic = str(
            topics.get("visual_event", "/perception/visual_event")
        )
        self._camera_topic = str(
            topics.get("camera_image", "/camera/camera/color/image_raw")
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
        self._enrollment_pub = self.create_publisher(
            String, enrollment_topic, _EVENT_QOS
        )
        self._camera_sub = self.create_subscription(
            Image, self._camera_topic, self._on_camera, _CAMERA_QOS
        )
        rate = float(topics.get("observation_rate_hz", 10.0))
        self._timer = self.create_timer(1.0 / max(rate, 0.1), self._publish_visual)

        service_name = str(
            topics.get("vision_task", "/perception/vision/task")
        )
        self._service = (
            self.create_service(VisionTask, service_name, self._handle_task)
            if VisionTask is not None else None
        )
        logger.info(
            "Vision node ready: camera=%s visual=%s service=%s",
            self._camera_topic,
            self._visual_topic,
            service_name if self._service is not None else "unavailable",
        )

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
                from marsdog_vision_interaction.providers.mock_vision import (
                    MockVisionProvider,
                )
                vision = MockVisionProvider(vision_config.get("config", {}))
                vision.start()
            self._providers["vision"] = vision

        object_config = providers.get("object", {})
        if object_config.get("enabled", True):
            from marsdog_vision_interaction.providers.object_detector import (
                ObjectDetectorProvider,
            )
            config = dict(object_config.get("config", {}))
            if object_config.get("type") == "mock":
                config["object_rknn_model"] = ""
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
            encoding = str(getattr(message, "encoding", "bgr8")).lower()
            if encoding in ("mono8", "8uc1"):
                frame = np.frombuffer(message.data, np.uint8).reshape(
                    message.height, message.width
                )
                import cv2
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                frame = np.frombuffer(message.data, np.uint8).reshape(
                    message.height, message.width, 3
                )
                if encoding == "rgb8":
                    import cv2
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            self._latest_frame = frame
            vision = self._providers.get("vision")
            if vision is not None and hasattr(vision, "process_frame"):
                vision.process_frame(frame)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.debug("Camera frame rejected: %s", exc)

    def _publish_visual(self) -> None:
        if self._enrollment.face_session is not None:
            self._process_enrollment_frame()

        vision = self._providers.get("vision")
        try:
            raw = (
                vision.get_observation()  # type: ignore[attr-defined]
                if vision is not None and vision.is_available()
                else {}
            )
        except Exception as exc:
            logger.error("Visual observation failed: %s", exc)
            raw = {}
        self._target_manager.update_vision(
            raw.get("humans", []),
            raw.get("faces", []),
        )
        event = normalize_visual_event(raw)
        event["active_target"] = self._target_manager.get_active_dict()
        event["events"] = self._derive_events(event)
        message = String()
        message.data = json.dumps(event, ensure_ascii=False)
        self._visual_pub.publish(message)

    @staticmethod
    def _derive_events(observation: dict[str, Any]) -> list[str]:
        events: list[str] = []
        active = observation.get("active_target", {})
        identity = str(active.get("identity", "unknown"))
        identity_known = identity not in ("", "unknown")
        if observation.get("faces"):
            events.append(face_identity_to_vision_event(identity))
        action = str(active.get("pose_action", ""))
        action_event = pose_action_to_vision_event(action, identity_known)
        if action_event:
            events.append(action_event)
        for hand in observation.get("hands", []):
            hand_event = pose_action_to_vision_event(
                str(hand.get("hand_action", "")), identity_known
            )
            if hand_event and hand_event not in events:
                events.append(hand_event)
        for item in observation.get("tracked_objects", []):
            object_event = object_to_vision_event(str(item.get("label", "")))
            if object_event and object_event not in events:
                events.append(object_event)
        return events

    def _process_enrollment_frame(self) -> None:
        frame = self._latest_frame
        if frame is None:
            return
        if frame.shape[1] > frame.shape[0]:
            frame = frame[:, : frame.shape[1] // 2]
        result = self._enrollment.process_face_frame(frame)
        if result.get("done"):
            self._sync_face_registry()
        message = String()
        message.data = json.dumps(result, ensure_ascii=False)
        self._enrollment_pub.publish(message)

    def _handle_task(self, request: Any, response: Any) -> Any:
        response.task_id = request.task_id
        response.task_type = request.task_type
        response.success = False
        response.result_json = ""
        response.error_message = ""
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
        return response

    def _run_task(self, task_type: str, params: dict[str, Any]) -> dict[str, Any]:
        vision = self._providers.get("vision")
        if task_type == "check_person":
            if vision is None or not hasattr(vision, "check_person"):
                return {"ok": False, "error": "vision unavailable"}
            return {"ok": True, **vision.check_person()}  # type: ignore[attr-defined]
        if task_type == "detect_objects":
            detector = self._providers.get("object")
            if detector is None:
                return {"ok": False, "error": "object detector unavailable"}
            frame = self._latest_frame
            if frame is not None and frame.shape[1] > frame.shape[0]:
                frame = frame[:, : frame.shape[1] // 2]
            return {
                "ok": True,
                "objects": detector.detect_objects(frame, [params]),  # type: ignore[attr-defined]
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
            return self._enrollment.start_face(
                str(params.get("name", "")),
                int(params.get("required_shots", 3)),
            )
        if task_type == "cancel_face_enrollment":
            return self._enrollment.cancel_face()
        if task_type == "upload_face":
            payload = base64.b64decode(
                str(params.get("image_base64", "")), validate=True
            )
            result = self._enrollment.enroll_face_from_image(
                str(params.get("name", "")), payload
            )
            if result.get("ok"):
                self._sync_face_registry()
            return result
        if task_type == "list_faces":
            return {
                "ok": True,
                "faces": self._enrollment.list_enrolled_faces(),
            }
        if task_type == "delete_face":
            return self._enrollment.delete_face(str(params.get("name", "")))
        return {"ok": False, "error": f"unsupported task_type: {task_type}"}

    def _crop_largest_face(self) -> np.ndarray | None:
        frame = self._latest_frame
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

    def _wire_face_enrollment(self) -> None:
        vision = self._providers.get("vision")
        detector = getattr(vision, "_face_detector", None)
        if detector is not None:
            self._enrollment.set_face_detector(detector)

    def _sync_face_registry(self) -> None:
        vision = self._providers.get("vision")
        if vision is not None and hasattr(vision, "sync_enrolled_to_throttle"):
            vision.sync_enrolled_to_throttle()  # type: ignore[attr-defined]
        recognizer = self._providers.get("face_recognition")
        if recognizer is None or not hasattr(recognizer, "enroll"):
            return
        try:
            import cv2
            for name in self._enrollment.list_enrolled_faces():
                for path in self._enrollment.get_face_paths(name):
                    image = cv2.imread(path)
                    if image is None:
                        continue
                    result = recognizer.enroll(image, name)  # type: ignore[attr-defined]
                    if result.get("success"):
                        break
        except Exception as exc:
            logger.warning("Face registry sync failed: %s", exc)

    def destroy_node(self) -> None:
        for provider in self._providers.values():
            if provider is not None:
                provider.stop()
        super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = VisionInteractionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
