"""Ultralytics YOLOE object detection provider.

Ultralytics owns image preprocessing, the exported-model backend, NMS, and
``Results`` decoding.  The configured model is currently an RKNN export, so
Ultralytics still executes it on the Rockchip NPU through rknn-toolkit-lite2.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from marsdog_vision_interaction.core.object_detection_session import (
    ObjectDetectionSessionManager,
)
from marsdog_vision_interaction.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class ObjectDetectorProvider(BaseProvider):
    """Lazy-loaded Ultralytics YOLOE detector with a stable ROS result shape."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        # ``object_rknn_model`` remains accepted for existing local configs.
        self._model_path = str(
            config.get("object_model", config.get("object_rknn_model", ""))
        )
        self._det_threshold = float(config.get("det_threshold", 0.5))
        self._nms_threshold = float(config.get("nms_threshold", 0.45))
        self._max_detections = max(
            1, int(config.get("max_detections", 100))
        )
        self._image_size = max(32, int(config.get("image_size", 640)))
        self._mock_mode = bool(config.get("mock_mode", False))

        self._model: Any = None
        self._loaded = False
        self.last_error = ""

    def start(self) -> None:
        self.last_error = ""
        if self._mock_mode:
            self.available = True
            logger.info("ObjectDetectorProvider started (explicit mock)")
            return
        if not self._model_path:
            self.available = False
            self.last_error = "object model path is not configured"
            logger.error(self.last_error)
            return
        self.available = True
        logger.info(
            "ObjectDetectorProvider — Ultralytics lazy-load, model=%s",
            self._model_path,
        )

    def stop(self) -> None:
        self._release_runtime()
        self._model = None
        self._loaded = False
        self.available = False
        logger.info("ObjectDetectorProvider stopped")

    def detect_objects(
        self,
        frame: Any = None,
        params: dict[str, Any] | list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect objects and return normalized boxes for the ROS contract."""
        self.last_error = ""
        try:
            target_labels = ObjectDetectionSessionManager.normalize_target_labels(
                params.get("target_labels") if isinstance(params, dict) else None
            )
        except ValueError as exc:
            self.last_error = str(exc)
            return []
        if self._mock_mode:
            return self._filter_target_labels(
                self._mock_objects(), target_labels
            )

        threshold = self._confidence_from_params(params)
        if not self.available:
            self.last_error = self.last_error or "object detector unavailable"
            return []
        if frame is None:
            self.last_error = "camera frame unavailable or stale"
            return []
        if not self._loaded and not self._load_model():
            return []

        try:
            return self._filter_target_labels(
                self._run_inference(frame, threshold),
                target_labels,
            )
        except Exception as exc:
            logger.error(
                "Ultralytics object detection failed: %s",
                exc,
                exc_info=True,
            )
            self.available = False
            self.last_error = f"object inference failed: {exc}"
            return []

    def _confidence_from_params(
        self,
        params: dict[str, Any] | list[dict[str, Any]] | None,
    ) -> float:
        threshold = self._det_threshold
        if isinstance(params, dict):
            try:
                threshold = float(params.get("confidence", threshold))
            except (ValueError, TypeError):
                pass
        elif params:
            for item in params:
                if item.get("key") == "confidence" or "confidence" in item:
                    try:
                        threshold = float(
                            item.get("value", item.get("confidence"))
                        )
                    except (ValueError, TypeError):
                        pass
        return min(1.0, max(0.0, threshold))

    def _load_model(self) -> bool:
        """Create the YOLOE wrapper; its exported backend loads on predict."""
        try:
            from ultralytics import YOLOE

            started = time.perf_counter()
            self._model = YOLOE(self._model_path)
            self._loaded = True
            logger.info(
                "Ultralytics YOLOE initialized: %s (%.0fms)",
                self._model_path,
                (time.perf_counter() - started) * 1000.0,
            )
            return True
        except Exception as exc:
            logger.error("Ultralytics YOLOE load failed: %s", exc)
            self._loaded = True
            self.available = False
            self.last_error = f"object model unavailable: {exc}"
            return False

    def _run_inference(
        self,
        frame: np.ndarray,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Run Ultralytics preprocessing, inference, NMS, and decoding."""
        started = time.perf_counter()
        results = self._model.predict(
            source=frame,
            conf=threshold,
            iou=self._nms_threshold,
            max_det=self._max_detections,
            imgsz=self._image_size,
            save=False,
            verbose=False,
        )
        objects = self._results_to_objects(results)
        logger.info(
            "Ultralytics YOLOE: %d objects (%.1fms, conf=%.2f, iou=%.2f)",
            len(objects),
            (time.perf_counter() - started) * 1000.0,
            threshold,
            self._nms_threshold,
        )
        return objects

    def _results_to_objects(self, results: Any) -> list[dict[str, Any]]:
        """Convert the first Ultralytics ``Results.boxes`` to ROS fields."""
        if not results:
            return []
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        xyxyn = self._as_numpy(getattr(boxes, "xyxyn", [])).reshape(-1, 4)
        confidences = self._as_numpy(
            getattr(boxes, "conf", [])
        ).reshape(-1)
        class_ids = self._as_numpy(getattr(boxes, "cls", [])).reshape(-1)
        names = getattr(result, "names", {})

        count = min(
            len(xyxyn),
            len(confidences),
            len(class_ids),
            self._max_detections,
        )
        objects: list[dict[str, Any]] = []
        for index in range(count):
            x1, y1, x2, y2 = np.clip(xyxyn[index], 0.0, 1.0)
            width = max(0.0, float(x2 - x1))
            height = max(0.0, float(y2 - y1))
            if width <= 0.0 or height <= 0.0:
                continue
            class_id = int(class_ids[index])
            if isinstance(names, dict):
                label = str(names.get(class_id, f"class_{class_id}"))
            elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
                label = str(names[class_id])
            else:
                label = f"class_{class_id}"
            objects.append({
                "label": label,
                "x": round(float(x1), 4),
                "y": round(float(y1), 4),
                "w": round(width, 4),
                "h": round(height, 4),
                "confidence": round(float(confidences[index]), 4),
                "center_x": round(float((x1 + x2) / 2.0), 4),
                "center_y": round(float((y1 + y2) / 2.0), 4),
            })
        return objects

    @staticmethod
    def _as_numpy(value: Any) -> np.ndarray:
        detach = getattr(value, "detach", None)
        if callable(detach):
            value = detach()
        cpu = getattr(value, "cpu", None)
        if callable(cpu):
            value = cpu()
        to_numpy = getattr(value, "numpy", None)
        if callable(to_numpy):
            value = to_numpy()
        return np.asarray(value)

    def _release_runtime(self) -> None:
        """Best-effort release of RKNNLite hidden behind AutoBackend."""
        try:
            predictor = getattr(self._model, "predictor", None)
            auto_backend = getattr(predictor, "model", None)
            backend = getattr(auto_backend, "backend", None)
            runtime = getattr(backend, "model", None)
            release = getattr(runtime, "release", None)
            if callable(release):
                release()
        except Exception as exc:
            logger.debug("Ultralytics backend release failed: %s", exc)

    @staticmethod
    def _filter_target_labels(
        objects: list[dict[str, Any]],
        target_labels: list[str],
    ) -> list[dict[str, Any]]:
        if not target_labels:
            return objects
        requested = {label.casefold() for label in target_labels}
        return [
            item
            for item in objects
            if str(item.get("label", "")).strip().casefold() in requested
        ]

    @staticmethod
    def _mock_objects() -> list[dict[str, Any]]:
        """Return deterministic data only when mock mode was explicit."""
        return [{
            "label": "dog toy ball",
            "x": 0.3,
            "y": 0.45,
            "w": 0.12,
            "h": 0.12,
            "confidence": 0.88,
            "center_x": 0.36,
            "center_y": 0.51,
        }]
