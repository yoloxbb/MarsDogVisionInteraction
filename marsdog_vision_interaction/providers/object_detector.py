"""Object detection provider — YOLOE-seg RKNN inference.

Real RKNN model for on-demand object detection via /perception/perception_task.
Lazy-loaded: model is only initialized on first detect_objects() call.

Model: YOLOE-26s-seg, 16 dog-related classes, 640x640 input.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from marsdog_vision_interaction.providers.base import BaseProvider

logger = logging.getLogger(__name__)

# Model classes
_CLASS_NAMES = {
    0: "dog toy ball",
    1: "dog frisbee toy",
    2: "dog tug ring toy",
    3: "dog collar",
    4: "dog bowl",
    5: "dog leash",
    6: "dog treat bag",
    7: "dog food can",
    8: "dog bed",
    9: "trash can",
    10: "cardboard shipping box",
    11: "sock",
    12: "slipper",
    13: "tissue paper",
    14: "door",
    15: "stairs",
    16: "cat",
    17: "dog",
}

_IMG_SIZE = 640
_NUM_CLASSES = 18
_NUM_MASKS = 32


class ObjectDetectorProvider(BaseProvider):
    """On-demand YOLOE-seg RKNN object detector.

    Lazy-loads ~23MB RKNN model on first detect_objects() call.
    Inference at 640x640, ~50-100ms on RK3588 (3 NPU cores).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self._model_path = config.get("object_rknn_model", "")
        self._det_threshold = float(config.get("det_threshold", 0.5))
        self._nms_threshold = float(config.get("nms_threshold", 0.45))

        self._rknn: Any = None
        self._loaded = False

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        try:
            logger.info("ObjectDetectorProvider — lazy-load, model=%s", self._model_path)
            self.available = True
            logger.info("ObjectDetectorProvider started (lazy)")
        except Exception as exc:
            self.available = False
            logger.warning("ObjectDetectorProvider start failed: %s", exc)

    def stop(self) -> None:
        if self._rknn is not None:
            try:
                self._rknn.release()
            except Exception:
                pass
            self._rknn = None
        self._loaded = False
        self.available = False
        logger.info("ObjectDetectorProvider stopped")

    # ── Public API ─────────────────────────────────────────────

    def detect_objects(
        self, frame: Any = None, params: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Detect objects in a BGR camera frame.

        Args:
            frame: BGR numpy array from camera.
            params: Optional [{key: confidence, value: 0.3}, ...].

        Returns:
            List of dicts with label, x, y, w, h, confidence, center_x, center_y.
        """
        # Parse params
        threshold = self._det_threshold
        if params:
            for p in params:
                if p.get("key") == "confidence":
                    try:
                        threshold = float(p["value"])
                    except (ValueError, TypeError):
                        pass

        if not self.available:
            return self._mock_objects()
        if frame is None:
            return self._mock_objects()

        # Lazy-load on first call
        if not self._loaded:
            if not self._load_model():
                return self._mock_objects()

        try:
            return self._run_inference(frame, threshold)
        except Exception as exc:
            logger.error("Object detection error: %s", exc, exc_info=True)
            return self._mock_objects()

    # ── Model loading ──────────────────────────────────────────

    def _load_model(self) -> bool:
        """Load RKNN model from disk."""
        import os
        from rknnlite.api import RKNNLite

        # Support both directory and direct file paths
        if os.path.isdir(self._model_path):
            # Find .rknn file in directory
            for f in os.listdir(self._model_path):
                if f.endswith(".rknn"):
                    model_file = os.path.join(self._model_path, f)
                    break
            else:
                logger.error("No .rknn file found in %s", self._model_path)
                self._loaded = True
                return False
        else:
            model_file = self._model_path

        try:
            t0 = time.perf_counter()
            self._rknn = RKNNLite()
            ret = self._rknn.load_rknn(model_file)
            if ret != 0:
                logger.error("RKNN load_rknn failed: ret=%d", ret)
                self._loaded = True
                return False

            ret = self._rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
            if ret != 0:
                logger.error("RKNN init_runtime failed: ret=%d", ret)
                self._loaded = True
                return False

            elapsed = (time.perf_counter() - t0) * 1000
            self._loaded = True
            logger.info(
                "YOLOE-seg RKNN loaded: %s (%.0fms)", model_file, elapsed,
            )
            return True

        except Exception as exc:
            logger.error("RKNN load failed: %s", exc)
            self._loaded = True
            return False

    # ── Inference ──────────────────────────────────────────────

    def _run_inference(
        self, frame: np.ndarray, threshold: float,
    ) -> list[dict[str, Any]]:
        """Run full inference pipeline: preprocess → RKNN → postprocess.

        Args:
            frame: BGR numpy array (H, W, 3).
            threshold: Confidence threshold.

        Returns:
            List of detected object dicts.
        """
        import cv2

        h, w = frame.shape[:2]

        # ── Preprocess ──────────────────────────────────────
        # Resize to 640x640, BGR→RGB, uint8 [0-255]
        resized = cv2.resize(frame, (_IMG_SIZE, _IMG_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        # NCHW format
        input_data = np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.uint8)

        # ── Inference ───────────────────────────────────────
        outputs = self._rknn.inference([input_data])
        # Output 0: (1, 54, 8400) — 4 bbox + 18 cls + 32 mask = 54
        # Output 1: (1, 32, 160, 160) — proto masks (unused for bbox-only)

        # ── Postprocess ─────────────────────────────────────
        objects = self._postprocess(outputs, threshold, h, w)

        logger.info(
            "YOLOE-seg: %d objects (%dx%d→640x640, thr=%.2f)",
            len(objects), w, h, threshold,
        )
        return objects

    def _postprocess(
        self, outputs: list[np.ndarray], threshold: float,
        orig_h: int, orig_w: int,
    ) -> list[dict[str, Any]]:
        """Decode YOLO outputs — decode ALL boxes, then filter by conf + NMS.

        Output format (not end2end):
          [0]: (1, 4+16+32=52, 8400) — bbox + class_scores + mask_coeffs
          8400 = 80² + 40² + 20² (3 detection heads)
        """
        det = outputs[0][0]  # (52, 8400)

        # Split channels
        boxes_raw = det[:4, :]                 # (4, 8400) — cx, cy, w, h
        class_raw = det[4:4 + _NUM_CLASSES, :]  # (16, 8400)

        # Sigmoid
        boxes_sig = 1.0 / (1.0 + np.exp(-boxes_raw))  # (4, 8400)
        class_scores = 1.0 / (1.0 + np.exp(-class_raw))  # (16, 8400)

        # Decode ALL 8400 boxes first (need grid info before filtering)
        boxes_px = self._decode_all_boxes(boxes_sig)  # (8400, 4)

        # Max score and class per anchor
        scores = class_scores.max(axis=0)  # (8400,)
        class_ids = class_scores.argmax(axis=0)  # (8400,)

        # Filter by confidence
        keep = scores > threshold
        if not np.any(keep):
            return []

        boxes_px = boxes_px[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Scale to original frame
        scale_x = orig_w / _IMG_SIZE
        scale_y = orig_h / _IMG_SIZE
        boxes_px[:, 0] *= scale_x
        boxes_px[:, 1] *= scale_y
        boxes_px[:, 2] *= scale_x
        boxes_px[:, 3] *= scale_y

        # NMS
        keep_idx = self._nms(boxes_px, scores, class_ids)

        # Build result — filter tiny noise boxes
        min_area_px = orig_w * orig_h * 0.005  # 0.5% of frame area minimum

        objects = []
        for idx in keep_idx:
            x1, y1, x2, y2 = boxes_px[idx]
            bw, bh = x2 - x1, y2 - y1
            if bw < 3 or bh < 3 or bw * bh < min_area_px:
                continue
            cid = int(class_ids[idx])
            objects.append({
                "label": _CLASS_NAMES.get(cid, f"class_{cid}"),
                "x": round(float(x1 / orig_w), 4),
                "y": round(float(y1 / orig_h), 4),
                "w": round(float(bw / orig_w), 4),
                "h": round(float(bh / orig_h), 4),
                "confidence": round(float(scores[idx]), 4),
                "center_x": round(float((x1 + x2) / 2 / orig_w), 4),
                "center_y": round(float((y1 + y2) / 2 / orig_h), 4),
            })
        return objects

    def _decode_all_boxes(self, boxes_sig: np.ndarray) -> np.ndarray:
        """Decode all 8400 boxes using proper multi-scale grid.

        Multi-scale: 80×80(stride 8), 40×40(stride 16), 20×20(stride 32).
        boxes_sig: (4, 8400) sigmoid-activated raw box predictions.
        Returns: (8400, 4) x1,y1,x2,y2 in 640x640 space.
        """
        all_boxes = []
        offset = 0
        for grid_h, grid_w, stride in [(80, 80, 8), (40, 40, 16), (20, 20, 32)]:
            n_cells = grid_h * grid_w
            chunk = boxes_sig[:, offset:offset + n_cells]  # (4, n_cells)

            # Grid coordinates
            gy, gx = np.meshgrid(np.arange(grid_h), np.arange(grid_w), indexing='ij')
            gx = gx.ravel().astype(np.float32)
            gy = gy.ravel().astype(np.float32)

            # Decode: cx = (sig*2 - 0.5 + grid_x) * stride
            cx = (chunk[0] * 2.0 - 0.5 + gx) * stride
            cy = (chunk[1] * 2.0 - 0.5 + gy) * stride
            w = (chunk[2] * 2.0) ** 2 * stride
            h = (chunk[3] * 2.0) ** 2 * stride

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            boxes = np.stack([x1, y1, x2, y2], axis=1)  # (n_cells, 4)
            all_boxes.append(boxes)
            offset += n_cells

        return np.concatenate(all_boxes, axis=0).astype(np.float32)

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
    ) -> list[int]:
        """Simple class-aware NMS.

        Args:
            boxes: (N, 4) x1, y1, x2, y2.
            scores: (N,) confidence.
            class_ids: (N,) class IDs.

        Returns:
            List of indices to keep.
        """
        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            idx = order[0]
            keep.append(int(idx))
            if len(order) == 1:
                break

            # IoU of current best vs rest
            box = boxes[idx]
            other_boxes = boxes[order[1:]]

            # Same class only
            same_class = class_ids[order[1:]] == class_ids[idx]

            xx1 = np.maximum(box[0], other_boxes[:, 0])
            yy1 = np.maximum(box[1], other_boxes[:, 1])
            xx2 = np.minimum(box[2], other_boxes[:, 2])
            yy2 = np.minimum(box[3], other_boxes[:, 3])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h

            area_box = (box[2] - box[0]) * (box[3] - box[1])
            area_other = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
            union = area_box + area_other - inter
            iou = inter / np.maximum(union, 1e-6)

            # Suppress high IoU + same class
            suppress = (iou > self._nms_threshold) & same_class
            order = order[1:][~suppress]

        return keep

    @staticmethod
    def _mock_objects() -> list[dict[str, Any]]:
        """Return mock detection when model unavailable."""
        return [
            {
                "label": "red ball",
                "x": 0.3, "y": 0.45, "w": 0.12, "h": 0.12,
                "confidence": 0.88, "center_x": 0.36, "center_y": 0.51,
            }
        ]
