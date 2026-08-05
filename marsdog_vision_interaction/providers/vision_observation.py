"""Observation vision provider — face detection + MediaPipe pose + hands.

Continuous detection at observation rate, caching the latest result.

Models:
  - YuNet             face detection (OpenCV ONNX, 320x320)
  - MediaPipe PoseLandmarker  (33-point pose + human detection, Lite)
  - MediaPipe HandLandmarker  (21-point hand keypoints, Lite)

Total per frame: ~20ms, well within 100ms budget at 10Hz.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from marsdog_vision_interaction.providers.base import BaseProvider
from marsdog_vision_interaction.utils.stereo_view import select_camera_view

logger = logging.getLogger(__name__)

_FACE_INPUT_SIZE = (320, 320)


class VisionObservationProvider(BaseProvider):
    """Continuous vision — YuNet face + MediaPipe pose + hands.

    Accepts camera frames from the bridge node via process_frame().
    Runs detection on each frame, caches the result.
    get_observation() returns the latest cached dict.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)

        self._face_detect_model = config.get("face_detect_model", "")
        self._mediapipe_model = config.get("mediapipe_model", "")
        self._hand_landmark_model = config.get(
            "hand_landmark_model",
            "/home/cat/xbb/models/vision/hand_landmarker.task",
        )
        self._det_threshold = float(config.get("det_threshold", 0.5))
        self._nms_threshold = float(config.get("nms_threshold", 0.45))
        self._pose_confidence_threshold = float(
            config.get("pose_confidence_threshold", 0.5),
        )

        # Stereo fusion — single-target constraint for binocular cameras
        self._stereo_enabled = bool(config.get("stereo_enabled", True))
        self._stereo_view = str(config.get("stereo_view", "left")).lower()
        self._stereo_min_aspect_ratio = max(
            1.0, float(config.get("stereo_min_aspect_ratio", 2.2))
        )
        if self._stereo_view not in ("left", "right"):
            logger.warning(
                "Invalid stereo_view=%r; falling back to left",
                self._stereo_view,
            )
            self._stereo_view = "left"
        self._single_target = bool(config.get("single_target_mode", True))
        self._last_input_layout: tuple[int, int, bool] | None = None

        # Vision task control — individual tasks can be toggled at runtime
        # via the HTTP task endpoint (useful for enrollment scenarios)
        self.task_face_enabled = True
        self.task_pose_enabled = True
        self.task_hand_enabled = True

        # Detectors
        self._face_detector: Any = None   # cv2.FaceDetectorYN
        self._pose_landmarker: Any = None  # mediapipe PoseLandmarker
        self._hand_landmarker: Any = None  # mediapipe HandLandmarker

        # Face tracking + throttled recognition
        self._face_tracker: Any = None     # FaceByteTracker
        self._face_rec_throttle: Any = None  # FaceRecognitionThrottle
        self._face_rec_model: Any = None   # cv2.FaceRecognizerSF (for alignCrop + feature)
        self._use_byte_track: bool = bool(config.get("face_tracking", {}).get("use_bytetrack", True))

        # Pose/Hand action classifier (mock — random low-frequency triggers)
        from marsdog_vision_interaction.providers.pose_action import PoseActionClassifier
        self._action_classifier = PoseActionClassifier(
            trigger_chance=float(config.get("action_trigger_chance", 0.003)),
            min_duration_sec=float(config.get("action_min_duration_sec", 2.0)),
            max_duration_sec=float(config.get("action_max_duration_sec", 4.0)),
            min_cooldown_sec=float(config.get("action_min_cooldown_sec", 5.0)),
            max_cooldown_sec=float(config.get("action_max_cooldown_sec", 15.0)),
        )

        # Cache
        self._lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        self._cached_observation: dict[str, Any] = {}
        self._processing = False

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        try:
            import cv2

            total = 3  # face + pose + hand
            loaded = 0

            # YuNet face detector
            if self._face_detect_model:
                try:
                    self._face_detector = cv2.FaceDetectorYN.create(
                        model=self._face_detect_model,
                        config="",
                        input_size=_FACE_INPUT_SIZE,
                        score_threshold=self._det_threshold,
                        nms_threshold=self._nms_threshold,
                        top_k=5000,
                    )
                    loaded += 1
                    logger.info("YuNet loaded: %s", self._face_detect_model)
                except Exception as exc:
                    logger.warning("YuNet failed: %s", exc)
            else:
                total -= 1

            # MediaPipe PoseLandmarker (human detection + pose in one pass)
            if self._mediapipe_model:
                try:
                    import mediapipe as mp
                    from mediapipe.tasks.python import vision
                    from mediapipe.tasks.python.vision import RunningMode

                    options = vision.PoseLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=self._mediapipe_model,
                        ),
                        running_mode=RunningMode.IMAGE,
                        num_poses=5,
                        min_pose_detection_confidence=self._det_threshold,
                        min_pose_presence_confidence=self._det_threshold,
                        min_tracking_confidence=0.5,
                    )
                    self._pose_landmarker = vision.PoseLandmarker.create_from_options(
                        options,
                    )
                    loaded += 1
                    logger.info("MediaPipe PoseLandmarker loaded: %s", self._mediapipe_model)
                except Exception as exc:
                    logger.warning("MediaPipe PoseLandmarker failed: %s", exc)
            else:
                total -= 1

            # MediaPipe HandLandmarker (21-point hand keypoints)
            if self._hand_landmark_model:
                try:
                    import mediapipe as mp
                    from mediapipe.tasks.python import vision
                    from mediapipe.tasks.python.vision import RunningMode

                    hand_options = vision.HandLandmarkerOptions(
                        base_options=mp.tasks.BaseOptions(
                            model_asset_path=self._hand_landmark_model,
                        ),
                        running_mode=RunningMode.IMAGE,
                        num_hands=2,
                        min_hand_detection_confidence=0.5,
                        min_hand_presence_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self._hand_landmarker = vision.HandLandmarker.create_from_options(
                        hand_options,
                    )
                    loaded += 1
                    logger.info("MediaPipe HandLandmarker loaded: %s", self._hand_landmark_model)
                except Exception as exc:
                    logger.warning("MediaPipe HandLandmarker failed: %s", exc)
            else:
                total -= 1

            # ── ByteTrack face tracker ──
            if self._use_byte_track:
                try:
                    from marsdog_vision_interaction.providers.face_tracker import FaceByteTracker
                    ft_cfg = self.config.get("face_tracking", {})
                    self._face_tracker = FaceByteTracker(
                        track_activation_threshold=float(ft_cfg.get("track_activation_threshold", 0.25)),
                        lost_track_buffer=int(ft_cfg.get("lost_track_buffer", 30)),
                        minimum_matching_threshold=float(ft_cfg.get("minimum_matching_threshold", 0.8)),
                        minimum_consecutive_frames=int(ft_cfg.get("minimum_consecutive_frames", 2)),
                    )
                    loaded += 1; total += 1
                    logger.info("ByteTrack face tracker initialized")
                except Exception as exc:
                    logger.warning("ByteTrack init failed: %s — tracking disabled", exc)
                    self._use_byte_track = False

            # ── SFace recognizer for throttled recognition ──
            face_rec_model_path = self.config.get("face_recogn_model", "")
            face_rec_cfg = self.config.get("face_recognition_throttle", {})
            if face_rec_model_path:
                try:
                    self._face_rec_model = cv2.FaceRecognizerSF.create(
                        model=face_rec_model_path, config="",
                    )
                    from marsdog_vision_interaction.providers.face_tracker import FaceRecognitionThrottle
                    self._face_rec_throttle = FaceRecognitionThrottle(
                        face_recognizer=self._face_rec_model,
                        min_face_score=float(face_rec_cfg.get("min_face_score", 0.85)),
                        min_face_size_px=int(face_rec_cfg.get("min_face_size_px", 40)),
                        unknown_active_retry_sec=float(face_rec_cfg.get("unknown_active_retry_sec", 0.5)),
                        unknown_inactive_retry_sec=float(face_rec_cfg.get("unknown_inactive_retry_sec", 1.5)),
                        known_active_reverify_sec=float(face_rec_cfg.get("known_active_reverify_sec", 3.0)),
                        known_inactive_reverify_sec=float(face_rec_cfg.get("known_inactive_reverify_sec", 8.0)),
                        confirm_known_count=int(face_rec_cfg.get("confirm_known_count", 2)),
                        confirm_unknown_count=int(face_rec_cfg.get("confirm_unknown_count", 4)),
                        sface_cosine_threshold=float(face_rec_cfg.get("sface_cosine_threshold", 0.36)),
                    )
                    loaded += 1; total += 1
                    logger.info("SFace recognition throttle initialized")
                except Exception as exc:
                    logger.warning("SFace recognizer init failed: %s", exc)

            if loaded > 0:
                self.available = True
                logger.info(
                    "VisionObservationProvider started — %d/%d models loaded",
                    loaded, total,
                )
            else:
                self.available = False
                logger.warning("VisionObservationProvider — no models, unavailable")

        except Exception as exc:
            self.available = False
            logger.warning("VisionObservationProvider start failed: %s", exc, exc_info=True)

    def stop(self) -> None:
        self._face_detector = None
        self._face_tracker = None
        self._face_rec_throttle = None
        self._face_rec_model = None
        if self._pose_landmarker is not None:
            try:
                self._pose_landmarker.close()
            except Exception:
                pass
            self._pose_landmarker = None
        if self._hand_landmarker is not None:
            try:
                self._hand_landmarker.close()
            except Exception:
                pass
            self._hand_landmarker = None
        with self._lock:
            self._latest_frame = None
            self._cached_observation = {}
        self.available = False
        logger.info("VisionObservationProvider stopped")

    # ── Frame injection ────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> None:
        """Accept a camera frame — crop to left half, detect, feed TargetManager."""
        if not self.available:
            return
        if self._processing:
            return

        self._processing = True
        try:
            self._latest_frame = frame
            obs = self._process_frame_impl(frame)
            with self._lock:
                self._cached_observation = obs
        except Exception as exc:
            logger.error("Frame processing error: %s", exc, exc_info=True)
        finally:
            self._processing = False

    def _process_frame_impl(self, frame: np.ndarray) -> dict[str, Any]:
        """Select one stereo eye and run all 2D models on that view.

        All 2D vision models (YuNet face, MediaPipe pose) operate on the
        configured eye only. Coordinates remain normalized to that single
        view, so x=0.5 always means straight ahead for the selected camera.
        """
        if not self._stereo_enabled:
            obs = self._run_inference(frame, detect_faces=True)
        else:
            selected_frame, stereo_split = select_camera_view(
                frame,
                stereo_enabled=True,
                view=self._stereo_view,
                min_aspect_ratio=self._stereo_min_aspect_ratio,
            )
            input_height, input_width = frame.shape[:2]
            layout = (input_width, input_height, stereo_split)
            if layout != self._last_input_layout:
                if stereo_split:
                    logger.info(
                        "Stereo input %dx%d detected; using %s eye %dx%d",
                        input_width, input_height, self._stereo_view,
                        selected_frame.shape[1], selected_frame.shape[0],
                    )
                else:
                    logger.warning(
                        "Input %dx%d is not side-by-side stereo (ratio %.2f < %.2f); "
                        "using the complete frame",
                        input_width, input_height, input_width / input_height,
                        self._stereo_min_aspect_ratio,
                    )
                self._last_input_layout = layout
            obs = self._run_inference(selected_frame, detect_faces=True)

        # ── Pose/Hand action classifier update ──────────────
        self._action_classifier.update()
        pose_key, pose_label = self._action_classifier.pose_action
        hand_key, hand_label = self._action_classifier.hand_action

        # Pose actions belong to the detected human. Attach them before target
        # selection so they survive into ActiveTarget and downstream events.
        for human in obs.get("humans", []):
            human["pose_action"] = pose_key
            human["pose_action_label"] = pose_label

        # Feed to TargetManager for single-target selection
        from marsdog_vision_interaction.fusion.stereo_fusion import get_target_manager

        mgr = get_target_manager()
        mgr.update_vision(
            humans=obs.get("humans", []),
            faces=obs.get("faces", []),
        )
        active = mgr.get_active_target()
        active_dict = active.to_dict()
        target_is_current = active_dict["tracking_state"] == "tracking"

        # Build unified output (matches the bridge's expected format)
        humans_out = []
        if active.confidence > 0 and target_is_current:
            humans_out = [{
                "x": round(active.bbox[0], 4),
                "y": round(active.bbox[1], 4),
                "w": round(active.bbox[2], 4),
                "h": round(active.bbox[3], 4),
                "confidence": round(active.confidence, 4),
                "pose_state": active.pose_state,
                "pose_action": pose_key,
                "pose_action_label": pose_label,
                "keypoints": active.keypoints,
                "track_id": active.track_id,
            }]

        faces_out = []
        if active.face_confidence > 0 and target_is_current:
            faces_out = [{
                "track_id": active.face_track_id,
                "x": round(active.face_bbox[0], 4),
                "y": round(active.face_bbox[1], 4),
                "w": round(active.face_bbox[2], 4),
                "h": round(active.face_bbox[3], 4),
                "confidence": round(active.face_confidence, 4),
                "recognized_user": active.identity,
                "identity_confidence": round(active.identity_confidence, 4),
                "identity_state": active.identity_state,
                "quality": round(active.face_confidence, 4),
            }]
        else:
            # Keep any detected faces even if not yet bound to a human
            for f in obs.get("faces", []):
                faces_out.append({
                    "track_id": int(f.get("track_id", -1)),
                    "x": round(f.get("x", 0), 4),
                    "y": round(f.get("y", 0), 4),
                    "w": round(f.get("w", 0), 4),
                    "h": round(f.get("h", 0), 4),
                    "confidence": round(f.get("confidence", 0), 4),
                    "recognized_user": f.get("recognized_user", ""),
                    "identity_confidence": round(
                        f.get("identity_confidence", 0), 4,
                    ),
                    "identity_state": f.get(
                        "identity_state", "unverified",
                    ),
                    "quality": round(f.get("quality", 0), 4),
                })

        # Attach hand action labels to each hand
        hands_out = []
        for h in obs.get("hands", []):
            hands_out.append({
                "handedness": h.get("handedness", ""),
                "hand_action": hand_key,
                "hand_action_label": hand_label,
                "landmarks": h.get("landmarks", []),
            })

        return {
            "active_target": active_dict,
            "faces": faces_out,
            "humans": humans_out,
            "hands": hands_out,
            "tracked_objects": obs.get("tracked_objects", []),
        }

    @staticmethod
    def _select_stereo_view(
        frame: np.ndarray,
        view: str = "left",
        min_aspect_ratio: float = 2.2,
    ) -> np.ndarray:
        """Select one eye for stereo input, preserving normal mono frames."""
        selected, _ = select_camera_view(
            frame,
            stereo_enabled=True,
            view=view,
            min_aspect_ratio=min_aspect_ratio,
        )
        return selected

    @staticmethod
    def _scale_obs_to_region(
        obs: dict[str, Any],
        ox: float, oy: float, ow: float, oh: float,
    ) -> None:
        """Scale normalized coordinates from a sub-region to full-frame space.

        Args:
            obs: Observation dict with faces/humans in sub-region coords.
            ox, oy: Offset of sub-region in full frame (normalized 0-1).
            ow, oh: Size of sub-region relative to full frame.
        """
        for det_list in (obs.get("faces", []), obs.get("humans", [])):
            for det in det_list:
                det["x"] = round(ox + float(det.get("x", 0)) * ow, 4)
                det["y"] = round(oy + float(det.get("y", 0)) * oh, 4)
                det["w"] = round(float(det.get("w", 0)) * ow, 4)
                det["h"] = round(float(det.get("h", 0)) * oh, 4)
                for kp in det.get("keypoints", []):
                    kp["x"] = round(ox + float(kp.get("x", 0)) * ow, 4)
                    kp["y"] = round(oy + float(kp.get("y", 0)) * oh, 4)

        # Scale hand landmarks (left half → full frame)
        for hand in obs.get("hands", []):
            for lm in hand.get("landmarks", []):
                lm["x"] = round(ox + float(lm.get("x", 0)) * ow, 4)
                lm["y"] = round(oy + float(lm.get("y", 0)) * oh, 4)

    # ── Public API ─────────────────────────────────────────────

    def get_observation(self) -> dict[str, Any]:
        with self._lock:
            if self._cached_observation:
                return dict(self._cached_observation)
        return {}

    def check_person(self) -> dict[str, Any]:
        obs = self.get_observation()
        humans = obs.get("humans", [])
        return {"present": len(humans) > 0, "count": len(humans)}

    def enroll_face(self, face_image: Any = None) -> dict[str, Any]:
        _ = face_image
        return {"success": True, "user_id": "mock_user_001"}

    def recognize_face(self, face_image: Any = None) -> dict[str, Any]:
        _ = face_image
        return {"user_id": "unknown", "confidence": 0.0}

    # ── Inference ──────────────────────────────────────────────

    def _run_inference(
        self, frame: np.ndarray, detect_faces: bool = True,
    ) -> dict[str, Any]:
        """Run YuNet + MediaPipe on a single frame.

        Args:
            frame: BGR numpy array.
            detect_faces: If False, skip YuNet (e.g. right stereo half).

        Returns normalized coordinates [0-1] relative to the input frame.
        Caller is responsible for mapping to full-frame space if needed.
        """
        h, w = frame.shape[:2]

        faces = self._detect_faces(frame, w, h) if (detect_faces and self.task_face_enabled) else []
        humans = self._detect_pose_mediapipe(frame, w, h) if self.task_pose_enabled else []
        hands = self._detect_hands(frame, w, h) if self.task_hand_enabled else []

        return {
            "faces": faces,
            "humans": humans,
            "hands": hands,
            "tracked_objects": [],
        }

    # ── YuNet face detection ───────────────────────────────────

    def _detect_faces(self, frame: np.ndarray, w: int, h: int) -> list[dict[str, Any]]:
        if self._face_detector is None:
            return []

        try:
            self._face_detector.setInputSize((w, h))
            _, results = self._face_detector.detect(frame)
            faces = []
            if results is None or len(results) == 0:
                # Feed empty detections to tracker to keep state consistent
                if self._face_tracker is not None:
                    self._face_tracker.update(
                        np.empty((0, 4), dtype=np.float32),
                        np.empty((0,), dtype=np.float32),
                    )
                return []

            # Build face list with xyxy for tracking
            detections_xyxy = []
            detections_scores = []
            face_data = []
            for det in results:
                fx, fy, fw, fh = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                # FaceDetectorYN layout is bbox(4), five landmarks(10),
                # confidence(1). det[4] is a landmark coordinate, not score.
                conf = float(det[-1])
                if conf < self._det_threshold:
                    continue
                x1, y1, x2, y2 = fx, fy, fx + fw, fy + fh
                detections_xyxy.append([x1, y1, x2, y2])
                detections_scores.append(conf)
                face_data.append({
                    "x": round(fx / w, 4), "y": round(fy / h, 4),
                    "w": round(fw / w, 4), "h": round(fh / h, 4),
                    "confidence": round(conf, 4),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,  # pixel coords for tracking
                    "_yunet_detection": np.asarray(det, dtype=np.float32),
                })

            if not face_data:
                if self._face_tracker is not None:
                    self._face_tracker.update(
                        np.empty((0, 4), dtype=np.float32),
                        np.empty((0,), dtype=np.float32),
                    )
                return []

            # ── ByteTrack ──
            track_ids = None
            if self._face_tracker is not None:
                xyxy_arr = np.array(detections_xyxy, dtype=np.float32)
                scores_arr = np.array(detections_scores, dtype=np.float32)
                track_ids = self._face_tracker.update(xyxy_arr, scores_arr)

            # ── Throttled SFace recognition ──
            now = time.time()
            for i, fd in enumerate(face_data):
                tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else -1
                fd["track_id"] = tid
                fd["identity_state"] = "unverified"
                fd["identity_confidence"] = 0.0
                fd["recognized_user"] = ""

                if tid < 0:
                    continue

                # Update tracker with current bbox
                bw = int(fd["x2"] - fd["x1"])
                bh = int(fd["y2"] - fd["y1"])

                if self._face_rec_throttle is not None:
                    self._face_rec_throttle.mark_seen(tid, np.array(detections_xyxy[i]), now)

                    # Check if we should recognize this face now
                    # is_active = this is the active target
                    is_active = False  # will be refined later in fusion
                    if self._face_rec_throttle.should_recognize(
                        tid, fd["confidence"], bw, bh, is_active, now,
                    ):
                        identity, identity_conf = self._run_sface(frame, fd)
                        self._face_rec_throttle.update_identity(tid, identity, identity_conf, now)

                # Read current track state
                track_state = self._face_rec_throttle.get_track_state(tid) if self._face_rec_throttle else None
                if track_state is not None:
                    fd["recognized_user"] = track_state.identity if track_state.is_known else ""
                    fd["identity_confidence"] = round(track_state.identity_confidence, 4)
                    fd["identity_state"] = track_state.identity_state

                # Quality score: face_score * size_ratio
                quality = fd["confidence"]
                fd["quality"] = round(quality, 4)

            # Remove internal pixel fields
            for fd in face_data:
                fd.pop("x1", None); fd.pop("y1", None); fd.pop("x2", None); fd.pop("y2", None)
                fd.pop("_yunet_detection", None)

            logger.debug(
                "YuNet: %d raw, %d passed, track_ids=%s",
                len(results), len(face_data),
                [f["track_id"] for f in face_data] if face_data else [],
            )
            faces = face_data
            return faces

        except Exception as exc:
            logger.debug("Face detection error: %s", exc)
            return []

    def _run_sface(
        self, frame: np.ndarray, face: dict[str, Any],
    ) -> tuple[str, float]:
        """Run SFace recognition on a single face ROI.

        Returns (identity, confidence) tuple.
        """
        if self._face_rec_model is None or self._face_rec_throttle is None:
            return ("unknown", 0.0)
        if self._face_rec_throttle.enrolled_count == 0:
            return ("unknown", 0.0)

        try:
            x1, y1 = int(face["x1"]), int(face["y1"])
            x2, y2 = int(face["x2"]), int(face["y2"])
            h_img, w_img = frame.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                return ("unknown", 0.0)

            # alignCrop requires the original 15-value YuNet detection row.
            # Fall back to the bounded ROI when alignment is unavailable.
            aligned = None
            yunet_detection = face.get("_yunet_detection")
            if yunet_detection is not None:
                try:
                    aligned = self._face_rec_model.alignCrop(
                        frame, yunet_detection,
                    )
                except Exception as exc:
                    logger.debug("SFace alignCrop failed, using ROI: %s", exc)
            if aligned is None or aligned.size == 0:
                aligned = frame[y1:y2, x1:x2]
                if aligned.size == 0:
                    return ("unknown", 0.0)

            feature = self._face_rec_model.feature(aligned)
            if feature is None:
                return ("unknown", 0.0)

            # Compare against enrolled embeddings
            best_name = "unknown"
            best_score = 0.0
            for name, emb in self._face_rec_throttle._enrolled_embeddings.items():
                score = self._cosine_sim(feature, emb)
                if score > best_score:
                    best_score = score
                    best_name = name

            threshold = self._face_rec_throttle._cosine_threshold
            if best_score >= threshold:
                return (best_name, round(float(best_score), 4))
            return ("unknown", round(float(best_score), 4))

        except Exception as exc:
            logger.debug("SFace recognition error: %s", exc)
            return ("unknown", 0.0)

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        a_f = a.ravel().astype(np.float64)
        b_f = b.ravel().astype(np.float64)
        dot = np.dot(a_f, b_f)
        norm = np.linalg.norm(a_f) * np.linalg.norm(b_f)
        return float(dot / norm) if norm > 1e-8 else 0.0

    def sync_enrolled_to_throttle(self) -> None:
        """Copy enrolled embeddings from FaceRecognitionProvider to throttle."""
        if self._face_rec_throttle is None:
            return
        try:
            from marsdog_vision_interaction.core.face_enrollment_manager import EnrollmentManager
            import cv2
            names = EnrollmentManager.list_enrolled_faces()
            enrolled: dict[str, np.ndarray] = {}
            for name in names:
                paths = EnrollmentManager.get_face_paths(name)
                for p_str in paths:
                    from pathlib import Path
                    if not Path(p_str).exists():
                        continue
                    img = cv2.imread(p_str)
                    if img is None:
                        continue
                    feature = self._face_rec_model.feature(img)
                    if feature is not None:
                        enrolled[name] = feature.ravel()
                        break  # one embedding per person
            self._face_rec_throttle.set_enrolled_embeddings(enrolled)
            if enrolled:
                logger.info("Throttle synced: %d enrolled faces", len(enrolled))
        except Exception as exc:
            logger.debug("Throttle sync error: %s", exc)

    # ── MediaPipe PoseLandmarker ───────────────────────────────

    def _detect_pose_mediapipe(
        self, frame: np.ndarray, w: int, h: int,
    ) -> list[dict[str, Any]]:
        """Run MediaPipe PoseLandmarker — combined human detection + 33-point pose.

        Returns:
            List of human dicts with bbox, confidence, pose_state, keypoints.
        """
        if self._pose_landmarker is None:
            return []

        try:
            import mediapipe as mp
            import cv2

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = self._pose_landmarker.detect(mp_image)

            if not result.pose_landmarks:
                return []

            humans = []
            for landmarks in result.pose_landmarks:
                if len(landmarks) == 0:
                    continue

                # Compute bounding box from landmarks
                xs = [lm.x for lm in landmarks if lm.visibility > 0.1]
                ys = [lm.y for lm in landmarks if lm.visibility > 0.1]
                vs = [lm.visibility for lm in landmarks]

                if not xs or not ys:
                    continue

                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                avg_vis = sum(vs) / len(vs)

                bw = max_x - min_x
                bh = max_y - min_y

                if bw < 0.01 or bh < 0.01:
                    continue

                # Map 33 landmarks to keypoints
                keypoints = []
                for i, lm in enumerate(landmarks):
                    keypoints.append({
                        "id": i,
                        "x": round(float(lm.x), 4),
                        "y": round(float(lm.y), 4),
                        "confidence": round(float(lm.visibility), 4),
                    })

                # Classify pose state from key landmarks
                pose_state = self._classify_pose(landmarks)

                humans.append({
                    "x": round(min_x, 4),
                    "y": round(min_y, 4),
                    "w": round(bw, 4),
                    "h": round(bh, 4),
                    "confidence": round(float(avg_vis), 4),
                    "pose_state": pose_state,
                    "keypoints": keypoints,
                })

            return humans

        except Exception as exc:
            logger.debug("MediaPipe pose error: %s", exc)
            return []

    # ── MediaPipe HandLandmarker ──────────────────────────────

    def _detect_hands(
        self, frame: np.ndarray, w: int, h: int,
    ) -> list[dict[str, Any]]:
        """Run MediaPipe HandLandmarker — 21-point hand keypoints.

        Returns:
            List of hand dicts with handedness and landmarks.
            Coordinates normalized [0-1] relative to input frame.
        """
        if self._hand_landmarker is None:
            return []

        try:
            import mediapipe as mp
            import cv2

            # MediaPipe expects RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb,
            )

            result = self._hand_landmarker.detect(mp_image)

            if not result.hand_landmarks:
                return []

            hands = []
            for i, landmarks in enumerate(result.hand_landmarks):
                if len(landmarks) == 0:
                    continue

                # Handedness: "Left" or "Right"
                handedness = ""
                if result.handedness and i < len(result.handedness):
                    cats = result.handedness[i]
                    if cats:
                        handedness = cats[0].category_name  # "Left" / "Right"

                # Build 21 landmarks
                hand_landmarks = []
                for j, lm in enumerate(landmarks):
                    hand_landmarks.append({
                        "id": j,
                        "x": round(float(lm.x), 4),
                        "y": round(float(lm.y), 4),
                    })

                if hand_landmarks:
                    hands.append({
                        "handedness": handedness,
                        "landmarks": hand_landmarks,
                    })

            return hands

        except Exception as exc:
            logger.debug("MediaPipe hand error: %s", exc)
            return []

    # ── Pose classification ────────────────────────────────────

    @staticmethod
    def _classify_pose(landmarks) -> str:
        """Simple pose classifier based on key landmarks.

        Uses shoulder-hip-eye vertical ratios to distinguish
        standing / sitting / lying.
        """
        try:
            # Key indices (MediaPipe pose topology):
            # 11=left_shoulder, 12=right_shoulder
            # 23=left_hip, 24=right_hip
            # 0=nose
            nose_y = landmarks[0].y
            shoulder_y = (landmarks[11].y + landmarks[12].y) / 2
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
            knee_y = (landmarks[25].y + landmarks[26].y) / 2

            # Torso length
            torso_len = abs(hip_y - shoulder_y)
            if torso_len < 0.02:
                return "unknown"

            # Vertical position of hips relative to shoulders
            # Standing: hips well below shoulders, knees below hips
            if hip_y > shoulder_y + torso_len * 0.3:
                if knee_y > hip_y:
                    return "standing"
                return "sitting"

            # Lying: hips and shoulders at similar height
            return "lying"

        except (IndexError, ZeroDivisionError):
            return "unknown"
