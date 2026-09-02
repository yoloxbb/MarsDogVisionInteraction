"""Face enrollment and visual identity storage."""

from __future__ import annotations

import json
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from marsdog_vision_interaction.messages.face_identity import (
    ALLOWED_FACE_IDENTITIES,
    face_identity_display_name,
    face_identity_role,
    validate_face_identity,
)


_STORAGE_ROOT = Path("data")
_FACES_DIR = _STORAGE_ROOT / "faces"
_REGISTRY_PATH = _STORAGE_ROOT / "face_registry.json"
MAX_FACES = len(ALLOWED_FACE_IDENTITIES)
MAX_SAMPLES_PER_FACE = 5
_STORAGE_LOCK = threading.RLock()


def set_storage_root(path: str | Path) -> None:
    global _STORAGE_ROOT, _FACES_DIR, _REGISTRY_PATH
    _STORAGE_ROOT = Path(path)
    _FACES_DIR = _STORAGE_ROOT / "faces"
    _REGISTRY_PATH = _STORAGE_ROOT / "face_registry.json"
    _FACES_DIR.mkdir(parents=True, exist_ok=True)


def _to_registry_path(path: str | Path) -> str:
    """Store face images relative to the configured data directory."""
    return Path(path).resolve().relative_to(_STORAGE_ROOT.resolve()).as_posix()


def _from_registry_path(value: str | Path) -> Path | None:
    """Resolve current and legacy registry entries inside the data directory."""
    path = Path(value)
    if not path.is_absolute():
        candidate = _STORAGE_ROOT / path
    else:
        try:
            candidate = _STORAGE_ROOT / path.resolve().relative_to(
                _STORAGE_ROOT.resolve()
            )
        except ValueError:
            # Older registries stored a developer-specific absolute path. Keep
            # only the portable suffix starting at ``faces``.
            try:
                faces_index = path.parts.index("faces")
            except ValueError:
                return None
            candidate = _STORAGE_ROOT / Path(*path.parts[faces_index:])
    try:
        candidate.resolve().relative_to(_STORAGE_ROOT.resolve())
    except ValueError:
        return None
    return candidate


def _load_registry() -> dict[str, Any]:
    if not _REGISTRY_PATH.exists():
        return {"schema_version": 1, "faces": {}}
    try:
        value = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value.setdefault("schema_version", 1)
            value.setdefault("faces", {})
            return value
    except (OSError, json.JSONDecodeError):
        pass
    return {"schema_version": 1, "faces": {}}


def _save_registry(value: dict[str, Any]) -> None:
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _REGISTRY_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(_REGISTRY_PATH)


def _face_sample_ids(directory: Path) -> list[int]:
    sample_ids: set[int] = set()
    if directory.is_dir():
        for item in directory.glob("*.jpg"):
            try:
                sample_id = int(item.stem)
            except ValueError:
                continue
            if 1 <= sample_id <= MAX_SAMPLES_PER_FACE:
                sample_ids.add(sample_id)
    return sorted(sample_ids)


def _validate_sample_id(value: int) -> int:
    sample_id = int(value)
    if not 1 <= sample_id <= MAX_SAMPLES_PER_FACE:
        raise ValueError(
            f"sample_id 必须在 1 到 {MAX_SAMPLES_PER_FACE} 之间"
        )
    return sample_id


def _available_sample_ids(directory: Path) -> list[int]:
    occupied = set(_face_sample_ids(directory))
    return [
        sample_id
        for sample_id in range(1, MAX_SAMPLES_PER_FACE + 1)
        if sample_id not in occupied
    ]


def _known_face_names(registry: dict[str, Any] | None = None) -> list[str]:
    value = registry if registry is not None else _load_registry()
    registered = value.get("faces", {})
    return [
        name
        for name in ALLOWED_FACE_IDENTITIES
        if (
            name in registered
            and bool(_face_sample_ids(_FACES_DIR / name))
        )
    ]


def _sample_record(name: str, sample_id: int) -> dict[str, Any]:
    path = _FACES_DIR / name / f"{sample_id:03d}.jpg"
    available = path.is_file()
    return {
        "sample_id": sample_id,
        "sample_key": f"{sample_id:03d}",
        "image_filename": path.name,
        "image_path": str(path),
        "image_url": f"/api/v1/faces/{name}/samples/{sample_id}/image",
        "image_available": available,
        "ready": available,
        "image_size_bytes": path.stat().st_size if available else 0,
        "updated_at": path.stat().st_mtime if available else 0.0,
    }


def _update_registry_identity(
    registry: dict[str, Any],
    name: str,
    *,
    enrolled_at: float | None = None,
) -> list[int]:
    directory = _FACES_DIR / name
    sample_ids = _face_sample_ids(directory)
    if not sample_ids:
        registry["faces"].pop(name, None)
        return []
    previous = registry["faces"].get(name, {})
    now = time.time()
    registry["faces"][name] = {
        "paths": [
            _to_registry_path(directory / f"{sample_id:03d}.jpg")
            for sample_id in sample_ids
        ],
        "shots": len(sample_ids),
        "enrolled_at": float(
            enrolled_at
            if enrolled_at is not None
            else previous.get("enrolled_at", now)
        ),
        "updated_at": now,
    }
    return sample_ids


@dataclass
class FaceEnrollment:
    name: str
    required_shots: int
    started_at: float
    current_step: int = 1
    shots_collected: int = 0
    framed_count: int = 0
    framed_required: int = 3
    captured_paths: list[str] = field(default_factory=list)
    planned_sample_ids: list[int] = field(default_factory=list)
    done: bool = False
    poses: list[str] = field(default_factory=list)


_CONTINUOUS_POSE = "continuous"
_CONTINUOUS_PROMPT = "请自然面对摄像头并保持稳定，系统将连续采集人脸样本"


class FaceEnrollmentManager:
    """Own all face enrollment state; no voice-print data is stored here."""

    def __init__(self) -> None:
        self._face_detector: Any = None
        self._session: FaceEnrollment | None = None
        _FACES_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def face_session(self) -> FaceEnrollment | None:
        return self._session

    def set_face_detector(self, detector: Any) -> None:
        self._face_detector = detector

    def start_face(self, name: str, required_shots: int = 3) -> dict[str, Any]:
        try:
            name = validate_face_identity(name)
            shot_count = int(required_shots)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "status": 422, "error": str(exc)}
        if not 1 <= shot_count <= MAX_SAMPLES_PER_FACE:
            return {
                "ok": False,
                "status": 422,
                "error": (
                    f"required_shots 必须在 1 到 "
                    f"{MAX_SAMPLES_PER_FACE} 之间"
                ),
            }
        if self._session is not None:
            return {
                "ok": False,
                "status": 409,
                "error": "已有进行中的人脸注册会话",
            }
        available_ids = _available_sample_ids(_FACES_DIR / name)
        if len(available_ids) < shot_count:
            return {
                "ok": False,
                "status": 409,
                "code": "face_sample_limit_reached",
                "error": f"单人人脸样本最多 {MAX_SAMPLES_PER_FACE} 张",
                "name": name,
                "shots": MAX_SAMPLES_PER_FACE - len(available_ids),
                "available_slots": len(available_ids),
                "max_samples_per_face": MAX_SAMPLES_PER_FACE,
            }
        poses = [_CONTINUOUS_POSE] * shot_count
        self._session = FaceEnrollment(
            name=name,
            required_shots=shot_count,
            started_at=time.time(),
            poses=poses,
            planned_sample_ids=available_ids[:shot_count],
        )
        return {
            "ok": True,
            "name": name,
            "display_name": face_identity_display_name(name),
            "step": 1,
            "total_steps": self._session.required_shots,
            "pose": poses[0],
            "prompt": _CONTINUOUS_PROMPT,
            "max_faces": MAX_FACES,
            "max_samples_per_face": MAX_SAMPLES_PER_FACE,
        }

    def process_face_frame(self, frame: np.ndarray) -> dict[str, Any]:
        session = self._session
        if session is None or session.done:
            return {"ok": False, "error": "没有进行中的人脸注册会话"}

        faces = self._detect_faces(frame)
        if not faces:
            session.framed_count = 0
            return {
                "ok": True,
                "name": session.name,
                "step": session.current_step,
                "total_steps": session.required_shots,
                "status": "searching",
                "pose": self._expected_pose(session),
                "prompt": "未检测到人脸；" + self._current_prompt(session),
                "done": False,
            }

        if len(faces) != 1:
            session.framed_count = 0
            return self._waiting_result(session, "画面中必须只有一张人脸")

        best = faces[0]
        quality_error = self._quality_error(frame, best)
        if quality_error:
            session.framed_count = 0
            return self._waiting_result(session, quality_error)
        expected_pose = self._expected_pose(session)
        session.framed_count += 1
        if session.framed_count < session.framed_required:
            return {
                "ok": True,
                "name": session.name,
                "step": session.current_step,
                "total_steps": session.required_shots,
                "status": "tracking",
                "pose": expected_pose,
                "confidence": best["confidence"],
                "progress_pct": int(
                    session.framed_count / session.framed_required * 100
                ),
                "done": False,
            }

        height, width = frame.shape[:2]
        x1 = max(0, int(best["x"] * width))
        y1 = max(0, int(best["y"] * height))
        x2 = min(width, int((best["x"] + best["w"]) * width))
        y2 = min(height, int((best["y"] + best["h"]) * height))
        if x2 <= x1 or y2 <= y1:
            session.framed_count = 0
            return {"ok": True, "status": "searching", "done": False}

        import cv2

        face_dir = _FACES_DIR / session.name
        face_dir.mkdir(parents=True, exist_ok=True)
        sample_id = session.planned_sample_ids[session.shots_collected]
        path = face_dir / f"{sample_id:03d}.jpg"
        if not cv2.imwrite(str(path), frame[y1:y2, x1:x2]):
            session.framed_count = 0
            return self._waiting_result(session, "保存人脸样本失败，请重试")
        session.captured_paths.append(_to_registry_path(path))
        session.shots_collected += 1
        session.framed_count = 0

        if session.shots_collected >= session.required_shots:
            session.done = True
            with _STORAGE_LOCK:
                registry = _load_registry()
                sample_ids = _update_registry_identity(
                    registry,
                    session.name,
                )
                _save_registry(registry)
            result = {
                "ok": True,
                "name": session.name,
                "display_name": face_identity_display_name(session.name),
                "step": session.required_shots,
                "total_steps": session.required_shots,
                "status": "done",
                "pose": expected_pose,
                "shots": len(sample_ids),
                "sample_ids": sample_ids,
                "done": True,
            }
            # Completion is terminal.  Do not keep publishing a completed/error
            # enrollment result on every 10 Hz observation tick.
            self._session = None
            return result

        session.current_step = session.shots_collected + 1
        return {
            "ok": True,
            "name": session.name,
            "step": session.current_step,
            "total_steps": session.required_shots,
            "status": "captured",
            "pose": self._expected_pose(session),
            "captured_pose": expected_pose,
            "shots": session.shots_collected,
            "prompt": self._current_prompt(session),
            "done": False,
        }

    @staticmethod
    def _expected_pose(session: FaceEnrollment) -> str:
        index = min(session.shots_collected, len(session.poses) - 1)
        return session.poses[index]

    @classmethod
    def _current_prompt(cls, session: FaceEnrollment) -> str:
        _ = session
        return _CONTINUOUS_PROMPT

    @classmethod
    def _waiting_result(
        cls,
        session: FaceEnrollment,
        prompt: str,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "name": session.name,
            "step": session.current_step,
            "total_steps": session.required_shots,
            "status": "searching",
            "pose": cls._expected_pose(session),
            "prompt": prompt,
            "done": False,
        }

    @staticmethod
    def _has_valid_landmarks(face: dict[str, Any]) -> bool:
        landmarks = face.get("landmarks")
        if not isinstance(landmarks, list) or len(landmarks) != 5:
            return False
        try:
            right_eye, left_eye, _nose, right_mouth, left_mouth = landmarks
            eye_mid_y = (float(right_eye[1]) + float(left_eye[1])) / 2.0
            mouth_mid_y = (float(right_mouth[1]) + float(left_mouth[1])) / 2.0
            eye_distance = abs(float(left_eye[0]) - float(right_eye[0]))
            face_height = mouth_mid_y - eye_mid_y
        except (IndexError, TypeError, ValueError):
            return False
        return eye_distance >= 1e-4 and face_height >= 1e-4

    @staticmethod
    def _quality_error(frame: np.ndarray, face: dict[str, Any]) -> str | None:
        # Landmark-less adapters keep legacy behavior. YuNet sessions receive
        # the stricter production quality gates below.
        if not FaceEnrollmentManager._has_valid_landmarks(face):
            return None
        height, width = frame.shape[:2]
        face_w = int(float(face.get("w", 0.0)) * width)
        face_h = int(float(face.get("h", 0.0)) * height)
        if float(face.get("confidence", 0.0)) < 0.85:
            return "人脸检测置信度不足，请调整光线"
        if min(face_w, face_h) < 80:
            return "人脸太小，请靠近摄像头"
        x1 = max(0, int(float(face["x"]) * width))
        y1 = max(0, int(float(face["y"]) * height))
        x2 = min(width, x1 + face_w)
        y2 = min(height, y1 + face_h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "人脸区域无效，请重新站位"
        import cv2
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        if brightness < 35.0:
            return "画面过暗，请增加正面光线"
        if brightness > 225.0:
            return "画面过曝，请避开强光"
        if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 35.0:
            return "画面模糊，请保持头部稳定"
        return None

    def enroll_face_from_image(
        self,
        name: str,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        try:
            name = validate_face_identity(name)
        except ValueError as exc:
            return {
                "ok": False,
                "status": 422,
                "code": "invalid_face_identity",
                "error": str(exc),
                "allowed_names": list(ALLOWED_FACE_IDENTITIES),
            }
        with _STORAGE_LOCK:
            face_dir = _FACES_DIR / name
            available_ids = _available_sample_ids(face_dir)
            if not available_ids:
                return self._sample_capacity_error(name)

        prepared = self._prepare_uploaded_face(image_bytes)
        if not prepared.get("ok", False):
            return prepared
        jpeg_bytes = bytes(prepared["jpeg_bytes"])

        with _STORAGE_LOCK:
            available_ids = _available_sample_ids(face_dir)
            if not available_ids:
                return self._sample_capacity_error(name)
            sample_id = available_ids[0]
            face_dir.mkdir(parents=True, exist_ok=True)
            path = face_dir / f"{sample_id:03d}.jpg"
            self._write_sample_atomic(path, jpeg_bytes)
            registry = _load_registry()
            sample_ids = _update_registry_identity(registry, name)
            _save_registry(registry)
        return {
            "ok": True,
            "name": name,
            "display_name": face_identity_display_name(name),
            "face_role": face_identity_role(name),
            "shots": len(sample_ids),
            "sample_id": sample_id,
            "sample_key": f"{sample_id:03d}",
            "image_path": str(path),
            "max_faces": MAX_FACES,
            "max_samples_per_face": MAX_SAMPLES_PER_FACE,
        }

    def _prepare_uploaded_face(self, image_bytes: bytes) -> dict[str, Any]:
        if self._face_detector is None:
            return {
                "ok": False,
                "status": 503,
                "error": "人脸检测模型不可用",
            }
        import cv2

        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            return {"ok": False, "status": 422, "error": "无法解码图片"}
        faces = self._detect_faces(image)
        if not faces:
            return {
                "ok": False,
                "status": 422,
                "error": "未检测到清晰人脸",
            }
        best = max(faces, key=lambda face: face["w"] * face["h"])
        quality_error = self._quality_error(image, best)
        if quality_error:
            return {"ok": False, "status": 422, "error": quality_error}
        height, width = image.shape[:2]
        x1 = max(0, int(best["x"] * width))
        y1 = max(0, int(best["y"] * height))
        x2 = min(width, int((best["x"] + best["w"]) * width))
        y2 = min(height, int((best["y"] + best["h"]) * height))
        if x2 <= x1 or y2 <= y1:
            return {"ok": False, "status": 422, "error": "人脸区域无效"}
        encoded, jpeg = cv2.imencode(".jpg", image[y1:y2, x1:x2])
        if not encoded:
            return {"ok": False, "status": 422, "error": "人脸图片编码失败"}
        return {
            "ok": True,
            "jpeg_bytes": jpeg.tobytes(),
            "source_width": width,
            "source_height": height,
            "face_confidence": float(best.get("confidence", 0.0)),
        }

    @staticmethod
    def _write_sample_atomic(path: Path, payload: bytes) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_bytes(payload)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _sample_capacity_error(name: str) -> dict[str, Any]:
        shots = len(_face_sample_ids(_FACES_DIR / name))
        return {
            "ok": False,
            "status": 409,
            "code": "face_sample_limit_reached",
            "error": f"单人人脸样本已达到上限 {MAX_SAMPLES_PER_FACE} 张",
            "name": name,
            "shots": shots,
            "max_samples_per_face": MAX_SAMPLES_PER_FACE,
        }

    def cancel_face(self) -> dict[str, Any]:
        if self._session is None:
            return {"ok": False, "error": "没有进行中的人脸注册会话"}
        session = self._session
        name = session.name
        for value in session.captured_paths:
            path = _from_registry_path(value)
            if path is not None:
                path.unlink(missing_ok=True)
        directory = _FACES_DIR / name
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
        self._session = None
        return {"ok": True, "name": name, "cancelled": True}

    def _detect_faces(self, frame: np.ndarray) -> list[dict[str, Any]]:
        if self._face_detector is None:
            return []
        try:
            height, width = frame.shape[:2]
            self._face_detector.setInputSize((width, height))
            _, detections = self._face_detector.detect(frame)
            if detections is None:
                return []
            return [
                {
                    "x": float(item[0]) / width,
                    "y": float(item[1]) / height,
                    "w": float(item[2]) / width,
                    "h": float(item[3]) / height,
                    "confidence": float(item[-1]),
                    "landmarks": [
                        [float(item[index]) / width, float(item[index + 1]) / height]
                        for index in range(4, 14, 2)
                    ],
                }
                for item in detections
                if float(item[-1]) >= 0.3
            ]
        except Exception:
            return []

    @staticmethod
    def list_enrolled_faces() -> list[str]:
        with _STORAGE_LOCK:
            return _known_face_names()

    @staticmethod
    def get_face_paths(name: str) -> list[str]:
        try:
            normalized = validate_face_identity(name)
        except ValueError:
            return []
        with _STORAGE_LOCK:
            directory = _FACES_DIR / normalized
            return [
                str(directory / f"{sample_id:03d}.jpg")
                for sample_id in _face_sample_ids(directory)
            ]

    def list_face_records(self) -> dict[str, Any]:
        with _STORAGE_LOCK:
            registry = _load_registry()
            records: list[dict[str, Any]] = []
            for name in _known_face_names(registry):
                metadata = registry["faces"].get(name, {})
                sample_ids = _face_sample_ids(_FACES_DIR / name)
                records.append({
                    "name": name,
                    "display_name": face_identity_display_name(name),
                    "role": face_identity_role(name),
                    "shots": len(sample_ids),
                    "sample_ids": sample_ids,
                    "samples_url": f"/api/v1/faces/{name}/samples",
                    "enrolled_at": float(metadata.get("enrolled_at", 0.0)),
                    "updated_at": float(metadata.get("updated_at", 0.0)),
                    "ready": bool(sample_ids),
                })
            occupied = {record["name"] for record in records}
            return {
                "ok": True,
                "count": len(records),
                "max_faces": MAX_FACES,
                "max_samples_per_face": MAX_SAMPLES_PER_FACE,
                "allowed_names": list(ALLOWED_FACE_IDENTITIES),
                "available_names": [
                    name for name in ALLOWED_FACE_IDENTITIES
                    if name not in occupied
                ],
                "faces": records,
            }

    def list_face_samples(self, name: str) -> dict[str, Any]:
        try:
            normalized = validate_face_identity(name)
        except ValueError as exc:
            return {"ok": False, "status": 422, "error": str(exc)}
        with _STORAGE_LOCK:
            directory = _FACES_DIR / normalized
            sample_ids = _face_sample_ids(directory)
            if not sample_ids:
                return {"ok": False, "status": 404, "error": "face not found"}
            return {
                "ok": True,
                "name": normalized,
                "display_name": face_identity_display_name(normalized),
                "role": face_identity_role(normalized),
                "shots": len(sample_ids),
                "max_samples_per_face": MAX_SAMPLES_PER_FACE,
                "sample_ids": sample_ids,
                "samples": [
                    _sample_record(normalized, sample_id)
                    for sample_id in sample_ids
                ],
            }

    def get_face_sample(self, name: str, sample_id: int) -> dict[str, Any]:
        try:
            normalized = validate_face_identity(name)
            normalized_sample_id = _validate_sample_id(sample_id)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "status": 422, "error": str(exc)}
        with _STORAGE_LOCK:
            if normalized_sample_id not in _face_sample_ids(
                _FACES_DIR / normalized
            ):
                return {
                    "ok": False,
                    "status": 404,
                    "error": "face sample not found",
                }
            return {
                "ok": True,
                "name": normalized,
                "display_name": face_identity_display_name(normalized),
                "role": face_identity_role(normalized),
                **_sample_record(normalized, normalized_sample_id),
            }

    def replace_face_sample(
        self,
        name: str,
        sample_id: int,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        try:
            normalized = validate_face_identity(name)
            normalized_sample_id = _validate_sample_id(sample_id)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "status": 422, "error": str(exc)}
        with _STORAGE_LOCK:
            directory = _FACES_DIR / normalized
            if normalized_sample_id not in _face_sample_ids(directory):
                return {
                    "ok": False,
                    "status": 404,
                    "error": "face sample not found",
                }
        prepared = self._prepare_uploaded_face(image_bytes)
        if not prepared.get("ok", False):
            return prepared
        with _STORAGE_LOCK:
            if normalized_sample_id not in _face_sample_ids(directory):
                return {
                    "ok": False,
                    "status": 404,
                    "error": "face sample not found",
                }
            path = directory / f"{normalized_sample_id:03d}.jpg"
            self._write_sample_atomic(path, bytes(prepared["jpeg_bytes"]))
            registry = _load_registry()
            sample_ids = _update_registry_identity(registry, normalized)
            _save_registry(registry)
        return {
            "ok": True,
            "name": normalized,
            "display_name": face_identity_display_name(normalized),
            "face_role": face_identity_role(normalized),
            "shots": len(sample_ids),
            "sample_id": normalized_sample_id,
            "sample_key": f"{normalized_sample_id:03d}",
            "image_path": str(path),
            "replaced": True,
        }

    def delete_face_sample(self, name: str, sample_id: int) -> dict[str, Any]:
        try:
            normalized = validate_face_identity(name)
            normalized_sample_id = _validate_sample_id(sample_id)
        except (TypeError, ValueError) as exc:
            return {"ok": False, "status": 422, "error": str(exc)}
        with _STORAGE_LOCK:
            directory = _FACES_DIR / normalized
            if normalized_sample_id not in _face_sample_ids(directory):
                return {
                    "ok": False,
                    "status": 404,
                    "error": "face sample not found",
                }
            path = directory / f"{normalized_sample_id:03d}.jpg"
            path.unlink(missing_ok=True)
            registry = _load_registry()
            remaining_ids = _update_registry_identity(registry, normalized)
            face_removed = not remaining_ids
            if face_removed and directory.is_dir():
                shutil.rmtree(directory)
            _save_registry(registry)
        return {
            "ok": True,
            "name": normalized,
            "display_name": face_identity_display_name(normalized),
            "face_role": face_identity_role(normalized),
            "deleted_sample_id": normalized_sample_id,
            "deleted_sample_key": f"{normalized_sample_id:03d}",
            "shots": len(remaining_ids),
            "remaining_sample_ids": remaining_ids,
            "face_removed": face_removed,
        }

    @staticmethod
    def delete_face(name: str) -> dict[str, Any]:
        try:
            normalized = validate_face_identity(name)
        except ValueError as exc:
            return {"ok": False, "status": 422, "error": str(exc)}
        with _STORAGE_LOCK:
            target = _FACES_DIR / normalized
            if not target.exists():
                return {"ok": False, "status": 404, "error": "face not found"}
            shutil.rmtree(target)
            registry = _load_registry()
            registry["faces"].pop(normalized, None)
            _save_registry(registry)
        return {"ok": True, "name": normalized}


# Provider compatibility while the migrated code is being stabilized.
EnrollmentManager = FaceEnrollmentManager
