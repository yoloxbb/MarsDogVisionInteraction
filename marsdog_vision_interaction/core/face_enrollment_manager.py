"""Face enrollment and visual identity storage."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


_STORAGE_ROOT = Path("data")
_FACES_DIR = _STORAGE_ROOT / "faces"
_REGISTRY_PATH = _STORAGE_ROOT / "face_registry.json"


def set_storage_root(path: str | Path) -> None:
    global _STORAGE_ROOT, _FACES_DIR, _REGISTRY_PATH
    _STORAGE_ROOT = Path(path)
    _FACES_DIR = _STORAGE_ROOT / "faces"
    _REGISTRY_PATH = _STORAGE_ROOT / "face_registry.json"
    _FACES_DIR.mkdir(parents=True, exist_ok=True)


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
    done: bool = False


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
        name = name.strip()
        if not name:
            return {"ok": False, "error": "名称不能为空"}
        self._session = FaceEnrollment(
            name=name,
            required_shots=max(1, int(required_shots)),
            started_at=time.time(),
        )
        return {
            "ok": True,
            "name": name,
            "step": 1,
            "total_steps": self._session.required_shots,
            "prompt": "请正对摄像头并保持稳定",
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
                "prompt": "未检测到人脸，请正对摄像头",
                "done": False,
            }

        best = max(faces, key=lambda face: face["w"] * face["h"])
        session.framed_count += 1
        if session.framed_count < session.framed_required:
            return {
                "ok": True,
                "name": session.name,
                "step": session.current_step,
                "total_steps": session.required_shots,
                "status": "tracking",
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
        path = face_dir / f"{session.shots_collected + 1:03d}.jpg"
        cv2.imwrite(str(path), frame[y1:y2, x1:x2])
        session.captured_paths.append(str(path))
        session.shots_collected += 1
        session.framed_count = 0

        if session.shots_collected >= session.required_shots:
            session.done = True
            registry = _load_registry()
            registry["faces"][session.name] = {
                "paths": list(session.captured_paths),
                "enrolled_at": time.time(),
            }
            _save_registry(registry)
            return {
                "ok": True,
                "name": session.name,
                "step": session.required_shots,
                "total_steps": session.required_shots,
                "status": "done",
                "shots": session.shots_collected,
                "done": True,
            }

        session.current_step = session.shots_collected + 1
        return {
            "ok": True,
            "name": session.name,
            "step": session.current_step,
            "total_steps": session.required_shots,
            "status": "captured",
            "shots": session.shots_collected,
            "done": False,
        }

    def enroll_face_from_image(
        self,
        name: str,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        name = name.strip()
        if not name:
            return {"ok": False, "error": "名称不能为空"}
        import cv2

        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            return {"ok": False, "error": "无法解码图片"}
        faces = self._detect_faces(image)
        if not faces:
            return {"ok": False, "error": "未检测到清晰人脸"}
        best = max(faces, key=lambda face: face["w"] * face["h"])
        height, width = image.shape[:2]
        x1 = max(0, int(best["x"] * width))
        y1 = max(0, int(best["y"] * height))
        x2 = min(width, int((best["x"] + best["w"]) * width))
        y2 = min(height, int((best["y"] + best["h"]) * height))

        face_dir = _FACES_DIR / name
        face_dir.mkdir(parents=True, exist_ok=True)
        path = face_dir / f"{len(list(face_dir.glob('*.jpg'))) + 1:03d}.jpg"
        cv2.imwrite(str(path), image[y1:y2, x1:x2])

        registry = _load_registry()
        paths = [str(item) for item in sorted(face_dir.glob("*.jpg"))]
        registry["faces"][name] = {
            "paths": paths,
            "enrolled_at": time.time(),
        }
        _save_registry(registry)
        return {"ok": True, "name": name, "shots": len(paths), "path": str(path)}

    def cancel_face(self) -> dict[str, Any]:
        if self._session is None:
            return {"ok": False, "error": "没有进行中的人脸注册会话"}
        name = self._session.name
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
                }
                for item in detections
                if float(item[-1]) >= 0.3
            ]
        except Exception:
            return []

    @staticmethod
    def list_enrolled_faces() -> list[str]:
        return sorted(_load_registry()["faces"])

    @staticmethod
    def get_face_paths(name: str) -> list[str]:
        return list(
            _load_registry()["faces"].get(name, {}).get("paths", [])
        )

    @staticmethod
    def delete_face(name: str) -> dict[str, Any]:
        name = name.strip()
        target = _FACES_DIR / name
        if not name or not target.exists():
            return {"ok": False, "status": 404, "error": "face not found"}
        shutil.rmtree(target)
        registry = _load_registry()
        registry["faces"].pop(name, None)
        _save_registry(registry)
        return {"ok": True, "name": name}


# Provider compatibility while the migrated code is being stabilized.
EnrollmentManager = FaceEnrollmentManager
