from __future__ import annotations

import asyncio
import json
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest

import marsdog_vision_interaction.core.face_enrollment_manager as storage
from marsdog_vision_interaction.api.face_api import FaceApiServer
from marsdog_vision_interaction.core.face_enrollment_manager import (
    FaceEnrollmentManager,
    set_storage_root,
)
from marsdog_vision_interaction.messages.face_identity import (
    ALLOWED_FACE_IDENTITIES,
)


class _FaceDetector:
    def setInputSize(self, size) -> None:
        self.size = size

    def detect(self, frame):
        height, width = frame.shape[:2]
        detection = np.zeros((1, 15), dtype=np.float32)
        detection[0, :4] = (0.0, 0.0, float(width), float(height))
        detection[0, -1] = 0.99
        return None, detection


@pytest.fixture
def face_manager(tmp_path: Path):  # type: ignore[no-untyped-def]
    original = (
        storage._STORAGE_ROOT,
        storage._FACES_DIR,
        storage._REGISTRY_PATH,
    )
    set_storage_root(tmp_path)
    manager = FaceEnrollmentManager()
    manager.set_face_detector(_FaceDetector())
    try:
        yield manager
    finally:
        storage._STORAGE_ROOT = original[0]
        storage._FACES_DIR = original[1]
        storage._REGISTRY_PATH = original[2]


def _image(level: int = 64) -> bytes:
    image = np.full((120, 120, 3), level, dtype=np.uint8)
    ok, payload = cv2.imencode(".jpg", image)
    assert ok
    return payload.tobytes()


def _request(
    server: FaceApiServer,
    method: str,
    path: str,
    **kwargs: object,
) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=server.create_app())
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(request())


def _server(manager: FaceEnrollmentManager) -> FaceApiServer:
    return FaceApiServer(
        {"enabled": True, "max_upload_mb": 1},
        manager.enroll_face_from_image,
        manager.list_face_records,
        manager.list_face_samples,
        manager.get_face_sample,
        manager.replace_face_sample,
        manager.delete_face_sample,
    )


def test_fixed_five_identities_and_five_stable_samples(
    face_manager: FaceEnrollmentManager,
) -> None:
    for expected_id in range(1, 6):
        result = face_manager.enroll_face_from_image("owner", _image(expected_id))
        assert result["ok"] is True
        assert result["sample_id"] == expected_id

    rejected = face_manager.enroll_face_from_image("owner", b"not-an-image")
    invalid_identity = face_manager.enroll_face_from_image("friend", _image())

    assert rejected["status"] == 409
    assert rejected["code"] == "face_sample_limit_reached"
    assert invalid_identity["status"] == 422
    assert face_manager.list_face_records()["allowed_names"] == list(
        ALLOWED_FACE_IDENTITIES
    )


def test_sample_delete_does_not_renumber_and_add_reuses_smallest_gap(
    face_manager: FaceEnrollmentManager,
) -> None:
    for level in (30, 60, 90):
        assert face_manager.enroll_face_from_image("owner", _image(level))["ok"]

    deleted = face_manager.delete_face_sample("owner", 2)
    listed = face_manager.list_face_samples("owner")
    added = face_manager.enroll_face_from_image("owner", _image(120))

    assert deleted["remaining_sample_ids"] == [1, 3]
    assert listed["sample_ids"] == [1, 3]
    assert added["sample_id"] == 2
    assert face_manager.list_face_samples("owner")["sample_ids"] == [1, 2, 3]


def test_replace_is_in_place_and_last_delete_releases_identity(
    face_manager: FaceEnrollmentManager,
) -> None:
    created = face_manager.enroll_face_from_image("family_member_1", _image(20))
    replaced = face_manager.replace_face_sample(
        "family_member_1", created["sample_id"], _image(180)
    )
    deleted = face_manager.delete_face_sample("family_member_1", 1)

    assert replaced["sample_id"] == 1
    assert replaced["replaced"] is True
    assert deleted["face_removed"] is True
    assert face_manager.list_face_records()["count"] == 0


def test_camera_enrollment_obeys_remaining_sample_capacity(
    face_manager: FaceEnrollmentManager,
) -> None:
    for level in (30, 60, 90, 120):
        assert face_manager.enroll_face_from_image("owner", _image(level))["ok"]

    rejected = face_manager.start_face("owner", required_shots=2)
    admitted = face_manager.start_face("owner", required_shots=1)

    assert rejected["status"] == 409
    assert rejected["available_slots"] == 1
    assert admitted["ok"] is True
    assert face_manager.face_session is not None
    assert face_manager.face_session.planned_sample_ids == [5]


def test_registry_ignores_but_does_not_delete_legacy_free_names(
    face_manager: FaceEnrollmentManager,
    tmp_path: Path,
) -> None:
    legacy_dir = tmp_path / "faces" / "legacy-name"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "001.jpg").write_bytes(_image())
    (tmp_path / "face_registry.json").write_text(
        json.dumps({
            "schema_version": 1,
            "faces": {
                "legacy-name": {"paths": ["faces/legacy-name/001.jpg"]},
            },
        }),
        encoding="utf-8",
    )

    assert face_manager.list_enrolled_faces() == []
    assert legacy_dir.exists()


def test_face_api_without_token_exposes_sample_crud(
    face_manager: FaceEnrollmentManager,
) -> None:
    server = _server(face_manager)
    assert _request(server, "GET", "/api/v1/faces").status_code == 200

    created = _request(
        server,
        "POST",
        "/api/v1/faces/owner/samples",
        files={"image": ("owner.jpg", _image(), "image/jpeg")},
    )
    listed = _request(
        server,
        "GET",
        "/api/v1/faces/owner/samples",
    )
    downloaded = _request(
        server,
        "GET",
        "/api/v1/faces/owner/samples/1/image",
    )
    replaced = _request(
        server,
        "PUT",
        "/api/v1/faces/owner/samples/1",
        files={"image": ("replacement.png", _image(150), "image/png")},
    )
    deleted = _request(
        server,
        "DELETE",
        "/api/v1/faces/owner/samples/1",
    )

    assert created.status_code == 201
    assert created.json()["sample_id"] == 1
    assert listed.json()["sample_ids"] == [1]
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"] == "image/jpeg"
    assert replaced.json()["replaced"] is True
    assert deleted.json()["face_removed"] is True


def test_remote_bind_without_token_is_allowed(
    face_manager: FaceEnrollmentManager,
) -> None:
    server = _server(face_manager)
    server._host = "0.0.0.0"
    server._validate_bind()
