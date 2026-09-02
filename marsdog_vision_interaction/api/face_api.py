"""FastAPI server for sample-level face image management."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path as FilePath
from typing import Any

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Path,
    Response,
    UploadFile,
)
from marsdog_vision_interaction.messages.face_identity import FaceIdentity


logger = logging.getLogger(__name__)


class FaceApiServer:
    """Run a bounded FastAPI server in the ROS process."""

    def __init__(
        self,
        config: dict[str, Any],
        upload_handler: Callable[[str, bytes], dict[str, Any]],
        list_handler: Callable[[], dict[str, Any]],
        sample_list_handler: Callable[[str], dict[str, Any]],
        sample_get_handler: Callable[[str, int], dict[str, Any]],
        sample_replace_handler: Callable[[str, int, bytes], dict[str, Any]],
        sample_delete_handler: Callable[[str, int], dict[str, Any]],
    ) -> None:
        self._config = dict(config)
        self._upload_handler = upload_handler
        self._list_handler = list_handler
        self._sample_list_handler = sample_list_handler
        self._sample_get_handler = sample_get_handler
        self._sample_replace_handler = sample_replace_handler
        self._sample_delete_handler = sample_delete_handler
        self._enabled = bool(config.get("enabled", True))
        self._host = str(config.get("host", "127.0.0.1")).strip()
        self._port = int(config.get("port", 8092))
        self._max_upload_bytes = max(
            1,
            int(float(config.get("max_upload_mb", 10.0)) * 1024 * 1024),
        )
        self._server: Any = None
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def address(self) -> str:
        display_host = (
            "127.0.0.1" if self._host in ("0.0.0.0", "::") else self._host
        )
        return f"http://{display_host}:{self._port}"

    def _validate_bind(self) -> None:
        if not self._host:
            raise RuntimeError("face_api.host 不能为空")

    def start(self) -> bool:
        if not self._enabled:
            return False
        self._validate_bind()

        import uvicorn

        uvicorn_config = uvicorn.Config(
            self.create_app(),
            host=self._host,
            port=self._port,
            log_level="info",
            log_config=None,
            access_log=True,
        )
        self._server = uvicorn.Server(uvicorn_config)
        self._thread = threading.Thread(
            target=self._server.run,
            name="face-fastapi",
            daemon=True,
        )
        self._thread.start()
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            if bool(getattr(self._server, "started", False)):
                logger.info("Face FastAPI ready: %s/docs", self.address)
                return True
            if not self._thread.is_alive():
                break
            time.sleep(0.02)
        self.stop()
        raise RuntimeError(f"人脸 FastAPI 启动失败: {self.address}")

    def stop(self) -> None:
        server = self._server
        if server is not None:
            server.should_exit = True
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=3.0)
        self._thread = None
        self._server = None

    def create_app(self) -> Any:
        app = FastAPI(
            title="MarsDog Vision Face API",
            version="1.0.0",
            description=(
                "Manage individual face image samples in five fixed owner/family "
                "identity slots. Each identity accepts at most five samples."
            ),
        )
        def raise_for_result(result: dict[str, Any]) -> None:
            if result.get("ok", False):
                return
            error = str(result.get("error", "face operation failed"))
            configured_status = int(result.get("status", 0))
            if 400 <= configured_status <= 599:
                status = configured_status
            else:
                status = 503 if "不可用" in error else 422
            raise HTTPException(status_code=status, detail=error)

        async def read_image(image: UploadFile) -> bytes:
            filename = str(image.filename or "")
            suffix = FilePath(filename).suffix.lower()
            if suffix and suffix not in (".jpg", ".jpeg", ".png"):
                raise HTTPException(
                    status_code=415,
                    detail="only JPG, JPEG and PNG are supported",
                )
            chunks: list[bytes] = []
            total = 0
            try:
                while True:
                    chunk = await image.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self._max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail="uploaded image is too large",
                        )
                    chunks.append(chunk)
            finally:
                await image.close()
            if not chunks:
                raise HTTPException(status_code=400, detail="image file is empty")
            return b"".join(chunks)

        @app.get("/health")
        async def health() -> dict[str, Any]:
            return {"ok": True, "service": "marsdog-vision-face-api"}

        @app.post(
            "/api/v1/faces/{name}/samples",
            status_code=201,
        )
        async def add_face_sample(
            image: UploadFile = File(...),
            name: FaceIdentity = Path(
                ...,
                description="owner 或 family_member_1 到 family_member_4",
            ),
        ) -> dict[str, Any]:
            payload = await read_image(image)
            result = self._upload_handler(name.value, payload)
            raise_for_result(result)
            return {"request_id": uuid.uuid4().hex, **result}

        @app.get("/api/v1/faces")
        async def list_faces() -> dict[str, Any]:
            result = self._list_handler()
            raise_for_result(result)
            return result

        @app.get("/api/v1/faces/{name}/samples")
        async def list_face_samples(
            name: FaceIdentity = Path(...),
        ) -> dict[str, Any]:
            result = self._sample_list_handler(name.value)
            raise_for_result(result)
            return result

        @app.get("/api/v1/faces/{name}/samples/{sample_id}")
        async def get_face_sample(
            name: FaceIdentity = Path(...),
            sample_id: int = Path(..., ge=1, le=5),
        ) -> dict[str, Any]:
            result = self._sample_get_handler(name.value, sample_id)
            raise_for_result(result)
            return result

        @app.get("/api/v1/faces/{name}/samples/{sample_id}/image")
        async def download_face_sample_image(
            name: FaceIdentity = Path(...),
            sample_id: int = Path(..., ge=1, le=5),
        ) -> Any:
            result = self._sample_get_handler(name.value, sample_id)
            raise_for_result(result)
            if not bool(result.get("image_available", False)):
                raise HTTPException(status_code=404, detail="face sample not found")
            try:
                payload = FilePath(str(result["image_path"])).read_bytes()
            except OSError as exc:
                raise HTTPException(
                    status_code=404,
                    detail="face sample image not found",
                ) from exc
            return Response(
                content=payload,
                media_type="image/jpeg",
                headers={
                    "Content-Disposition": (
                        "inline; filename="
                        f'"{name.value}_{sample_id:03d}.jpg"'
                    ),
                },
            )

        @app.put("/api/v1/faces/{name}/samples/{sample_id}")
        async def replace_face_sample(
            image: UploadFile = File(...),
            name: FaceIdentity = Path(...),
            sample_id: int = Path(..., ge=1, le=5),
        ) -> dict[str, Any]:
            payload = await read_image(image)
            result = self._sample_replace_handler(name.value, sample_id, payload)
            raise_for_result(result)
            return {"request_id": uuid.uuid4().hex, **result}

        @app.delete("/api/v1/faces/{name}/samples/{sample_id}")
        async def delete_face_sample(
            name: FaceIdentity = Path(...),
            sample_id: int = Path(..., ge=1, le=5),
        ) -> dict[str, Any]:
            result = self._sample_delete_handler(name.value, sample_id)
            raise_for_result(result)
            return {"request_id": uuid.uuid4().hex, **result}

        return app
