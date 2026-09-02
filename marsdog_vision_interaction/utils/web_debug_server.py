"""Small dependency-free HTTP server for the vision debug dashboard."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
import json
import logging
import threading
from typing import Any, Callable
from urllib.parse import urlsplit


logger = logging.getLogger(__name__)


class _DashboardHttpServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class VisionDebugWebServer:
    """Serve one dashboard, a JSON state endpoint and an MJPEG stream."""

    def __init__(
        self,
        host: str,
        port: int,
        snapshot_provider: Callable[[], dict[str, Any]],
        task_handler: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        event_history_clear_handler: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.host = str(host)
        self.port = int(port)
        self._snapshot_provider = snapshot_provider
        self._task_handler = task_handler
        self._event_history_clear_handler = event_history_clear_handler
        self._condition = threading.Condition()
        self._jpeg = b""
        self._frame_sequence = 0
        self._stopping = threading.Event()
        self._httpd: _DashboardHttpServer | None = None
        self._thread: threading.Thread | None = None
        self._html = (
            files("marsdog_vision_interaction.web")
            .joinpath("dashboard.html")
            .read_bytes()
        )

    @property
    def bound_port(self) -> int:
        if self._httpd is None:
            return self.port
        return int(self._httpd.server_address[1])

    def start(self) -> None:
        if self._httpd is not None:
            return
        self._stopping.clear()
        self._httpd = _DashboardHttpServer(
            (self.host, self.port), self._make_handler()
        )
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="vision-debug-web",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stopping.set()
        with self._condition:
            self._condition.notify_all()
        httpd = self._httpd
        self._httpd = None
        if httpd is not None:
            try:
                httpd.shutdown()
            except KeyboardInterrupt:
                # ROS launch can relay SIGINT while node teardown is already
                # in progress.  The listening socket must still be closed.
                pass
            finally:
                httpd.server_close()
        if self._thread is not None:
            try:
                self._thread.join(timeout=2.0)
            except KeyboardInterrupt:
                pass
            self._thread = None

    def update_jpeg(self, value: bytes) -> None:
        if not value:
            return
        with self._condition:
            self._jpeg = bytes(value)
            self._frame_sequence += 1
            self._condition.notify_all()

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
                path = urlsplit(self.path).path
                if path in ("/", "/index.html"):
                    self._send_bytes("text/html; charset=utf-8", owner._html)
                elif path == "/api/state":
                    self._send_state()
                elif path == "/healthz":
                    self._send_bytes("text/plain; charset=utf-8", b"ok\n")
                elif path == "/stream.mjpg":
                    self._send_stream()
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
                path = urlsplit(self.path).path
                if path == "/api/debug/event-history/clear":
                    if owner._event_history_clear_handler is None:
                        self.send_error(HTTPStatus.NOT_FOUND)
                        return
                    try:
                        result = owner._event_history_clear_handler()
                        self._send_json_result(result)
                    except Exception as exc:
                        logger.warning("Event history clear failed: %s", exc)
                        self._send_json_result(
                            {"ok": False, "error": str(exc)},
                            status=HTTPStatus.INTERNAL_SERVER_ERROR,
                        )
                    return
                if path != "/api/vision/task" or owner._task_handler is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 2 * 1024 * 1024:
                        raise ValueError("invalid request size")
                    value = json.loads(self.rfile.read(length))
                    if not isinstance(value, dict):
                        raise ValueError("request must be a JSON object")
                    task_type = str(value.get("task_type", ""))
                    params = value.get("params", {})
                    if not isinstance(params, dict):
                        raise ValueError("params must be a JSON object")
                    result = owner._task_handler(task_type, params)
                    status = (
                        HTTPStatus.OK
                        if result.get("ok", False)
                        else HTTPStatus.BAD_REQUEST
                    )
                    self._send_json_result(result, status=status)
                except (ValueError, json.JSONDecodeError) as exc:
                    payload = json.dumps(
                        {"ok": False, "error": str(exc)}, ensure_ascii=False
                    ).encode("utf-8")
                    self._send_bytes(
                        "application/json; charset=utf-8",
                        payload,
                        status=HTTPStatus.BAD_REQUEST,
                    )

            def _send_json_result(
                self,
                result: dict[str, Any],
                *,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                payload = json.dumps(
                    result, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                self._send_bytes(
                    "application/json; charset=utf-8", payload, status=status
                )

            def _send_state(self) -> None:
                try:
                    value = owner._snapshot_provider()
                    payload = json.dumps(
                        value,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        default=str,
                    ).encode("utf-8")
                    self._send_bytes(
                        "application/json; charset=utf-8", payload
                    )
                except Exception as exc:  # defensive boundary for debug server
                    logger.warning("Vision debug snapshot failed: %s", exc)
                    payload = json.dumps(
                        {"ok": False, "error": str(exc)},
                        ensure_ascii=False,
                    ).encode("utf-8")
                    self._send_bytes(
                        "application/json; charset=utf-8",
                        payload,
                        status=HTTPStatus.INTERNAL_SERVER_ERROR,
                    )

            def _send_bytes(
                self,
                content_type: str,
                payload: bytes,
                *,
                status: HTTPStatus = HTTPStatus.OK,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(payload)

            def _send_stream(self) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.send_header("Cache-Control", "no-store, max-age=0")
                self.end_headers()
                sequence = -1
                try:
                    while not owner._stopping.is_set():
                        with owner._condition:
                            owner._condition.wait_for(
                                lambda: (
                                    owner._frame_sequence != sequence
                                    or owner._stopping.is_set()
                                ),
                                timeout=1.0,
                            )
                            if owner._stopping.is_set():
                                return
                            jpeg = owner._jpeg
                            sequence = owner._frame_sequence
                        if not jpeg:
                            continue
                        self.wfile.write(b"--frame\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            f"Content-Length: {len(jpeg)}\r\n\r\n".encode()
                        )
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                except OSError:
                    return

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug("Vision debug HTTP: " + format, *args)

        return Handler
