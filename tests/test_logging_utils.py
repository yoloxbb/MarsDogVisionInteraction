import json
import io
import logging

from marsdog_vision_interaction.utils import logging_utils


def test_vision_trace_is_correlated_json(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(logging_utils, "time_monotonic_ms", lambda: 1234.5)
    logging_utils.configure_event_trace(
        enabled=True,
        log_dir=str(tmp_path),
        run_id="run-01",
        case_id="STOP-01-r1",
    )

    logging_utils.vision_trace(
        "event_publish",
        result="published",
        event_type="EVT_VISION_STOP_GESTURE",
        latency_ms=12.3,
    )
    logging.shutdown()

    trace_file = next(tmp_path.glob("vision_trace_*.jsonl"))
    line = trace_file.read_text(encoding="utf-8").strip()
    assert line.startswith("VISION_TRACE ")
    payload = json.loads(line.removeprefix("VISION_TRACE "))
    assert payload["schema_version"] == 1
    assert payload["record"] == "event_publish"
    assert payload["run_id"] == "run-01"
    assert payload["case_id"] == "STOP-01-r1"
    assert payload["monotonic_ms"] == 1234.5
    assert payload["event_type"] == "EVT_VISION_STOP_GESTURE"


def test_disabled_vision_trace_creates_no_file(tmp_path) -> None:
    logging_utils.configure_event_trace(enabled=False, log_dir=str(tmp_path))
    logging_utils.vision_trace("runtime_start", result="ready")
    assert list(tmp_path.iterdir()) == []


def test_continuous_timing_trace_is_rate_limited_but_failures_are_not(
    tmp_path, monkeypatch,
) -> None:
    monkeypatch.setattr(logging_utils, "time_monotonic_ms", lambda: 1000.0)
    logging_utils.configure_event_trace(
        enabled=True,
        log_dir=str(tmp_path),
        timing_interval_sec=5.0,
    )

    assert logging_utils.vision_timing_trace(
        node="vision_observation",
        module="pose_landmarker",
        stage="inference",
        latency_ms=12.3456,
    )
    assert not logging_utils.vision_timing_trace(
        node="vision_observation",
        module="pose_landmarker",
        stage="inference",
        latency_ms=13.0,
    )
    assert logging_utils.vision_timing_trace(
        node="vision_observation",
        module="pose_landmarker",
        stage="inference",
        latency_ms=1.0,
        result="failure",
    )
    logging.shutdown()

    trace_file = next(tmp_path.glob("vision_trace_*.jsonl"))
    payloads = [
        json.loads(line.removeprefix("VISION_TRACE "))
        for line in trace_file.read_text(encoding="utf-8").splitlines()
    ]
    assert len(payloads) == 2
    assert payloads[0]["record"] == "stage_complete"
    assert payloads[0]["latency_ms"] == 12.346
    assert payloads[0]["sampled"] is True
    assert payloads[1]["result"] == "failure"
    assert payloads[1]["sampled"] is False


def test_structured_logger_preserves_standard_extra_formatting() -> None:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging_utils.StructuredLogger("uvicorn-compat")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    logger.info(
        "Started server process [%d]",
        148848,
        extra={"color_message": "Started server process [\x1b[36m%d\x1b[0m]"},
    )

    assert stream.getvalue().strip() == "Started server process [148848]"


def test_structured_logger_supports_custom_and_standard_kwargs() -> None:
    stream = io.StringIO()
    logger = logging_utils.StructuredLogger("mixed-kwargs")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(stream))

    logger.info("camera_init", device="0", width=640, stacklevel=1)

    assert stream.getvalue().strip() == "camera_init  device='0'  width=640"
