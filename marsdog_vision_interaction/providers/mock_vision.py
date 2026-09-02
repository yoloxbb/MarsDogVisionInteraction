"""Mock vision provider for Phase 1.

Produces stable, predictable mock observations and object detection results
when no real camera or vision model is available.
"""

from __future__ import annotations

import logging
import copy
from typing import Any

from marsdog_vision_interaction.fusion.stereo_fusion import get_target_manager
from marsdog_vision_interaction.providers.base import BaseProvider

logger = logging.getLogger(__name__)


class MockVisionProvider(BaseProvider):
    """Mock vision provider that returns a standing human and a red ball."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._last_observation: dict[str, Any] = {}

    def start(self) -> None:
        try:
            logger.info("MockVisionProvider starting — camera_id mock")
            self.available = True
            logger.info("MockVisionProvider started (mock)")
        except Exception as exc:
            self.available = False
            logger.warning(
                "MockVisionProvider start failed (unexpected): %s",
                exc,
                exc_info=True,
            )

    def stop(self) -> None:
        self.available = False
        logger.info("MockVisionProvider stopped")

    def get_observation(self) -> dict[str, Any]:
        """Return a mock observation with one standing human."""
        obs = {
            "faces": [],
            "humans": [
                {
                    "x": 0.25,
                    "y": 0.1,
                    "w": 0.4,
                    "h": 0.8,
                    "confidence": 0.8,
                    "pose_state": "standing",
                    "keypoints": [],
                }
            ],
            "hands": [],
            "tracked_objects": [],
        }
        manager = get_target_manager()
        manager.update_vision(obs["humans"], obs["faces"])
        snapshot = manager.get_snapshot()
        active = snapshot["active_target"]
        active_dict = active.to_dict()
        obs.update({
            "vision_epoch": snapshot["vision_epoch"],
            "active_target": active_dict,
            "human_candidates": snapshot["human_candidates"],
        })
        if active.track_id > 0:
            obs["humans"][0]["track_id"] = active.track_id
        self._last_observation = obs
        return copy.deepcopy(obs)

    def detect_objects(self, params: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Return mock object detection — a red ball.

        Args:
            params: Optional detection parameters (unused in mock).

        Returns:
            List of detected object dicts.
        """
        _ = params
        logger.debug("MockVisionProvider detect_objects called — returning mock red ball")
        return [
            {
                "label": "red ball",
                "x": 0.3,
                "y": 0.45,
                "w": 0.12,
                "h": 0.12,
                "confidence": 0.88,
                "center_x": 0.36,
                "center_y": 0.51,
            }
        ]

    def check_person(self) -> dict[str, Any]:
        """Check if a person is present based on last observation.

        Returns:
            Dict with present and count keys.
        """
        obs = self._last_observation or self.get_observation()
        candidates = [
            item
            for item in obs.get("human_candidates", [])
            if item.get("tracking_state") == "tracking"
        ]
        humans = obs.get("humans", [])
        count = len(candidates) if candidates else len(humans)
        return {
            "present": count > 0,
            "count": count,
            "identity": str(
                obs.get("active_target", {}).get("identity", "unknown")
            ),
            "target": dict(obs.get("active_target", {})),
        }

    def enroll_face(self, face_image: Any = None) -> dict[str, Any]:
        """Mock face enrollment — always succeeds."""
        _ = face_image
        logger.info("MockVisionProvider enroll_face — mock success")
        return {"success": True, "user_id": "mock_user_001"}

    def recognize_face(self, face_image: Any = None) -> dict[str, Any]:
        """Mock face recognition — returns unknown."""
        _ = face_image
        logger.debug("MockVisionProvider recognize_face — returning unknown")
        return {"user_id": "unknown", "confidence": 0.0}
