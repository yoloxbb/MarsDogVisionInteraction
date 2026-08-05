"""Pose and hand action classifier — mock implementation.

Defines action labels for body pose and hand gestures grouped by emotional state.
Real implementation will classify actions from keypoint sequences.
Mock: randomly triggers actions at low frequency with persistence + cooldown.

Action groups:
  - happy:   positive/celebratory body language
  - angry:   confrontational/dominant gestures
  - sad:     withdrawn/depressed posture
  - neutral: normal idle behavior
  - alert:   emergency/stop signals
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Action definitions ──────────────────────────────────────────────

# Each action: (key, label_cn, category)
# category: "pose" = body-level, "hand" = hand gesture

_HAPPY_ACTIONS: list[tuple[str, str, str]] = [
    ("arm_raise_wave",    "手臂高举/挥舞",    "pose"),
    ("jump",              "跳跃",             "pose"),
    ("lean_forward_arms_open", "身体前倾/张开双臂", "pose"),
    ("nodding",           "快速点头",          "pose"),
    ("clapping",          "双手鼓掌",          "hand"),
    ("thumbs_up",         "点赞手势",          "hand"),
]

_ANGRY_ACTIONS: list[tuple[str, str, str]] = [
    ("hands_on_hips",     "双手叉腰",          "pose"),
    ("rapid_wave_slap",   "快速挥手/拍打",      "hand"),
    ("finger_pointing",   "用手指点",          "hand"),
    ("stomping",          "急促跺脚",          "pose"),
    ("arms_crossed",      "双臂交叉于胸前",     "pose"),
]

_SAD_ACTIONS: list[tuple[str, str, str]] = [
    ("head_down_slumped", "低头/肩膀下垂",     "pose"),
    ("hands_covering_face", "双手掩面或抱头",   "hand"),
    ("body_curled_up",    "身体蜷缩",          "pose"),
    ("hunched_back",      "驼背/弓着背走路",    "pose"),
]

_NEUTRAL_ACTIONS: list[tuple[str, str, str]] = [
    ("neutral_stand_sit", "自然站立/端坐",     "pose"),
]

_ALERT_ACTIONS: list[tuple[str, str, str]] = [
    ("stop_gesture",      "停止/别动手势",     "hand"),
    ("fallen_down",       "跌倒姿态",          "pose"),
]

# Flattened lists for random selection
_POSE_ACTIONS: list[tuple[str, str]] = []
_HAND_ACTIONS: list[tuple[str, str]] = []

for _actions in [_HAPPY_ACTIONS, _ANGRY_ACTIONS, _SAD_ACTIONS, _NEUTRAL_ACTIONS, _ALERT_ACTIONS]:
    for key, label, cat in _actions:
        if cat == "pose":
            _POSE_ACTIONS.append((key, label))
        else:
            _HAND_ACTIONS.append((key, label))

# Lookup tables
_ACTION_LABEL: dict[str, str] = {}
_ACTION_CATEGORY: dict[str, str] = {}
for _actions in [_HAPPY_ACTIONS, _ANGRY_ACTIONS, _SAD_ACTIONS, _NEUTRAL_ACTIONS, _ALERT_ACTIONS]:
    for key, label, cat in _actions:
        _ACTION_LABEL[key] = label
        _ACTION_CATEGORY[key] = cat


def get_action_label(key: str) -> str:
    """Get Chinese label for an action key."""
    return _ACTION_LABEL.get(key, "")


def get_action_category(key: str) -> str:
    """Get category (pose/hand) for an action key."""
    return _ACTION_CATEGORY.get(key, "")


# ── Mock action classifier ──────────────────────────────────────────


class PoseActionClassifier:
    """Mock action classifier — random triggers with persistence.

    Behavior:
      - IDLE phase: small random chance (~0.3% per frame ≈ once per 33s)
        to trigger a random action.
      - ACTIVE phase: action label persists for duration_sec (2-4s).
      - COOLDOWN phase: wait cooldown_sec (5-15s) before allowing next trigger.
      - Trigger probability is clamped so actions appear infrequently.
    """

    def __init__(
        self,
        trigger_chance: float = 0.003,   # per-frame probability (~1/300 frames)
        min_duration_sec: float = 2.0,
        max_duration_sec: float = 4.0,
        min_cooldown_sec: float = 5.0,
        max_cooldown_sec: float = 15.0,
    ) -> None:
        self._trigger_chance = trigger_chance
        self._min_duration = min_duration_sec
        self._max_duration = max_duration_sec
        self._min_cooldown = min_cooldown_sec
        self._max_cooldown = max_cooldown_sec

        # State
        self._state: str = "idle"  # idle | active | cooldown
        self._state_entered_at: float = 0.0
        self._pose_action: str = ""
        self._pose_label: str = ""
        self._hand_action: str = ""
        self._hand_label: str = ""
        self._active_until: float = 0.0
        self._cooldown_until: float = 0.0

    def update(self, now: float | None = None) -> None:
        """Update state machine. Call once per frame (~10Hz).

        Args:
            now: Current time (seconds). Uses time.time() if None.
        """
        if now is None:
            now = time.time()

        if self._state == "cooldown":
            if now >= self._cooldown_until:
                self._state = "idle"
                self._state_entered_at = now
            return

        if self._state == "active":
            if now >= self._active_until:
                self._pose_action = ""
                self._pose_label = ""
                self._hand_action = ""
                self._hand_label = ""
                self._state = "cooldown"
                self._state_entered_at = now
                self._cooldown_until = now + random.uniform(
                    self._min_cooldown, self._max_cooldown,
                )
                logger.debug(
                    "PoseAction: cleared → cooldown %.1fs",
                    self._cooldown_until - now,
                )
            return

        # ── State: idle — random trigger check ──
        if random.random() > self._trigger_chance:
            return

        # Trigger a random action
        self._state = "active"
        self._state_entered_at = now
        self._active_until = now + random.uniform(
            self._min_duration, self._max_duration,
        )

        # Pick a random pose action and a random hand action
        if _POSE_ACTIONS and random.random() < 0.6:  # 60% chance of pose action
            key, label = random.choice(_POSE_ACTIONS)
            self._pose_action = key
            self._pose_label = label
        else:
            self._pose_action = ""
            self._pose_label = ""

        if _HAND_ACTIONS and random.random() < 0.6:  # 60% chance of hand action
            # Don't pick a hand action if we already have a pose action that
            # is also a hand action (shouldn't happen with current data)
            key, label = random.choice(_HAND_ACTIONS)
            self._hand_action = key
            self._hand_label = label
        else:
            self._hand_action = ""
            self._hand_label = ""

        # Ensure at least one action is set
        if not self._pose_action and not self._hand_action:
            all_actions = _POSE_ACTIONS + _HAND_ACTIONS
            key, label = random.choice(all_actions)
            cat = get_action_category(key)
            if cat == "pose":
                self._pose_action = key
                self._pose_label = label
            else:
                self._hand_action = key
                self._hand_label = label

        logger.info(
            "PoseAction triggered: pose=%s hand=%s (until +%.1fs)",
            self._pose_action or "none",
            self._hand_action or "none",
            self._active_until - now,
        )

    # ── Read API ────────────────────────────────────────────────

    @property
    def pose_action(self) -> tuple[str, str]:
        """Return (action_key, action_label) for current pose action."""
        return (self._pose_action, self._pose_label)

    @property
    def hand_action(self) -> tuple[str, str]:
        """Return (action_key, action_label) for current hand action."""
        return (self._hand_action, self._hand_label)

    def is_active(self) -> bool:
        """Check if any action is currently active."""
        return self._state == "active" and (bool(self._pose_action) or bool(self._hand_action))

    def snapshot(self) -> dict[str, Any]:
        """Return current state as a dict for JSON serialization."""
        return {
            "pose_action": self._pose_action,
            "pose_action_label": self._pose_label,
            "hand_action": self._hand_action,
            "hand_action_label": self._hand_label,
            "action_state": self._state,
        }
