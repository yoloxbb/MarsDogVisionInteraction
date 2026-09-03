from marsdog_vision_interaction.providers.gesture_pose_engine import (
    ActionName,
    BehaviorEngine,
    FallEventManager,
    HandLandmark,
    LandmarkFrame,
    PoseLandmark,
)
from marsdog_vision_interaction.providers.pose_action import PoseActionClassifier


def _pose(kind: str = "standing") -> tuple[PoseLandmark, ...]:
    points = [PoseLandmark(0.5, 0.5, 0.0, 0.0, 0.0) for _ in range(33)]

    def set_point(index: int, x: float, y: float, z: float = 0.0) -> None:
        points[index] = PoseLandmark(x, y, z, 1.0, 1.0)

    if kind == "standing":
        values = (
            (0, 0.50, 0.18), (7, 0.46, 0.20), (8, 0.54, 0.20),
            (11, 0.40, 0.30), (12, 0.60, 0.30),
            (13, 0.38, 0.48), (14, 0.62, 0.48),
            (15, 0.38, 0.65), (16, 0.62, 0.65),
            (23, 0.45, 0.55), (24, 0.55, 0.55),
            (25, 0.45, 0.75), (26, 0.55, 0.75),
            (27, 0.45, 0.95), (28, 0.55, 0.95),
        )
    else:
        values = (
            (0, 0.18, 0.75), (7, 0.20, 0.71), (8, 0.20, 0.79),
            (11, 0.30, 0.65), (12, 0.30, 0.85),
            (13, 0.45, 0.65), (14, 0.45, 0.85),
            (15, 0.58, 0.65), (16, 0.58, 0.85),
            (23, 0.70, 0.65), (24, 0.70, 0.85),
            (25, 0.82, 0.65), (26, 0.82, 0.85),
            (27, 0.95, 0.65), (28, 0.95, 0.85),
        )
    for index, x, y in values:
        set_point(index, x, y)
    return tuple(points)


def _pose_with_nose_y(y: float) -> tuple[PoseLandmark, ...]:
    points = list(_pose("standing"))
    nose = points[0]
    points[0] = PoseLandmark(
        nose.x, y, nose.z, nose.visibility, nose.presence
    )
    return tuple(points)


def _pose_with_arms(
    *,
    left_elbow: tuple[float, float],
    right_elbow: tuple[float, float],
    left_wrist: tuple[float, float],
    right_wrist: tuple[float, float],
    wrist_confidence: float = 1.0,
    hip_confidence: float = 1.0,
) -> tuple[PoseLandmark, ...]:
    points = list(_pose("standing"))
    for index, (x, y) in (
        (13, left_elbow),
        (14, right_elbow),
        (15, left_wrist),
        (16, right_wrist),
    ):
        confidence = wrist_confidence if index in (15, 16) else 1.0
        points[index] = PoseLandmark(
            x, y, 0.0, confidence, confidence
        )
    for index in (23, 24):
        point = points[index]
        points[index] = PoseLandmark(
            point.x,
            point.y,
            point.z,
            hip_confidence,
            hip_confidence,
        )
    return tuple(points)


def _forward_palm(*, curled_ring_finger: bool = False) -> tuple[HandLandmark, ...]:
    """Synthetic camera-facing left palm placed at the left Pose wrist."""

    coordinates = [
        (0.42, 0.42),
        (0.38, 0.39), (0.35, 0.36), (0.33, 0.33), (0.31, 0.30),
        (0.38, 0.36), (0.38, 0.31), (0.38, 0.26), (0.38, 0.21),
        (0.42, 0.35), (0.42, 0.30), (0.42, 0.25), (0.42, 0.19),
        (0.46, 0.36), (0.46, 0.31), (0.46, 0.26), (0.46, 0.21),
        (0.50, 0.37), (0.50, 0.33), (0.50, 0.29), (0.50, 0.24),
    ]
    if curled_ring_finger:
        coordinates[14:17] = [(0.46, 0.32), (0.44, 0.35), (0.46, 0.38)]
    return tuple(HandLandmark(x, y, -0.30) for x, y in coordinates)


def _hand_at(
    center_x: float,
    center_y: float,
) -> tuple[HandLandmark, ...]:
    hand = _forward_palm()
    current_x = sum(point.x for point in hand) / len(hand)
    current_y = sum(point.y for point in hand) / len(hand)
    return tuple(
        HandLandmark(
            point.x + center_x - current_x,
            point.y + center_y - current_y,
            point.z,
        )
        for point in hand
    )


def _forward_palm_at_wrist(
    wrist_x: float,
    wrist_y: float,
) -> tuple[HandLandmark, ...]:
    hand = _forward_palm()
    return tuple(
        HandLandmark(
            point.x + wrist_x - hand[0].x,
            point.y + wrist_y - hand[0].y,
            point.z,
        )
        for point in hand
    )


def _shift_pose(
    pose: tuple[PoseLandmark, ...],
    *,
    dy: float,
    keep_ankles_planted: bool = False,
) -> tuple[PoseLandmark, ...]:
    shifted: list[PoseLandmark] = []
    for index, point in enumerate(pose):
        point_dy = 0.0 if keep_ankles_planted and index in (27, 28) else dy
        shifted.append(
            PoseLandmark(
                point.x,
                point.y + point_dy,
                point.z,
                point.visibility,
                point.presence,
            )
        )
    return tuple(shifted)


def _hide_pose_landmarks(
    pose: tuple[PoseLandmark, ...],
    *indices: int,
) -> tuple[PoseLandmark, ...]:
    hidden = set(indices)
    return tuple(
        PoseLandmark(
            point.x,
            point.y,
            point.z,
            0.0 if index in hidden else point.visibility,
            0.0 if index in hidden else point.presence,
        )
        for index, point in enumerate(pose)
    )


def _shift_selected_pose_landmarks(
    pose: tuple[PoseLandmark, ...],
    indices: tuple[int, ...],
    *,
    dy: float,
) -> tuple[PoseLandmark, ...]:
    selected = set(indices)
    return tuple(
        PoseLandmark(
            point.x,
            point.y + (dy if index in selected else 0.0),
            point.z,
            point.visibility,
            point.presence,
        )
        for index, point in enumerate(pose)
    )


def _face_covering_pose() -> tuple[PoseLandmark, ...]:
    return _pose_with_arms(
        left_elbow=(0.42, 0.31),
        right_elbow=(0.58, 0.31),
        left_wrist=(0.46, 0.21),
        right_wrist=(0.54, 0.21),
    )


def test_standing_action_is_deterministic_and_keeps_public_key() -> None:
    classifier = PoseActionClassifier()
    snapshots = [
        classifier.update(
            track_id=7,
            pose_landmarks=_pose("standing"),
            now=10.0 + index * 0.1,
        )
        for index in range(10)
    ]
    assert snapshots[-1]["pose_action"] == "neutral_stand_sit"
    assert snapshots[-1]["recognized_actions"][0]["name"] == "standing"
    assert {
        item["name"] for item in snapshots[-1]["raw_scores"]
    } == {action.value for action in ActionName}
    assert snapshots[-1]["recognized_actions"][0]["priority"] == "P4"
    assert snapshots[-1]["recognized_actions"][0]["group"] == "posture"
    assert "head_vertical_range_ratio" in snapshots[-1]["temporal_features"]
    assert "ankle_vertical_velocity" in snapshots[-1]["temporal_features"]
    assert "shoulder_vertical_velocity" in snapshots[-1]["temporal_features"]
    assert "torso_scale_change_ratio" in snapshots[-1]["temporal_features"]
    assert snapshots[-1]["fall_detector"]["armed"] is False
    assert snapshots[-1]["fall_detector"]["lying_score"] == 0.0
    assert snapshots[-1]["jump_detector"]["active"] is False
    assert snapshots[-1]["jump_detector"]["mode"] == "full_body"
    assert snapshots[-1]["jump_detector"]["phase"] == "monitoring"
    assert snapshots[-1]["jump_detector"]["missing_landmarks"] == []
    assert set(snapshots[-1]["jump_detector"]["component_scores"]) == {
        "hip_upward",
        "shoulder_upward",
        "ankle_upward",
        "motion_coherence",
        "torso_stability",
        "baseline_displacement",
        "position_return",
    }
    assert snapshots[-1]["hand_features"]["left"]["detected"] is False
    assert (
        snapshots[-1]["hand_features"]["left"][
            "stop_command_zone_score"
        ]
        == 0.0
    )


def test_entering_frame_while_lying_never_fabricates_fall_event() -> None:
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=_pose("lying"),
            )
        )
        for index in range(20)
    ]
    assert not any(result.fall_status.event_triggered for result in results)
    assert all(
        ActionName.FALL not in {action.name for action in result.actions}
        for result in results
    )


def test_upright_to_lying_transition_emits_one_fall_edge() -> None:
    engine = BehaviorEngine()
    for index in range(14):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=_pose("standing"),
            )
        )

    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.5 + index * 0.1,
                pose_landmarks=_pose("lying"),
            )
        )
        for index in range(15)
    ]
    assert sum(result.fall_status.event_triggered for result in results) == 1
    assert any(result.fall_status.alert_active for result in results)


def test_recent_upright_to_lying_fallback_survives_missing_motion_score(
    monkeypatch,
) -> None:
    # Real PoseLandmarker output can jump directly from a valid upright body to
    # a valid horizontal body without enough adjacent landmarks for velocity.
    monkeypatch.setattr(FallEventManager, "_transition_score", lambda *args: 0.0)
    engine = BehaviorEngine()
    for index in range(14):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=_pose("standing"),
            )
        )

    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.5 + index * 0.1,
                pose_landmarks=_pose("lying"),
            )
        )
        for index in range(12)
    ]

    assert sum(result.fall_status.event_triggered for result in results) == 1
    assert any(result.fall_status.phase.value == "falling" for result in results)


def test_fall_transition_survives_brief_pose_occlusion() -> None:
    """A low-FPS fall commonly hides the body before Pose reacquires it lying."""

    engine = BehaviorEngine()
    for index in range(14):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=_pose("standing"),
            )
        )

    for timestamp in (1.6, 1.8, 2.0, 2.2):
        engine.update(LandmarkFrame(monotonic_s=timestamp, pose_landmarks=None))
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=2.4 + index * 0.2,
                pose_landmarks=_pose("lying"),
            )
        )
        for index in range(8)
    ]

    assert sum(result.fall_status.event_triggered for result in results) == 1
    assert max(result.fall_status.transition_score for result in results) >= 0.55


def test_fall_confirmation_tolerates_one_noisy_lying_frame() -> None:
    engine = BehaviorEngine()
    for index in range(14):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=_pose("standing"),
            )
        )

    kinds = ("lying", "lying", "standing", "lying", "lying", "lying")
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.6 + index * 0.2,
                pose_landmarks=_pose(kind),
            )
        )
        for index, kind in enumerate(kinds)
    ]

    assert sum(result.fall_status.event_triggered for result in results) == 1


def test_fall_confirmation_tolerates_short_upright_looking_pose_noise() -> None:
    engine = BehaviorEngine()
    for index in range(14):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=_pose("standing"),
            )
        )

    observations = (
        (1.6, "lying"),
        (1.8, "lying"),
        (2.0, "standing"),
        (2.2, "standing"),
        (2.4, "lying"),
        (2.6, "lying"),
        (2.8, "lying"),
        (3.0, "lying"),
        (3.2, "lying"),
    )
    results = [
        engine.update(
            LandmarkFrame(monotonic_s=timestamp, pose_landmarks=_pose(kind))
        )
        for timestamp, kind in observations
    ]

    assert sum(result.fall_status.event_triggered for result in results) == 1


def test_fall_keeps_armed_track_across_brief_missing_active_target() -> None:
    classifier = PoseActionClassifier(track_timeout_sec=0.75)
    for index in range(14):
        classifier.update(
            track_id=7,
            pose_landmarks=_pose("standing"),
            now=0.1 + index * 0.1,
        )

    for timestamp in (1.6, 1.8, 2.0, 2.2):
        classifier.update(track_id=0, pose_landmarks=None, now=timestamp)
    results = [
        classifier.update(
            track_id=7,
            pose_landmarks=_pose("lying"),
            now=2.4 + index * 0.2,
        )
        for index in range(8)
    ]

    assert sum(bool(result["fall_event_triggered"]) for result in results) == 1
    assert any(result["pose_action"] == "fallen_down" for result in results)


def test_stop_requires_all_four_non_thumb_fingers_to_be_straight() -> None:
    pose = _pose_with_arms(
        left_elbow=(0.41, 0.34),
        right_elbow=(0.62, 0.48),
        left_wrist=(0.42, 0.42),
        right_wrist=(0.62, 0.65),
    )
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index * 0.1,
                pose_landmarks=pose,
                left_hand=_forward_palm(curled_ring_finger=True),
            )
        )
        for index in range(8)
    ]

    assert max(result.raw_score_map[ActionName.STOP_GESTURE] for result in results) == 0.0
    assert all(
        ActionName.STOP_GESTURE not in {action.name for action in result.actions}
        for result in results
    )


def test_camera_facing_open_palm_still_triggers_stop() -> None:
    pose = _pose_with_arms(
        left_elbow=(0.41, 0.34),
        right_elbow=(0.62, 0.48),
        left_wrist=(0.42, 0.42),
        right_wrist=(0.62, 0.65),
    )
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index * 0.1,
                pose_landmarks=pose,
                left_hand=_forward_palm(),
            )
        )
        for index in range(8)
    ]

    assert any(
        ActionName.STOP_GESTURE in {action.name for action in result.actions}
        for result in results
    )


def test_naturally_lowered_open_hand_does_not_trigger_stop() -> None:
    """A straight resting arm must not count as an intentional command."""

    lowered_wrist = (0.38, 0.67)
    pose = _pose_with_arms(
        left_elbow=(0.38, 0.49),
        right_elbow=(0.62, 0.48),
        left_wrist=lowered_wrist,
        right_wrist=(0.62, 0.65),
    )
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index * 0.1,
                pose_landmarks=pose,
                left_hand=_forward_palm_at_wrist(*lowered_wrist),
            )
        )
        for index in range(8)
    ]

    assert max(
        result.raw_score_map[ActionName.STOP_GESTURE]
        for result in results
    ) == 0.0
    assert all(
        ActionName.STOP_GESTURE not in {action.name for action in result.actions}
        for result in results
    )


def test_short_takeoff_is_confirmed_and_held_as_jumping() -> None:
    """Two take-off frames must survive long enough for stable publication."""

    standing = _pose("standing")
    engine = BehaviorEngine()
    for index in range(8):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=standing,
            )
        )

    observations = (
        _shift_pose(standing, dy=-0.05),
        _shift_pose(standing, dy=-0.10),
        _shift_pose(standing, dy=-0.10),
        _shift_pose(standing, dy=-0.08),
        _shift_pose(standing, dy=-0.04),
    )
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=0.9 + index * 0.1,
                pose_landmarks=pose,
            )
        )
        for index, pose in enumerate(observations)
    ]

    assert sum(result.jump_status.event_triggered for result in results) == 1
    assert any(result.jump_status.active for result in results)
    assert any(
        ActionName.JUMPING in {action.name for action in result.actions}
        for result in results
    )


def test_standing_up_with_planted_feet_is_not_jumping() -> None:
    """Hip-only upward motion is a stand-up, not an airborne take-off."""

    standing = _pose("standing")
    engine = BehaviorEngine()
    for index in range(8):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=standing,
            )
        )
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=0.9 + index * 0.1,
                pose_landmarks=_shift_pose(
                    standing,
                    dy=-0.05 * (index + 1),
                    keep_ankles_planted=True,
                ),
            )
        )
        for index in range(3)
    ]

    assert not any(result.jump_status.event_triggered for result in results)
    assert all(
        ActionName.JUMPING not in {action.name for action in result.actions}
        for result in results
    )


def test_cropped_upper_body_jump_confirms_after_downward_return() -> None:
    """Missing ankles use coherent shoulder/hip lift plus return evidence."""

    cropped = _hide_pose_landmarks(_pose("standing"), 27, 28)
    engine = BehaviorEngine()
    for index in range(8):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=cropped,
            )
        )

    offsets = (-0.05, -0.10, -0.10, -0.05, -0.02, 0.0, 0.0)
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=0.9 + index * 0.1,
                pose_landmarks=_shift_pose(cropped, dy=offset),
            )
        )
        for index, offset in enumerate(offsets)
    ]

    triggered = [result for result in results if result.jump_status.event_triggered]
    assert len(triggered) == 1
    assert triggered[0].jump_status.mode == "upper_body_fallback"
    assert triggered[0].jump_status.missing_landmarks == (
        "left_ankle",
        "right_ankle",
    )
    assert any(result.jump_status.phase == "awaiting_return" for result in results)
    assert any(
        ActionName.JUMPING in {action.name for action in result.actions}
        for result in results
    )


def test_low_fps_cropped_jump_accepts_one_strong_takeoff_sample() -> None:
    """A short jump must remain observable near the runtime's 5 FPS cadence."""

    cropped = _hide_pose_landmarks(_pose("standing"), 27, 28)
    engine = BehaviorEngine()
    for index in range(4):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.2 + index * 0.2,
                pose_landmarks=cropped,
            )
        )

    observations = (
        (1.0, -0.015),
        (1.2, -0.015),
        (1.4, 0.0),
        (1.6, 0.0),
        (1.8, 0.0),
    )
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=timestamp,
                pose_landmarks=_shift_pose(cropped, dy=offset),
            )
        )
        for timestamp, offset in observations
    ]

    triggered = [result for result in results if result.jump_status.event_triggered]
    assert len(triggered) == 1
    assert triggered[0].jump_status.mode == "upper_body_fallback"
    assert any(
        ActionName.JUMPING in {action.name for action in result.actions}
        for result in results
    )


def test_cropped_standing_up_without_return_is_not_jumping() -> None:
    """A persistent height change must not pass the cropped-body fallback."""

    cropped = _hide_pose_landmarks(_pose("standing"), 27, 28)
    engine = BehaviorEngine()
    for index in range(8):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=cropped,
            )
        )
    offsets = (-0.05, -0.10) + (-0.10,) * 12
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=0.9 + index * 0.1,
                pose_landmarks=_shift_pose(cropped, dy=offset),
            )
        )
        for index, offset in enumerate(offsets)
    ]

    assert not any(result.jump_status.event_triggered for result in results)
    assert any(
        result.jump_status.rejection_reason == "return_not_observed"
        for result in results
    )
    assert all(
        ActionName.JUMPING not in {action.name for action in result.actions}
        for result in results
    )


def test_missing_hips_cannot_use_upper_body_jump_fallback() -> None:
    cropped = _hide_pose_landmarks(_pose("standing"), 23, 24, 27, 28)
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=_shift_pose(cropped, dy=-0.04 * index),
            )
        )
        for index in range(8)
    ]

    assert results[-1].jump_status.mode == "unavailable"
    assert all(not result.jump_status.event_triggered for result in results)
    assert results[-1].jump_status.rejection_reason == "insufficient_upper_body"


def test_cropped_knee_lift_and_return_triggers_stomping_at_low_fps() -> None:
    """Knee motion is the fallback when a close crop removes both ankles."""

    cropped = _hide_pose_landmarks(_pose("standing"), 27, 28)
    engine = BehaviorEngine()
    for index in range(5):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.2 + index * 0.2,
                pose_landmarks=cropped,
            )
        )

    offsets = (-0.03, -0.06, -0.02, 0.0, 0.0, 0.0, 0.0)
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.2 + index * 0.2,
                pose_landmarks=_shift_selected_pose_landmarks(
                    cropped,
                    (25,),
                    dy=offset,
                ),
            )
        )
        for index, offset in enumerate(offsets)
    ]

    assert max(
        result.raw_score_map[ActionName.STOMPING] for result in results
    ) >= 0.55
    assert any(
        ActionName.STOMPING in {action.name for action in result.actions}
        for result in results
    )


def test_cropped_knee_raise_without_return_is_not_stomping() -> None:
    cropped = _hide_pose_landmarks(_pose("standing"), 27, 28)
    engine = BehaviorEngine()
    offsets = (0.0, 0.0, 0.0, -0.05, -0.05, -0.05, -0.05, -0.05)
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=0.2 + index * 0.2,
                pose_landmarks=_shift_selected_pose_landmarks(
                    cropped,
                    (25,),
                    dy=offset,
                ),
            )
        )
        for index, offset in enumerate(offsets)
    ]

    assert max(
        result.raw_score_map[ActionName.STOMPING] for result in results
    ) == 0.0
    assert all(
        ActionName.STOMPING not in {action.name for action in result.actions}
        for result in results
    )


def test_cropped_stomp_diagnostics_report_knee_fallback() -> None:
    cropped = _hide_pose_landmarks(_pose("standing"), 27, 28)
    classifier = PoseActionClassifier()
    snapshots = []
    offsets = (0.0, 0.0, 0.0, -0.03, -0.06, -0.02, 0.0, 0.0, 0.0)
    for index, offset in enumerate(offsets):
        snapshots.append(
            classifier.update(
                track_id=7,
                pose_landmarks=_shift_selected_pose_landmarks(
                    cropped,
                    (25,),
                    dy=offset,
                ),
                now=0.2 + index * 0.2,
            )
        )

    assert snapshots[-1]["stomp_detector"]["source"] == "knee_fallback"
    assert snapshots[-1]["stomp_detector"]["knee_direction_changes"] >= 1
    assert any(snapshot["stomp_detector"]["recognized"] for snapshot in snapshots)


def test_single_frame_full_body_shift_does_not_trigger_jump() -> None:
    """One camera/body-position discontinuity is not a confirmed take-off."""

    standing = _pose("standing")
    engine = BehaviorEngine()
    for index in range(8):
        engine.update(
            LandmarkFrame(
                monotonic_s=0.1 + index * 0.1,
                pose_landmarks=standing,
            )
        )
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=0.9,
                pose_landmarks=_shift_pose(standing, dy=-0.08),
            )
        ),
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0,
                pose_landmarks=standing,
            )
        ),
    ]

    assert not any(result.jump_status.event_triggered for result in results)
    assert all(
        ActionName.JUMPING not in {action.name for action in result.actions}
        for result in results
    )


def test_elevated_camera_bent_elbow_open_palm_triggers_stop() -> None:
    """A dog-head camera must tolerate foreshortening and a bent raised arm."""

    raised_palm = _hand_at(0.42, 0.30)
    pose = _pose_with_arms(
        left_elbow=(0.50, 0.28),
        right_elbow=(0.62, 0.48),
        left_wrist=(raised_palm[0].x, raised_palm[0].y),
        right_wrist=(0.62, 0.65),
        wrist_confidence=0.62,
    )
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index * 0.1,
                pose_landmarks=pose,
                left_hand=raised_palm,
            )
        )
        for index in range(8)
    ]

    assert any(
        ActionName.STOP_GESTURE in {action.name for action in result.actions}
        for result in results
    )


def test_crossed_arms_open_palm_does_not_trigger_stop() -> None:
    crossed_pose = _pose_with_arms(
        left_elbow=(0.58, 0.42),
        right_elbow=(0.42, 0.42),
        left_wrist=(0.42, 0.42),
        right_wrist=(0.58, 0.42),
    )
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index * 0.1,
                pose_landmarks=crossed_pose,
                left_hand=_forward_palm(),
            )
        )
        for index in range(8)
    ]

    assert min(
        result.raw_score_map[ActionName.ARMS_CROSSED]
        for result in results
    ) >= 0.40
    assert max(
        result.raw_score_map[ActionName.STOP_GESTURE]
        for result in results
    ) == 0.0
    assert all(
        ActionName.STOP_GESTURE not in {action.name for action in result.actions}
        for result in results
    )


def test_face_covering_requires_two_current_hands() -> None:
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index * 0.1,
                pose_landmarks=_face_covering_pose(),
                left_hand=_hand_at(0.46, 0.21),
            )
        )
        for index in range(12)
    ]

    assert max(
        result.raw_score_map[ActionName.FACE_COVERING]
        for result in results
    ) == 0.0
    assert all(
        ActionName.FACE_COVERING not in {
            action.name for action in result.actions
        }
        for result in results
    )


def test_face_covering_clears_immediately_when_only_hands_remain() -> None:
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index * 0.1,
                pose_landmarks=_face_covering_pose(),
                left_hand=_hand_at(0.46, 0.21),
                right_hand=_hand_at(0.54, 0.21),
            )
        )
        for index in range(12)
    ]
    assert ActionName.FACE_COVERING in {
        action.name for action in results[-1].actions
    }

    hands_without_person = engine.update(
        LandmarkFrame(
            monotonic_s=2.2,
            pose_landmarks=None,
            left_hand=_hand_at(0.46, 0.21),
            right_hand=_hand_at(0.54, 0.21),
        )
    )

    assert (
        hands_without_person.raw_score_map[ActionName.FACE_COVERING]
        == 0.0
    )
    assert ActionName.FACE_COVERING not in {
        action.name for action in hands_without_person.actions
    }


def test_stop_is_cleared_immediately_when_no_hand_is_observed() -> None:
    pose = _pose_with_arms(
        left_elbow=(0.41, 0.34),
        right_elbow=(0.62, 0.48),
        left_wrist=(0.42, 0.42),
        right_wrist=(0.62, 0.65),
    )
    engine = BehaviorEngine()
    for index in range(6):
        result = engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index * 0.1,
                pose_landmarks=pose,
                left_hand=_forward_palm(),
            )
        )
    assert ActionName.STOP_GESTURE in {action.name for action in result.actions}

    no_hand_result = engine.update(
        LandmarkFrame(monotonic_s=1.6, pose_landmarks=pose)
    )

    assert no_hand_result.raw_score_map[ActionName.STOP_GESTURE] == 0.0
    assert ActionName.STOP_GESTURE not in {
        action.name for action in no_hand_result.actions
    }


def test_stop_rejects_low_confidence_wrist_from_cropped_upper_body() -> None:
    """A HandLandmarker false positive must not pair with a guessed Pose wrist."""

    cropped_pose = _pose_with_arms(
        left_elbow=(0.41, 0.34),
        right_elbow=(0.62, 0.48),
        left_wrist=(0.42, 0.42),
        right_wrist=(0.62, 0.65),
        wrist_confidence=0.55,
    )
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index * 0.1,
                pose_landmarks=cropped_pose,
                left_hand=_forward_palm(),
            )
        )
        for index in range(8)
    ]

    assert max(result.raw_score_map[ActionName.STOP_GESTURE] for result in results) == 0.0
    assert all(
        ActionName.STOP_GESTURE not in {action.name for action in result.actions}
        for result in results
    )


def test_small_nose_keypoint_jitter_does_not_trigger_fast_nod() -> None:
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index / 15.0,
                pose_landmarks=_pose_with_nose_y(
                    0.174 if index % 2 else 0.186
                ),
            )
        )
        for index in range(20)
    ]
    assert all(
        ActionName.FAST_NOD not in {action.name for action in result.actions}
        for result in results
    )
    assert results[-1].raw_score_map[ActionName.FAST_NOD] < 0.55
    assert results[-1].features.temporal.head_vertical_direction_changes > 0


def test_deliberate_head_vertical_cycle_can_trigger_fast_nod() -> None:
    engine = BehaviorEngine()
    cycle = (0.18, 0.21, 0.24, 0.21, 0.18)
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index / 15.0,
                pose_landmarks=_pose_with_nose_y(cycle[index % len(cycle)]),
            )
        )
        for index in range(20)
    ]
    assert any(
        ActionName.FAST_NOD in {action.name for action in result.actions}
        for result in results
    )


def test_hand_only_pose_false_positive_cannot_trigger_fast_nod() -> None:
    engine = BehaviorEngine()
    cycle = (0.18, 0.21, 0.24, 0.21, 0.18)
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index / 15.0,
                # Simulate PoseLandmarker fitting a moving hand as a person.
                pose_landmarks=_pose_with_nose_y(
                    cycle[index % len(cycle)]
                ),
                left_hand=_hand_at(0.50, 0.30),
                face_observed=False,
            )
        )
        for index in range(20)
    ]

    assert max(
        result.raw_score_map[ActionName.FAST_NOD]
        for result in results
    ) == 0.0
    assert all(
        ActionName.FAST_NOD not in {action.name for action in result.actions}
        for result in results
    )


def test_fast_nod_clears_immediately_when_face_disappears() -> None:
    engine = BehaviorEngine()
    cycle = (0.18, 0.21, 0.24, 0.21, 0.18)
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index / 15.0,
                pose_landmarks=_pose_with_nose_y(
                    cycle[index % len(cycle)]
                ),
                face_observed=True,
            )
        )
        for index in range(20)
    ]
    assert ActionName.FAST_NOD in {
        action.name for action in results[-1].actions
    }

    hand_only = engine.update(
        LandmarkFrame(
            monotonic_s=2.4,
            pose_landmarks=_pose_with_nose_y(0.24),
            left_hand=_hand_at(0.50, 0.30),
            face_observed=False,
        )
    )

    assert hand_only.raw_score_map[ActionName.FAST_NOD] == 0.0
    assert ActionName.FAST_NOD not in {
        action.name for action in hand_only.actions
    }


def test_crossing_arms_does_not_trigger_clapping() -> None:
    engine = BehaviorEngine()
    wrist_pairs = (
        ((0.38, 0.44), (0.62, 0.44)),
        ((0.43, 0.44), (0.57, 0.44)),
        ((0.48, 0.44), (0.52, 0.44)),
        ((0.52, 0.44), (0.48, 0.44)),
        ((0.57, 0.44), (0.43, 0.44)),
        ((0.59, 0.44), (0.41, 0.44)),
    )
    frames = list(wrist_pairs) + [wrist_pairs[-1]] * 10
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index / 15.0,
                pose_landmarks=_pose_with_arms(
                    left_elbow=(0.34, 0.42),
                    right_elbow=(0.66, 0.42),
                    left_wrist=left_wrist,
                    right_wrist=right_wrist,
                ),
            )
        )
        for index, (left_wrist, right_wrist) in enumerate(frames)
    ]
    assert max(
        result.features.temporal.wrist_distance_direction_changes
        for result in results
    ) == 1
    assert all(
        ActionName.CLAPPING not in {action.name for action in result.actions}
        for result in results
    )
    assert ActionName.ARMS_CROSSED in {
        action.name for action in results[-1].actions
    }


def test_occluded_hands_on_hips_remains_recognizable() -> None:
    pose = _pose_with_arms(
        left_elbow=(0.27, 0.43),
        right_elbow=(0.73, 0.43),
        left_wrist=(0.36, 0.50),
        right_wrist=(0.64, 0.50),
        wrist_confidence=0.35,
        hip_confidence=0.40,
    )
    engine = BehaviorEngine()
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index / 15.0,
                pose_landmarks=pose,
            )
        )
        for index in range(12)
    ]
    recognized = {action.name for action in results[-1].actions}
    assert ActionName.HANDS_ON_HIPS in recognized
    assert ActionName.ARMS_CROSSED not in recognized
    assert ActionName.ARMS_OPEN not in recognized


def test_repeated_hand_open_close_still_triggers_clapping() -> None:
    engine = BehaviorEngine()
    hand_distances = (
        0.20,
        0.12,
        0.04,
        0.12,
        0.20,
        0.12,
        0.04,
        0.12,
        0.20,
        0.12,
        0.04,
        0.12,
        0.20,
    ) * 2
    results = [
        engine.update(
            LandmarkFrame(
                monotonic_s=1.0 + index / 15.0,
                pose_landmarks=_pose_with_arms(
                    left_elbow=(0.37, 0.42),
                    right_elbow=(0.63, 0.42),
                    left_wrist=(0.50 - distance / 2.0, 0.44),
                    right_wrist=(0.50 + distance / 2.0, 0.44),
                ),
            )
        )
        for index, distance in enumerate(hand_distances)
    ]
    assert any(
        ActionName.CLAPPING in {action.name for action in result.actions}
        for result in results
    )
    assert all(
        ActionName.ARMS_CROSSED not in {action.name for action in result.actions}
        for result in results
    )
