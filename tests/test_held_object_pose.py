from marsdog_vision_interaction.core.held_object_pose import (
    HOLDING_DOG_FOOD,
    HOLDING_TOY,
    HeldObjectPoseManager,
)


def _target(*, target_id: str = "epoch:human:7", action: str = "") -> dict:
    keypoints = [
        {
            "id": index,
            "x": 0.40 if index == 15 else 0.60,
            "y": 0.48,
            "confidence": 1.0,
            "presence": 1.0,
        }
        for index in (15, 16)
    ]
    return {
        "target_id": target_id,
        "track_id": 7,
        "tracking_state": "tracking",
        "confidence": 0.9,
        "bbox": [0.25, 0.10, 0.50, 0.80],
        "keypoints": keypoints,
        "pose_action": action,
    }


def _object(
    label: str,
    *,
    sequence: int,
    x: float = 0.37,
    y: float = 0.45,
    confidence: float = 0.9,
) -> dict:
    return {
        "label": label,
        "object_track_id": 3,
        "source_sequence": sequence,
        "tracking_state": "tracking",
        "confidence": confidence,
        "bbox": [x, y, 0.08, 0.08],
    }


def _update(
    manager: HeldObjectPoseManager,
    *,
    now: float,
    sequence: int,
    objects: list[dict],
    target: dict | None = None,
):
    return manager.update(
        now=now,
        active_target=target or _target(),
        hands=[],
        objects=objects,
        object_result_sequence=sequence,
    )


def test_two_distinct_near_wrist_toy_results_confirm_holding_pose() -> None:
    manager = HeldObjectPoseManager()
    first = _update(
        manager,
        now=1.0,
        sequence=10,
        objects=[_object("dog toy ball", sequence=10)],
    )
    repeated_snapshot = _update(
        manager,
        now=1.1,
        sequence=10,
        objects=[_object("dog toy ball", sequence=10)],
    )
    confirmed = _update(
        manager,
        now=1.5,
        sequence=11,
        objects=[_object("dog toy ball", sequence=11)],
    )

    assert first.state == "candidate"
    assert first.evidence_hits == 1
    assert first.object_result_sequence == 10
    assert repeated_snapshot.evidence_hits == 1
    assert repeated_snapshot.object_result_sequence == 10
    assert confirmed.state == "confirmed"
    assert confirmed.object_result_sequence == 11
    assert confirmed.action == HOLDING_TOY
    assert confirmed.action_label == "手持玩具"
    assert confirmed.evidence_hits == confirmed.required_hits == 2


def test_food_can_confirms_dog_food_pose() -> None:
    manager = HeldObjectPoseManager()
    _update(
        manager,
        now=2.0,
        sequence=20,
        objects=[_object("dog food can", sequence=20)],
    )
    status = _update(
        manager,
        now=2.5,
        sequence=21,
        objects=[_object("dog food can", sequence=21)],
    )

    assert status.action == HOLDING_DOG_FOOD
    assert status.action_label == "手持狗粮"


def test_object_in_person_box_but_far_from_both_wrists_is_rejected() -> None:
    manager = HeldObjectPoseManager()
    statuses = [
        _update(
            manager,
            now=3.0 + index * 0.5,
            sequence=30 + index,
            objects=[_object(
                "dog treat bag",
                sequence=30 + index,
                x=0.48,
                y=0.80,
            )],
        )
        for index in range(3)
    ]

    assert all(status.state == "inactive" for status in statuses)
    assert statuses[-1].rejection_reason == "wrist_too_far"
    assert statuses[-1].evaluated_wrist_distance_ratio is not None
    assert (
        statuses[-1].evaluated_wrist_distance_ratio
        > statuses[-1].wrist_distance_threshold_ratio
    )


def test_hand_landmark_outside_current_person_cannot_supply_wrist() -> None:
    manager = HeldObjectPoseManager()
    target = _target()
    target["keypoints"] = []
    foreign_hand = [{
        "handedness": "right",
        "landmarks": [{"id": 0, "x": 0.90, "y": 0.50}],
    }]
    statuses = [
        manager.update(
            now=3.0 + index * 0.5,
            active_target=target,
            hands=foreign_hand,
            objects=[_object(
                "dog toy ball",
                sequence=35 + index,
                x=0.72,
                y=0.46,
            )],
            object_result_sequence=35 + index,
        )
        for index in range(2)
    ]

    assert all(status.state == "inactive" for status in statuses)


def test_low_confidence_and_stale_object_track_are_rejected() -> None:
    manager = HeldObjectPoseManager(min_object_confidence=0.35)
    low = _update(
        manager,
        now=4.0,
        sequence=40,
        objects=[_object(
            "dog toy ball", sequence=40, confidence=0.30
        )],
    )
    stale = _object("dog toy ball", sequence=40)
    stale["tracking_state"] = "temporarily_lost"
    old_sequence = _update(
        manager,
        now=4.5,
        sequence=41,
        objects=[stale],
    )

    assert low.state == "inactive"
    assert old_sequence.state == "inactive"


def test_pose_and_object_from_different_camera_times_are_rejected() -> None:
    manager = HeldObjectPoseManager(max_pose_object_sync_delta_s=0.45)
    item = _object("dog food can", sequence=45)
    item["header"] = {"stamp": 100.0}
    status = manager.update(
        now=4.5,
        active_target=_target(),
        hands=[],
        objects=[item],
        object_result_sequence=45,
        pose_observation_stamp=100.6,
    )

    assert status.state == "inactive"
    assert status.rejection_reason == "timestamp_mismatch"
    assert status.pose_object_sync_delta_ms == 600.0


def test_normal_async_pose_object_skew_is_accepted() -> None:
    manager = HeldObjectPoseManager(max_pose_object_sync_delta_s=0.45)
    first = _object("dog food can", sequence=46)
    first["header"] = {"stamp": 100.0}
    second = _object("dog food can", sequence=47)
    second["header"] = {"stamp": 100.5}
    initial = manager.update(
        now=4.5,
        active_target=_target(),
        hands=[],
        objects=[first],
        object_result_sequence=46,
        pose_observation_stamp=100.4,
    )
    confirmed = manager.update(
        now=5.0,
        active_target=_target(),
        hands=[],
        objects=[second],
        object_result_sequence=47,
        pose_observation_stamp=100.8,
    )

    assert initial.state == "candidate"
    assert confirmed.action == HOLDING_DOG_FOOD


def test_one_detector_miss_does_not_erase_recent_positive_hit() -> None:
    manager = HeldObjectPoseManager(confirmation_window_s=1.5)
    first = _update(
        manager,
        now=10.0,
        sequence=80,
        objects=[_object("dog toy ball", sequence=80)],
    )
    missed = _update(
        manager,
        now=10.5,
        sequence=81,
        objects=[],
    )
    confirmed = _update(
        manager,
        now=11.0,
        sequence=82,
        objects=[_object("dog toy ball", sequence=82)],
    )

    assert first.evidence_hits == 1
    assert missed.state == "candidate"
    assert missed.evidence_hits == 1
    assert confirmed.action == HOLDING_TOY


def test_two_detector_misses_break_confirmation() -> None:
    manager = HeldObjectPoseManager(confirmation_window_s=2.0)
    _update(
        manager,
        now=11.0,
        sequence=83,
        objects=[_object("dog toy ball", sequence=83)],
    )
    _update(manager, now=11.4, sequence=84, objects=[])
    second_miss = _update(manager, now=11.8, sequence=85, objects=[])
    new_positive = _update(
        manager,
        now=12.2,
        sequence=86,
        objects=[_object("dog toy ball", sequence=86)],
    )

    assert second_miss.state == "inactive"
    assert new_positive.state == "candidate"
    assert new_positive.evidence_hits == 1


def test_held_object_may_extend_beyond_person_box_when_near_valid_wrist() -> None:
    manager = HeldObjectPoseManager(wrist_distance_ratio=0.16)
    target = _target()
    target["keypoints"][1]["x"] = 0.80
    outside_center = _object(
        "dog treat bag",
        sequence=90,
        x=0.82,
        y=0.45,
    )
    outside_center["bbox"] = [0.82, 0.45, 0.12, 0.12]
    first = _update(
        manager,
        now=12.0,
        sequence=90,
        objects=[outside_center],
        target=target,
    )
    outside_center = dict(outside_center, source_sequence=91)
    confirmed = _update(
        manager,
        now=12.5,
        sequence=91,
        objects=[outside_center],
        target=target,
    )

    assert first.state == "candidate"
    assert confirmed.action == HOLDING_DOG_FOOD


def test_negative_result_breaks_confirmation_but_active_pose_has_short_hold() -> None:
    manager = HeldObjectPoseManager(hold_s=1.25)
    _update(
        manager,
        now=5.0,
        sequence=50,
        objects=[_object("dog frisbee toy", sequence=50)],
    )
    confirmed = _update(
        manager,
        now=5.5,
        sequence=51,
        objects=[_object("dog frisbee toy", sequence=51)],
    )
    held_through_one_miss = _update(
        manager,
        now=6.0,
        sequence=52,
        objects=[],
    )
    released = _update(
        manager,
        now=6.8,
        sequence=53,
        objects=[],
    )

    assert confirmed.action == HOLDING_TOY
    assert held_through_one_miss.action == HOLDING_TOY
    assert released.state == "inactive"


def test_target_change_or_loss_immediately_clears_held_pose() -> None:
    manager = HeldObjectPoseManager()
    _update(
        manager,
        now=7.0,
        sequence=70,
        objects=[_object("dog tug ring toy", sequence=70)],
    )
    confirmed = _update(
        manager,
        now=7.5,
        sequence=71,
        objects=[_object("dog tug ring toy", sequence=71)],
    )
    changed = _update(
        manager,
        now=7.6,
        sequence=71,
        objects=[_object("dog tug ring toy", sequence=71)],
        target=_target(target_id="epoch:human:8"),
    )
    lost_target = _target(target_id="epoch:human:8")
    lost_target["tracking_state"] = "temporarily_lost"
    lost = _update(
        manager,
        now=7.7,
        sequence=71,
        objects=[],
        target=lost_target,
    )

    assert confirmed.action == HOLDING_TOY
    assert changed.state == "inactive"
    assert lost.state == "inactive"
