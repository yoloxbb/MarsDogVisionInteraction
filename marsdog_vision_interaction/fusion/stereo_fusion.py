"""Visual stereo fusion and single-target constraint for binocular cameras.

Takes visual detections from camera halves and delegates visual target
selection to ``VisualTargetManager``. Voice state is deliberately absent;
cross-modal association happens downstream through ROS2 topics.
"""

from __future__ import annotations

from typing import Any

from marsdog_vision_interaction.core.visual_target_manager import TargetManager


# Module-level singleton — one TargetManager per process
_target_manager: TargetManager | None = None


def get_target_manager() -> TargetManager:
    """Get or create the global TargetManager singleton."""
    global _target_manager
    if _target_manager is None:
        _target_manager = TargetManager()
    return _target_manager


def reset_target_manager() -> None:
    """Reset the global TargetManager (for testing)."""
    global _target_manager
    _target_manager = TargetManager()


def _iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    if a[2] <= 0 or a[3] <= 0 or b[2] <= 0 or b[3] <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def fuse_stereo_observation(
    left_obs: dict[str, Any],
    right_obs: dict[str, Any],
    single_target: bool = True,
    face_iou_threshold: float = 0.1,
) -> dict[str, Any]:
    """Fuse left+right stereo detections via TargetManager.

    Decision flow:
      1. Collect all humans and faces from both halves.
      2. Feed to TargetManager.update_vision() for single-target selection.
      3. TargetManager applies priority rules and binds identity.
      4. Return unified observation with active_target.

    Args:
        left_obs:  Detections from left camera half.
        right_obs: Detections from right camera half.
        single_target: Enforce single-person constraint.
        face_iou_threshold: IoU threshold for face→human association.

    Returns:
        Unified observation dict with "active_target", "faces", "humans", etc.
    """
    mgr = get_target_manager()

    # Collect
    all_humans: list[dict[str, Any]] = []
    all_humans.extend(left_obs.get("humans", []))
    all_humans.extend(right_obs.get("humans", []))

    all_faces: list[dict[str, Any]] = []
    all_faces.extend(left_obs.get("faces", []))
    all_faces.extend(right_obs.get("faces", []))

    # Deduplicate faces: faces close to each other are the same person
    deduped_faces = _deduplicate_faces(all_faces, iou_threshold=0.5)

    # Feed to TargetManager for single-target decision
    mgr.update_vision(
        humans=all_humans,
        faces=deduped_faces,
    )

    # Get the active target
    active = mgr.get_active_target()

    # Build unified observation
    humans_out = []
    if active.confidence > 0:
        humans_out = [{
            "x": round(active.bbox[0], 4),
            "y": round(active.bbox[1], 4),
            "w": round(active.bbox[2], 4),
            "h": round(active.bbox[3], 4),
            "confidence": round(active.confidence, 4),
            "pose_state": active.pose_state,
            "pose_action": active.pose_action,
            "pose_action_label": active.pose_action_label,
            "keypoints": active.keypoints,
            "track_id": active.track_id,
        }]

    faces_out = []
    # Always include the active target's face if it has one
    if active.face_confidence > 0:
        faces_out.append({
            "track_id": active.face_track_id,
            "x": round(active.face_bbox[0], 4),
            "y": round(active.face_bbox[1], 4),
            "w": round(active.face_bbox[2], 4),
            "h": round(active.face_bbox[3], 4),
            "confidence": round(active.face_confidence, 4),
            "recognized_user": active.identity,
            "identity_confidence": round(active.identity_confidence, 4),
            "identity_state": active.identity_state,
            "quality": round(active.face_confidence, 4),
        })

    # Also include any deduped faces that were NOT already covered by the
    # active target — this prevents dropping faces when MediaPipe fails to
    # detect a human body but YuNet still finds faces.
    seen_keys = {(round(f["x"], 3), round(f["y"], 3)) for f in faces_out}
    for f in deduped_faces:
        fkey = (round(f.get("x", 0), 3), round(f.get("y", 0), 3))
        if fkey not in seen_keys:
            seen_keys.add(fkey)
            faces_out.append({
                "track_id": int(f.get("track_id", -1)),
                "x": round(f.get("x", 0), 4),
                "y": round(f.get("y", 0), 4),
                "w": round(f.get("w", 0), 4),
                "h": round(f.get("h", 0), 4),
                "confidence": round(f.get("confidence", 0), 4),
                "recognized_user": f.get("recognized_user", ""),
                "identity_confidence": round(
                    f.get("identity_confidence", 0), 4,
                ),
                "identity_state": f.get("identity_state", "unverified"),
                "quality": round(f.get("quality", 0), 4),
            })

    return {
        "active_target": active.to_dict(),
        "faces": faces_out,
        "humans": humans_out,
        "hands": [],
        "tracked_objects": [],
        "_fusion_meta": {
            "total_humans_raw": len(all_humans),
            "total_faces_raw": len(all_faces),
            "deduped_faces": len(deduped_faces),
            "target_locked": True,
            "selection_reason": active.selection_reason,
        },
    }


def _deduplicate_faces(
    faces: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Remove duplicate face detections via IoU.

    Since faces are now only detected on the left stereo half,
    simple IoU dedup is sufficient (no cross-half matching needed).
    """
    if len(faces) <= 1:
        return faces

    sorted_faces = sorted(faces, key=lambda f: f.get("confidence", 0), reverse=True)
    kept: list[dict[str, Any]] = []

    for face in sorted_faces:
        fb = (face.get("x", 0), face.get("y", 0),
              face.get("w", 0), face.get("h", 0))
        is_dup = False
        for k in kept:
            kb = (k.get("x", 0), k.get("y", 0),
                  k.get("w", 0), k.get("h", 0))
            if _iou(fb, kb) > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(face)

    return kept
