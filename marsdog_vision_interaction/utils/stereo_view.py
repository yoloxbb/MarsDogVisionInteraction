"""Select one eye only when an image really looks side-by-side stereo."""

from __future__ import annotations

import numpy as np


def select_camera_view(
    frame: np.ndarray,
    *,
    stereo_enabled: bool,
    view: str = "left",
    min_aspect_ratio: float = 2.2,
) -> tuple[np.ndarray, bool]:
    """Return the inference view and whether a stereo split was applied.

    The WN binocular camera normally publishes two 4:3 320x240 eyes as a
    640x240 image (aspect ratio 2.67).  Some V4L2 profiles instead negotiate a
    normal 16:9 image such as 424x240.  Blindly splitting that image produces
    the 212x240 crop that makes a person disappear at the edge of the frame.
    """
    if not stereo_enabled or frame.ndim < 2:
        return frame, False
    height, width = frame.shape[:2]
    threshold = max(1.0, float(min_aspect_ratio))
    if height <= 0 or width < 2 or width / height < threshold:
        return frame, False
    mid_x = width // 2
    selected = frame[:, mid_x:] if str(view).lower() == "right" else frame[:, :mid_x]
    return selected, True
