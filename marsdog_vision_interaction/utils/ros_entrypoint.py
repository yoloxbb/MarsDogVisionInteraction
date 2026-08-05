"""Run installed ROS executables with the project's Python environment."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys


def _candidate_python() -> Path | None:
    """Resolve an explicit, activated, or workspace-local Python runtime."""
    explicit = os.environ.get("MARSDOG_PYTHON", "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate

    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if virtual_env:
        candidate = Path(virtual_env) / "bin" / "python"
        if candidate.is_file():
            return candidate

    # Support ``ros2 launch`` after sourcing the ROS workspace even when the
    # uv virtual environment was not explicitly activated.
    anchors = (Path.cwd(), Path(__file__).resolve())
    checked: set[Path] = set()
    for anchor in anchors:
        for parent in (anchor, *anchor.parents):
            if parent in checked:
                continue
            checked.add(parent)
            candidate = parent / ".venv" / "bin" / "python"
            if candidate.is_file():
                return candidate
    return None


def run_ros_entrypoint(module_name: str) -> None:
    """Re-exec a ROS node module with the project venv when available."""
    candidate = _candidate_python()
    if (
        candidate is not None
        # Do not resolve symlinks here: ``.venv/bin/python`` commonly points
        # to /usr/bin/python3, but invoking through the venv path is what
        # activates the virtual environment's site-packages.
        and candidate.absolute() != Path(sys.executable).absolute()
    ):
        os.execv(
            str(candidate),
            [str(candidate), "-m", module_name, *sys.argv[1:]],
        )

    module = importlib.import_module(module_name)
    module.main()
