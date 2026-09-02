from pathlib import Path
import sys

from marsdog_vision_interaction.utils.ros_entrypoint import _candidate_python


def test_candidate_python_follows_installed_executable_symlink(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkout = tmp_path / "checkout"
    python = checkout / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    source_script = checkout / "scripts" / "vision_interaction"
    source_script.parent.mkdir()
    source_script.write_text("", encoding="utf-8")
    installed_script = tmp_path / "ros2_ws" / "install" / "vision_interaction"
    installed_script.parent.mkdir(parents=True)
    installed_script.symlink_to(source_script)

    monkeypatch.delenv("MARSDOG_PYTHON", raising=False)
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setattr(sys, "argv", [str(installed_script)])
    monkeypatch.chdir(tmp_path / "ros2_ws")

    assert _candidate_python() == python
