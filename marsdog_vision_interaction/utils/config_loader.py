"""YAML configuration loader for MarsDog perception.

Loads perception.yaml and merges with defaults, providing typed access
to provider configs, topic settings, and debug options.
"""

from __future__ import annotations

import os
from pathlib import Path
from string import Template
from typing import Any

import yaml


def _find_project_root(config_path: Path) -> Path:
    """Find a source checkout without assuming a developer-specific path."""
    candidates = (config_path.parent, *config_path.parents, Path.cwd().resolve())
    for candidate in candidates:
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "marsdog_vision_interaction").is_dir()
        ):
            return candidate
    return Path.cwd().resolve()


def _path_variables(config_path: Path) -> dict[str, str]:
    project_override = os.environ.get("MARSDOG_VISION_PROJECT_DIR")
    project_dir = Path(
        project_override or _find_project_root(config_path)
    ).expanduser().resolve()

    model_override = os.environ.get("MARSDOG_VISION_MODEL_DIR")
    if model_override:
        model_dir = Path(model_override).expanduser().resolve()
    else:
        candidates = (
            project_dir / "models" / "vision",
            project_dir.parent / "models" / "vision",
        )
        model_dir = next(
            (item for item in candidates if item.is_dir()),
            candidates[0],
        )

    data_dir = Path(
        os.environ.get("MARSDOG_VISION_DATA_DIR", project_dir / "data")
    ).expanduser().resolve()
    return {
        "MARSDOG_VISION_PROJECT_DIR": str(project_dir),
        "MARSDOG_VISION_MODEL_DIR": str(model_dir),
        "MARSDOG_VISION_DATA_DIR": str(data_dir),
    }


def _expand_variables(value: Any, variables: dict[str, str]) -> Any:
    if isinstance(value, str):
        return Template(value).safe_substitute(variables)
    if isinstance(value, list):
        return [_expand_variables(item, variables) for item in value]
    if isinstance(value, dict):
        return {
            key: _expand_variables(item, variables)
            for key, item in value.items()
        }
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a perception YAML config file.

    Args:
        path: Path to a YAML config file (e.g. config/perception.yaml).

    Returns:
        Parsed config dict. Empty dict if file not found or invalid.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    config_path = Path(path).expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(
            f"Config {config_path} must be a YAML mapping, "
            f"got {type(data).__name__}"
        )

    return _expand_variables(data, _path_variables(config_path))


def load_config_safe(
    path: str | Path,
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load config with fallback to defaults on any error.

    Args:
        path: Path to a YAML config file.
        defaults: Fallback dict if load fails.

    Returns:
        Parsed config dict or defaults.
    """
    try:
        return load_config(path)
    except Exception:
        return defaults if defaults is not None else {}
