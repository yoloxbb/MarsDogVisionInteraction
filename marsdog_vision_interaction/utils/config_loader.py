"""YAML configuration loader for MarsDog perception.

Loads perception.yaml and merges with defaults, providing typed access
to provider configs, topic settings, and debug options.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


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
        raise ValueError(f"Config {config_path} must be a YAML mapping, got {type(data).__name__}")

    return data


def load_config_safe(path: str | Path, defaults: dict[str, Any] | None = None) -> dict[str, Any]:
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
