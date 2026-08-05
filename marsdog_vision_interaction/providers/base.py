"""Base provider interface for MarsDog perception."""

from __future__ import annotations

from typing import Any


class BaseProvider:
    """Base class for all perception providers.

    Providers handle specific perception tasks (wakeup, vision, audio, intent, etc.)
    and may wrap hardware adapters or produce mock data.

    Attributes:
        config: Provider configuration dict from YAML.
        available: Whether the provider is initialized and operational.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.available = False

    def start(self) -> None:
        """Initialize and become ready.

        Subclasses must call super().start() after successful initialization.
        Failures should be caught, logged as warnings, and leave
        self.available == False — never raise to the node.
        """
        self.available = True

    def stop(self) -> None:
        """Release resources and become unavailable."""
        self.available = False

    def is_available(self) -> bool:
        """Check if this provider is ready for use."""
        return self.available
