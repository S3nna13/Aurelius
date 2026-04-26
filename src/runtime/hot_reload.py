"""Hot-reload manager for configuration and model weights in Aurelius runtime."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReloadEvent:
    resource_type: str
    path: str
    triggered_at: float
    success: bool
    error: str = ""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])


@dataclass(frozen=True)
class ReloadConfig:
    watch_interval_s: float = 5.0
    max_retries: int = 3
    rollback_on_failure: bool = True


@dataclass
class _ResourceEntry:
    resource_type: str
    path: str
    loader_fn: Callable[[str], object]


class HotReloader:
    """Manages hot-reloading of configuration files and model weights."""

    def __init__(self, config: ReloadConfig | None = None) -> None:
        self.config: ReloadConfig = config if config is not None else ReloadConfig()
        self._resources: dict[str, _ResourceEntry] = {}
        self._cache: dict[str, object] = {}
        self._history: list[ReloadEvent] = []

    def register_resource(
        self,
        resource_type: str,
        path: str,
        loader_fn: Callable[[str], object],
    ) -> None:
        """Register a resource type with a path and a loader function."""
        self._resources[resource_type] = _ResourceEntry(
            resource_type=resource_type,
            path=path,
            loader_fn=loader_fn,
        )

    def reload(self, resource_type: str) -> ReloadEvent:
        """Reload the resource by calling its loader_fn. Returns a ReloadEvent."""
        entry = self._resources.get(resource_type)
        if entry is None:
            event = ReloadEvent(
                resource_type=resource_type,
                path="",
                triggered_at=time.monotonic(),
                success=False,
                error=f"Resource type '{resource_type}' is not registered.",
            )
            self._history.append(event)
            return event

        previous_value = self._cache.get(resource_type)
        triggered_at = time.monotonic()

        try:
            result = entry.loader_fn(entry.path)
            self._cache[resource_type] = result
            event = ReloadEvent(
                resource_type=resource_type,
                path=entry.path,
                triggered_at=triggered_at,
                success=True,
            )
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            if self.config.rollback_on_failure and previous_value is not None:
                self._cache[resource_type] = previous_value
            event = ReloadEvent(
                resource_type=resource_type,
                path=entry.path,
                triggered_at=triggered_at,
                success=False,
                error=error_msg,
            )

        self._history.append(event)
        return event

    def get(self, resource_type: str) -> object | None:
        """Return the cached value for a resource type, or None."""
        return self._cache.get(resource_type)

    def reload_history(self) -> list[ReloadEvent]:
        """Return the full list of ReloadEvents in order."""
        return list(self._history)

    def resource_count(self) -> int:
        """Return the number of registered resources."""
        return len(self._resources)


HOT_RELOAD_REGISTRY: dict[str, type[HotReloader]] = {"default": HotReloader}
