"""Plugin Hook Registry — Claude Code-inspired extensible agent lifecycle hooks.
Enables logging, rate-limiting, safety filtering without modifying the agent loop.

.. warning::
    Hooks execute in-process with full access to ``**kwargs``.  Do **not**
    register untrusted callables in security-sensitive contexts.  For
    untrusted plugins, create a scoped :class:`PluginHookRegistry`
    instance rather than using the global singleton.
"""
from __future__ import annotations

from typing import Callable, Any

HOOK_POINTS = (
    "pre_tool_call",
    "post_tool_call",
    "pre_generation",
    "post_generation",
    "on_error",
)


class PluginHookRegistry:
    def __init__(self):
        self._hooks: dict[str, list[Callable]] = {k: [] for k in HOOK_POINTS}

    def register(self, point: str, fn: Callable) -> None:
        if point not in self._hooks:
            raise ValueError(
                f"Unknown hook point {point!r}. Valid points: {HOOK_POINTS}"
            )
        self._hooks[point].append(fn)

    def fire(self, point: str, **kwargs: Any) -> None:
        if point not in self._hooks:
            raise ValueError(
                f"Unknown hook point {point!r}. Valid points: {HOOK_POINTS}"
            )
        for fn in self._hooks[point]:
            try:
                fn(**kwargs)
            except Exception:
                # Hooks must not break the caller.  Log and continue.
                import logging
                logging.getLogger(__name__).exception(
                    "Hook %r at %r raised an exception; continuing", fn, point
                )

    def clear(self, point: str | None = None) -> None:
        if point is None:
            for k in self._hooks:
                self._hooks[k].clear()
        elif point in self._hooks:
            self._hooks[point].clear()
        else:
            raise ValueError(f"Unknown hook point {point!r}.")

    def hook_count(self, point: str) -> int:
        return len(self._hooks.get(point, []))

    def all_points(self) -> tuple[str, ...]:
        return HOOK_POINTS


# Module-level singleton — for trusted / first-party use only.
# Untrusted plugins should receive a scoped registry instance.
HOOK_REGISTRY = PluginHookRegistry()
