"""Plugin sandbox for restricted agent extension execution.

Provides import denylisting and callable inspection before execution.
"""

from __future__ import annotations

import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.agent.plugin_hook import PluginHookRegistry


class SandboxViolationError(Exception):
    """Raised when a plugin violates sandbox policy."""


@dataclass
class SandboxConfig:
    """Configuration for plugin sandbox restrictions."""

    timeout_seconds: float = 5.0
    max_memory_mb: int | None = None
    denied_imports: list[str] = field(default_factory=lambda: ["os", "subprocess", "sys", "socket"])
    allow_network: bool = False

    def __post_init__(self) -> None:
        if self.max_memory_mb is not None and self.max_memory_mb < 0:
            raise ValueError("max_memory_mb must be >= 0")


@dataclass
class SandboxResult:
    """Result of a sandboxed execution attempt."""

    success: bool
    output: Any | None = None
    violation: str | None = None
    duration_ms: float = 0.0


@dataclass
class PluginSandbox:
    """Inspects and optionally executes callables under sandbox policy."""

    config: SandboxConfig = field(default_factory=SandboxConfig)

    def run(
        self,
        callable_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> SandboxResult:
        """Execute *callable_fn* after inspecting its globals for denied imports."""
        if not callable(callable_fn):
            raise TypeError(f"callable_fn must be callable, got {type(callable_fn).__name__}")

        # Check globals for denied imports
        violation = self._check_callable_globals(callable_fn)
        if violation:
            return SandboxResult(
                success=False,
                violation=violation,
                duration_ms=0.0,
            )

        start = time.perf_counter()
        try:
            output = callable_fn(*args, **kwargs)
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            return SandboxResult(
                success=False,
                violation=str(exc),
                duration_ms=duration_ms,
            )

        duration_ms = (time.perf_counter() - start) * 1000
        return SandboxResult(
            success=True,
            output=output,
            duration_ms=duration_ms,
        )

    def run_hook(
        self,
        registry: PluginHookRegistry,
        point: str,
        **kwargs: Any,
    ) -> SandboxResult:
        """Run all hooks registered at *point* through the sandbox individually.

        Returns ``success=True`` only if every hook passes sandbox inspection
        and executes without error.
        """
        if point not in registry.all_points():
            return SandboxResult(
                success=False,
                violation=f"unknown hook point: {point!r}",
                duration_ms=0.0,
            )

        start = time.perf_counter()
        overall_success = True
        for fn in registry._hooks.get(point, []):
            result = self.run(fn, **kwargs)
            if not result.success:
                overall_success = False
                duration_ms = (time.perf_counter() - start) * 1000
                return SandboxResult(
                    success=False,
                    violation=result.violation,
                    duration_ms=duration_ms,
                )

        duration_ms = (time.perf_counter() - start) * 1000
        return SandboxResult(
            success=overall_success,
            duration_ms=duration_ms,
        )

    def check_module(self, module: types.ModuleType) -> list[str]:
        """Inspect *module* for any imported names matching denied imports.

        Returns a list of violation strings like ``"denied import: os"``.
        """
        violations: list[str] = []
        module_dict = getattr(module, "__dict__", {})
        for name in self.config.denied_imports:
            if name in module_dict:
                violations.append(f"denied import: {name}")
        return violations

    def _check_callable_globals(self, callable_fn: Callable[..., Any]) -> str | None:
        """Return a violation string if *callable_fn*'s globals contain a denied import."""
        globs = getattr(callable_fn, "__globals__", {})
        for name in self.config.denied_imports:
            if name in globs:
                return f"denied import: {name}"
        return None


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

DEFAULT_PLUGIN_SANDBOX: PluginSandbox = PluginSandbox()

PLUGIN_SANDBOX_REGISTRY: dict[str, PluginSandbox] = {
    "default": DEFAULT_PLUGIN_SANDBOX,
}


__all__ = [
    "DEFAULT_PLUGIN_SANDBOX",
    "PLUGIN_SANDBOX_REGISTRY",
    "PluginSandbox",
    "SandboxConfig",
    "SandboxResult",
    "SandboxViolationError",
]
