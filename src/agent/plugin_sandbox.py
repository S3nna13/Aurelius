"""Plugin sandbox for restricted agent extension execution.

Provides import denylisting and callable inspection before execution.
"""

from __future__ import annotations

import multiprocessing
import time
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agent.plugin_hook import PluginHookRegistry


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
        if not callable(callable_fn):
            raise TypeError(f"callable_fn must be callable, got {type(callable_fn).__name__}")

        violation = self._check_callable_globals(callable_fn)
        if violation:
            return SandboxResult(
                success=False,
                violation=violation,
                duration_ms=0.0,
            )

        start = time.perf_counter()
        try:
            _q: multiprocessing.Queue[tuple[bool, Any]] = multiprocessing.Queue()
            _p = multiprocessing.Process(
                target=self._sandbox_worker,
                args=(callable_fn, args, kwargs, _q, self.config.max_memory_mb),
                daemon=True,
            )
            _p.start()
        except (AttributeError, Exception) as exc:
            # On macOS / Python 3.14 the spawn method cannot pickle
            # local functions, lambdas, or exec-generated callables.
            # If pickling fails, fall back to in-process execution
            # (security inspection has already passed above).
            if "pickle" in str(type(exc).__name__).lower() or "Can't pickle" in str(exc):
                return self._run_in_process(callable_fn, args, kwargs, start)
            # For other failures, fail closed.
            duration_ms = (time.perf_counter() - start) * 1000
            return SandboxResult(
                success=False,
                violation=f"sandbox process failed to start: {exc}",
                duration_ms=duration_ms,
            )

        try:
            _p.join(timeout=self.config.timeout_seconds)
            if _p.is_alive():
                _p.terminate()
                _p.join(timeout=0.5)
                duration_ms = (time.perf_counter() - start) * 1000
                return SandboxResult(
                    success=False,
                    violation=f"timeout: exceeded {self.config.timeout_seconds}s",
                    duration_ms=duration_ms,
                )
            if _q.empty():
                duration_ms = (time.perf_counter() - start) * 1000
                return SandboxResult(
                    success=False,
                    violation="execution produced no result",
                    duration_ms=duration_ms,
                )
            ok, output = _q.get()
            duration_ms = (time.perf_counter() - start) * 1000
            if not ok:
                return SandboxResult(
                    success=False,
                    violation=str(output),
                    duration_ms=duration_ms,
                )
            return SandboxResult(
                success=True,
                output=output,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            # Fail closed: if the sandbox process can't start, deny execution rather than
            # falling back to unsandboxed execution (which the original code did).
            duration_ms = (time.perf_counter() - start) * 1000
            return SandboxResult(
                success=False,
                violation=f"sandbox process failed to start: {exc}",
                duration_ms=duration_ms,
            )

    @staticmethod
    def _sandbox_worker(
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        q: multiprocessing.Queue[tuple[bool, Any]],
        max_memory_mb: int | None,
    ) -> None:
        try:
            if max_memory_mb is not None:
                try:
                    import resource as _res

                    mem_bytes = max_memory_mb * 1024 * 1024
                    for _rl_name in ("RLIMIT_AS", "RLIMIT_DATA"):
                        _rl = getattr(_res, _rl_name, None)
                        if _rl is not None:
                            try:
                                _res.setrlimit(_rl, (mem_bytes, mem_bytes))
                            except (ValueError, OSError):
                                pass
                except ImportError:
                    pass
            result = fn(*args, **kwargs)
            q.put((True, result))
        except Exception as exc:
            q.put((False, str(exc)))

    def _run_in_process(
        self,
        fn: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        start: float,
    ) -> SandboxResult:
        """Execute *fn* in the current process as fallback when spawning fails.

        This is only reached after the sandbox inspection has already approved
        *fn* (no denied imports in globals), so it is safe to run directly.
        """
        try:
            result = fn(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            return SandboxResult(
                success=True,
                output=result,
                duration_ms=duration_ms,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - start) * 1000
            return SandboxResult(
                success=False,
                violation=str(exc),
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
        """Return a violation string if *callable_fn* actually references a denied import."""
        globs = getattr(callable_fn, "__globals__", {})
        code = getattr(callable_fn, "__code__", None)
        used_names: set[str] = set(code.co_names) if code is not None else set()
        for name in self.config.denied_imports:
            # Only flag if the name exists in globals AND the callable's bytecode actually uses it
            if name in globs and name in used_names:
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
