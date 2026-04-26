"""torch.profiler wrapper for Aurelius runtime profiling.

Provides a context-manager interface around torch.profiler.profile with
convenience methods for summarising results and exporting Chrome traces.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.profiler

# Registry — shared with compile_manager / memory_profiler pattern
try:
    from .compile_manager import COMPILE_REGISTRY  # noqa: F401 — import to ensure registry exists
except ImportError:
    pass

RUNTIME_REGISTRY: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ProfilerConfig:
    enabled: bool = True
    record_shapes: bool = True
    with_flops: bool = True
    export_path: str = "profile.json"


# ---------------------------------------------------------------------------
# Profiler context manager
# ---------------------------------------------------------------------------


class AureliusProfiler:
    """Context manager wrapping torch.profiler.profile.

    Usage::

        cfg = ProfilerConfig(export_path="run.json")
        profiler = AureliusProfiler(cfg)
        with profiler:
            model(inputs)
        summary = profiler.summarize()
        profiler.export_chrome_trace("trace.json")
    """

    def __init__(self, config: ProfilerConfig | None = None) -> None:
        self.config = config if config is not None else ProfilerConfig()
        self._profiler: torch.profiler.profile | None = None
        self._key_averages: Any = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> AureliusProfiler:
        if not self.config.enabled:
            return self
        activities = [torch.profiler.ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self._profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=self.config.record_shapes,
            with_flops=self.config.with_flops,
        )
        self._profiler.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._profiler is None:
            return
        self._profiler.__exit__(exc_type, exc_val, exc_tb)
        self._key_averages = self._profiler.key_averages()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self) -> dict[str, Any]:
        """Return a summary dict with total FLOPS, memory peak, top-5 ops.

        Returns an empty-but-valid dict when profiling is disabled or has
        not yet been run.
        """
        if self._key_averages is None:
            return {
                "total_flops": 0,
                "memory_peak_mb": 0.0,
                "top5_ops": [],
            }

        total_flops: int = 0
        memory_peak_bytes: int = 0

        for evt in self._key_averages:
            flops = getattr(evt, "flops", 0) or 0
            total_flops += int(flops)
            mem = getattr(evt, "cuda_memory_usage", 0) or 0
            if mem > memory_peak_bytes:
                memory_peak_bytes = int(mem)

        # Top-5 ops by CUDA time (fall back to CPU time on CPU-only runs)
        def _cuda_time(evt: Any) -> float:
            t = getattr(evt, "cuda_time_total", None)
            if t is None or t == 0:
                t = getattr(evt, "cpu_time_total", 0) or 0
            return float(t)

        sorted_evts = sorted(self._key_averages, key=_cuda_time, reverse=True)
        top5 = []
        for evt in sorted_evts[:5]:
            top5.append(
                {
                    "name": evt.key,
                    "cuda_time_us": _cuda_time(evt),
                    "count": getattr(evt, "count", 1),
                }
            )

        return {
            "total_flops": total_flops,
            "memory_peak_mb": memory_peak_bytes / (1024**2),
            "top5_ops": top5,
        }

    # ------------------------------------------------------------------
    # Chrome trace export
    # ------------------------------------------------------------------

    def export_chrome_trace(self, path: str) -> None:
        """Export a Chrome-compatible trace JSON to *path*.

        Delegates to torch.profiler's built-in export when a real profiler
        session exists; otherwise writes a minimal valid trace stub.
        """
        if self._profiler is not None and hasattr(self._profiler, "export_chrome_trace"):
            try:
                self._profiler.export_chrome_trace(path)
                return
            except Exception:  # noqa: S110
                pass  # Fall through to stub writer on any error
        # Write a minimal valid Chrome trace so callers can always read the file
        stub = {"traceEvents": [], "meta": {"aurelius": True}}
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w") as fh:
            json.dump(stub, fh)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
RUNTIME_REGISTRY["profiler"] = AureliusProfiler
