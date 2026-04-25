"""JIT compilation cache for Aurelius runtime.

Caches torch.jit.script / torch.compile artefacts keyed by a content hash
derived from module source code and example input shapes.  Evicts entries
using an LRU policy when the cache grows beyond max_entries.
"""

from __future__ import annotations

import hashlib
import inspect
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

try:
    from .torch_profiler_wrapper import RUNTIME_REGISTRY
except ImportError:
    RUNTIME_REGISTRY: dict[str, object] = {}  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class JITCacheConfig:
    cache_dir: str = ".aurelius_jit_cache"
    max_entries: int = 64


# ---------------------------------------------------------------------------
# Cache implementation
# ---------------------------------------------------------------------------

class JITCache:
    """LRU cache for JIT-compiled nn.Module instances.

    The cache key is the SHA-256 of:
      - module source code (via ``inspect.getsource`` or ``str(module)``)
      - stringified example input shapes

    When a cached entry exists the compiled module is returned directly.
    Otherwise the module is compiled (via ``torch.compile`` if available,
    falling back to ``torch.jit.script``), stored, and returned.  LRU
    eviction maintains at most *max_entries* in memory.
    """

    def __init__(self, config: JITCacheConfig | None = None) -> None:
        self.config = config if config is not None else JITCacheConfig()
        # OrderedDict acts as LRU: most-recently-used at the end
        self._cache: OrderedDict[str, nn.Module] = OrderedDict()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_compile(
        self,
        module: nn.Module,
        example_inputs: list[Any],
    ) -> nn.Module:
        """Return a compiled version of *module*, using cached result if available.

        Args:
            module:         The nn.Module to compile.
            example_inputs: List of example tensors (used for shape hashing and
                            torch.compile tracing).

        Returns:
            A compiled nn.Module (may be the same object if compilation is
            unavailable or if a cached compiled module is returned).
        """
        key = self._make_key(module, example_inputs)

        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]

        # Compile
        compiled = self._compile(module, example_inputs)

        # Store with LRU eviction
        self._cache[key] = compiled
        self._cache.move_to_end(key)
        if len(self._cache) > self.config.max_entries:
            self._cache.popitem(last=False)  # evict LRU (oldest)

        return compiled

    def cache_size(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)

    def clear(self) -> None:
        """Evict all cached entries."""
        self._cache.clear()

    def invalidate(self, module: nn.Module, example_inputs: list[Any]) -> bool:
        """Remove a specific entry from the cache.  Returns True if found."""
        key = self._make_key(module, example_inputs)
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_key(module: nn.Module, example_inputs: list[Any]) -> str:
        """Compute sha256(module_source + input_shapes)."""
        try:
            src = inspect.getsource(type(module))
        except (OSError, TypeError):
            src = str(module)

        shapes = str([
            tuple(t.shape) if isinstance(t, torch.Tensor) else repr(t)
            for t in example_inputs
        ])
        raw = (src + shapes).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _compile(module: nn.Module, example_inputs: list[Any]) -> nn.Module:
        """Attempt torch.compile; fall back to torch.jit.script; fall back to identity."""
        if hasattr(torch, "compile"):
            try:
                return torch.compile(module)  # type: ignore[return-value]
            except Exception:
                pass
        try:
            return torch.jit.script(module)
        except Exception:
            pass
        # Last resort: return module uncompiled
        return module


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
RUNTIME_REGISTRY["jit_cache"] = JITCache
