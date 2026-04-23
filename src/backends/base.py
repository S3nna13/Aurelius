"""Aurelius-native backend adapter base classes.

This module defines the abstract surface that every backend/engine adapter
must implement. It is deliberately backend-agnostic and depends only on the
Python standard library.

A *backend* is a training/math backend (e.g. pytorch, jax). An *engine* is
an inference engine (e.g. vllm, sglang, llama.cpp). Both share the same
:class:`BackendContract` shape for uniform reasoning about versioning and
capability tags.

No foreign imports are permitted in this file.
"""

from __future__ import annotations

import abc
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "BackendAdapterError",
    "BackendContract",
    "BackendAdapter",
    "EngineAdapter",
]


_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")
_BACKEND_NAME_RE = re.compile(r"^[a-z0-9_\-]+$")


class BackendAdapterError(Exception):
    """Raised for any backend-adapter contract or lookup failure."""


@dataclass(frozen=True)
class BackendContract:
    """Versioned identity + capability record for a backend/engine adapter.

    ``backend_name`` is the registry key (lower-snake-or-dash charset).
    ``engine_contract`` and ``adapter_contract`` are independent semver
    tracks so engine-facing code and adapter-facing code can bump
    compatibility separately.
    """

    backend_name: str
    engine_contract: str
    adapter_contract: str
    supports_training: bool
    supports_inference: bool
    capability_tags: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.backend_name, str) or not self.backend_name:
            raise BackendAdapterError(
                f"backend_name must be a non-empty string, got "
                f"{self.backend_name!r}"
            )
        if not _BACKEND_NAME_RE.match(self.backend_name):
            raise BackendAdapterError(
                f"backend_name must match [a-z0-9_-]+, got "
                f"{self.backend_name!r}"
            )

        for vfield in ("engine_contract", "adapter_contract"):
            v = getattr(self, vfield)
            if not isinstance(v, str) or not _SEMVER_RE.match(v):
                raise BackendAdapterError(
                    f"{vfield} must match semver X.Y.Z, got {v!r}"
                )

        for bfield in ("supports_training", "supports_inference"):
            v = getattr(self, bfield)
            if not isinstance(v, bool):
                raise BackendAdapterError(
                    f"{bfield} must be a bool, got {type(v).__name__}"
                )

        if not isinstance(self.capability_tags, tuple):
            raise BackendAdapterError(
                f"capability_tags must be a tuple, got "
                f"{type(self.capability_tags).__name__}"
            )
        for tag in self.capability_tags:
            if not isinstance(tag, str):
                raise BackendAdapterError(
                    f"capability_tags entries must be str, got "
                    f"{type(tag).__name__}"
                )


class BackendAdapter(abc.ABC):
    """Abstract base class for training/math backend adapters.

    Concrete subclasses expose a minimal tensor namespace plus enough
    introspection for the runtime to reason about device placement and
    dtype without touching foreign frameworks directly.
    """

    @property
    @abc.abstractmethod
    def contract(self) -> BackendContract:
        """Versioned identity + capability record for this adapter."""

    @abc.abstractmethod
    def tensor_ns(self) -> object:
        """Return a namespace object exposing ``zeros``, ``ones``, ``as_array``."""

    @abc.abstractmethod
    def dtype_of(self, obj: Any) -> str:
        """Return the dtype of ``obj`` as a native string representation."""

    @abc.abstractmethod
    def device_of(self, obj: Any) -> str:
        """Return the device placement of ``obj`` as a native string."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True iff this adapter's underlying backend is usable."""

    def runtime_info(self) -> dict:
        """Return a dict with at least backend_name + contract + availability."""
        c = self.contract
        return {
            "backend_name": c.backend_name,
            "engine_contract": c.engine_contract,
            "adapter_contract": c.adapter_contract,
            "available": self.is_available(),
        }


class EngineAdapter(abc.ABC):
    """Abstract base class for inference-engine adapters.

    This is a contract adapter only -- real engines subclass this and
    implement their own serving loops. The adapter surface exists so that
    Aurelius core code can discover engine capability without importing
    the engine's Python package at module load time.
    """

    @property
    @abc.abstractmethod
    def contract(self) -> BackendContract:
        """Versioned identity + capability record for this engine adapter."""

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True iff this engine can be invoked in the current env."""

    @abc.abstractmethod
    def describe(self) -> dict:
        """Return a JSON-safe description of this engine adapter."""

    @abc.abstractmethod
    def supported_ops(self) -> tuple[str, ...]:
        """Return the tuple of op names this engine adapter supports."""
