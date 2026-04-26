"""Variant adapter descriptors (data-only).

This module defines metadata-only descriptors for variant specialization
adapters (LoRA, prompt-tuning, prefix-tuning, head-swap, full-weights delta).
No weight math happens here; the factory dispatches on this metadata.

Pure stdlib (``dataclasses``, ``enum``, ``typing``). Additive-within-file only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AdapterKind(Enum):
    """Taxonomy of variant adapter kinds."""

    LORA = "lora"
    PROMPT_TUNE = "prompt_tune"
    PREFIX_TUNE = "prefix_tune"
    HEAD_SWAP = "head_swap"
    FULL_WEIGHTS_DELTA = "full_weights_delta"
    NONE = "none"


class AdapterValidationError(Exception):
    """Raised when a VariantAdapter fails validation."""


@dataclass(frozen=True)
class VariantAdapter:
    """Descriptor for a variant adapter (data only)."""

    id: str
    kind: AdapterKind
    target_modules: tuple[str, ...] = ()
    rank: int | None = None
    params_count: int | None = None
    weights_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise AdapterValidationError("adapter id must be a non-empty string")
        if not isinstance(self.kind, AdapterKind):
            raise AdapterValidationError("adapter kind must be an AdapterKind")
        if not isinstance(self.target_modules, tuple):
            raise AdapterValidationError("target_modules must be a tuple")
        if self.kind is AdapterKind.LORA:
            if self.rank is None or not isinstance(self.rank, int) or self.rank <= 0:
                raise AdapterValidationError("LORA adapter requires rank > 0")
            if len(self.target_modules) == 0:
                raise AdapterValidationError("LORA adapter requires non-empty target_modules")
        if self.kind is AdapterKind.PREFIX_TUNE:
            if len(self.target_modules) == 0:
                raise AdapterValidationError(
                    "PREFIX_TUNE adapter requires non-empty target_modules"
                )
        if self.kind is AdapterKind.HEAD_SWAP:
            if not self.weights_path:
                raise AdapterValidationError("HEAD_SWAP adapter requires weights_path")


VARIANT_ADAPTER_REGISTRY: dict[str, VariantAdapter] = {}
VARIANT_ADAPTER_ATTACHMENTS: dict[str, list[str]] = {}


def register_adapter(adapter: VariantAdapter) -> VariantAdapter:
    """Register an adapter. Raises on duplicate id."""
    if not isinstance(adapter, VariantAdapter):
        raise AdapterValidationError("register_adapter expects a VariantAdapter")
    if adapter.id in VARIANT_ADAPTER_REGISTRY:
        raise AdapterValidationError(f"adapter id already registered: {adapter.id!r}")
    VARIANT_ADAPTER_REGISTRY[adapter.id] = adapter
    return adapter


def get_adapter(id: str) -> VariantAdapter:
    """Look up adapter by id. Raises KeyError if missing."""
    if id not in VARIANT_ADAPTER_REGISTRY:
        raise KeyError(f"unknown adapter id: {id!r}")
    return VARIANT_ADAPTER_REGISTRY[id]


def list_adapters() -> tuple[str, ...]:
    """Return registered adapter ids in deterministic (sorted) order."""
    return tuple(sorted(VARIANT_ADAPTER_REGISTRY.keys()))


def adapters_for_variant(variant_id: str) -> tuple[VariantAdapter, ...]:
    """Return adapters whose id starts with ``"{variant_id}/"``."""
    prefix = f"{variant_id}/"
    matches = [
        adapter for key, adapter in VARIANT_ADAPTER_REGISTRY.items() if key.startswith(prefix)
    ]
    matches.sort(key=lambda a: a.id)
    return tuple(matches)


def attach_to_variant(variant_id: str, adapter_id: str) -> None:
    """Record that ``adapter_id`` is attached to ``variant_id``."""
    if adapter_id not in VARIANT_ADAPTER_REGISTRY:
        raise KeyError(f"unknown adapter id: {adapter_id!r}")
    VARIANT_ADAPTER_ATTACHMENTS.setdefault(variant_id, []).append(adapter_id)


__all__ = [
    "AdapterKind",
    "AdapterValidationError",
    "VARIANT_ADAPTER_ATTACHMENTS",
    "VARIANT_ADAPTER_REGISTRY",
    "VariantAdapter",
    "adapters_for_variant",
    "attach_to_variant",
    "get_adapter",
    "list_adapters",
    "register_adapter",
]
