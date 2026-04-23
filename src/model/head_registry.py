"""Swappable output-head registry.

This module provides metadata descriptors and a factory for the final
output heads attached to variants of the Aurelius backbone (language
modelling, reward, classification, DPO dual-head, value head, etc.).

The registry holds ``HeadSpec`` descriptors; ``build_head`` turns a spec
plus a ``d_model`` into an ``nn.Module`` suitable for attaching on top of
the backbone hidden state.

Additive-within-file only. No new config flags (family infra).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import torch
from torch import nn


class HeadKind(Enum):
    """Taxonomy of output-head kinds."""

    LM = "lm"
    REWARD = "reward"
    CLASSIFIER = "classifier"
    DUAL = "dual"
    EMBEDDING = "embedding"
    VALUE = "value"
    MULTI_HEAD = "multi_head"


class HeadFactoryError(Exception):
    """Raised when a HeadSpec cannot be validated or built."""


@dataclass(frozen=True)
class HeadSpec:
    """Descriptor for a single output head."""

    id: str
    kind: HeadKind
    output_dim: int
    bias: bool = False
    tied_to_embedding: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise HeadFactoryError("head id must be a non-empty string")
        if not isinstance(self.kind, HeadKind):
            raise HeadFactoryError("head kind must be a HeadKind")
        if not isinstance(self.output_dim, int) or isinstance(self.output_dim, bool):
            raise HeadFactoryError("output_dim must be an int")
        if not isinstance(self.bias, bool):
            raise HeadFactoryError("bias must be a bool")
        if not isinstance(self.tied_to_embedding, bool):
            raise HeadFactoryError("tied_to_embedding must be a bool")
        if not isinstance(self.metadata, dict):
            raise HeadFactoryError("metadata must be a dict")

        if self.kind is HeadKind.REWARD and self.output_dim != 1:
            raise HeadFactoryError("REWARD head requires output_dim == 1")
        if self.kind is HeadKind.VALUE and self.output_dim != 1:
            raise HeadFactoryError("VALUE head requires output_dim == 1")
        if self.kind is HeadKind.CLASSIFIER and self.output_dim <= 0:
            raise HeadFactoryError(
                "CLASSIFIER head requires output_dim > 0 (number of classes)"
            )
        if self.kind is not HeadKind.LM and self.tied_to_embedding:
            raise HeadFactoryError(
                "tied_to_embedding is only valid for LM heads"
            )
        if self.output_dim <= 0:
            raise HeadFactoryError("output_dim must be > 0")
        if self.kind is HeadKind.LM and self.tied_to_embedding:
            # Propagate the tied flag into metadata for downstream consumers.
            self.metadata.setdefault("tied_to_embedding", True)
        if self.kind is HeadKind.MULTI_HEAD:
            sub = self.metadata.get("subhead_names")
            if not isinstance(sub, (list, tuple)) or len(sub) == 0:
                raise HeadFactoryError(
                    "MULTI_HEAD requires metadata['subhead_names'] (non-empty)"
                )
            for name in sub:
                if not isinstance(name, str) or not name:
                    raise HeadFactoryError(
                        "MULTI_HEAD subhead_names entries must be non-empty strings"
                    )


HEAD_REGISTRY: dict[str, HeadSpec] = {}


def register_head(spec: HeadSpec) -> HeadSpec:
    """Register a HeadSpec into the global registry."""
    if not isinstance(spec, HeadSpec):
        raise HeadFactoryError("spec must be a HeadSpec")
    if spec.id in HEAD_REGISTRY:
        raise HeadFactoryError(f"duplicate head id: {spec.id!r}")
    HEAD_REGISTRY[spec.id] = spec
    return spec


def get_head(head_id: str) -> HeadSpec:
    """Fetch a registered HeadSpec by id."""
    if head_id not in HEAD_REGISTRY:
        raise HeadFactoryError(f"unknown head id: {head_id!r}")
    return HEAD_REGISTRY[head_id]


def list_heads() -> tuple[str, ...]:
    """Return the registered head ids in insertion order."""
    return tuple(HEAD_REGISTRY.keys())


def heads_by_kind(kind: HeadKind) -> tuple[HeadSpec, ...]:
    """Return all registered HeadSpecs of a given kind."""
    if not isinstance(kind, HeadKind):
        raise HeadFactoryError("kind must be a HeadKind")
    return tuple(s for s in HEAD_REGISTRY.values() if s.kind is kind)


class _DualHead(nn.Module):
    """DPO-style dual head: LM logits + scalar value."""

    def __init__(self, d_model: int, output_dim: int, bias: bool) -> None:
        super().__init__()
        self.lm_logits = nn.Linear(d_model, output_dim, bias=bias)
        self.value = nn.Linear(d_model, 1, bias=bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover - trivial
        return self.lm_logits(x), self.value(x)


class _MultiHead(nn.Module):
    """Container of named sub-heads."""

    def __init__(
        self,
        d_model: int,
        output_dim: int,
        bias: bool,
        subhead_names: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(d_model, output_dim, bias=bias) for name in subhead_names}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # pragma: no cover
        return {name: head(x) for name, head in self.heads.items()}


def build_head(spec: HeadSpec, d_model: int) -> nn.Module:
    """Construct an ``nn.Module`` from a HeadSpec and backbone width."""
    if not isinstance(spec, HeadSpec):
        raise HeadFactoryError("spec must be a HeadSpec")
    if not isinstance(d_model, int) or d_model <= 0:
        raise HeadFactoryError("d_model must be a positive int")

    if spec.kind is HeadKind.DUAL:
        return _DualHead(d_model, spec.output_dim, spec.bias)
    if spec.kind is HeadKind.MULTI_HEAD:
        names = tuple(spec.metadata["subhead_names"])
        return _MultiHead(d_model, spec.output_dim, spec.bias, names)
    return nn.Linear(d_model, spec.output_dim, bias=spec.bias)


# ---------------------------------------------------------------------------
# Seed heads
# ---------------------------------------------------------------------------

_SEED_HEADS: tuple[HeadSpec, ...] = (
    HeadSpec(
        id="aurelius/default-lm",
        kind=HeadKind.LM,
        output_dim=128000,
        bias=False,
        tied_to_embedding=True,
        metadata={"tied_to_embedding": True},
    ),
    HeadSpec(
        id="aurelius/reward-v1",
        kind=HeadKind.REWARD,
        output_dim=1,
        bias=True,
    ),
    HeadSpec(
        id="aurelius/classifier-binary",
        kind=HeadKind.CLASSIFIER,
        output_dim=2,
        bias=True,
    ),
    HeadSpec(
        id="aurelius/value-head",
        kind=HeadKind.VALUE,
        output_dim=1,
        bias=True,
    ),
)


for _seed in _SEED_HEADS:
    register_head(_seed)


__all__ = [
    "HEAD_REGISTRY",
    "HeadFactoryError",
    "HeadKind",
    "HeadSpec",
    "build_head",
    "get_head",
    "heads_by_kind",
    "list_heads",
    "register_head",
]
