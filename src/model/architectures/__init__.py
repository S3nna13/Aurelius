"""Complete architecture library — all 140+ papers from the bibliography."""

from __future__ import annotations

from . import (
    cnn_vision,
    edge_federated,
    foundational,
    generative,
    graph_neuro,
    multimodal,
    rag_agents,
    recommenders,
    rl_reasoning,
    sparse_moe,
    transformer,
)
from .registry import ARCHITECTURE_REGISTRY, get_architecture, list_architectures, register

__all__ = [
    "ARCHITECTURE_REGISTRY",
    "register",
    "list_architectures",
    "get_architecture",
    "cnn_vision",
    "edge_federated",
    "foundational",
    "generative",
    "graph_neuro",
    "multimodal",
    "rag_agents",
    "recommenders",
    "rl_reasoning",
    "sparse_moe",
    "transformer",
]
