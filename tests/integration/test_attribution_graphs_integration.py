"""Integration tests for attribution_graphs via src.interpretability."""

from __future__ import annotations

import torch
import torch.nn as nn

import src.interpretability as interp


def test_exposed_via_package():
    for name in (
        "AttributionNode",
        "AttributionEdge",
        "AttributionGraph",
        "AttributionGraphBuilder",
    ):
        assert hasattr(interp, name), f"interp missing {name}"


def test_build_graph_on_tiny_model():
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(3, 5),
        nn.Linear(5, 4),
    )
    builder = interp.AttributionGraphBuilder(model, top_k_per_node=2)
    x = torch.randn(1, 3)
    g = builder.build(x, target_layer=-1, target_unit=0)
    assert isinstance(g, interp.AttributionGraph)
    assert len(g.nodes) == 5 + 4
    assert len(g.edges) == 5 * 2
    assert g.is_acyclic_forward()


def test_existing_interp_entries_intact():
    # A couple of known existing modules should still be importable.
    from src.interpretability import (
        logit_lens,  # noqa: F401
        token_attribution,  # noqa: F401
    )
