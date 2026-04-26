"""Integration test for StripedAttention (Brandon et al. 2023).

Builds a StripedAttention with the canonical tiny config (d_model=64, n_heads=4,
window_size=4), feeds a [2, 16, 64] input through the full forward pass, and
verifies that:
  - the output shape is correct
  - a backward pass runs without errors
  - head_type_mask alternates correctly
  - the module is registered in MODEL_COMPONENT_REGISTRY
"""

from __future__ import annotations

import pytest
import torch

from src.model import MODEL_COMPONENT_REGISTRY
from src.model.striped_attention import StripedAttention, StripedAttentionConfig

# ---------------------------------------------------------------------------
# Integration config (tiny, matches spec)
# ---------------------------------------------------------------------------

D_MODEL = 64
N_HEADS = 4
WINDOW_SIZE = 4
BATCH = 2
SEQ = 16


@pytest.fixture(scope="module")
def integration_model() -> StripedAttention:
    cfg = StripedAttentionConfig(d_model=D_MODEL, n_heads=N_HEADS, window_size=WINDOW_SIZE)
    model = StripedAttention(cfg)
    model.train(False)  # inference mode
    return model


@pytest.fixture(scope="module")
def integration_input() -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(BATCH, SEQ, D_MODEL)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_integration_output_shape(
    integration_model: StripedAttention,
    integration_input: torch.Tensor,
) -> None:
    """Forward pass produces the expected output shape [2, 16, 64]."""
    with torch.no_grad():
        out = integration_model(integration_input)
    assert out.shape == (BATCH, SEQ, D_MODEL), (
        f"Expected shape ({BATCH}, {SEQ}, {D_MODEL}), got {tuple(out.shape)}"
    )


def test_integration_backward(integration_input: torch.Tensor) -> None:
    """A backward pass through the full integration model runs without error."""
    cfg = StripedAttentionConfig(d_model=D_MODEL, n_heads=N_HEADS, window_size=WINDOW_SIZE)
    model = StripedAttention(cfg)
    model.train()

    out = model(integration_input)
    loss = out.sum()
    loss.backward()  # should not raise

    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient for parameter: {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for parameter: {name}"


def test_integration_head_type_mask_alternates(
    integration_model: StripedAttention,
) -> None:
    """head_type_mask correctly alternates global/local for n_heads=4."""
    mask = integration_model.head_type_mask()
    assert mask == ["global", "local", "global", "local"], f"Expected alternating mask, got {mask}"


def test_integration_registry() -> None:
    """StripedAttention is registered under 'striped_attention' in MODEL_COMPONENT_REGISTRY."""
    assert "striped_attention" in MODEL_COMPONENT_REGISTRY, (
        "Key 'striped_attention' missing from MODEL_COMPONENT_REGISTRY"
    )
    assert MODEL_COMPONENT_REGISTRY["striped_attention"] is StripedAttention, (
        "Registry value is not StripedAttention class"
    )


def test_integration_output_not_all_zeros(
    integration_model: StripedAttention,
    integration_input: torch.Tensor,
) -> None:
    """Integration forward pass produces non-trivial output."""
    with torch.no_grad():
        out = integration_model(integration_input)
    assert out.abs().max().item() > 1e-6, "Integration output is unexpectedly all zeros"


def test_integration_global_vs_local_heads_differ(
    integration_model: StripedAttention,
) -> None:
    """Global and local heads produce different per-head outputs for long inputs."""
    torch.manual_seed(99)
    T = 20  # longer than window_size=4
    head_dim = D_MODEL // N_HEADS

    q = torch.randn(1, T, head_dim)
    k = torch.randn(1, T, head_dim)
    v = torch.randn(1, T, head_dim)

    out_global = integration_model._full_attention(q, k, v)
    out_local = integration_model._local_attention(q, k, v, WINDOW_SIZE)

    assert not torch.allclose(out_global, out_local, atol=1e-4), (
        "Expected global and local heads to produce different outputs for T > window_size"
    )
