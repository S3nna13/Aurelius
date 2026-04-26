"""Tests for nGPT (Normalized Transformer with Representation Learning on the
Hypersphere).

Tiny config used throughout:
    d_model=64, n_heads=4, head_dim=16, d_ff=128, n_layers=2,
    vocab_size=256, max_seq_len=32
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from aurelius.model.ngpt import (
    NGPTAttention,
    NGPTBlock,
    NGPTConfig,
    NGPTModel,
    NormalizedEmbedding,
    normalize_weights,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

TINY_CFG = NGPTConfig(
    d_model=64,
    n_heads=4,
    head_dim=16,
    d_ff=128,
    n_layers=2,
    vocab_size=256,
    max_seq_len=32,
    alpha_attn=0.05,
    alpha_mlp=0.05,
)

B, T = 2, 8  # batch size, sequence length


@pytest.fixture
def cfg() -> NGPTConfig:
    return TINY_CFG


@pytest.fixture
def model(cfg: NGPTConfig) -> NGPTModel:
    torch.manual_seed(0)
    return NGPTModel(cfg).eval()


@pytest.fixture
def input_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, TINY_CFG.vocab_size, (B, T))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _row_norms(weight: torch.Tensor) -> torch.Tensor:
    """Return per-row (output-neuron) L2 norms of a 2-D weight matrix."""
    return weight.norm(dim=1)


# ---------------------------------------------------------------------------
# Test 1 — Hidden states after each block have L2 norm ~= 1.0 (+/-0.01)
# ---------------------------------------------------------------------------


def test_hidden_state_norms_unit(cfg: NGPTConfig, input_ids: torch.Tensor) -> None:
    """All hidden states emitted by every nGPTBlock must lie on the unit sphere."""
    torch.manual_seed(0)
    model = NGPTModel(cfg).eval()

    with torch.no_grad():
        x = model.embedding(input_ids)  # (B, T, d_model)
        for block in model.blocks:
            x = block(x)
            norms = x.norm(dim=-1)  # (B, T)
            assert norms.allclose(torch.ones_like(norms), atol=0.01), (
                f"Hidden-state norms not ~= 1.0: min={norms.min():.4f}, max={norms.max():.4f}"
            )


# ---------------------------------------------------------------------------
# Test 2 — Output logits are NOT normalized (norms should vary)
# ---------------------------------------------------------------------------


def test_output_logits_not_normalized(model: NGPTModel, input_ids: torch.Tensor) -> None:
    """The lm_head output must NOT be unit-normalized; norms should be non-constant."""
    with torch.no_grad():
        logits = model(input_ids)  # (B, T, vocab_size)
    norms = logits.norm(dim=-1)  # (B, T)
    # norms should not all equal 1.0 -- check that variance is non-trivial
    assert norms.std() > 1e-3 or not norms.allclose(torch.ones_like(norms), atol=0.01), (
        "Logit row norms appear to be unit-normalized -- expected raw logits"
    )


# ---------------------------------------------------------------------------
# Test 3 — Forward pass output shape is (B, T, vocab_size)
# ---------------------------------------------------------------------------


def test_output_shape(model: NGPTModel, cfg: NGPTConfig, input_ids: torch.Tensor) -> None:
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (B, T, cfg.vocab_size), (
        f"Expected shape {(B, T, cfg.vocab_size)}, got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# Test 4 — No NaN or Inf in forward pass
# ---------------------------------------------------------------------------


def test_no_nan_inf(model: NGPTModel, input_ids: torch.Tensor) -> None:
    with torch.no_grad():
        logits = model(input_ids)
    assert torch.isfinite(logits).all(), "Forward pass produced NaN or Inf values"


# ---------------------------------------------------------------------------
# Test 5 — loss.backward() succeeds
# ---------------------------------------------------------------------------


def test_backward_succeeds(cfg: NGPTConfig, input_ids: torch.Tensor) -> None:
    torch.manual_seed(0)
    model = NGPTModel(cfg).train()
    logits = model(input_ids)  # (B, T, vocab_size)
    targets = input_ids  # teacher-force on itself
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, cfg.vocab_size),
        targets.view(-1),
    )
    loss.backward()
    # Check that at least some gradients exist and are finite
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed after backward()"
    for g in grads:
        assert torch.isfinite(g).all(), "NaN/Inf gradient encountered"


# ---------------------------------------------------------------------------
# Test 6 — normalize_weights makes weight row norms ~= 1.0
# ---------------------------------------------------------------------------


def test_normalize_weights(cfg: NGPTConfig) -> None:
    torch.manual_seed(42)
    # Create a model with intentionally scaled weights
    model = NGPTModel(cfg)
    # Corrupt weights so norms are not 1
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.mul_(5.0)

    normalize_weights(model)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            row_norms = _row_norms(m.weight)
            assert row_norms.allclose(torch.ones_like(row_norms), atol=0.01), (
                f"After normalize_weights, row norms not ~= 1.0: "
                f"min={row_norms.min():.4f}, max={row_norms.max():.4f}"
            )


# ---------------------------------------------------------------------------
# Test 7 — NormalizedEmbedding returns unit vectors
# ---------------------------------------------------------------------------


def test_normalized_embedding_unit_norms(cfg: NGPTConfig) -> None:
    torch.manual_seed(0)
    emb = NormalizedEmbedding(cfg.vocab_size, cfg.d_model)
    ids = torch.randint(0, cfg.vocab_size, (B, T))
    with torch.no_grad():
        out = emb(ids)  # (B, T, d_model)
    norms = out.norm(dim=-1)  # (B, T)
    assert norms.allclose(torch.ones_like(norms), atol=1e-5), (
        f"NormalizedEmbedding norms not ~= 1.0: min={norms.min():.6f}, max={norms.max():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 8 — NGPTAttention output shape is correct
# ---------------------------------------------------------------------------


def test_attention_output_shape(cfg: NGPTConfig) -> None:
    torch.manual_seed(0)
    attn = NGPTAttention(cfg).eval()
    x = torch.randn(B, T, cfg.d_model)
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    with torch.no_grad():
        out = attn(x)
    assert out.shape == (B, T, cfg.d_model), (
        f"Expected attention output shape {(B, T, cfg.d_model)}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 9 — NGPTBlock preserves hidden dim
# ---------------------------------------------------------------------------


def test_block_preserves_hidden_dim(cfg: NGPTConfig) -> None:
    torch.manual_seed(0)
    block = NGPTBlock(cfg).eval()
    x = torch.randn(B, T, cfg.d_model)
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    with torch.no_grad():
        out = block(x)
    assert out.shape == (B, T, cfg.d_model), (
        f"NGPTBlock changed hidden dim: expected {(B, T, cfg.d_model)}, got {out.shape}"
    )


# ---------------------------------------------------------------------------
# Test 10 — Deterministic output with same seed
# ---------------------------------------------------------------------------


def test_deterministic_with_seed(cfg: NGPTConfig, input_ids: torch.Tensor) -> None:
    torch.manual_seed(7)
    model_a = NGPTModel(cfg).eval()
    with torch.no_grad():
        logits_a = model_a(input_ids)

    torch.manual_seed(7)
    model_b = NGPTModel(cfg).eval()
    with torch.no_grad():
        logits_b = model_b(input_ids)

    assert torch.allclose(logits_a, logits_b, atol=1e-6), (
        "Models initialized with the same seed produced different outputs"
    )


# ---------------------------------------------------------------------------
# Test 11 — Different inputs give different outputs
# ---------------------------------------------------------------------------


def test_different_inputs_different_outputs(cfg: NGPTConfig) -> None:
    torch.manual_seed(0)
    model = NGPTModel(cfg).eval()

    ids_a = torch.zeros(1, T, dtype=torch.long)
    ids_b = torch.ones(1, T, dtype=torch.long)

    with torch.no_grad():
        out_a = model(ids_a)
        out_b = model(ids_b)

    assert not torch.allclose(out_a, out_b, atol=1e-4), (
        "Different input IDs produced identical logits"
    )


# ---------------------------------------------------------------------------
# Test 12 — Gradient flows to embedding weights
# ---------------------------------------------------------------------------


def test_gradient_flows_to_embedding(cfg: NGPTConfig, input_ids: torch.Tensor) -> None:
    torch.manual_seed(0)
    model = NGPTModel(cfg).train()

    logits = model(input_ids)
    loss = logits.sum()
    loss.backward()

    assert model.embedding.weight.grad is not None, (
        "Embedding weight has no gradient after backward()"
    )
    assert torch.isfinite(model.embedding.weight.grad).all(), (
        "Embedding weight gradient contains NaN/Inf"
    )
    assert model.embedding.weight.grad.abs().sum() > 0, "Embedding weight gradient is all zeros"
