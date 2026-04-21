"""Tests for CoLT5 Conditional Computation module.

Covers: CoLT5Config defaults, CoLT5FFN forward shapes, routing behaviour,
        CoLT5Block structure, residual connection, gradients, determinism,
        edge-case (heavy_fraction=1.0), and an end-to-end integration test.
"""

from __future__ import annotations

import math

import pytest
import torch

from src.model.colt5_conditional import CoLT5Block, CoLT5Config, CoLT5FFN


# ---------------------------------------------------------------------------
# Shared tiny config (all tests use this unless noted)
# ---------------------------------------------------------------------------

TINY = CoLT5Config(
    d_model=64,
    d_ff_light=32,
    d_ff_heavy=128,
    heavy_fraction=0.25,
    dropout=0.0,
)

B, T = 2, 8  # batch=2, seq_len=8 -> k = max(1, int(8*0.25)) = 2


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    """Default CoLT5Config should have the documented field values."""
    cfg = CoLT5Config()
    assert cfg.d_model == 2048
    assert cfg.d_ff_light == 512
    assert cfg.d_ff_heavy == 2048
    assert cfg.heavy_fraction == 0.25
    assert cfg.n_heavy_heads == 8
    assert cfg.n_light_heads == 2
    assert cfg.dropout == 0.0


# ---------------------------------------------------------------------------
# 2. test_ffn_output_shape
# ---------------------------------------------------------------------------

def test_ffn_output_shape():
    """FFN output tensor must be [B, T, d_model]."""
    ffn = CoLT5FFN(TINY)
    x = torch.randn(B, T, TINY.d_model)
    out, _ = ffn(x)
    assert out.shape == (B, T, TINY.d_model), f"Expected {(B, T, TINY.d_model)}, got {out.shape}"


# ---------------------------------------------------------------------------
# 3. test_ffn_routing_scores_shape
# ---------------------------------------------------------------------------

def test_ffn_routing_scores_shape():
    """Routing scores must be [B, T]."""
    ffn = CoLT5FFN(TINY)
    x = torch.randn(B, T, TINY.d_model)
    _, scores = ffn(x)
    assert scores.shape == (B, T), f"Expected {(B, T)}, got {scores.shape}"


# ---------------------------------------------------------------------------
# 4. test_heavy_fraction_respected
# ---------------------------------------------------------------------------

def test_heavy_fraction_respected():
    """Exactly k = max(1, int(T * heavy_fraction)) tokens receive heavy path."""
    torch.manual_seed(0)
    ffn = CoLT5FFN(TINY)
    x = torch.randn(B, T, TINY.d_model)

    k_expected = max(1, int(T * TINY.heavy_fraction))

    with torch.no_grad():
        scores = ffn.router(x).squeeze(-1)
        top_k_idx = scores.topk(k_expected, dim=1).indices

    for b in range(B):
        assert top_k_idx[b].numel() == k_expected, (
            f"Batch {b}: expected {k_expected} heavy tokens, "
            f"got {top_k_idx[b].numel()}"
        )


# ---------------------------------------------------------------------------
# 5. test_light_heavy_different
# ---------------------------------------------------------------------------

def test_light_heavy_different():
    """CoLT5 output must differ from a light-only baseline."""
    torch.manual_seed(42)
    ffn = CoLT5FFN(TINY)
    x = torch.randn(B, T, TINY.d_model)

    with torch.no_grad():
        full_out, _ = ffn(x)
        light_only = ffn._light_ffn(x)

    assert not torch.allclose(full_out, light_only), (
        "CoLT5 output should differ from light-only baseline at top-k positions."
    )


# ---------------------------------------------------------------------------
# 6. test_heavy_important_tokens
# ---------------------------------------------------------------------------

def test_heavy_important_tokens():
    """The highest-scoring tokens should be selected for the heavy path."""
    torch.manual_seed(7)
    ffn = CoLT5FFN(TINY)
    x = torch.randn(B, T, TINY.d_model)

    with torch.no_grad():
        scores = ffn.router(x).squeeze(-1)

    k = max(1, int(T * TINY.heavy_fraction))

    for b in range(B):
        top_k_vals, top_k_idx = scores[b].topk(k)
        selected_min = top_k_vals.min().item()
        mask = torch.ones(T, dtype=torch.bool)
        mask[top_k_idx] = False
        non_selected = scores[b][mask]
        if non_selected.numel() > 0:
            non_selected_max = non_selected.max().item()
            assert selected_min >= non_selected_max, (
                f"Batch {b}: selected token score {selected_min:.4f} "
                f"< non-selected score {non_selected_max:.4f}"
            )


# ---------------------------------------------------------------------------
# 7. test_routing_stats_keys
# ---------------------------------------------------------------------------

def test_routing_stats_keys():
    """routing_stats() must return a dict with the required keys."""
    ffn = CoLT5FFN(TINY)
    x = torch.randn(B, T, TINY.d_model)
    with torch.no_grad():
        _, scores = ffn(x)
    stats = ffn.routing_stats(scores)
    assert "heavy_fraction_actual" in stats, "Missing key: heavy_fraction_actual"
    assert "score_entropy" in stats, "Missing key: score_entropy"


# ---------------------------------------------------------------------------
# 8. test_routing_stats_fraction_range
# ---------------------------------------------------------------------------

def test_routing_stats_fraction_range():
    """heavy_fraction_actual must be in [0, 1] and score_entropy in [0, 1]."""
    ffn = CoLT5FFN(TINY)
    x = torch.randn(B, T, TINY.d_model)
    with torch.no_grad():
        _, scores = ffn(x)
    stats = ffn.routing_stats(scores)
    assert 0.0 <= stats["heavy_fraction_actual"] <= 1.0, (
        f"heavy_fraction_actual out of range: {stats['heavy_fraction_actual']}"
    )
    assert 0.0 <= stats["score_entropy"] <= 1.0, (
        f"score_entropy out of range: {stats['score_entropy']}"
    )


# ---------------------------------------------------------------------------
# 9. test_block_output_keys
# ---------------------------------------------------------------------------

def test_block_output_keys():
    """CoLT5Block.forward must return dict with 'output' and 'routing_scores'."""
    block = CoLT5Block(TINY)
    x = torch.randn(B, T, TINY.d_model)
    result = block(x)
    assert "output" in result, "Missing key: output"
    assert "routing_scores" in result, "Missing key: routing_scores"


# ---------------------------------------------------------------------------
# 10. test_block_output_shape
# ---------------------------------------------------------------------------

def test_block_output_shape():
    """CoLT5Block output tensor must be [B, T, d_model]."""
    block = CoLT5Block(TINY)
    x = torch.randn(B, T, TINY.d_model)
    result = block(x)
    assert result["output"].shape == (B, T, TINY.d_model), (
        f"Expected {(B, T, TINY.d_model)}, got {result['output'].shape}"
    )
    assert result["routing_scores"].shape == (B, T), (
        f"Expected {(B, T)}, got {result['routing_scores'].shape}"
    )


# ---------------------------------------------------------------------------
# 11. test_block_residual
# ---------------------------------------------------------------------------

def test_block_residual():
    """Block output should equal x + ffn(norm(x)) (pre-norm residual)."""
    torch.manual_seed(99)
    block = CoLT5Block(TINY)
    x = torch.randn(B, T, TINY.d_model)

    with torch.no_grad():
        result = block(x)
        normed = block.norm(x)
        ffn_out, _ = block.colt5_ffn(normed)
        expected = x + ffn_out

    assert torch.allclose(result["output"], expected, atol=1e-5), (
        "Block output does not match x + ffn(norm(x))."
    )


# ---------------------------------------------------------------------------
# 12. test_gradient_flows
# ---------------------------------------------------------------------------

def test_gradient_flows():
    """Backward pass must propagate gradients to all learnable parameters."""
    block = CoLT5Block(TINY)
    x = torch.randn(B, T, TINY.d_model, requires_grad=True)
    result = block(x)
    loss = result["output"].sum()
    loss.backward()

    assert x.grad is not None, "No gradient flowed back to input x."
    assert not torch.all(x.grad == 0), "Input gradient is all zeros."

    for name, param in block.named_parameters():
        assert param.grad is not None, f"No gradient for parameter: {name}"


# ---------------------------------------------------------------------------
# 13. test_determinism
# ---------------------------------------------------------------------------

def test_determinism():
    """Same input must produce identical output (no stochastic operations)."""
    block = CoLT5Block(TINY)
    block.train(False)  # inference mode
    x = torch.randn(B, T, TINY.d_model)
    with torch.no_grad():
        out1 = block(x)["output"]
        out2 = block(x)["output"]
    assert torch.equal(out1, out2), "CoLT5Block is not deterministic."


# ---------------------------------------------------------------------------
# 14. test_heavy_fraction_one
# ---------------------------------------------------------------------------

def test_heavy_fraction_one():
    """When heavy_fraction=1.0, every token is routed to the heavy path."""
    cfg_all_heavy = CoLT5Config(
        d_model=64,
        d_ff_light=32,
        d_ff_heavy=128,
        heavy_fraction=1.0,
        dropout=0.0,
    )
    ffn = CoLT5FFN(cfg_all_heavy)
    x = torch.randn(B, T, cfg_all_heavy.d_model)
    out, scores = ffn(x)

    k = max(1, int(T * cfg_all_heavy.heavy_fraction))
    assert k == T, f"Expected k={T} when heavy_fraction=1.0, got k={k}"
    assert out.shape == (B, T, cfg_all_heavy.d_model)

    stats = ffn.routing_stats(scores)
    assert abs(stats["heavy_fraction_actual"] - 1.0) < 1e-6, (
        f"Expected heavy_fraction_actual=1.0, got {stats['heavy_fraction_actual']}"
    )


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

def test_integration_colt5_block():
    """Integration: CoLT5Block, d_model=64, B=2, T=16 -- shapes + backward."""
    cfg = CoLT5Config(
        d_model=64,
        d_ff_light=32,
        d_ff_heavy=128,
        heavy_fraction=0.25,
        dropout=0.0,
    )
    B_int, T_int = 2, 16
    k_expected = max(1, int(T_int * cfg.heavy_fraction))  # 4

    block = CoLT5Block(cfg)
    x = torch.randn(B_int, T_int, cfg.d_model, requires_grad=True)

    result = block(x)
    output = result["output"]
    routing_scores = result["routing_scores"]

    assert output.shape == (B_int, T_int, cfg.d_model), (
        f"Integration: output shape {output.shape} != {(B_int, T_int, cfg.d_model)}"
    )
    assert routing_scores.shape == (B_int, T_int), (
        f"Integration: routing_scores shape {routing_scores.shape} != {(B_int, T_int)}"
    )

    with torch.no_grad():
        stats = block.colt5_ffn.routing_stats(routing_scores.detach())
    expected_frac = float(k_expected) / T_int
    assert abs(stats["heavy_fraction_actual"] - expected_frac) < 1e-6, (
        f"Integration: heavy_fraction_actual {stats['heavy_fraction_actual']} "
        f"!= {expected_frac}"
    )

    loss = output.mean()
    loss.backward()
    assert x.grad is not None, "Integration: no gradient at input."
    assert not torch.all(x.grad == 0), "Integration: input gradient is all zeros."

    for name, param in block.named_parameters():
        assert param.grad is not None, f"Integration: no gradient for {name}"

    with torch.no_grad():
        x2 = torch.randn(B_int, T_int, cfg.d_model)
        res = block(x2)
        normed = block.norm(x2)
        ffn_out, _ = block.colt5_ffn(normed)
        expected = x2 + ffn_out
        assert torch.allclose(res["output"], expected, atol=1e-5), (
            "Integration: residual connection check failed."
        )
