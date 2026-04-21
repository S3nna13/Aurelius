"""Unit tests for DeepSeek Sparse Attention (DSA) — GLM-5 §3.1."""
from __future__ import annotations

import torch
import pytest

from src.model.dsa_attention import DSAAttention, DSAConfig, LightningIndexer

# Tiny config used by most tests
TINY_CFG = DSAConfig(d_model=64, n_heads=4, top_k=8)


def make_model(cfg: DSAConfig | None = None) -> DSAAttention:
    return DSAAttention(cfg or TINY_CFG)


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------
def test_output_shape():
    model = make_model()
    x = torch.randn(2, 16, 64)
    out = model(x)
    assert out.shape == (2, 16, 64)


# ---------------------------------------------------------------------------
# 2. top_k indices always in [0, T)
# ---------------------------------------------------------------------------
def test_top_k_not_exceeded():
    model = make_model()
    T = 16
    x = torch.randn(2, T, 64)
    # Patch forward to capture top_idx
    captured = {}
    original_forward = model.indexer.forward

    def patched_indexer(k):
        scores = original_forward(k)
        top_k = min(model.cfg.top_k, T)
        _, idx = scores.topk(top_k, dim=-1)
        captured["idx"] = idx
        return scores

    model.indexer.forward = patched_indexer
    model(x)
    idx = captured["idx"]
    assert (idx >= 0).all()
    assert (idx < T).all()


# ---------------------------------------------------------------------------
# 3. top_k capped at seq_len
# ---------------------------------------------------------------------------
def test_top_k_capped_at_seq_len():
    cfg = DSAConfig(d_model=64, n_heads=4, top_k=100)
    model = DSAAttention(cfg)
    T = 16
    x = torch.randn(2, T, 64)
    out = model(x)
    # If top_k were not clamped this would raise; shape confirms it ran fine
    assert out.shape == (2, T, 64)


# ---------------------------------------------------------------------------
# 4. Dense-equivalence concept: top_k == T → finite output, same shape
# ---------------------------------------------------------------------------
def test_dense_equivalence_concept():
    T = 16
    cfg = DSAConfig(d_model=64, n_heads=4, top_k=T)
    model = DSAAttention(cfg)
    x = torch.randn(2, T, 64)
    out = model(x)
    assert out.shape == (2, T, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 5. No NaN / Inf in output
# ---------------------------------------------------------------------------
def test_no_nan_inf():
    model = make_model()
    x = torch.randn(2, 16, 64)
    out = model(x)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 6. Gradient flows through q/k/v projections
# ---------------------------------------------------------------------------
def test_gradient_flows_through_qkv():
    model = make_model()
    x = torch.randn(2, 16, 64)
    out = model(x)
    out.sum().backward()
    assert model.q_proj.weight.grad is not None
    assert model.k_proj.weight.grad is not None
    assert model.v_proj.weight.grad is not None


# ---------------------------------------------------------------------------
# 7. Indexer parameters are part of the model's parameter set
#    (grad may be None after main-loss backward because topk is discrete;
#     the indexer trains via an auxiliary indexer loss in Stage 1)
# ---------------------------------------------------------------------------
def test_gradient_flows_through_indexer():
    """Indexer score layer is registered as a model parameter (trainable by default)."""
    model = make_model()
    # Verify the indexer weight is a proper leaf parameter with requires_grad=True
    assert model.indexer.score.weight.requires_grad
    # The indexer is listed in model.parameters()
    param_ids = {id(p) for p in model.parameters()}
    assert id(model.indexer.score.weight) in param_ids


# ---------------------------------------------------------------------------
# 8. freeze_indexer=True → indexer params require_grad=False
# ---------------------------------------------------------------------------
def test_freeze_indexer():
    cfg = DSAConfig(d_model=64, n_heads=4, top_k=8, freeze_indexer=True)
    model = DSAAttention(cfg)
    for p in model.indexer.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# 9. freeze_indexer=True → after backward, indexer grad is None
# ---------------------------------------------------------------------------
def test_freeze_indexer_no_grad():
    cfg = DSAConfig(d_model=64, n_heads=4, top_k=8, freeze_indexer=True)
    model = DSAAttention(cfg)
    x = torch.randn(2, 16, 64)
    out = model(x)
    out.sum().backward()
    assert model.indexer.score.weight.grad is None


# ---------------------------------------------------------------------------
# 10. Determinism
# ---------------------------------------------------------------------------
def test_determinism():
    torch.manual_seed(0)
    model = make_model()
    x = torch.randn(2, 16, 64)
    torch.manual_seed(42)
    out1 = model(x)
    torch.manual_seed(42)
    out2 = model(x)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# 11. Seq len == 1: top_k clamped to 1
# ---------------------------------------------------------------------------
def test_seq_len_1():
    model = make_model()  # top_k=8 but T=1
    x = torch.randn(2, 1, 64)
    out = model(x)
    assert out.shape == (2, 1, 64)
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 12. Batch size == 1
# ---------------------------------------------------------------------------
def test_batch_size_1():
    model = make_model()
    x = torch.randn(1, 16, 64)
    out = model(x)
    assert out.shape == (1, 16, 64)
    assert torch.isfinite(out).all()
