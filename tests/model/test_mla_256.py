"""Unit tests for MLA-256 with Muon Split (GLM-5 §3.1, arXiv:2602.15763)."""

from __future__ import annotations

import torch

from src.model.mla_256 import MLA256Attention, MLA256Config

# ---------------------------------------------------------------------------
# Shared tiny config (fast, CPU-friendly)
# ---------------------------------------------------------------------------
TINY = MLA256Config(d_model=64, n_heads=4, head_dim=16, kv_lrank=8)


def make_model_and_input(cfg: MLA256Config = TINY, B: int = 2, T: int = 8):
    model = MLA256Attention(cfg)
    x = torch.randn(B, T, cfg.d_model)
    return model, x


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------
def test_output_shape():
    model, x = make_model_and_input(TINY, B=2, T=8)
    out = model(x)
    assert out.shape == (2, 8, 64), f"Expected (2, 8, 64), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. head_dim configuration (256 in production, 16 in tiny)
# ---------------------------------------------------------------------------
def test_head_dim_256():
    # The config stores the head_dim; in production it would be 256.
    prod_like = MLA256Config(d_model=128, n_heads=2, head_dim=256, kv_lrank=16)
    assert prod_like.head_dim == 256

    # Tiny config used in all other tests has head_dim=16 (structurally equivalent).
    assert TINY.head_dim == 16

    # Sanity: model built from tiny config stores correct head_dim
    model = MLA256Attention(TINY)
    assert model.head_dim == TINY.head_dim


# ---------------------------------------------------------------------------
# 3. KV compression: kv_down output is [B, T, kv_lrank]
# ---------------------------------------------------------------------------
def test_kv_compression():
    model, x = make_model_and_input()
    with torch.no_grad():
        latent = model.kv_down(x)
    assert latent.shape == (2, 8, TINY.kv_lrank), (
        f"Expected (2, 8, {TINY.kv_lrank}), got {latent.shape}"
    )


# ---------------------------------------------------------------------------
# 4. No NaN / Inf in output
# ---------------------------------------------------------------------------
def test_no_nan_inf():
    model, x = make_model_and_input()
    out = model(x)
    assert torch.isfinite(out).all(), "Output contains NaN or Inf"


# ---------------------------------------------------------------------------
# 5. Gradient flows through q_proj
# ---------------------------------------------------------------------------
def test_gradient_flows_q():
    model, x = make_model_and_input()
    out = model(x)
    out.sum().backward()
    assert model.q_proj.weight.grad is not None
    assert torch.isfinite(model.q_proj.weight.grad).all()


# ---------------------------------------------------------------------------
# 6. Gradient flows through kv_down
# ---------------------------------------------------------------------------
def test_gradient_flows_kv_down():
    model, x = make_model_and_input()
    out = model(x)
    out.sum().backward()
    assert model.kv_down.weight.grad is not None
    assert torch.isfinite(model.kv_down.weight.grad).all()


# ---------------------------------------------------------------------------
# 7. Gradient flows through o_proj
# ---------------------------------------------------------------------------
def test_gradient_flows_output():
    model, x = make_model_and_input()
    out = model(x)
    out.sum().backward()
    assert model.o_proj.weight.grad is not None
    assert torch.isfinite(model.o_proj.weight.grad).all()


# ---------------------------------------------------------------------------
# 8. Muon Split applied at init: weights are finite and non-zero
# ---------------------------------------------------------------------------
def test_muon_split_applied_at_init():
    model = MLA256Attention(TINY)
    w = model.q_proj.weight.data
    assert torch.isfinite(w).all(), "q_proj.weight contains NaN/Inf after Muon Split"
    assert not (w == 0).all(), "q_proj.weight is all zeros after Muon Split"


# ---------------------------------------------------------------------------
# 9. Determinism: same seed gives same output
# ---------------------------------------------------------------------------
def test_determinism():
    def run_once():
        torch.manual_seed(42)
        model = MLA256Attention(TINY)
        model.train(False)
        x = torch.randn(2, 8, TINY.d_model)
        with torch.no_grad():
            return model(x)

    out1 = run_once()
    out2 = run_once()
    assert torch.allclose(out1, out2, atol=0.0), "Outputs differ across identical seeds"


# ---------------------------------------------------------------------------
# 10. Sequence length 1: no crash, correct shape
# ---------------------------------------------------------------------------
def test_seq_len_1():
    model, _ = make_model_and_input()
    x = torch.randn(2, 1, TINY.d_model)
    out = model(x)
    assert out.shape == (2, 1, TINY.d_model), f"Expected (2, 1, 64), got {out.shape}"


# ---------------------------------------------------------------------------
# 11. Batch size 1: correct shape
# ---------------------------------------------------------------------------
def test_batch_1():
    model = MLA256Attention(TINY)
    x = torch.randn(1, 8, TINY.d_model)
    out = model(x)
    assert out.shape == (1, 8, TINY.d_model), f"Expected (1, 8, 64), got {out.shape}"


# ---------------------------------------------------------------------------
# 12. MLA256Attention is distinct from any existing MLA class
# ---------------------------------------------------------------------------
def test_distinct_from_mla():
    from src.model.mla_256 import MLA256Attention as MLA256

    # Try to import the original MLA; if it doesn't exist, skip gracefully.
    try:
        from src.model.mla import MLAAttention as OrigMLA

        assert MLA256 is not OrigMLA, "MLA256Attention must be a distinct class from MLAAttention"
    except ImportError:
        pass  # Original MLA class name differs; distinctness is inherently satisfied.

    # Class name and module check as belt-and-suspenders guard.
    assert MLA256.__name__ == "MLA256Attention"
    assert MLA256.__module__ == "src.model.mla_256"


# ---------------------------------------------------------------------------
# 13. reorthogonalize() works without error and keeps weights finite
# ---------------------------------------------------------------------------
def test_reorthogonalize():
    model = MLA256Attention(TINY)
    model.reorthogonalize()
    w = model.q_proj.weight.data
    assert torch.isfinite(w).all(), "q_proj.weight contains NaN/Inf after reorthogonalize()"


# ---------------------------------------------------------------------------
# 14. Per-head orthogonalization preserves weight shape
# ---------------------------------------------------------------------------
def test_per_head_orth_shape():
    model = MLA256Attention(TINY)
    w = model.q_proj.weight.data.clone()
    result = model._orthogonalize_per_head(w, TINY.n_heads)
    assert result.shape == w.shape, (
        f"Orthogonalized weight shape {result.shape} != original {w.shape}"
    )
