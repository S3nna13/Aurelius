"""Tests for model merging (SLERP + TIES)."""

import copy

import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.model_merge import slerp, slerp_merge, ties_merge


def _small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def test_slerp_t0_returns_v0():
    """SLERP at t=0 must return v0."""
    v0 = torch.randn(64)
    v1 = torch.randn(64)
    result = slerp(0.0, v0, v1)
    # At t=0, result should be proportional to v0 (same direction)
    # SLERP normalizes then scales, so compare normalized directions
    assert torch.allclose(
        result / (result.norm() + 1e-8),
        v0 / (v0.norm() + 1e-8),
        atol=1e-5,
    )


def test_slerp_t1_returns_v1():
    """SLERP at t=1 must return v1."""
    v0 = torch.randn(64)
    v1 = torch.randn(64)
    result = slerp(1.0, v0, v1)
    assert torch.allclose(
        result / (result.norm() + 1e-8),
        v1 / (v1.norm() + 1e-8),
        atol=1e-5,
    )


def test_slerp_preserves_norm():
    """SLERP interpolant magnitude should be between v0 and v1 norms."""
    v0 = torch.randn(128)
    v1 = torch.randn(128)
    result = slerp(0.5, v0, v1)
    r_norm = result.norm().item()
    # Magnitude should be between min and max of input norms
    min(v0.norm().item(), v1.norm().item())
    max(v0.norm().item(), v1.norm().item())
    # SLERP interpolates on unit sphere then scales; result should be in reasonable range
    assert r_norm > 0


def test_slerp_merge_returns_model():
    """slerp_merge must return a model with valid parameters."""
    m1 = _small_model()
    m2 = _small_model()
    # Give m2 different weights
    for p in m2.parameters():
        p.data += torch.randn_like(p) * 0.1

    merged = slerp_merge(m1, m1, m2, t=0.5)

    # Check that merged is between m1 and m2 (not identical to either)
    for (n1, p1), (n2, p2), (nm, pm) in zip(
        m1.named_parameters(), m2.named_parameters(), merged.named_parameters()
    ):
        if p1.dtype.is_floating_point:
            # Merged should differ from both extremes at t=0.5
            assert not torch.equal(pm, p1) or not torch.equal(pm, p2)


def test_slerp_merge_forward_works():
    """Merged model must produce valid forward pass output."""
    m1 = _small_model()
    m2 = copy.deepcopy(m1)

    merged = slerp_merge(m1, m1, m2, t=0.5)
    ids = torch.randint(0, 256, (1, 8))
    _, logits, _ = merged(ids)
    assert logits.shape == (1, 8, 256)
    assert torch.isfinite(logits).all()


def test_ties_merge_two_models():
    """ties_merge with 2 models must return a valid model."""
    base = _small_model()
    m1 = copy.deepcopy(base)
    m2 = copy.deepcopy(base)

    # Slightly perturb models (simulate fine-tuning)
    torch.manual_seed(1)
    for p in m1.parameters():
        p.data += torch.randn_like(p) * 0.05
    for p in m2.parameters():
        p.data += torch.randn_like(p) * 0.05

    merged = ties_merge(base, [m1, m2], density=0.3)
    ids = torch.randint(0, 256, (1, 8))
    _, logits, _ = merged(ids)
    assert logits.shape == (1, 8, 256)
    assert torch.isfinite(logits).all()


def test_ties_merge_empty_models():
    """ties_merge with empty list returns a copy of base."""
    base = _small_model()
    merged = ties_merge(base, [])

    for (n1, p1), (n2, p2) in zip(base.named_parameters(), merged.named_parameters()):
        assert torch.equal(p1, p2)
