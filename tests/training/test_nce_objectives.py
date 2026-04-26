"""
Tests for src/training/nce_objectives.py
=========================================
10 unit tests + 4 additional unit tests + 1 integration test = 15 unit + 1 integration.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.training import TRAINING_REGISTRY
from src.training.nce_objectives import NCEConfig, NCEObjectives

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def default_cfg() -> NCEConfig:
    return NCEConfig(temperature=0.07, n_negatives=64, normalize=True)


@pytest.fixture()
def module(default_cfg: NCEConfig) -> NCEObjectives:
    return NCEObjectives(default_cfg)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = NCEConfig()
    assert cfg.temperature == 0.07
    assert cfg.n_negatives == 64
    assert cfg.normalize is True


# ---------------------------------------------------------------------------
# 2. test_maybe_normalize_unit_norm
# ---------------------------------------------------------------------------


def test_maybe_normalize_unit_norm(module):
    x = torch.randn(8, 32)
    out = module.maybe_normalize(x)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ---------------------------------------------------------------------------
# 3. test_nce_loss_scalar
# ---------------------------------------------------------------------------


def test_nce_loss_scalar(module):
    B, k = 4, 8
    real_scores = torch.randn(B)
    noise_scores = torch.randn(B, k)
    loss = module.nce_loss(real_scores, noise_scores)
    assert loss.shape == ()
    assert loss.ndim == 0


# ---------------------------------------------------------------------------
# 4. test_nce_loss_positive — random inputs → loss > 0
# ---------------------------------------------------------------------------


def test_nce_loss_positive(module):
    torch.manual_seed(0)
    real_scores = torch.randn(16)
    noise_scores = torch.randn(16, 8)
    loss = module.nce_loss(real_scores, noise_scores)
    assert loss.item() > 0.0


# ---------------------------------------------------------------------------
# 5. test_nce_loss_perfect — real=+5, noise=-5 → loss near 0
# ---------------------------------------------------------------------------


def test_nce_loss_perfect(module):
    B, k = 8, 16
    real_scores = torch.full((B,), 5.0)
    noise_scores = torch.full((B, k), -5.0)
    loss = module.nce_loss(real_scores, noise_scores)
    assert loss.item() < 0.02, f"Expected near-0 loss but got {loss.item()}"


# ---------------------------------------------------------------------------
# 6. test_infonce_loss_scalar
# ---------------------------------------------------------------------------


def test_infonce_loss_scalar(module):
    B, D = 6, 16
    q = torch.randn(B, D)
    k = torch.randn(B, D)
    loss = module.infonce_loss(q, k)
    assert loss.shape == ()


# ---------------------------------------------------------------------------
# 7. test_infonce_diagonal_correct — queries == keys → loss ≤ log(B)
# ---------------------------------------------------------------------------


def test_infonce_diagonal_correct(module):
    """When queries == keys the diagonal logits are highest; loss should be small."""
    torch.manual_seed(42)
    B, D = 8, 32
    # Use orthogonal-ish vectors so diagonal is clearly the best match
    q = F.normalize(torch.randn(B, D), dim=-1)
    loss = module.infonce_loss(q, q)
    # loss should be less than log(B) (the chance-level upper bound)
    assert loss.item() < math.log(B)


# ---------------------------------------------------------------------------
# 8. test_infonce_temperature_effect — lower τ → sharper (lower) loss
# ---------------------------------------------------------------------------


def test_infonce_temperature_effect():
    """Lower temperature should produce lower loss when diagonal is clearly best."""
    torch.manual_seed(7)
    B, D = 8, 64
    q = F.normalize(torch.randn(B, D), dim=-1)

    cfg_low = NCEConfig(temperature=0.01, normalize=True)
    cfg_high = NCEConfig(temperature=1.0, normalize=True)

    loss_low = NCEObjectives(cfg_low).infonce_loss(q, q).item()
    loss_high = NCEObjectives(cfg_high).infonce_loss(q, q).item()

    # Lower temperature makes the correct class more dominant → lower loss
    assert loss_low < loss_high


# ---------------------------------------------------------------------------
# 9. test_nt_xent_scalar
# ---------------------------------------------------------------------------


def test_nt_xent_scalar(module):
    B, D = 6, 16
    z1 = torch.randn(B, D)
    z2 = torch.randn(B, D)
    loss = module.nt_xent_loss(z1, z2)
    assert loss.shape == ()


# ---------------------------------------------------------------------------
# 10. test_nt_xent_positive_pairs — z1 == z2 → lower NT-Xent than random
# ---------------------------------------------------------------------------


def test_nt_xent_positive_pairs(module):
    torch.manual_seed(0)
    B, D = 8, 32
    z = torch.randn(B, D)
    loss_identical = module.nt_xent_loss(z, z).item()
    z2_random = torch.randn(B, D)
    loss_random = module.nt_xent_loss(z, z2_random).item()
    assert loss_identical < loss_random


# ---------------------------------------------------------------------------
# 11. test_nt_xent_symmetric — loss(z1,z2) ≈ loss(z2,z1)
# ---------------------------------------------------------------------------


def test_nt_xent_symmetric(module):
    torch.manual_seed(1)
    B, D = 8, 32
    z1 = torch.randn(B, D)
    z2 = torch.randn(B, D)
    loss_a = module.nt_xent_loss(z1, z2).item()
    loss_b = module.nt_xent_loss(z2, z1).item()
    assert abs(loss_a - loss_b) < 1e-5, (
        f"NT-Xent should be symmetric but got {loss_a:.6f} vs {loss_b:.6f}"
    )


# ---------------------------------------------------------------------------
# 12. test_forward_keys — returns "loss", "alignment", "uniformity"
# ---------------------------------------------------------------------------


def test_forward_keys(module):
    B, D = 6, 16
    z1 = torch.randn(B, D)
    z2 = torch.randn(B, D)
    out = module(z1, z2, mode="nt_xent")
    assert set(out.keys()) == {"loss", "alignment", "uniformity"}
    for key, val in out.items():
        assert val.shape == (), f"{key} should be scalar"


# ---------------------------------------------------------------------------
# 13. test_alignment_identical_pairs — alignment near 0 when z1 == z2
# ---------------------------------------------------------------------------


def test_alignment_identical_pairs(module):
    B, D = 8, 32
    z = torch.randn(B, D)
    out = module(z, z, mode="nt_xent")
    # alignment = -mean(||z1 - z2||^2); when identical, norm is 0 → alignment = 0
    assert abs(out["alignment"].item()) < 1e-5


# ---------------------------------------------------------------------------
# 14. test_uniformity_spread — spread embeddings have better (lower) uniformity
# ---------------------------------------------------------------------------


def test_uniformity_spread(module):
    """Uniformly spread embeddings on the sphere yield a more-negative uniformity."""
    B = 16
    # Clustered: all near the same point
    z_clustered = torch.ones(B, 8) / (8**0.5)
    z_clustered = z_clustered + torch.randn_like(z_clustered) * 0.001

    # Spread: random Gaussian → after normalisation, approximately uniform on sphere
    torch.manual_seed(99)
    z_spread = torch.randn(B, 8)

    cfg = NCEConfig(normalize=True)
    m = NCEObjectives(cfg)

    out_clust = m(z_clustered, z_clustered.clone(), mode="nt_xent")
    out_spread = m(z_spread, z_spread.clone(), mode="nt_xent")

    # More spread → more negative uniformity score
    assert out_spread["uniformity"].item() < out_clust["uniformity"].item()


# ---------------------------------------------------------------------------
# 15. test_registry
# ---------------------------------------------------------------------------


def test_registry():
    assert "nce_objectives" in TRAINING_REGISTRY
    assert TRAINING_REGISTRY["nce_objectives"] is NCEObjectives


# ---------------------------------------------------------------------------
# Integration test — B=16, D=64, all three objectives, finite + backward
# ---------------------------------------------------------------------------


def test_integration_all_objectives():
    """
    Integration: B=16, D=64.
    Run NCE, InfoNCE (in-batch + explicit negatives), and NT-Xent.
    Verify:
      - all losses are finite scalars
      - backward pass completes without error
      - forward() dict keys correct for both modes
    """
    torch.manual_seed(42)
    B, D, N = 16, 64, 8

    cfg = NCEConfig(temperature=0.07, n_negatives=N, normalize=True)
    model = NCEObjectives(cfg)

    # ---- NCE ----
    real_scores = torch.randn(B, requires_grad=True)
    noise_scores = torch.randn(B, N, requires_grad=True)
    loss_nce = model.nce_loss(real_scores, noise_scores)
    assert torch.isfinite(loss_nce)
    loss_nce.backward()
    assert real_scores.grad is not None
    assert noise_scores.grad is not None

    # ---- InfoNCE in-batch ----
    q = torch.randn(B, D, requires_grad=True)
    k = torch.randn(B, D, requires_grad=True)
    loss_infonce = model.infonce_loss(q, k)
    assert torch.isfinite(loss_infonce)
    loss_infonce.backward()
    assert q.grad is not None

    # ---- InfoNCE with explicit negatives ----
    q2 = torch.randn(B, D, requires_grad=True)
    k2 = torch.randn(B, D, requires_grad=True)
    negs = torch.randn(B, N, D, requires_grad=True)
    loss_infonce_neg = model.infonce_loss(q2, k2, negatives=negs)
    assert torch.isfinite(loss_infonce_neg)
    loss_infonce_neg.backward()
    assert q2.grad is not None

    # ---- NT-Xent ----
    z1 = torch.randn(B, D, requires_grad=True)
    z2 = torch.randn(B, D, requires_grad=True)
    loss_ntxent = model.nt_xent_loss(z1, z2)
    assert torch.isfinite(loss_ntxent)
    loss_ntxent.backward()
    assert z1.grad is not None
    assert z2.grad is not None

    # ---- forward() nt_xent mode ----
    z1b = torch.randn(B, D, requires_grad=True)
    z2b = torch.randn(B, D, requires_grad=True)
    out_ntxent = model(z1b, z2b, mode="nt_xent")
    assert set(out_ntxent.keys()) == {"loss", "alignment", "uniformity"}
    assert torch.isfinite(out_ntxent["loss"])
    assert torch.isfinite(out_ntxent["alignment"])
    assert torch.isfinite(out_ntxent["uniformity"])
    out_ntxent["loss"].backward()

    # ---- forward() infonce mode ----
    z1c = torch.randn(B, D, requires_grad=True)
    z2c = torch.randn(B, D, requires_grad=True)
    out_infonce = model(z1c, z2c, mode="infonce")
    assert set(out_infonce.keys()) == {"loss", "alignment", "uniformity"}
    assert torch.isfinite(out_infonce["loss"])
    out_infonce["loss"].backward()
