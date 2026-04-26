"""Tests for contrastive representation learning module."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.contrastive_repr import (
    ContrastiveConfig,
    ContrastiveTrainer,
    MomentumEncoder,
    ProjectionHead,
    moco_loss,
    nt_xent_loss,
    supervised_contrastive_loss,
)

torch.manual_seed(0)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

B = 4
D = 64
SEQ_LEN = 16


@pytest.fixture(scope="module")
def small_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture(scope="module")
def small_model(small_cfg: AureliusConfig) -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture(scope="module")
def contrastive_cfg() -> ContrastiveConfig:
    return ContrastiveConfig()


@pytest.fixture
def input_ids(small_cfg: AureliusConfig) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, small_cfg.vocab_size, (B, SEQ_LEN))


# ---------------------------------------------------------------------------
# 1. ContrastiveConfig defaults
# ---------------------------------------------------------------------------


def test_contrastive_config_defaults():
    cfg = ContrastiveConfig()
    assert cfg.temperature == 0.07
    assert cfg.momentum == 0.999
    assert cfg.queue_size == 256
    assert cfg.projection_dim == 128
    assert cfg.use_l2_normalize is True
    assert cfg.n_positives == 2


# ---------------------------------------------------------------------------
# 2. ProjectionHead output shape
# ---------------------------------------------------------------------------


def test_projection_head_output_shape():
    torch.manual_seed(0)
    head = ProjectionHead(d_model=D, hidden_dim=D, output_dim=128)
    x = torch.randn(B, D)
    out = head(x)
    assert out.shape == (B, 128), f"Expected ({B}, 128), got {out.shape}"


# ---------------------------------------------------------------------------
# 3. ProjectionHead gradient flow
# ---------------------------------------------------------------------------


def test_projection_head_gradient_flow():
    torch.manual_seed(0)
    head = ProjectionHead(d_model=D, hidden_dim=D, output_dim=128)
    x = torch.randn(B, D, requires_grad=True)
    out = head(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Check that at least one parameter got a gradient
    for p in head.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


# ---------------------------------------------------------------------------
# 4. NT-Xent loss returns scalar
# ---------------------------------------------------------------------------


def test_nt_xent_loss_scalar():
    torch.manual_seed(0)
    z1 = F.normalize(torch.randn(B, D), dim=-1)
    z2 = F.normalize(torch.randn(B, D), dim=-1)
    loss = nt_xent_loss(z1, z2, temperature=0.07)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# 5. NT-Xent loss is low for identical views
# ---------------------------------------------------------------------------


def test_nt_xent_loss_identical_views():
    torch.manual_seed(0)
    z = F.normalize(torch.randn(B, D), dim=-1)
    loss_identical = nt_xent_loss(z, z.clone(), temperature=0.07)
    z2 = F.normalize(torch.randn(B, D), dim=-1)
    loss_random = nt_xent_loss(z, z2, temperature=0.07)
    # Identical views should yield strictly lower loss
    assert loss_identical.item() < loss_random.item(), (
        f"Expected identical-view loss ({loss_identical.item():.4f}) < "
        f"random-view loss ({loss_random.item():.4f})"
    )


# ---------------------------------------------------------------------------
# 6. Supervised contrastive loss returns scalar
# ---------------------------------------------------------------------------


def test_supervised_contrastive_loss_scalar():
    torch.manual_seed(0)
    features = F.normalize(torch.randn(B, D), dim=-1)
    labels = torch.randint(0, 3, (B,))
    loss = supervised_contrastive_loss(features, labels, temperature=0.07)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"
    assert torch.isfinite(loss).item()


# ---------------------------------------------------------------------------
# 7. Supervised contrastive: all same class (edge case — no positives cross-anchor)
# ---------------------------------------------------------------------------


def test_supervised_contrastive_all_same_class():
    """When all samples share the same class, every other sample is a positive.

    The loss should still be finite (no NaN/Inf).
    """
    torch.manual_seed(0)
    features = F.normalize(torch.randn(B, D), dim=-1)
    labels = torch.zeros(B, dtype=torch.long)
    loss = supervised_contrastive_loss(features, labels, temperature=0.07)
    assert torch.isfinite(loss).item(), f"Loss should be finite, got {loss.item()}"


# ---------------------------------------------------------------------------
# 8. MoCo loss returns scalar
# ---------------------------------------------------------------------------


def test_moco_loss_scalar():
    torch.manual_seed(0)
    K = 64  # queue size
    query = F.normalize(torch.randn(B, D), dim=-1)
    key = F.normalize(torch.randn(B, D), dim=-1)
    queue = F.normalize(torch.randn(K, D), dim=-1)
    loss = moco_loss(query, key, queue, temperature=0.07)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 9. MoCo loss value is finite
# ---------------------------------------------------------------------------


def test_moco_loss_value_finite():
    torch.manual_seed(0)
    K = 64
    query = F.normalize(torch.randn(B, D), dim=-1)
    key = F.normalize(torch.randn(B, D), dim=-1)
    queue = F.normalize(torch.randn(K, D), dim=-1)
    loss = moco_loss(query, key, queue, temperature=0.07)
    assert torch.isfinite(loss).item(), f"MoCo loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# 10. ContrastiveTrainer SimCLR step returns correct keys
# ---------------------------------------------------------------------------


def test_contrastive_trainer_simclr_step_keys(
    small_model: AureliusTransformer,
    small_cfg: AureliusConfig,
    contrastive_cfg: ContrastiveConfig,
):
    torch.manual_seed(0)
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
    trainer = ContrastiveTrainer(
        model=small_model,
        config=contrastive_cfg,
        optimizer=optimizer,
        d_model=small_cfg.d_model,
    )
    input_ids = torch.randint(0, small_cfg.vocab_size, (B, SEQ_LEN))
    aug_ids = torch.randint(0, small_cfg.vocab_size, (B, SEQ_LEN))
    result = trainer.train_step_simclr(input_ids, aug_ids)

    assert "loss" in result, "Result must contain 'loss'"
    assert "temperature" in result, "Result must contain 'temperature'"
    assert isinstance(result["loss"], float)
    assert isinstance(result["temperature"], float)
    assert result["temperature"] == contrastive_cfg.temperature


# ---------------------------------------------------------------------------
# 11. ContrastiveTrainer SupCon step returns correct keys
# ---------------------------------------------------------------------------


def test_contrastive_trainer_supcon_step_keys(
    small_model: AureliusTransformer,
    small_cfg: AureliusConfig,
    contrastive_cfg: ContrastiveConfig,
):
    torch.manual_seed(0)
    optimizer = torch.optim.Adam(small_model.parameters(), lr=1e-4)
    trainer = ContrastiveTrainer(
        model=small_model,
        config=contrastive_cfg,
        optimizer=optimizer,
        d_model=small_cfg.d_model,
    )
    input_ids = torch.randint(0, small_cfg.vocab_size, (B, SEQ_LEN))
    labels = torch.randint(0, 3, (B,))
    result = trainer.train_step_supcon(input_ids, labels)

    assert "loss" in result, "Result must contain 'loss'"
    assert isinstance(result["loss"], float)


# ---------------------------------------------------------------------------
# 12. MomentumEncoder update changes momentum encoder params
# ---------------------------------------------------------------------------


def test_momentum_encoder_update_changes_params():
    torch.manual_seed(0)
    cfg = ContrastiveConfig(projection_dim=D)
    # Simple linear encoder for testing
    encoder = nn.Linear(D, D)
    momentum_enc = MomentumEncoder(encoder=encoder, config=cfg)

    # Record initial momentum encoder params
    initial_params = {
        name: param.clone() for name, param in momentum_enc.momentum_encoder.named_parameters()
    }

    # Modify the online encoder parameters (simulate a gradient update)
    with torch.no_grad():
        for param in momentum_enc.encoder.parameters():
            param.add_(torch.randn_like(param) * 0.1)

    # Apply EMA update
    momentum_enc.update_momentum_encoder()

    # Momentum encoder params should have changed
    changed = False
    for name, param in momentum_enc.momentum_encoder.named_parameters():
        if not torch.allclose(param, initial_params[name]):
            changed = True
            break

    assert changed, "Momentum encoder parameters should change after EMA update"
