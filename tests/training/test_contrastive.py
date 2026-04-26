"""Tests for src/training/contrastive.py (SimCSE-style contrastive learning)."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.contrastive import (
    ContrastiveConfig,
    HardNegativeSimCSETrainer,
    SimCSETrainer,
    TextEncoder,
    pool_hidden_states,
    simcse_loss,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_cfg() -> AureliusConfig:
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
def tiny_model(tiny_cfg: AureliusConfig) -> AureliusTransformer:
    model = AureliusTransformer(tiny_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def cc() -> ContrastiveConfig:
    return ContrastiveConfig()


@pytest.fixture(scope="module")
def encoder(tiny_model: AureliusTransformer, cc: ContrastiveConfig) -> TextEncoder:
    return TextEncoder(backbone=tiny_model, config=cc)


@pytest.fixture()
def input_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 256, (4, 16))  # B=4, T=16


# ---------------------------------------------------------------------------
# 1. ContrastiveConfig defaults
# ---------------------------------------------------------------------------


def test_contrastive_config_defaults():
    cfg = ContrastiveConfig()
    assert cfg.temperature == 0.05
    assert cfg.pooling == "mean"
    assert cfg.projection_dim is None
    assert cfg.hard_negative_weight == 0.0


# ---------------------------------------------------------------------------
# 2-4. pool_hidden_states
# ---------------------------------------------------------------------------


def test_pool_mean_shape():
    hidden = torch.randn(4, 16, 64)
    out = pool_hidden_states(hidden, "mean")
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"


def test_pool_cls_matches_first_token():
    hidden = torch.randn(4, 16, 64)
    out = pool_hidden_states(hidden, "cls")
    assert out.shape == (4, 64)
    assert torch.allclose(out, hidden[:, 0, :])


def test_pool_last_matches_last_token():
    hidden = torch.randn(4, 16, 64)
    out = pool_hidden_states(hidden, "last")
    assert out.shape == (4, 64)
    assert torch.allclose(out, hidden[:, -1, :])


# ---------------------------------------------------------------------------
# 5-10. simcse_loss
# ---------------------------------------------------------------------------


def test_simcse_loss_returns_tensor_and_dict():
    z1 = torch.randn(4, 64)
    z2 = torch.randn(4, 64)
    result = simcse_loss(z1, z2, temperature=0.05)
    assert isinstance(result, tuple) and len(result) == 2
    loss, metrics = result
    assert isinstance(loss, torch.Tensor)
    assert isinstance(metrics, dict)


def test_simcse_loss_scalar_and_finite():
    z1 = torch.randn(4, 64)
    z2 = torch.randn(4, 64)
    loss, _ = simcse_loss(z1, z2, temperature=0.05)
    assert loss.ndim == 0, "loss should be a scalar"
    assert torch.isfinite(loss), "loss should be finite"


def test_simcse_loss_dict_has_required_keys():
    z1 = torch.randn(4, 64)
    z2 = torch.randn(4, 64)
    _, metrics = simcse_loss(z1, z2, temperature=0.05)
    assert "alignment" in metrics
    assert "uniformity" in metrics
    assert "accuracy" in metrics


def test_simcse_loss_accuracy_in_range():
    z1 = torch.randn(4, 64)
    z2 = torch.randn(4, 64)
    _, metrics = simcse_loss(z1, z2, temperature=0.05)
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_simcse_loss_alignment_in_range():
    z1 = torch.randn(4, 64)
    z2 = torch.randn(4, 64)
    _, metrics = simcse_loss(z1, z2, temperature=0.05)
    # cosine similarity is in [-1, 1]
    assert -1.0 - 1e-5 <= metrics["alignment"] <= 1.0 + 1e-5


def test_simcse_loss_identical_pairs_high_accuracy():
    """When z1 == z2 (perfect positive pairs), accuracy should be 1.0."""
    torch.manual_seed(42)
    z = torch.randn(8, 64)
    _, metrics = simcse_loss(z, z.clone(), temperature=0.05)
    assert metrics["accuracy"] == 1.0, (
        f"Expected accuracy=1.0 for identical pairs, got {metrics['accuracy']}"
    )


# ---------------------------------------------------------------------------
# 11-12. TextEncoder
# ---------------------------------------------------------------------------


def test_text_encoder_encode_shape(encoder: TextEncoder, input_ids: torch.Tensor):
    with torch.no_grad():
        out = encoder.encode(input_ids)
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"


def test_text_encoder_with_projection_dim(tiny_model: AureliusTransformer, input_ids: torch.Tensor):
    cfg = ContrastiveConfig(projection_dim=32)
    enc = TextEncoder(backbone=tiny_model, config=cfg)
    with torch.no_grad():
        out = enc.encode(input_ids)
    assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"


# ---------------------------------------------------------------------------
# 13-14. SimCSETrainer.train_step
# ---------------------------------------------------------------------------


def _make_trainer(tiny_model: AureliusTransformer) -> SimCSETrainer:
    cfg = ContrastiveConfig(temperature=0.05)
    enc = TextEncoder(backbone=tiny_model, config=cfg)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-4)
    return SimCSETrainer(encoder=enc, config=cfg, optimizer=opt)


def test_simcse_trainer_train_step_keys(tiny_model: AureliusTransformer, input_ids: torch.Tensor):
    trainer = _make_trainer(tiny_model)
    result = trainer.train_step(input_ids)
    for key in ("loss", "alignment", "uniformity", "accuracy"):
        assert key in result, f"Missing key: {key}"


def test_simcse_trainer_train_step_loss_finite(
    tiny_model: AureliusTransformer, input_ids: torch.Tensor
):
    trainer = _make_trainer(tiny_model)
    result = trainer.train_step(input_ids)
    assert isinstance(result["loss"], float)
    assert torch.isfinite(torch.tensor(result["loss"])), "train_step loss should be finite"


# ---------------------------------------------------------------------------
# 15. HardNegativeSimCSETrainer
# ---------------------------------------------------------------------------


def test_hard_negative_trainer_returns_correct_keys(
    tiny_model: AureliusTransformer, input_ids: torch.Tensor
):
    cfg = ContrastiveConfig(temperature=0.05, hard_negative_weight=1.0)
    enc = TextEncoder(backbone=tiny_model, config=cfg)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-4)
    trainer = HardNegativeSimCSETrainer(encoder=enc, config=cfg, optimizer=opt)

    torch.manual_seed(1)
    neg_ids = torch.randint(0, 256, (4, 16))
    result = trainer.train_step_with_negatives(input_ids, neg_ids)

    for key in ("loss", "alignment", "uniformity", "accuracy"):
        assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 16. Dropout produces different outputs on two forward passes in train mode
# ---------------------------------------------------------------------------


def test_train_mode_dropout_produces_different_outputs(input_ids: torch.Tensor):
    """Two calls to encode() in train mode should differ when dropout > 0."""
    cfg_drop = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
        dropout=0.5,
    )
    model_drop = AureliusTransformer(cfg_drop)
    model_drop.train()  # dropout active in train mode

    cc = ContrastiveConfig(pooling="mean")
    enc = TextEncoder(backbone=model_drop, config=cc)
    enc.train()

    torch.manual_seed(7)
    with torch.no_grad():
        z1 = enc.encode(input_ids)
        z2 = enc.encode(input_ids)

    assert not torch.allclose(z1, z2), (
        "Expected different outputs from two forward passes with dropout in train mode"
    )
