"""Tests for Multi-Token Prediction (MTP) training objective.

Uses a tiny 2-layer model (d_model=64) so tests run in seconds on CPU.
"""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.mtp import MTPHead, MTPObjective, MTPTrainer

# ---------------------------------------------------------------------------
# Tiny model fixture (2 layers, d_model=64)
# ---------------------------------------------------------------------------


def _tiny_config() -> AureliusConfig:
    """Return a minimal AureliusConfig that satisfies all assertions."""
    return AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        tie_embeddings=False,
    )


@pytest.fixture()
def tiny_model() -> AureliusTransformer:
    cfg = _tiny_config()
    model = AureliusTransformer(cfg)
    model.eval()
    return model


@pytest.fixture()
def tiny_mtp_objective() -> MTPObjective:
    cfg = _tiny_config()
    return MTPObjective(k=3, d_model=cfg.d_model, vocab_size=cfg.vocab_size)


# ---------------------------------------------------------------------------
# Test 1 — MTPHead output shape
# ---------------------------------------------------------------------------


def test_mtp_head_output_shape():
    """MTPHead(depth=1): (2, 16, 2048) hidden -> (2, 15, 128000) logits."""
    d_model = 2048
    vocab_size = 128_000
    depth = 1
    B, S = 2, 16

    head = MTPHead(d_model=d_model, vocab_size=vocab_size, depth=depth)
    h_slice = torch.randn(B, S - depth, d_model)
    logits = head(h_slice)

    assert logits.shape == (B, S - depth, vocab_size), (
        f"Expected ({B}, {S - depth}, {vocab_size}), got {logits.shape}"
    )


# ---------------------------------------------------------------------------
# Test 2 — MTPObjective returns a scalar tensor
# ---------------------------------------------------------------------------


def test_mtp_objective_loss_scalar(tiny_mtp_objective):
    """compute_mtp_loss should return a 0-d (scalar) tensor."""
    cfg = _tiny_config()
    B, S = 2, 16

    hidden_states = torch.randn(B, S, cfg.d_model, requires_grad=True)
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))

    loss = tiny_mtp_objective.compute_mtp_loss(hidden_states, input_ids, mtp_weight=0.3)

    assert loss.ndim == 0, f"Expected scalar (0-d tensor), got shape {loss.shape}"
    assert isinstance(loss, torch.Tensor)


# ---------------------------------------------------------------------------
# Test 3 — MTP loss is positive
# ---------------------------------------------------------------------------


def test_mtp_loss_positive(tiny_mtp_objective):
    """Cross-entropy loss must be > 0 for random logits."""
    cfg = _tiny_config()
    B, S = 2, 16

    hidden_states = torch.randn(B, S, cfg.d_model)
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))

    loss = tiny_mtp_objective.compute_mtp_loss(hidden_states, input_ids, mtp_weight=0.3)

    assert loss.item() > 0.0, f"MTP loss should be > 0, got {loss.item()}"


# ---------------------------------------------------------------------------
# Test 4 — mtp_weight scales loss linearly
# ---------------------------------------------------------------------------


def test_mtp_weight_scales_loss(tiny_mtp_objective):
    """mtp_weight=0.1 should give 10x smaller loss than mtp_weight=1.0."""
    cfg = _tiny_config()
    B, S = 2, 16

    torch.manual_seed(42)
    hidden_states = torch.randn(B, S, cfg.d_model)
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))

    loss_1_0 = tiny_mtp_objective.compute_mtp_loss(hidden_states, input_ids, mtp_weight=1.0)
    loss_0_1 = tiny_mtp_objective.compute_mtp_loss(hidden_states, input_ids, mtp_weight=0.1)

    ratio = loss_1_0.item() / loss_0_1.item()
    assert abs(ratio - 10.0) < 1e-4, (
        f"Expected ratio ~10x, got {ratio:.6f} "
        f"(loss@1.0={loss_1_0.item():.6f}, loss@0.1={loss_0_1.item():.6f})"
    )


# ---------------------------------------------------------------------------
# Test 5 — MTPTrainer.train_step returns the correct dict of floats
# ---------------------------------------------------------------------------


def test_mtp_trainer_step(tiny_model, tiny_mtp_objective):
    """train_step must return a dict with 'main_loss', 'mtp_loss', 'total_loss'."""
    cfg = _tiny_config()
    tiny_model.train()

    optimizer = torch.optim.AdamW(
        list(tiny_model.parameters()) + list(tiny_mtp_objective.parameters()),
        lr=1e-4,
    )
    trainer = MTPTrainer(
        model=tiny_model,
        mtp_objective=tiny_mtp_objective,
        optimizer=optimizer,
        mtp_weight=0.3,
    )

    B, S = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))

    metrics = trainer.train_step(input_ids)
    trainer.remove_hook()

    assert set(metrics.keys()) == {"main_loss", "mtp_loss", "total_loss"}, (
        f"Unexpected keys: {set(metrics.keys())}"
    )
    for key, val in metrics.items():
        assert isinstance(val, float), f"{key} should be a float, got {type(val)}"


# ---------------------------------------------------------------------------
# Test 6 — Gradients flow through MTPHead parameters after train_step
# ---------------------------------------------------------------------------


def test_mtp_gradients_flow(tiny_model, tiny_mtp_objective):
    """After train_step, all MTPHead parameters must have non-None gradients."""
    cfg = _tiny_config()
    tiny_model.train()

    optimizer = torch.optim.AdamW(
        list(tiny_model.parameters()) + list(tiny_mtp_objective.parameters()),
        lr=1e-4,
    )
    trainer = MTPTrainer(
        model=tiny_model,
        mtp_objective=tiny_mtp_objective,
        optimizer=optimizer,
        mtp_weight=0.3,
    )

    B, S = 2, 16
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    trainer.train_step(input_ids)
    trainer.remove_hook()

    for i, head in enumerate(tiny_mtp_objective.heads):
        for name, param in head.named_parameters():
            assert param.grad is not None, f"head[{i}].{name} has no gradient after train_step"
            assert param.grad.abs().sum().item() > 0.0, (
                f"head[{i}].{name} gradient is all-zero after train_step"
            )
