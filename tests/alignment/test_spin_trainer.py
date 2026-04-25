"""Unit tests for src/alignment/spin_trainer.py — 20 tests.

Uses a tiny model (vocab=256, d=64, n_heads=4, n_layers=2, max_seq=64) for
all forward and training-step tests.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.alignment.spin_trainer import (
    SPINBatch,
    SPINConfig,
    SPINLoss,
    SPINTrainer,
)
from src.alignment import ALIGNMENT_REGISTRY


# ---------------------------------------------------------------------------
# Tiny causal LM for testing
# ---------------------------------------------------------------------------

VOCAB = 256
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
MAX_SEQ = 64


class TinyLM(nn.Module):
    """Minimal causal LM: embedding → transformer → linear head."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D_MODEL)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=128, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, VOCAB)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = self.transformer(x)
        return self.head(x)


def _make_models() -> tuple[TinyLM, TinyLM]:
    torch.manual_seed(42)
    policy = TinyLM()
    ref = TinyLM()
    ref.load_state_dict(policy.state_dict())
    return policy, ref


def _make_batch(B: int = 4, L: int = 16) -> SPINBatch:
    torch.manual_seed(42)
    real_ids = torch.randint(0, VOCAB, (B, L))
    gen_ids = torch.randint(0, VOCAB, (B, L))
    real_mask = torch.ones(B, L)
    gen_mask = torch.ones(B, L)
    return SPINBatch(
        real_ids=real_ids,
        generated_ids=gen_ids,
        real_mask=real_mask,
        generated_mask=gen_mask,
    )


# ---------------------------------------------------------------------------
# 1–3. SPINConfig defaults
# ---------------------------------------------------------------------------


def test_spin_config_default_learning_rate():
    cfg = SPINConfig()
    assert cfg.learning_rate == pytest.approx(1e-6)


def test_spin_config_default_beta():
    cfg = SPINConfig()
    assert cfg.beta == pytest.approx(0.1)


def test_spin_config_default_n_iterations():
    cfg = SPINConfig()
    assert cfg.n_iterations == 3


def test_spin_config_default_lambda_reg():
    cfg = SPINConfig()
    assert cfg.lambda_reg == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# 4–6. SPINLoss basic properties
# ---------------------------------------------------------------------------


def test_spin_loss_returns_scalar():
    torch.manual_seed(42)
    loss_fn = SPINLoss(beta=0.1)
    B = 4
    pi_real = torch.randn(B)
    pi_gen = torch.randn(B)
    ref_real = torch.randn(B)
    ref_gen = torch.randn(B)
    out = loss_fn(pi_real, pi_gen, ref_real, ref_gen)
    assert out.shape == torch.Size([])


def test_spin_loss_is_finite():
    torch.manual_seed(42)
    loss_fn = SPINLoss(beta=0.1)
    B = 4
    pi_real = torch.randn(B)
    pi_gen = torch.randn(B)
    ref_real = torch.randn(B)
    ref_gen = torch.randn(B)
    out = loss_fn(pi_real, pi_gen, ref_real, ref_gen)
    assert torch.isfinite(out)


def test_spin_loss_beta_stored_as_buffer():
    loss_fn = SPINLoss(beta=0.2)
    # beta should be in buffers, not parameters
    assert "beta" in dict(loss_fn.named_buffers())
    param_names = [n for n, _ in loss_fn.named_parameters()]
    assert "beta" not in param_names


def test_spin_loss_positive():
    """SPIN loss should be non-negative (it is -logsigmoid, which is >= 0)."""
    torch.manual_seed(42)
    loss_fn = SPINLoss(beta=0.1)
    B = 8
    pi_real = torch.randn(B)
    pi_gen = torch.randn(B)
    ref_real = torch.randn(B)
    ref_gen = torch.randn(B)
    out = loss_fn(pi_real, pi_gen, ref_real, ref_gen)
    assert out.item() >= 0.0


# ---------------------------------------------------------------------------
# 7–9. SPINTrainer.compute_logprobs
# ---------------------------------------------------------------------------


def test_compute_logprobs_output_shape():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    mask = torch.ones(B, L)

    lp = trainer.compute_logprobs(policy, input_ids, labels, mask)
    assert lp.shape == (B,)


def test_compute_logprobs_is_finite():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    lp = trainer.compute_logprobs(policy, input_ids, labels, None)
    assert torch.all(torch.isfinite(lp))


def test_compute_logprobs_with_mask():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)

    B, L = 2, 16
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    mask = torch.ones(B, L)
    mask[0, -4:] = 0.0  # Pad last 4 tokens of first sequence

    lp = trainer.compute_logprobs(policy, input_ids, labels, mask)
    assert lp.shape == (B,)
    assert torch.all(torch.isfinite(lp))


# ---------------------------------------------------------------------------
# 10–13. SPINTrainer.train_step
# ---------------------------------------------------------------------------


def test_train_step_returns_correct_keys():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)
    batch = _make_batch()
    result = trainer.train_step(batch)
    assert "loss" in result
    assert "logr_real" in result
    assert "logr_gen" in result


def test_train_step_loss_is_finite():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)
    batch = _make_batch()
    result = trainer.train_step(batch)
    assert math.isfinite(result["loss"])


def test_train_step_logr_are_finite():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)
    batch = _make_batch()
    result = trainer.train_step(batch)
    assert math.isfinite(result["logr_real"])
    assert math.isfinite(result["logr_gen"])


def test_train_step_loss_is_positive():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)
    batch = _make_batch()
    result = trainer.train_step(batch)
    assert result["loss"] >= 0.0


def test_train_step_no_mask():
    """train_step works when batch masks are None."""
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    real_ids = torch.randint(0, VOCAB, (B, L))
    gen_ids = torch.randint(0, VOCAB, (B, L))
    batch = SPINBatch(real_ids=real_ids, generated_ids=gen_ids)
    result = trainer.train_step(batch)
    assert math.isfinite(result["loss"])


# ---------------------------------------------------------------------------
# 14–16. SPINTrainer.iteration
# ---------------------------------------------------------------------------


def test_iteration_returns_mean_metrics():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)
    batches = [_make_batch(), _make_batch()]
    result = trainer.iteration(batches)
    assert "loss" in result
    assert "logr_real" in result
    assert "logr_gen" in result


def test_iteration_loss_is_finite():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)
    batches = [_make_batch(), _make_batch()]
    result = trainer.iteration(batches)
    assert math.isfinite(result["loss"])


def test_iteration_empty_batches():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)
    result = trainer.iteration([])
    assert result["loss"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 17. Ref model is frozen
# ---------------------------------------------------------------------------


def test_ref_model_parameters_frozen():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    SPINTrainer(policy, ref, cfg, optimizer)
    for p in ref.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# 18. ALIGNMENT_REGISTRY contains "spin"
# ---------------------------------------------------------------------------


def test_alignment_registry_contains_spin():
    assert "spin" in ALIGNMENT_REGISTRY


def test_alignment_registry_spin_is_spin_trainer():
    assert ALIGNMENT_REGISTRY["spin"] is SPINTrainer


# ---------------------------------------------------------------------------
# 19–20. SPINBatch dataclass
# ---------------------------------------------------------------------------


def test_spin_batch_stores_real_and_generated():
    torch.manual_seed(42)
    B, L = 2, 8
    real = torch.randint(0, VOCAB, (B, L))
    gen = torch.randint(0, VOCAB, (B, L))
    batch = SPINBatch(real_ids=real, generated_ids=gen)
    assert batch.real_ids is real
    assert batch.generated_ids is gen
    assert batch.real_mask is None
    assert batch.generated_mask is None


def test_spin_batch_optional_masks():
    B, L = 2, 8
    real = torch.ones(B, L, dtype=torch.long)
    gen = torch.ones(B, L, dtype=torch.long)
    mask = torch.ones(B, L)
    batch = SPINBatch(real_ids=real, generated_ids=gen, real_mask=mask, generated_mask=mask)
    assert batch.real_mask is not None
    assert batch.generated_mask is not None
