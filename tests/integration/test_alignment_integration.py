"""Integration tests for SPIN and KTO alignment trainers.

Verifies that SPINTrainer and KTOTrainer are correctly registered in the
ALIGNMENT_REGISTRY and can complete a full forward + gradient step on a tiny
model without producing NaN values.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim

from src.alignment import ALIGNMENT_REGISTRY
from src.alignment.spin_trainer import (
    SPINBatch,
    SPINConfig,
    SPINTrainer,
)
from src.alignment.kto_trainer import (
    KTOBatch,
    KTOConfig,
    KTOTrainer,
)


# ---------------------------------------------------------------------------
# Tiny causal LM
# ---------------------------------------------------------------------------

VOCAB = 256
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2


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


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_alignment_registry_has_spin():
    assert "spin" in ALIGNMENT_REGISTRY


def test_alignment_registry_has_kto():
    assert "kto" in ALIGNMENT_REGISTRY


def test_spin_registry_entry_is_spin_trainer():
    assert ALIGNMENT_REGISTRY["spin"] is SPINTrainer


def test_kto_registry_entry_is_kto_trainer():
    assert ALIGNMENT_REGISTRY["kto"] is KTOTrainer


# ---------------------------------------------------------------------------
# SPIN full step integration
# ---------------------------------------------------------------------------


def test_spin_trainer_full_step_no_nan():
    """SPINTrainer.train_step completes without NaN on a tiny model pair."""
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig(learning_rate=1e-4, beta=0.1)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    real_ids = torch.randint(0, VOCAB, (B, L))
    gen_ids = torch.randint(0, VOCAB, (B, L))
    mask = torch.ones(B, L)

    batch = SPINBatch(
        real_ids=real_ids,
        generated_ids=gen_ids,
        real_mask=mask,
        generated_mask=mask,
    )

    result = trainer.train_step(batch)
    assert "loss" in result
    assert "logr_real" in result
    assert "logr_gen" in result
    import math
    assert math.isfinite(result["loss"]), f"SPIN loss is not finite: {result['loss']}"
    assert math.isfinite(result["logr_real"])
    assert math.isfinite(result["logr_gen"])


def test_spin_trainer_iteration_no_nan():
    """SPINTrainer.iteration over 2 batches returns finite mean metrics."""
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = SPINConfig(learning_rate=1e-4, beta=0.1)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = SPINTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    batches = [
        SPINBatch(
            real_ids=torch.randint(0, VOCAB, (B, L)),
            generated_ids=torch.randint(0, VOCAB, (B, L)),
            real_mask=torch.ones(B, L),
            generated_mask=torch.ones(B, L),
        )
        for _ in range(2)
    ]

    result = trainer.iteration(batches)
    import math
    assert math.isfinite(result["loss"])
    assert math.isfinite(result["logr_real"])
    assert math.isfinite(result["logr_gen"])


# ---------------------------------------------------------------------------
# KTO full step integration
# ---------------------------------------------------------------------------


def test_kto_trainer_full_step_no_nan():
    """KTOTrainer.train_step completes without NaN on a tiny model pair."""
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig(learning_rate=1e-4, beta=0.1)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    mask = torch.ones(B, L)
    # Mixed desirable/undesirable
    desirable = torch.zeros(B, dtype=torch.bool)
    desirable[:2] = True

    batch = KTOBatch(input_ids=input_ids, labels=labels, mask=mask, desirable=desirable)
    result = trainer.train_step(batch)

    import math
    assert "loss" in result
    assert "kto_desirable" in result
    assert "kto_undesirable" in result
    assert "kl_proxy" in result
    assert math.isfinite(result["loss"]), f"KTO loss is not finite: {result['loss']}"
    assert result["kl_proxy"] >= 0.0


def test_kto_trainer_all_desirable_no_nan():
    """All-desirable batch: train_step is finite with zero undesirable component."""
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig(learning_rate=1e-4)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    mask = torch.ones(B, L)
    desirable = torch.ones(B, dtype=torch.bool)

    batch = KTOBatch(input_ids=input_ids, labels=labels, mask=mask, desirable=desirable)
    result = trainer.train_step(batch)

    import math
    assert math.isfinite(result["loss"])
    import pytest
    assert result["kto_undesirable"] == pytest.approx(0.0)


def test_kto_trainer_all_undesirable_no_nan():
    """All-undesirable batch: train_step is finite with zero desirable component."""
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig(learning_rate=1e-4)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    mask = torch.ones(B, L)
    desirable = torch.zeros(B, dtype=torch.bool)

    batch = KTOBatch(input_ids=input_ids, labels=labels, mask=mask, desirable=desirable)
    result = trainer.train_step(batch)

    import math
    assert math.isfinite(result["loss"])
    import pytest
    assert result["kto_desirable"] == pytest.approx(0.0)
