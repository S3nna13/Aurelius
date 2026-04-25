"""Unit tests for src/alignment/kto_trainer.py — 20 tests.

Uses a tiny model (vocab=256, d=64, n_heads=4, n_layers=2, max_seq=64) for
all forward and training-step tests. References:
    Ethayarajh et al. (2024) arXiv:2402.01306 (Apache-2.0)
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.alignment.kto_trainer import (
    KTOBatch,
    KTOConfig,
    KTOLoss,
    KTOTrainer,
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


def _make_batch(B: int = 4, L: int = 16, all_desirable: bool | None = None) -> KTOBatch:
    torch.manual_seed(42)
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    mask = torch.ones(B, L)

    if all_desirable is True:
        desirable = torch.ones(B, dtype=torch.bool)
    elif all_desirable is False:
        desirable = torch.zeros(B, dtype=torch.bool)
    else:
        # Mixed: first half desirable, second half not
        des = torch.zeros(B, dtype=torch.bool)
        des[: B // 2] = True
        desirable = des

    return KTOBatch(input_ids=input_ids, labels=labels, mask=mask, desirable=desirable)


# ---------------------------------------------------------------------------
# 1–4. KTOConfig defaults
# ---------------------------------------------------------------------------


def test_kto_config_default_beta():
    cfg = KTOConfig()
    assert cfg.beta == pytest.approx(0.1)


def test_kto_config_default_desirable_weight():
    cfg = KTOConfig()
    assert cfg.desirable_weight == pytest.approx(1.0)


def test_kto_config_default_undesirable_weight():
    cfg = KTOConfig()
    assert cfg.undesirable_weight == pytest.approx(1.0)


def test_kto_config_default_learning_rate():
    cfg = KTOConfig()
    assert cfg.learning_rate == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# 5–8. KTOLoss basic properties
# ---------------------------------------------------------------------------


def test_kto_loss_returns_scalar():
    torch.manual_seed(42)
    loss_fn = KTOLoss(beta=0.1)
    B = 4
    policy_lp = torch.randn(B)
    ref_lp = torch.randn(B)
    des = torch.tensor([True, False, True, False])
    out = loss_fn(policy_lp, ref_lp, des)
    assert out.shape == torch.Size([])


def test_kto_loss_is_finite():
    torch.manual_seed(42)
    loss_fn = KTOLoss(beta=0.1)
    B = 4
    policy_lp = torch.randn(B)
    ref_lp = torch.randn(B)
    des = torch.tensor([True, False, True, False])
    out = loss_fn(policy_lp, ref_lp, des)
    assert torch.isfinite(out)


def test_kto_loss_all_desirable_vs_all_undesirable():
    """KTO is not symmetric: all-desirable loss differs from all-undesirable."""
    torch.manual_seed(42)
    loss_fn = KTOLoss(beta=0.1)
    B = 4
    policy_lp = torch.randn(B)
    ref_lp = torch.randn(B)

    des_all = torch.ones(B, dtype=torch.bool)
    unds_all = torch.zeros(B, dtype=torch.bool)

    loss_des = loss_fn(policy_lp, ref_lp, des_all).item()
    loss_unds = loss_fn(policy_lp, ref_lp, unds_all).item()
    # Losses should differ (asymmetric objective)
    assert loss_des != pytest.approx(loss_unds, abs=1e-6)


def test_kto_loss_weights_as_buffers():
    loss_fn = KTOLoss(beta=0.1, desirable_weight=2.0, undesirable_weight=0.5)
    buf_names = {n for n, _ in loss_fn.named_buffers()}
    assert "beta" in buf_names
    assert "desirable_weight" in buf_names
    assert "undesirable_weight" in buf_names
    param_names = [n for n, _ in loss_fn.named_parameters()]
    assert "beta" not in param_names


# ---------------------------------------------------------------------------
# 9–11. KTOTrainer.compute_logprobs
# ---------------------------------------------------------------------------


def test_compute_logprobs_output_shape():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    mask = torch.ones(B, L)

    lp = trainer.compute_logprobs(policy, input_ids, labels, mask)
    assert lp.shape == (B,)


def test_compute_logprobs_is_finite():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)

    B, L = 4, 16
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    lp = trainer.compute_logprobs(policy, input_ids, labels, None)
    assert torch.all(torch.isfinite(lp))


def test_compute_logprobs_with_none_mask():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)

    B, L = 2, 16
    input_ids = torch.randint(0, VOCAB, (B, L))
    labels = input_ids.clone()
    lp = trainer.compute_logprobs(policy, input_ids, labels, None)
    assert lp.shape == (B,)


# ---------------------------------------------------------------------------
# 12–16. KTOTrainer.train_step
# ---------------------------------------------------------------------------


def test_train_step_returns_correct_keys():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)
    batch = _make_batch()
    result = trainer.train_step(batch)
    assert "loss" in result
    assert "kto_desirable" in result
    assert "kto_undesirable" in result
    assert "kl_proxy" in result


def test_train_step_loss_is_finite():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)
    batch = _make_batch()
    result = trainer.train_step(batch)
    assert math.isfinite(result["loss"])


def test_train_step_kl_proxy_nonnegative():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)
    batch = _make_batch()
    result = trainer.train_step(batch)
    assert result["kl_proxy"] >= 0.0


def test_train_step_all_desirable():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)
    batch = _make_batch(all_desirable=True)
    result = trainer.train_step(batch)
    assert math.isfinite(result["loss"])
    # Undesirable component should be zero
    assert result["kto_undesirable"] == pytest.approx(0.0)


def test_train_step_all_undesirable():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = KTOTrainer(policy, ref, cfg, optimizer)
    batch = _make_batch(all_desirable=False)
    result = trainer.train_step(batch)
    assert math.isfinite(result["loss"])
    # Desirable component should be zero
    assert result["kto_desirable"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 17. Ref model is frozen
# ---------------------------------------------------------------------------


def test_ref_model_parameters_frozen():
    torch.manual_seed(42)
    policy, ref = _make_models()
    cfg = KTOConfig()
    optimizer = optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    KTOTrainer(policy, ref, cfg, optimizer)
    for p in ref.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# 18–19. ALIGNMENT_REGISTRY
# ---------------------------------------------------------------------------


def test_alignment_registry_contains_kto():
    assert "kto" in ALIGNMENT_REGISTRY


def test_alignment_registry_kto_is_kto_trainer():
    assert ALIGNMENT_REGISTRY["kto"] is KTOTrainer


# ---------------------------------------------------------------------------
# 20. KTOBatch dataclass
# ---------------------------------------------------------------------------


def test_kto_batch_stores_fields():
    torch.manual_seed(42)
    B, L = 4, 16
    ids = torch.randint(0, VOCAB, (B, L))
    labels = ids.clone()
    mask = torch.ones(B, L)
    des = torch.ones(B, dtype=torch.bool)
    batch = KTOBatch(input_ids=ids, labels=labels, mask=mask, desirable=des)
    assert batch.input_ids is ids
    assert batch.labels is labels
    assert batch.mask is mask
    assert batch.desirable is des


def test_kto_batch_none_mask():
    B, L = 4, 16
    ids = torch.randint(0, VOCAB, (B, L))
    labels = ids.clone()
    des = torch.ones(B, dtype=torch.bool)
    batch = KTOBatch(input_ids=ids, labels=labels, mask=None, desirable=des)
    assert batch.mask is None
