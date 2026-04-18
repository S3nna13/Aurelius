"""Tests for src/security/adversarial_training.py."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.adversarial_training import (
    AdvTrainConfig,
    AdvTrainingStep,
    fgsm_attack,
    pgd_attack,
)

MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

BATCH, SEQ, D = 1, 8, 64
EPSILON = 0.01
ALPHA = 0.003
STEPS = 3


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(42)
    m = AureliusTransformer(MODEL_CFG)
    m.eval()
    return m


@pytest.fixture()
def embeddings() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, D)


@pytest.fixture()
def input_ids() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, MODEL_CFG.vocab_size, (BATCH, SEQ))


@pytest.fixture()
def labels(input_ids: torch.Tensor) -> torch.Tensor:
    return input_ids.clone()


def _dummy_loss(emb: torch.Tensor) -> torch.Tensor:
    return emb.sum()


def test_fgsm_changes_embeddings(embeddings: torch.Tensor) -> None:
    perturbed = fgsm_attack(embeddings, _dummy_loss, epsilon=EPSILON)
    assert not torch.allclose(perturbed, embeddings)


def test_pgd_changes_embeddings_more_than_fgsm(embeddings: torch.Tensor) -> None:
    # Single-step FGSM with small alpha vs multi-step PGD with full epsilon per step.
    # PGD with alpha=EPSILON saturates to the bound, matching or exceeding FGSM(alpha_small).
    fgsm_emb = fgsm_attack(embeddings, _dummy_loss, epsilon=ALPHA)
    pgd_emb = pgd_attack(embeddings, _dummy_loss, epsilon=EPSILON, alpha=EPSILON, steps=STEPS)
    fgsm_diff = (fgsm_emb - embeddings).abs().max().item()
    pgd_diff = (pgd_emb - embeddings).abs().max().item()
    assert pgd_diff >= fgsm_diff - 1e-6


def test_pgd_within_epsilon_bound(embeddings: torch.Tensor) -> None:
    perturbed = pgd_attack(embeddings, _dummy_loss, epsilon=EPSILON, alpha=ALPHA, steps=STEPS)
    diff = (perturbed - embeddings).abs().max().item()
    assert diff <= EPSILON + 1e-6


def test_fgsm_zero_epsilon_no_change(embeddings: torch.Tensor) -> None:
    perturbed = fgsm_attack(embeddings, _dummy_loss, epsilon=0.0)
    assert torch.allclose(perturbed, embeddings)


def test_adversarial_embs_no_grad(embeddings: torch.Tensor) -> None:
    perturbed = pgd_attack(embeddings, _dummy_loss, epsilon=EPSILON, alpha=ALPHA, steps=STEPS)
    assert not perturbed.requires_grad


def test_adv_training_step_returns_3_tuple(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    cfg = AdvTrainConfig()
    step = AdvTrainingStep(cfg)
    result = step.step(model, input_ids, labels, model.embed)
    assert isinstance(result, tuple)
    assert len(result) == 3


def test_total_loss_is_weighted_combo(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    cfg = AdvTrainConfig(adv_weight=0.5)
    step = AdvTrainingStep(cfg)
    total, clean, adv = step.step(model, input_ids, labels, model.embed)
    expected = 0.5 * clean + 0.5 * adv
    assert torch.allclose(total, expected, atol=1e-5)


def test_total_loss_backward(
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    torch.manual_seed(99)
    m = AureliusTransformer(MODEL_CFG)
    m.train()
    cfg = AdvTrainConfig()
    step = AdvTrainingStep(cfg)
    total, _, _ = step.step(m, input_ids, labels, m.embed)
    total.backward()


def test_adv_weight_zero_equals_clean_loss(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    cfg = AdvTrainConfig(adv_weight=0.0)
    step = AdvTrainingStep(cfg)
    total, clean, _ = step.step(model, input_ids, labels, model.embed)
    assert torch.allclose(total, clean, atol=1e-5)


def test_adv_weight_one_equals_adv_loss(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    cfg = AdvTrainConfig(adv_weight=1.0)
    step = AdvTrainingStep(cfg)
    total, _, adv = step.step(model, input_ids, labels, model.embed)
    assert torch.allclose(total, adv, atol=1e-5)
