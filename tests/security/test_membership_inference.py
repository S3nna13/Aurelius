"""Tests for src/security/membership_inference.py."""

from __future__ import annotations

import pytest
import torch

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.security.membership_inference import (
    EntropyMIA,
    MIAConfig,
    ThresholdMIA,
    compute_sample_entropy,
    compute_sample_loss,
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

BATCH, SEQ = 1, 8


@pytest.fixture(scope="module")
def model() -> AureliusTransformer:
    torch.manual_seed(42)
    m = AureliusTransformer(MODEL_CFG)
    m.eval()
    return m


@pytest.fixture()
def input_ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, MODEL_CFG.vocab_size, (BATCH, SEQ))


@pytest.fixture()
def labels(input_ids: torch.Tensor) -> torch.Tensor:
    return input_ids.clone()


def test_compute_sample_loss_returns_float(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    result = compute_sample_loss(model, input_ids, labels)
    assert isinstance(result, float)


def test_compute_sample_loss_positive(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    result = compute_sample_loss(model, input_ids, labels)
    assert result > 0.0


def test_compute_sample_entropy_returns_float(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
) -> None:
    result = compute_sample_entropy(model, input_ids)
    assert isinstance(result, float)


def test_compute_sample_entropy_nonnegative(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
) -> None:
    result = compute_sample_entropy(model, input_ids)
    assert result >= 0.0


def test_threshold_predict_returns_bool(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    cfg = MIAConfig()
    mia = ThresholdMIA(cfg)
    result = mia.predict_member(model, input_ids, labels)
    assert isinstance(result, bool)


def test_entropy_predict_returns_bool(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
) -> None:
    cfg = MIAConfig()
    mia = EntropyMIA(cfg)
    result = mia.predict_member(model, input_ids)
    assert isinstance(result, bool)


def test_attack_accuracy_in_range(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    cfg = MIAConfig()
    mia = ThresholdMIA(cfg)
    members = [(input_ids, labels)]
    nonmembers = [(input_ids, labels)]
    acc = mia.attack_accuracy(model, members, nonmembers)
    assert 0.0 <= acc <= 1.0


def test_predict_batch_same_length(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    cfg = MIAConfig()
    mia = ThresholdMIA(cfg)
    ids_list = [input_ids, input_ids, input_ids]
    lbl_list = [labels, labels, labels]
    result = mia.predict_batch(model, ids_list, lbl_list)
    assert isinstance(result, list)
    assert len(result) == 3


def test_high_threshold_accepts_all(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    # Very high threshold: all samples have loss below it → all predicted members
    cfg = MIAConfig(loss_threshold=1e9)
    mia = ThresholdMIA(cfg)
    ids_list = [input_ids, input_ids]
    lbl_list = [labels, labels]
    results = mia.predict_batch(model, ids_list, lbl_list)
    assert all(results)


def test_low_threshold_rejects_all(
    model: AureliusTransformer,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    # Very low threshold: no sample has loss below it → none predicted members
    cfg = MIAConfig(loss_threshold=-1.0)
    mia = ThresholdMIA(cfg)
    ids_list = [input_ids, input_ids]
    lbl_list = [labels, labels]
    results = mia.predict_batch(model, ids_list, lbl_list)
    assert not any(results)
