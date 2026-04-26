"""Tests for src/training/nce_loss.py — NCE loss and sampled softmax."""

from __future__ import annotations

import pytest
import torch

from src.training.nce_loss import (
    NCEConfig,
    NCELanguageModelLoss,
    UnigramSampler,
    compare_nce_vs_softmax,
    nce_loss,
    sampled_softmax_loss,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

V = 64  # vocab size
D = 32  # model dimension
B = 2  # batch size
T = 4  # sequence length
K = 5  # noise samples


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def token_freqs() -> torch.Tensor:
    return torch.arange(1, V + 1, dtype=torch.float32)


@pytest.fixture()
def sampler(token_freqs) -> UnigramSampler:
    return UnigramSampler(token_freqs, exponent=0.75)


@pytest.fixture()
def nce_cfg() -> NCEConfig:
    return NCEConfig(n_noise_samples=K)


@pytest.fixture()
def nce_lm(token_freqs, nce_cfg) -> NCELanguageModelLoss:
    return NCELanguageModelLoss(
        vocab_size=V,
        d_model=D,
        config=nce_cfg,
        token_freqs=token_freqs,
    )


@pytest.fixture()
def hidden() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(B, T, D)


@pytest.fixture()
def targets() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, V, (B, T))


# ---------------------------------------------------------------------------
# 1. NCEConfig defaults
# ---------------------------------------------------------------------------


def test_nce_config_defaults():
    cfg = NCEConfig()
    assert cfg.n_noise_samples == 20
    assert cfg.noise_dist == "unigram"
    assert cfg.noise_exponent == 0.75
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# 2. UnigramSampler probs sum to ~1.0
# ---------------------------------------------------------------------------


def test_unigram_sampler_probs_sum(sampler):
    total = sampler.probs.sum().item()
    assert abs(total - 1.0) < 1e-5, f"probs sum {total} != 1.0"


# ---------------------------------------------------------------------------
# 3. UnigramSampler sample shape
# ---------------------------------------------------------------------------


def test_unigram_sampler_sample_shape(sampler):
    n = 50
    samples = sampler.sample(n)
    assert samples.shape == (n,), f"Expected ({n},), got {samples.shape}"


# ---------------------------------------------------------------------------
# 4. UnigramSampler sample range
# ---------------------------------------------------------------------------


def test_unigram_sampler_sample_range(sampler):
    samples = sampler.sample(200, seed=42)
    assert samples.dtype == torch.int64
    assert samples.min().item() >= 0
    assert samples.max().item() < V


# ---------------------------------------------------------------------------
# 5. UnigramSampler log_prob shape
# ---------------------------------------------------------------------------


def test_unigram_sampler_log_prob_shape(sampler):
    ids = torch.randint(0, V, (B, T))
    log_probs = sampler.log_prob(ids)
    assert log_probs.shape == ids.shape, f"Expected shape {ids.shape}, got {log_probs.shape}"


# ---------------------------------------------------------------------------
# 6. nce_loss returns scalar
# ---------------------------------------------------------------------------


def test_nce_loss_scalar():
    torch.manual_seed(0)
    N, M = 8, 40
    scores_pos = torch.randn(N)
    scores_neg = torch.randn(M)
    log_noise_pos = torch.full((N,), -3.0)
    log_noise_neg = torch.full((M,), -3.0)
    loss = nce_loss(scores_pos, scores_neg, log_noise_pos, log_noise_neg, k=5)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 7. nce_loss non-negative
# ---------------------------------------------------------------------------


def test_nce_loss_nonnegative():
    torch.manual_seed(1)
    N, M = 8, 40
    scores_pos = torch.randn(N)
    scores_neg = torch.randn(M)
    log_noise_pos = torch.full((N,), -3.0)
    log_noise_neg = torch.full((M,), -3.0)
    loss = nce_loss(scores_pos, scores_neg, log_noise_pos, log_noise_neg, k=5)
    assert loss.item() >= 0.0, f"NCE loss {loss.item()} is negative"


# ---------------------------------------------------------------------------
# 8. sampled_softmax_loss returns scalar
# ---------------------------------------------------------------------------


def test_sampled_softmax_loss_scalar():
    torch.manual_seed(2)
    logits = torch.randn(B, V)
    tgts = torch.randint(0, V, (B,))
    sampled_ids = torch.randint(0, V, (K,))
    log_probs_sampled = torch.full((K,), -4.0)
    loss = sampled_softmax_loss(logits, tgts, sampled_ids, log_probs_sampled)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 9. NCELanguageModelLoss forward returns correct keys
# ---------------------------------------------------------------------------


def test_nce_lm_loss_forward_keys(nce_lm, hidden, targets):
    loss, metrics = nce_lm(hidden, targets)
    assert "nce_loss" in metrics, "metrics must contain 'nce_loss'"
    assert "n_noise" in metrics, "metrics must contain 'n_noise'"


# ---------------------------------------------------------------------------
# 10. NCELanguageModelLoss forward loss is positive
# ---------------------------------------------------------------------------


def test_nce_lm_loss_forward_loss_positive(nce_lm, hidden, targets):
    loss, _ = nce_lm(hidden, targets)
    assert loss.item() > 0.0, f"Expected positive loss, got {loss.item()}"


# ---------------------------------------------------------------------------
# 11. compare_nce_vs_softmax returns correct keys
# ---------------------------------------------------------------------------


def test_compare_nce_vs_softmax_keys():
    torch.manual_seed(3)
    model_scores = torch.randn(B, V)
    tgts = torch.randint(0, V, (B,))
    result = compare_nce_vs_softmax(model_scores, tgts, vocab_size=V)
    assert "softmax_loss" in result
    assert "nce_approx_loss" in result
    assert "relative_error" in result
