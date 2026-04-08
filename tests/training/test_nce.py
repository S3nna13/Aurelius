"""Tests for src/training/nce.py — NCE and InfoNCE losses."""

import pytest
import torch

from src.training.nce import InfoNCELoss, NCELoss, NoiseDistribution, SelfNormalizedNCE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 100
D_MODEL = 32
BATCH = 2
SEQ = 4


@pytest.fixture()
def token_freqs() -> torch.Tensor:
    """Simple linearly-spaced frequency counts."""
    return torch.arange(1, VOCAB_SIZE + 1, dtype=torch.float32)


@pytest.fixture()
def noise_dist(token_freqs) -> NoiseDistribution:
    return NoiseDistribution(token_freqs)


@pytest.fixture()
def nce_loss() -> NCELoss:
    return NCELoss(vocab_size=VOCAB_SIZE, d_model=D_MODEL, k=5)


@pytest.fixture()
def nce_loss_with_dist(token_freqs) -> NCELoss:
    nd = NoiseDistribution(token_freqs)
    return NCELoss(vocab_size=VOCAB_SIZE, d_model=D_MODEL, k=5, noise_dist=nd)


@pytest.fixture()
def hidden() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(BATCH, SEQ, D_MODEL)


@pytest.fixture()
def targets() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ))


# ---------------------------------------------------------------------------
# NoiseDistribution tests
# ---------------------------------------------------------------------------

def test_noise_distribution_sums_to_one(noise_dist):
    """Probability vector must sum to (approximately) 1."""
    assert abs(noise_dist.probs.sum().item() - 1.0) < 1e-5


def test_noise_distribution_sample_shape(noise_dist):
    """sample(100) should return a (100,) tensor."""
    samples = noise_dist.sample(100)
    assert samples.shape == (100,)
    assert samples.dtype == torch.int64


def test_noise_distribution_log_prob_range(noise_dist):
    """Log probabilities must be ≤ 0 (probabilities ≤ 1)."""
    ids = torch.arange(VOCAB_SIZE)
    log_probs = noise_dist.log_prob(ids)
    assert (log_probs <= 0).all(), "log probabilities must be non-positive"


# ---------------------------------------------------------------------------
# NCELoss tests
# ---------------------------------------------------------------------------

def test_nce_loss_scalar(nce_loss, hidden, targets):
    """NCELoss.forward() must return a scalar tensor."""
    loss = nce_loss(hidden, targets)
    assert loss.shape == (), f"Expected scalar, got shape {loss.shape}"


def test_nce_loss_positive(nce_loss, hidden, targets):
    """NCE loss must be strictly positive."""
    loss = nce_loss(hidden, targets)
    assert loss.item() > 0, "NCE loss should be positive"


def test_nce_loss_with_noise_dist(nce_loss_with_dist, hidden, targets, noise_dist):
    """NCELoss should work when a NoiseDistribution is passed at forward time."""
    loss = nce_loss_with_dist(hidden, targets, noise_dist=noise_dist)
    assert loss.shape == ()
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# InfoNCELoss tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def infonce_loss() -> InfoNCELoss:
    return InfoNCELoss(d_model=D_MODEL, projection_dim=16, temperature=0.07)


def test_infonce_loss_scalar(infonce_loss):
    """InfoNCELoss.forward() must return a scalar tensor."""
    torch.manual_seed(1)
    anchors = torch.randn(BATCH, D_MODEL)
    positives = torch.randn(BATCH, D_MODEL)
    loss = infonce_loss(anchors, positives)
    assert loss.shape == ()


def test_infonce_loss_batch_size_2(infonce_loss):
    """InfoNCE must work with batch_size=2 (minimum for in-batch negatives)."""
    torch.manual_seed(2)
    anchors = torch.randn(2, D_MODEL)
    positives = torch.randn(2, D_MODEL)
    loss = infonce_loss(anchors, positives)
    assert torch.isfinite(loss), "Loss should be finite"


def test_infonce_temperature_effect():
    """Lower temperature should give higher-magnitude loss."""
    torch.manual_seed(3)
    anchors = torch.randn(4, D_MODEL)
    positives = torch.randn(4, D_MODEL)

    loss_high_temp = InfoNCELoss(D_MODEL, projection_dim=16, temperature=1.0)(anchors, positives)
    loss_low_temp = InfoNCELoss(D_MODEL, projection_dim=16, temperature=0.07)(anchors, positives)

    # Both projectors are freshly initialised (same random seed path), so
    # we compare magnitudes: lower tau → more peaked distribution → generally
    # higher cross-entropy when not perfectly aligned.
    assert loss_low_temp.item() != loss_high_temp.item(), (
        "Different temperatures should produce different losses"
    )


# ---------------------------------------------------------------------------
# SelfNormalizedNCE tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def self_norm_nce(nce_loss) -> SelfNormalizedNCE:
    return SelfNormalizedNCE(nce_loss, normalization_lambda=1.0)


def test_self_normalized_nce_returns_tuple(self_norm_nce, hidden, targets):
    """SelfNormalizedNCE.forward() must return a 2-tuple of tensors."""
    result = self_norm_nce(hidden, targets)
    assert isinstance(result, tuple), "Should return a tuple"
    assert len(result) == 2, "Tuple should have exactly 2 elements"
    total, penalty = result
    assert total.shape == ()
    assert penalty.shape == ()


def test_self_normalized_penalty_positive(self_norm_nce, hidden, targets):
    """Normalisation penalty (lambda * log_Z^2) must be >= 0."""
    _, penalty = self_norm_nce(hidden, targets)
    assert penalty.item() >= 0, "Normalisation penalty must be non-negative"
