"""Tests for src/data/domain_reweighting.py"""

import random
import pytest
import torch

from src.data.domain_reweighting import (
    DomainConfig,
    DomainReweighter,
    build_reweighted_batch,
    doro_update,
    ema_update,
    exp3_update,
    normalize_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DOMAINS = ["web", "books", "code"]


def _uniform_weights(n: int) -> torch.Tensor:
    return torch.ones(n) / n


def _losses(values: list[float]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float)


def _simple_domain_data(n_seqs: int = 5, seq_len: int = 8) -> dict[str, list[torch.Tensor]]:
    rng = random.Random(0)
    return {
        domain: [torch.randint(0, 100, (rng.randint(4, seq_len),)) for _ in range(n_seqs)]
        for domain in DOMAINS
    }


# ---------------------------------------------------------------------------
# 1. DomainConfig initializes correctly
# ---------------------------------------------------------------------------

def test_domain_config_defaults():
    cfg = DomainConfig(domain_names=DOMAINS)
    assert cfg.domain_names == DOMAINS
    assert cfg.initial_weights is None
    assert cfg.learning_rate == 1.0
    assert cfg.smoothing_alpha == 0.1
    assert cfg.min_weight == 0.01
    assert cfg.normalize is True
    assert cfg.strategy == "ema"


# ---------------------------------------------------------------------------
# 2. normalize_weights clips to min_weight and sums to 1
# ---------------------------------------------------------------------------

def test_normalize_weights_clips_and_sums_to_one():
    w = torch.tensor([0.0, 0.001, 0.5])
    out = normalize_weights(w, min_weight=0.05)
    assert out.min().item() >= 0.05 - 1e-6
    assert abs(out.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 3. normalize_weights uniform input stays approximately uniform
# ---------------------------------------------------------------------------

def test_normalize_weights_uniform_stays_uniform():
    n = 4
    w = torch.ones(n)
    out = normalize_weights(w, min_weight=0.01)
    expected = 1.0 / n
    for val in out.tolist():
        assert abs(val - expected) < 1e-5


# ---------------------------------------------------------------------------
# 4. ema_update increases weight for high-loss domain
# ---------------------------------------------------------------------------

def test_ema_update_upweights_high_loss_domain():
    n = 3
    w = _uniform_weights(n)
    # Domain 2 has much higher loss
    losses = _losses([0.1, 0.1, 10.0])
    updated = ema_update(w, losses, alpha=0.5, lr=1.0)
    assert updated[2].item() > updated[0].item()
    assert updated[2].item() > updated[1].item()


# ---------------------------------------------------------------------------
# 5. ema_update result sums approximately to 1
# ---------------------------------------------------------------------------

def test_ema_update_sums_to_one():
    w = _uniform_weights(4)
    losses = _losses([1.0, 2.0, 0.5, 3.0])
    updated = ema_update(w, losses, alpha=0.3, lr=1.0)
    assert abs(updated.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 6. exp3_update output is a valid probability distribution
# ---------------------------------------------------------------------------

def test_exp3_update_valid_prob_distribution():
    w = _uniform_weights(3)
    losses = _losses([1.0, 2.0, 0.5])
    out = exp3_update(w, losses, lr=0.1)
    assert out.min().item() >= 0.0
    assert abs(out.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 7. exp3_update upweights high-loss domain
# ---------------------------------------------------------------------------

def test_exp3_update_upweights_high_loss():
    w = _uniform_weights(3)
    # Domain 0 has by far the highest loss
    losses = _losses([50.0, 1.0, 1.0])
    out = exp3_update(w, losses, lr=0.1)
    assert out[0].item() > out[1].item()
    assert out[0].item() > out[2].item()


# ---------------------------------------------------------------------------
# 8. doro_update result sums approximately to 1
# ---------------------------------------------------------------------------

def test_doro_update_sums_to_one():
    w = _uniform_weights(3)
    losses = _losses([1.0, 3.0, 0.5])
    out = doro_update(w, losses, lr=0.5)
    assert abs(out.sum().item() - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# 9. DomainReweighter.get_weights returns dict with all domains
# ---------------------------------------------------------------------------

def test_reweighter_get_weights_has_all_domains():
    cfg = DomainConfig(domain_names=DOMAINS)
    rw = DomainReweighter(cfg)
    weights = rw.get_weights()
    assert set(weights.keys()) == set(DOMAINS)
    assert all(isinstance(v, float) for v in weights.values())


# ---------------------------------------------------------------------------
# 10. DomainReweighter.sample_domain returns a valid domain name
# ---------------------------------------------------------------------------

def test_reweighter_sample_domain_valid():
    cfg = DomainConfig(domain_names=DOMAINS)
    rw = DomainReweighter(cfg)
    rng = random.Random(42)
    for _ in range(20):
        d = rw.sample_domain(rng)
        assert d in DOMAINS


# ---------------------------------------------------------------------------
# 11. DomainReweighter.update changes weights after high-loss signal
# ---------------------------------------------------------------------------

def test_reweighter_update_changes_weights():
    cfg = DomainConfig(domain_names=DOMAINS, learning_rate=2.0, smoothing_alpha=0.5)
    rw = DomainReweighter(cfg)
    before = rw.get_weights().copy()

    # Give "code" a very high loss many times to force a visible shift
    for _ in range(10):
        rw.update({"web": 0.1, "books": 0.1, "code": 10.0})

    after = rw.get_weights()
    assert after["code"] > before["code"]


# ---------------------------------------------------------------------------
# 12. build_reweighted_batch returns correct batch size
# ---------------------------------------------------------------------------

def test_build_batch_correct_size():
    cfg = DomainConfig(domain_names=DOMAINS)
    rw = DomainReweighter(cfg)
    data = _simple_domain_data()
    rng = random.Random(7)
    batch, labels = build_reweighted_batch(data, rw, batch_size=8, rng=rng)
    assert batch.shape[0] == 8
    assert len(labels) == 8


# ---------------------------------------------------------------------------
# 13. build_reweighted_batch domain_labels contains valid domain names
# ---------------------------------------------------------------------------

def test_build_batch_labels_valid():
    cfg = DomainConfig(domain_names=DOMAINS)
    rw = DomainReweighter(cfg)
    data = _simple_domain_data()
    rng = random.Random(99)
    _, labels = build_reweighted_batch(data, rw, batch_size=16, rng=rng)
    for label in labels:
        assert label in DOMAINS


# ---------------------------------------------------------------------------
# 14. DomainReweighter with exp3 strategy updates correctly
# ---------------------------------------------------------------------------

def test_reweighter_exp3_strategy():
    cfg = DomainConfig(
        domain_names=DOMAINS,
        strategy="exp3",
        learning_rate=0.5,
    )
    rw = DomainReweighter(cfg)
    before = rw.get_weights().copy()

    # Repeatedly signal high loss for "books"
    for _ in range(15):
        rw.update({"web": 0.1, "books": 8.0, "code": 0.1})

    after = rw.get_weights()
    # books weight should have increased
    assert after["books"] > before["books"]
    # result is still a valid distribution
    total = sum(after.values())
    assert abs(total - 1.0) < 1e-4
