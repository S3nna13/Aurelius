"""Tests for logit-fusion Mixture-of-Agents (MoA) inference."""

from __future__ import annotations

import pytest
import torch

from src.inference.mixture_of_agents import (
    MixtureOfAgents,
    MoAConfig,
    MoADecoder,
    aggregate_logits_max_prob,
    aggregate_logits_mean,
    aggregate_logits_weighted,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_cfg():
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
def model_a(small_cfg):
    torch.manual_seed(0)
    m = AureliusTransformer(small_cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def model_b(small_cfg):
    torch.manual_seed(99)
    m = AureliusTransformer(small_cfg)
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_ids():
    """(2, 5) random token tensor — batch=2, seq_len=5."""
    torch.manual_seed(7)
    return torch.randint(0, 256, (2, 5))


# Helpers
B, T, V = 2, 5, 256


def _random_logits(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, V)


# ---------------------------------------------------------------------------
# 1. MoAConfig defaults
# ---------------------------------------------------------------------------


def test_moa_config_default_aggregation():
    cfg = MoAConfig()
    assert cfg.aggregation == "mean"


def test_moa_config_default_temperature():
    cfg = MoAConfig()
    assert cfg.temperature == 1.0


def test_moa_config_default_weights_none():
    cfg = MoAConfig()
    assert cfg.weights is None


# ---------------------------------------------------------------------------
# 4. aggregate_logits_mean shape and correctness
# ---------------------------------------------------------------------------


def test_aggregate_logits_mean_shape():
    logits = [_random_logits(i) for i in range(3)]
    out = aggregate_logits_mean(logits)
    assert out.shape == (B, T, V)


def test_aggregate_logits_mean_single():
    """Mean of a single tensor is that tensor."""
    item = _random_logits(42)
    out = aggregate_logits_mean([item])
    assert torch.allclose(out, item)


def test_aggregate_logits_mean_correctness():
    """Mean of two tensors equals element-wise average."""
    l1 = _random_logits(1)
    l2 = _random_logits(2)
    out = aggregate_logits_mean([l1, l2])
    expected = (l1 + l2) / 2.0
    assert torch.allclose(out, expected)


# ---------------------------------------------------------------------------
# 5. aggregate_logits_weighted
# ---------------------------------------------------------------------------


def test_aggregate_logits_weighted_shape():
    logits = [_random_logits(i) for i in range(2)]
    out = aggregate_logits_weighted(logits, [0.3, 0.7])
    assert out.shape == (B, T, V)


def test_aggregate_logits_weighted_equal_weights_equals_mean():
    """Equal weights should give the same result as mean."""
    l1 = _random_logits(10)
    l2 = _random_logits(11)
    weighted = aggregate_logits_weighted([l1, l2], [1.0, 1.0])
    mean = aggregate_logits_mean([l1, l2])
    assert torch.allclose(weighted, mean, atol=1e-5)


def test_aggregate_logits_weighted_single_model():
    """Single model with any non-zero weight returns that model's logits."""
    item = _random_logits(20)
    out = aggregate_logits_weighted([item], [5.0])
    assert torch.allclose(out, item, atol=1e-5)


def test_aggregate_logits_weighted_correctness():
    """Manually verify weighted average formula."""
    l1 = _random_logits(3)
    l2 = _random_logits(4)
    w = [0.25, 0.75]
    out = aggregate_logits_weighted([l1, l2], w)
    expected = (0.25 * l1 + 0.75 * l2) / (0.25 + 0.75)
    assert torch.allclose(out, expected, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. aggregate_logits_max_prob
# ---------------------------------------------------------------------------


def test_aggregate_logits_max_prob_shape():
    logits = [_random_logits(i) for i in range(3)]
    out = aggregate_logits_max_prob(logits)
    assert out.shape == (B, T, V)


def test_aggregate_logits_max_prob_single():
    """With a single model, output equals that model's logits."""
    item = _random_logits(55)
    out = aggregate_logits_max_prob([item])
    assert torch.allclose(out, item)


def test_aggregate_logits_max_prob_selects_highest_prob():
    """Winning model at each position has highest max probability."""
    torch.manual_seed(77)
    l1 = torch.zeros(1, 1, 4)
    l2 = torch.zeros(1, 1, 4)
    # Make model 2 clearly dominant: very large logit at position 0
    l2[0, 0, 0] = 100.0
    out = aggregate_logits_max_prob([l1, l2])
    # Output should equal l2 at this single (B=0, T=0) position
    assert torch.allclose(out, l2, atol=1e-5)


# ---------------------------------------------------------------------------
# 7. MixtureOfAgents.forward shape
# ---------------------------------------------------------------------------


def test_mixture_forward_shape(model_a, model_b, input_ids):
    cfg = MoAConfig(aggregation="mean")
    moa = MixtureOfAgents([model_a, model_b], cfg)
    out = moa.forward(input_ids)
    B_in, T_in = input_ids.shape
    assert out.shape == (B_in, T_in, 256)


# ---------------------------------------------------------------------------
# 8. MixtureOfAgents with single model returns same logits
# ---------------------------------------------------------------------------


def test_mixture_single_model_equals_direct(model_a, input_ids):
    cfg = MoAConfig(aggregation="mean", temperature=1.0)
    moa = MixtureOfAgents([model_a], cfg)
    fused = moa.forward(input_ids)
    _, direct, _ = model_a(input_ids)
    assert torch.allclose(fused, direct, atol=1e-5)


# ---------------------------------------------------------------------------
# 9. MixtureOfAgents with all three aggregation modes
# ---------------------------------------------------------------------------


def test_mixture_mode_mean(model_a, model_b, input_ids):
    cfg = MoAConfig(aggregation="mean")
    moa = MixtureOfAgents([model_a, model_b], cfg)
    out = moa.forward(input_ids)
    assert out.shape[0] == input_ids.shape[0]


def test_mixture_mode_weighted(model_a, model_b, input_ids):
    cfg = MoAConfig(aggregation="weighted", weights=[0.4, 0.6])
    moa = MixtureOfAgents([model_a, model_b], cfg)
    out = moa.forward(input_ids)
    assert out.shape[0] == input_ids.shape[0]


def test_mixture_mode_max_prob(model_a, model_b, input_ids):
    cfg = MoAConfig(aggregation="max_prob")
    moa = MixtureOfAgents([model_a, model_b], cfg)
    out = moa.forward(input_ids)
    assert out.shape[0] == input_ids.shape[0]


# ---------------------------------------------------------------------------
# 10. MoADecoder.generate returns token tensor of correct length
# ---------------------------------------------------------------------------


def test_decoder_generate_length(model_a, input_ids):
    cfg = MoAConfig(aggregation="mean")
    decoder = MoADecoder([model_a], cfg)
    max_new = 6
    out = decoder.generate(input_ids, max_new_tokens=max_new)
    B_in, T_in = input_ids.shape
    assert out.shape == (B_in, T_in + max_new)


def test_decoder_generate_returns_integer_tensor(model_a, input_ids):
    cfg = MoAConfig(aggregation="mean")
    decoder = MoADecoder([model_a], cfg)
    out = decoder.generate(input_ids, max_new_tokens=3)
    assert out.dtype in (torch.int64, torch.int32, torch.long)


def test_decoder_generate_preserves_prompt(model_a, input_ids):
    """The first T tokens of the output should equal the prompt."""
    cfg = MoAConfig(aggregation="mean")
    decoder = MoADecoder([model_a], cfg)
    out = decoder.generate(input_ids, max_new_tokens=4)
    assert torch.equal(out[:, : input_ids.shape[1]], input_ids)


# ---------------------------------------------------------------------------
# 11. Temperature scaling effect
# ---------------------------------------------------------------------------


def test_temperature_scaling_changes_logits(model_a, input_ids):
    """Logits with temperature=0.5 should differ from temperature=1.0."""
    cfg_1 = MoAConfig(aggregation="mean", temperature=1.0)
    cfg_hot = MoAConfig(aggregation="mean", temperature=0.5)
    moa_1 = MixtureOfAgents([model_a], cfg_1)
    moa_hot = MixtureOfAgents([model_a], cfg_hot)
    out_1 = moa_1.forward(input_ids)
    out_hot = moa_hot.forward(input_ids)
    # Divided by 0.5 means values should be approximately doubled
    assert torch.allclose(out_hot, out_1 * 2.0, atol=1e-4)


def test_temperature_1_leaves_logits_unchanged(model_a, input_ids):
    """Temperature=1.0 should return raw aggregated logits."""
    cfg = MoAConfig(aggregation="mean", temperature=1.0)
    moa = MixtureOfAgents([model_a], cfg)
    out = moa.forward(input_ids)
    _, direct, _ = model_a(input_ids)
    assert torch.allclose(out, direct, atol=1e-5)
