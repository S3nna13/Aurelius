import pytest
import torch

from src.inference.uncertainty import (
    MCDropoutEstimator,
    UncertaintyConfig,
    mutual_information,
    predictive_entropy,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=64,
        max_seq_len=64,
    )
    torch.manual_seed(0)
    model = AureliusTransformer(cfg)
    return model


def test_predictive_entropy_shape():
    probs = torch.softmax(torch.randn(2, 8, 64), dim=-1)
    ent = predictive_entropy(probs)
    assert ent.shape == (2, 8)


def test_predictive_entropy_nonneg():
    probs = torch.softmax(torch.randn(1, 4, 16), dim=-1)
    ent = predictive_entropy(probs)
    assert (ent >= 0).all()


def test_predictive_entropy_deterministic_is_zero():
    """Deterministic distribution has zero entropy."""
    probs = torch.zeros(1, 1, 8)
    probs[0, 0, 3] = 1.0  # all mass on token 3
    ent = predictive_entropy(probs)
    assert ent[0, 0].item() < 0.01


def test_mutual_information_shape():
    sample_probs = torch.softmax(torch.randn(5, 2, 8, 32), dim=-1)
    mi = mutual_information(sample_probs)
    assert mi.shape == (2, 8)


def test_mutual_information_nonneg():
    sample_probs = torch.softmax(torch.randn(4, 1, 6, 16), dim=-1)
    mi = mutual_information(sample_probs)
    assert (mi >= 0).all()


def test_mc_dropout_estimator_shapes(small_model):
    cfg = UncertaintyConfig(n_samples=3, dropout_p=0.1)
    estimator = MCDropoutEstimator(small_model, cfg)
    input_ids = torch.randint(0, 64, (1, 8))
    result = estimator.estimate(input_ids)
    assert result.predictive_entropy.shape == (1, 8)
    assert result.mutual_information.shape == (1, 8)
    assert result.confidence.shape == (1, 8)


def test_mc_dropout_estimator_confidence_in_range(small_model):
    cfg = UncertaintyConfig(n_samples=3)
    estimator = MCDropoutEstimator(small_model, cfg)
    input_ids = torch.randint(0, 64, (1, 6))
    result = estimator.estimate(input_ids)
    assert (result.confidence >= 0).all()
    assert (result.confidence <= 1.0 + 1e-5).all()


def test_mc_dropout_summary(small_model):
    cfg = UncertaintyConfig(n_samples=3)
    estimator = MCDropoutEstimator(small_model, cfg)
    input_ids = torch.randint(0, 64, (1, 4))
    result = estimator.estimate(input_ids)
    s = result.summary()
    assert "entropy" in s.lower()
