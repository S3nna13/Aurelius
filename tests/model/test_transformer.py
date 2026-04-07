"""Tests for the full AureliusTransformer."""
import torch
import pytest
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer, count_parameters


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=256, n_heads=4, n_kv_heads=2,
        head_dim=64, d_ff=512, vocab_size=1000, max_seq_len=128,
    )


@pytest.fixture
def small_model(small_cfg):
    return AureliusTransformer(small_cfg)


def test_forward_pass_shape(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 32))
    logits = small_model(tokens)
    assert logits.shape == (2, 32, small_cfg.vocab_size)


def test_tied_embeddings(small_cfg):
    model = AureliusTransformer(small_cfg)
    assert model.lm_head.weight is model.embed.weight


def test_untied_embeddings():
    cfg = AureliusConfig(
        n_layers=2, d_model=256, n_heads=4, n_kv_heads=2,
        head_dim=64, d_ff=512, vocab_size=1000, tie_embeddings=False,
    )
    model = AureliusTransformer(cfg)
    assert model.lm_head.weight is not model.embed.weight


def test_sequence_length_limit(small_model, small_cfg):
    with pytest.raises(AssertionError):
        tokens = torch.randint(0, small_cfg.vocab_size, (1, small_cfg.max_seq_len + 1))
        small_model(tokens)


def test_full_model_parameter_count():
    """Full 1.3B model — verify parameter count is in expected range."""
    cfg = AureliusConfig()
    model = AureliusTransformer(cfg)
    n = count_parameters(model)
    # Allow 1.2B–1.5B range
    assert 1_200_000_000 < n < 1_500_000_000, f"Unexpected param count: {n:,}"


def test_parameter_count_breakdown(small_model):
    counts = small_model.count_parameters()
    assert "total" in counts
    assert counts["total"] > 0
    assert counts["all_layers"] > 0


def test_no_nan_in_output(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 16))
    with torch.no_grad():
        logits = small_model(tokens)
    assert not torch.isnan(logits).any(), "NaN in forward pass output"
    assert not torch.isinf(logits).any(), "Inf in forward pass output"


def test_batch_size_one(small_model, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (1, 16))
    logits = small_model(tokens)
    assert logits.shape == (1, 16, small_cfg.vocab_size)
