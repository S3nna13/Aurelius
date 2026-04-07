"""Tests for Best-of-N sampling with self-certainty scoring."""
import math
import torch
import pytest
from src.inference.best_of_n import BestOfN, self_certainty_score
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=128, max_seq_len=64,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


@pytest.fixture
def bon(small_model):
    return BestOfN(small_model, n=3)


def test_self_certainty_returns_float():
    logits = torch.randn(5, 64)
    token_ids = torch.randint(0, 64, (5,))
    score = self_certainty_score(logits, token_ids)
    assert isinstance(score, float)
    assert math.isfinite(score)


def test_self_certainty_empty_returns_neg_inf():
    score = self_certainty_score(torch.randn(0, 64), torch.tensor([], dtype=torch.long))
    assert score == float("-inf")


def test_self_certainty_peaked_higher_than_uniform():
    """Peaked distribution should score higher than uniform."""
    vocab = 64
    # Peaked: one-hot logits
    peaked_logits = torch.full((4, vocab), -100.0)
    peaked_logits[torch.arange(4), torch.arange(4)] = 100.0
    token_ids = torch.arange(4)
    peaked_score = self_certainty_score(peaked_logits, token_ids)

    # Uniform: all logits equal
    uniform_logits = torch.zeros(4, vocab)
    uniform_score = self_certainty_score(uniform_logits, token_ids)

    assert peaked_score > uniform_score


def test_generate_output_shape(bon, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (1, 4))
    out = bon.generate(tokens, max_new_tokens=6)
    assert out.shape[0] == 1
    assert out.shape[1] >= 5  # at least one new token


def test_generate_batch_size_one_enforced(bon, small_cfg):
    tokens = torch.randint(0, small_cfg.vocab_size, (2, 4))
    with pytest.raises(ValueError, match="batch_size=1"):
        bon.generate(tokens, max_new_tokens=4)


def test_model_in_eval_mode(bon):
    assert not bon.model.training


def test_generate_selects_from_n_completions(small_model, small_cfg):
    """BestOfN with n=1 should behave like regular generate."""
    bon1 = BestOfN(small_model, n=1)
    torch.manual_seed(5)
    tokens = torch.randint(0, small_cfg.vocab_size, (1, 4))
    out = bon1.generate(tokens, max_new_tokens=4, temperature=0.01)
    assert out.shape[1] >= 5
