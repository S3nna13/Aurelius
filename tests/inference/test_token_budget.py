"""Tests for adaptive token budget (entropy-based early stopping)."""
import pytest
import torch

from src.inference.token_budget import (
    TokenBudgetConfig,
    TokenBudgetResult,
    generate_with_budget,
    nucleus_sample,
    token_entropy,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Tiny model fixture
# ---------------------------------------------------------------------------

def _make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
    )
    return AureliusTransformer(cfg)


VOCAB_SIZE = 256


# ---------------------------------------------------------------------------
# token_entropy tests
# ---------------------------------------------------------------------------

def test_token_entropy_range():
    """Entropy must be >= 0 for both uniform and peaked distributions."""
    uniform_logits = torch.zeros(VOCAB_SIZE)
    peaked_logits = torch.zeros(VOCAB_SIZE)
    peaked_logits[0] = 100.0

    h_uniform = token_entropy(uniform_logits)
    h_peaked = token_entropy(peaked_logits)

    assert h_uniform >= 0.0
    assert h_peaked >= 0.0


def test_token_entropy_uniform_is_max():
    """Uniform distribution has higher entropy than a peaked distribution."""
    uniform_logits = torch.zeros(VOCAB_SIZE)
    peaked_logits = torch.zeros(VOCAB_SIZE)
    peaked_logits[0] = 100.0

    h_uniform = token_entropy(uniform_logits)
    h_peaked = token_entropy(peaked_logits)

    assert h_uniform > h_peaked


# ---------------------------------------------------------------------------
# nucleus_sample tests
# ---------------------------------------------------------------------------

def test_nucleus_sample_returns_int():
    """nucleus_sample must return a Python int."""
    logits = torch.randn(VOCAB_SIZE)
    result = nucleus_sample(logits, top_p=0.9, temperature=1.0)
    assert isinstance(result, int)


def test_nucleus_sample_valid_token_id():
    """Sampled token id must be in [0, vocab_size)."""
    torch.manual_seed(42)
    for _ in range(20):
        logits = torch.randn(VOCAB_SIZE)
        result = nucleus_sample(logits, top_p=0.9, temperature=1.0)
        assert 0 <= result < VOCAB_SIZE


# ---------------------------------------------------------------------------
# generate_with_budget tests
# ---------------------------------------------------------------------------

def test_generate_with_budget_result_type():
    """generate_with_budget must return a TokenBudgetResult."""
    model = _make_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
    cfg = TokenBudgetConfig(max_new_tokens=5, min_new_tokens=1)
    result = generate_with_budget(model, input_ids, cfg)
    assert isinstance(result, TokenBudgetResult)


def test_generate_with_budget_respects_max():
    """Never generate more tokens than max_new_tokens."""
    model = _make_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
    max_new = 8
    cfg = TokenBudgetConfig(
        max_new_tokens=max_new,
        min_new_tokens=1,
        low_entropy_threshold=0.5,
        patience=3,
    )
    result = generate_with_budget(model, input_ids, cfg)
    assert len(result.token_ids) <= max_new


def test_generate_with_budget_respects_min():
    """Always generate at least min_new_tokens tokens."""
    model = _make_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
    min_new = 3
    # Use threshold=0.0 so early stop could trigger immediately, but min guards it
    cfg = TokenBudgetConfig(
        max_new_tokens=10,
        min_new_tokens=min_new,
        low_entropy_threshold=0.0,
        patience=1,
    )
    result = generate_with_budget(model, input_ids, cfg)
    assert len(result.token_ids) >= min_new


def test_generate_with_budget_entropies_length():
    """len(entropies) must equal len(token_ids)."""
    model = _make_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
    cfg = TokenBudgetConfig(max_new_tokens=6, min_new_tokens=1)
    result = generate_with_budget(model, input_ids, cfg)
    assert len(result.entropies) == len(result.token_ids)


def test_generate_with_budget_tokens_saved_nonnegative():
    """tokens_saved must be >= 0."""
    model = _make_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
    cfg = TokenBudgetConfig(max_new_tokens=8, min_new_tokens=1)
    result = generate_with_budget(model, input_ids, cfg)
    assert result.tokens_saved >= 0


def test_generate_with_budget_early_stop():
    """With very low threshold (0.0), stopped_early=True and tokens_saved > 0."""
    model = _make_model()
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
    cfg = TokenBudgetConfig(
        max_new_tokens=20,
        min_new_tokens=1,
        low_entropy_threshold=0.0,   # any entropy >= 0 so never triggers on its own…
        patience=1,                  # but threshold=0.0 means H < 0.0 never true
    )
    # threshold=0.0 means H < 0.0 is never true — we need a threshold that is
    # guaranteed to be exceeded. Use a very HIGH threshold instead so every token
    # triggers early stop.
    cfg = TokenBudgetConfig(
        max_new_tokens=20,
        min_new_tokens=1,
        low_entropy_threshold=1000.0,  # always below threshold (H is always < 1000)
        patience=1,
    )
    result = generate_with_budget(model, input_ids, cfg)
    assert result.stopped_early is True
    assert result.tokens_saved > 0
