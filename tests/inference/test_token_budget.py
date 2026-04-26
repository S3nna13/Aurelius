"""Tests for Token Budget & Adaptive Generation."""

import pytest
import torch

from src.inference.token_budget import (
    BudgetConfig,
    BudgetedGenerator,
    TokenBudget,
    adaptive_max_tokens,
    budget_aware_generate,
    estimate_sequence_complexity,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Tiny model helper
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
        max_seq_len=512,
    )
    return AureliusTransformer(cfg)


VOCAB_SIZE = 256


# ===================================================================
# BudgetConfig tests
# ===================================================================


class TestBudgetConfig:
    def test_defaults(self):
        """BudgetConfig defaults match spec."""
        cfg = BudgetConfig()
        assert cfg.max_tokens == 512
        assert cfg.target_efficiency == 0.8
        assert cfg.adaptive is True
        assert cfg.eos_token_id == 0
        assert cfg.pad_token_id == 0

    def test_custom_values(self):
        """BudgetConfig accepts custom values."""
        cfg = BudgetConfig(
            max_tokens=1024, target_efficiency=0.5, adaptive=False, eos_token_id=2, pad_token_id=1
        )
        assert cfg.max_tokens == 1024
        assert cfg.target_efficiency == 0.5
        assert cfg.adaptive is False
        assert cfg.eos_token_id == 2
        assert cfg.pad_token_id == 1


# ===================================================================
# TokenBudget tests
# ===================================================================


class TestTokenBudget:
    def test_initial_state(self):
        """Fresh budget: consumed=0, remaining=max, not exhausted."""
        b = TokenBudget(100)
        assert b.consumed == 0
        assert b.remaining == 100
        assert b.is_exhausted is False

    def test_consume(self):
        """Consuming tokens updates consumed/remaining."""
        b = TokenBudget(10)
        b.consume(3)
        assert b.consumed == 3
        assert b.remaining == 7

    def test_exhaustion(self):
        """Budget becomes exhausted when consumed >= max."""
        b = TokenBudget(5)
        b.consume(5)
        assert b.is_exhausted is True
        assert b.remaining == 0

    def test_over_consume(self):
        """Consuming past the limit still works; remaining clamps to 0."""
        b = TokenBudget(3)
        b.consume(10)
        assert b.is_exhausted is True
        assert b.remaining == 0
        assert b.consumed == 10

    def test_reset(self):
        """Reset brings consumed back to 0."""
        b = TokenBudget(10)
        b.consume(7)
        b.reset()
        assert b.consumed == 0
        assert b.remaining == 10
        assert b.is_exhausted is False

    def test_consume_negative_raises(self):
        """Consuming a negative number raises ValueError."""
        b = TokenBudget(10)
        with pytest.raises(ValueError):
            b.consume(-1)

    def test_negative_max_raises(self):
        """Negative max_tokens raises ValueError."""
        with pytest.raises(ValueError):
            TokenBudget(-1)


# ===================================================================
# estimate_sequence_complexity tests
# ===================================================================


class TestEstimateSequenceComplexity:
    def test_returns_float_in_range(self):
        """Complexity score must be a float in [0, 1]."""
        model = _make_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        c = estimate_sequence_complexity(model, input_ids)
        assert isinstance(c, float)
        assert 0.0 <= c <= 1.0

    def test_deterministic(self):
        """Same model + same input -> same complexity."""
        model = _make_model()
        ids = torch.randint(0, VOCAB_SIZE, (1, 8))
        c1 = estimate_sequence_complexity(model, ids)
        c2 = estimate_sequence_complexity(model, ids)
        assert c1 == pytest.approx(c2)


# ===================================================================
# adaptive_max_tokens tests
# ===================================================================


class TestAdaptiveMaxTokens:
    def test_simple_halves(self):
        """complexity < 0.3 -> budget halved (times efficiency)."""
        result = adaptive_max_tokens(100, complexity=0.1, target_efficiency=1.0)
        assert result == 50

    def test_complex_doubles(self):
        """complexity > 0.7 -> budget doubled (times efficiency)."""
        result = adaptive_max_tokens(100, complexity=0.9, target_efficiency=1.0)
        assert result == 200

    def test_mid_complexity(self):
        """complexity == 0.5 -> between half and double."""
        result = adaptive_max_tokens(100, complexity=0.5, target_efficiency=1.0)
        assert 50 < result < 200

    def test_efficiency_scales(self):
        """target_efficiency < 1.0 reduces the budget."""
        full = adaptive_max_tokens(100, complexity=0.5, target_efficiency=1.0)
        half = adaptive_max_tokens(100, complexity=0.5, target_efficiency=0.5)
        assert half < full

    def test_minimum_is_one(self):
        """Result is always at least 1."""
        result = adaptive_max_tokens(1, complexity=0.0, target_efficiency=0.01)
        assert result >= 1

    def test_invalid_complexity_raises(self):
        """complexity outside [0,1] raises ValueError."""
        with pytest.raises(ValueError):
            adaptive_max_tokens(100, complexity=-0.1)
        with pytest.raises(ValueError):
            adaptive_max_tokens(100, complexity=1.5)

    def test_invalid_efficiency_raises(self):
        """target_efficiency outside (0,1] raises ValueError."""
        with pytest.raises(ValueError):
            adaptive_max_tokens(100, complexity=0.5, target_efficiency=0.0)
        with pytest.raises(ValueError):
            adaptive_max_tokens(100, complexity=0.5, target_efficiency=1.5)


# ===================================================================
# budget_aware_generate tests
# ===================================================================


class TestBudgetAwareGenerate:
    def test_respects_budget(self):
        """Never generates more tokens than the budget allows."""
        model = _make_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
        budget = TokenBudget(5)
        tokens = budget_aware_generate(model, input_ids, budget)
        assert len(tokens) <= 5

    def test_returns_tensor(self):
        """Return value is a 1-D tensor."""
        model = _make_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
        budget = TokenBudget(3)
        tokens = budget_aware_generate(model, input_ids, budget)
        assert isinstance(tokens, torch.Tensor)
        assert tokens.dim() == 1

    def test_budget_consumed(self):
        """Budget consumed count matches number of generated tokens."""
        model = _make_model()
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
        budget = TokenBudget(6)
        tokens = budget_aware_generate(model, input_ids, budget)
        assert budget.consumed == len(tokens)


# ===================================================================
# BudgetedGenerator tests
# ===================================================================


class TestBudgetedGenerator:
    def test_generate_returns_dict(self):
        """generate() returns a dict with expected keys."""
        model = _make_model()
        gen = BudgetedGenerator(model, BudgetConfig(max_tokens=8))
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
        result = gen.generate(input_ids)
        assert isinstance(result, dict)
        assert "tokens" in result
        assert "tokens_consumed" in result
        assert "efficiency" in result

    def test_tokens_consumed_matches(self):
        """tokens_consumed matches len(tokens)."""
        model = _make_model()
        gen = BudgetedGenerator(model, BudgetConfig(max_tokens=8))
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
        result = gen.generate(input_ids)
        assert result["tokens_consumed"] == len(result["tokens"])

    def test_efficiency_in_range(self):
        """efficiency is in [0, 1]."""
        model = _make_model()
        gen = BudgetedGenerator(model, BudgetConfig(max_tokens=10))
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
        result = gen.generate(input_ids)
        assert 0.0 <= result["efficiency"] <= 1.0

    def test_non_adaptive_uses_raw_budget(self):
        """When adaptive=False, budget equals max_tokens exactly."""
        model = _make_model()
        cfg = BudgetConfig(max_tokens=4, adaptive=False)
        gen = BudgetedGenerator(model, cfg)
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 4))
        result = gen.generate(input_ids)
        # With adaptive=False, max_tok == 4, so consumed <= 4
        assert result["tokens_consumed"] <= 4

    def test_default_config(self):
        """BudgetedGenerator works with default BudgetConfig when None passed."""
        model = _make_model()
        gen = BudgetedGenerator(model)
        assert gen.config.max_tokens == 512
        assert gen.config.adaptive is True
