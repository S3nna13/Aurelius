"""Tests for src/inference/token_budget_v2.py

Covers:
- BudgetConfig defaults
- TokenUsage defaults
- compute_token_usage correctness
- check_budget pass / fail cases
- TokenBudgetTracker accumulation, warn, reset, summary
- estimate_tokens_from_chars
- truncate_to_budget (left / right / no-op)
"""
from __future__ import annotations

import pytest

from src.inference.token_budget_v2 import (
    BudgetConfig,
    TokenUsage,
    TokenBudgetTracker,
    check_budget,
    compute_token_usage,
    estimate_tokens_from_chars,
    truncate_to_budget,
)


# ---------------------------------------------------------------------------
# BudgetConfig
# ---------------------------------------------------------------------------

class TestBudgetConfig:
    def test_defaults(self):
        cfg = BudgetConfig()
        assert cfg.max_tokens == 10_000
        assert cfg.max_prompt_tokens == 4_096
        assert cfg.max_completion_tokens == 2_048
        assert cfg.cost_per_prompt_token == pytest.approx(1e-6)
        assert cfg.cost_per_completion_token == pytest.approx(2e-6)
        assert cfg.warn_at_fraction == pytest.approx(0.8)

    def test_custom_values(self):
        cfg = BudgetConfig(max_tokens=500, warn_at_fraction=0.5)
        assert cfg.max_tokens == 500
        assert cfg.warn_at_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# TokenUsage
# ---------------------------------------------------------------------------

class TestTokenUsage:
    def test_defaults_all_zero(self):
        usage = TokenUsage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0
        assert usage.estimated_cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_token_usage
# ---------------------------------------------------------------------------

class TestComputeTokenUsage:
    def setup_method(self):
        self.cfg = BudgetConfig(
            cost_per_prompt_token=1e-3,
            cost_per_completion_token=2e-3,
        )

    def test_total_equals_prompt_plus_completion(self):
        usage = compute_token_usage(100, 50, self.cfg)
        assert usage.total_tokens == 150

    def test_cost_is_positive(self):
        usage = compute_token_usage(100, 50, self.cfg)
        assert usage.estimated_cost > 0.0

    def test_cost_formula(self):
        usage = compute_token_usage(100, 50, self.cfg)
        expected = 100 * 1e-3 + 50 * 2e-3
        assert usage.estimated_cost == pytest.approx(expected)

    def test_zero_tokens(self):
        usage = compute_token_usage(0, 0, self.cfg)
        assert usage.total_tokens == 0
        assert usage.estimated_cost == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# check_budget
# ---------------------------------------------------------------------------

class TestCheckBudget:
    def setup_method(self):
        self.cfg = BudgetConfig(
            max_tokens=200,
            max_prompt_tokens=100,
            max_completion_tokens=80,
        )

    def test_within_budget_returns_true(self):
        usage = compute_token_usage(50, 40, self.cfg)
        ok, msg = check_budget(usage, self.cfg)
        assert ok is True
        assert msg == ""

    def test_over_max_tokens_returns_false(self):
        usage = compute_token_usage(150, 100, self.cfg)  # total=250 > 200
        ok, msg = check_budget(usage, self.cfg)
        assert ok is False

    def test_over_max_tokens_message_mentions_total(self):
        usage = compute_token_usage(150, 100, self.cfg)
        _, msg = check_budget(usage, self.cfg)
        assert "total_tokens" in msg or "max_tokens" in msg

    def test_over_prompt_limit_returns_false(self):
        usage = TokenUsage(prompt_tokens=110, completion_tokens=10, total_tokens=120)
        ok, msg = check_budget(usage, self.cfg)
        assert ok is False
        assert "prompt" in msg

    def test_over_completion_limit_returns_false(self):
        usage = TokenUsage(prompt_tokens=50, completion_tokens=90, total_tokens=140)
        ok, msg = check_budget(usage, self.cfg)
        assert ok is False
        assert "completion" in msg

    def test_exactly_at_limit_is_within_budget(self):
        usage = compute_token_usage(100, 80, self.cfg)  # total=180 <= 200
        ok, _ = check_budget(usage, self.cfg)
        assert ok is True


# ---------------------------------------------------------------------------
# TokenBudgetTracker
# ---------------------------------------------------------------------------

class TestTokenBudgetTracker:
    def setup_method(self):
        self.cfg = BudgetConfig(max_tokens=1_000, warn_at_fraction=0.8)
        self.tracker = TokenBudgetTracker(self.cfg)

    def test_record_accumulates_correctly(self):
        self.tracker.record(100, 50)
        self.tracker.record(200, 30)
        usage = self.tracker.total_usage()
        assert usage.prompt_tokens == 300
        assert usage.completion_tokens == 80
        assert usage.total_tokens == 380

    def test_record_returns_current_usage(self):
        usage = self.tracker.record(10, 5)
        assert usage.total_tokens == 15

    def test_total_usage_after_multiple_records(self):
        for _ in range(5):
            self.tracker.record(20, 10)
        usage = self.tracker.total_usage()
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_is_within_budget_true_initially(self):
        assert self.tracker.is_within_budget() is True

    def test_is_within_budget_false_when_exceeded(self):
        self.tracker.record(600, 600)  # total=1200 > 1000
        assert self.tracker.is_within_budget() is False

    def test_should_warn_false_initially(self):
        assert self.tracker.should_warn() is False

    def test_should_warn_true_at_threshold(self):
        # warn_at_fraction=0.8, max_tokens=1000 => threshold=800
        self.tracker.record(400, 400)  # total=800 == threshold
        assert self.tracker.should_warn() is True

    def test_should_warn_false_below_threshold(self):
        self.tracker.record(300, 300)  # total=600 < 800
        assert self.tracker.should_warn() is False

    def test_reset_clears_state(self):
        self.tracker.record(500, 300)
        self.tracker.reset()
        usage = self.tracker.total_usage()
        assert usage.total_tokens == 0
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0

    def test_reset_clears_n_calls(self):
        self.tracker.record(10, 5)
        self.tracker.reset()
        summary = self.tracker.usage_summary()
        assert summary["n_calls"] == 0

    def test_usage_summary_has_all_required_keys(self):
        self.tracker.record(50, 25)
        summary = self.tracker.usage_summary()
        required_keys = {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "estimated_cost",
            "budget_fraction",
            "n_calls",
        }
        assert required_keys.issubset(summary.keys())

    def test_usage_summary_budget_fraction(self):
        self.tracker.record(500, 0)  # total=500, max=1000 => fraction=0.5
        summary = self.tracker.usage_summary()
        assert summary["budget_fraction"] == pytest.approx(0.5)

    def test_usage_summary_n_calls(self):
        self.tracker.record(10, 5)
        self.tracker.record(10, 5)
        self.tracker.record(10, 5)
        assert self.tracker.usage_summary()["n_calls"] == 3


# ---------------------------------------------------------------------------
# estimate_tokens_from_chars
# ---------------------------------------------------------------------------

class TestEstimateTokensFromChars:
    def test_returns_positive_int_for_nonempty(self):
        result = estimate_tokens_from_chars("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string_returns_zero(self):
        assert estimate_tokens_from_chars("") == 0

    def test_exactly_divisible(self):
        # 8 chars / 4.0 => 2 tokens exactly
        assert estimate_tokens_from_chars("abcdefgh") == 2

    def test_rounds_up(self):
        # 5 chars / 4.0 => 1.25 => ceil => 2
        assert estimate_tokens_from_chars("abcde") == 2

    def test_custom_chars_per_token(self):
        # 10 chars / 2.0 => 5 tokens
        assert estimate_tokens_from_chars("a" * 10, chars_per_token=2.0) == 5


# ---------------------------------------------------------------------------
# truncate_to_budget
# ---------------------------------------------------------------------------

class TestTruncateToBudget:
    def test_left_truncation_keeps_end(self):
        ids = list(range(10))  # [0..9]
        result = truncate_to_budget(ids, max_tokens=4, side="left")
        assert result == [6, 7, 8, 9]

    def test_right_truncation_keeps_start(self):
        ids = list(range(10))  # [0..9]
        result = truncate_to_budget(ids, max_tokens=4, side="right")
        assert result == [0, 1, 2, 3]

    def test_no_op_when_under_budget(self):
        ids = [1, 2, 3]
        result = truncate_to_budget(ids, max_tokens=10)
        assert result == [1, 2, 3]

    def test_no_op_when_exactly_at_budget(self):
        ids = [1, 2, 3, 4]
        result = truncate_to_budget(ids, max_tokens=4)
        assert result == [1, 2, 3, 4]

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError):
            truncate_to_budget([1, 2, 3], max_tokens=2, side="middle")

    def test_empty_list_returns_empty(self):
        assert truncate_to_budget([], max_tokens=5) == []

    def test_max_tokens_zero_left_returns_empty(self):
        result = truncate_to_budget([1, 2, 3], max_tokens=0, side="left")
        assert result == []

    def test_max_tokens_zero_right_returns_empty(self):
        result = truncate_to_budget([1, 2, 3], max_tokens=0, side="right")
        assert result == []
