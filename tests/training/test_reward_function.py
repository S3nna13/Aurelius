"""Tests for src.training.reward_function."""

from __future__ import annotations

import pytest

from src.training.reward_function import (
    REWARD_FUNCTION_REGISTRY,
    BaseRewardFunction,
    DirectionalRewardFn,
    FormatRewardFn,
    LengthRewardFn,
    RewardResult,
    RewardSignal,
)

# ---------------------------------------------------------------------------
# Signal / dataclass basics
# ---------------------------------------------------------------------------


def test_reward_signal_values():
    assert RewardSignal.CORRECT.value == 1.0
    assert RewardSignal.PARTIALLY_CORRECT.value == 0.5
    assert RewardSignal.INCORRECT.value == 0.0
    assert RewardSignal.PENALIZED.value == -0.5


def test_reward_result_frozen():
    r = RewardResult(score=1.0, signal=RewardSignal.CORRECT)
    with pytest.raises(Exception):
        r.score = 0.0  # type: ignore[misc]


def test_reward_result_defaults():
    r = RewardResult(score=0.0, signal=RewardSignal.INCORRECT)
    assert r.breakdown == {}
    assert r.notes == ""


# ---------------------------------------------------------------------------
# DirectionalRewardFn
# ---------------------------------------------------------------------------


def test_directional_correct():
    fn = DirectionalRewardFn()
    r = fn.score("I am bullish on BTC", "bullish")
    assert r.score >= 1.0 - 1e-9
    assert r.signal == RewardSignal.CORRECT


def test_directional_wrong():
    fn = DirectionalRewardFn()
    r = fn.score("bearish outlook", "bullish")
    assert r.score == pytest.approx(-0.5)
    assert r.signal == RewardSignal.PENALIZED


def test_directional_synonym_long():
    fn = DirectionalRewardFn()
    r = fn.score("go long", "bullish")
    assert r.breakdown.get("direction") == 1.0


def test_directional_synonym_sell_is_bearish():
    fn = DirectionalRewardFn()
    r = fn.score("sell now", "bearish")
    assert r.breakdown.get("direction") == 1.0


def test_directional_price_target_bonus():
    fn = DirectionalRewardFn()
    r = fn.score("bullish, target $50,000", "bullish")
    assert r.breakdown.get("price_target") == 0.2


def test_directional_reasoning_bonus():
    fn = DirectionalRewardFn()
    text = "bullish " + " ".join(["word"] * 25)
    r = fn.score(text, "bullish")
    assert r.breakdown.get("reasoning") == 0.1


def test_directional_score_clamped_upper():
    fn = DirectionalRewardFn()
    text = "bullish target $100 " + " ".join(["w"] * 30)
    r = fn.score(text, "bullish")
    assert r.score <= 1.0


def test_directional_score_clamped_lower():
    fn = DirectionalRewardFn()
    r = fn.score("bearish", "bullish")
    assert r.score >= -0.5


def test_directional_no_direction_found():
    fn = DirectionalRewardFn()
    r = fn.score("the weather is nice", "bullish")
    assert r.score == 0.0
    assert r.signal == RewardSignal.INCORRECT


def test_directional_batch():
    fn = DirectionalRewardFn()
    results = fn.score_batch(["bullish", "bearish", "neutral"], "bullish")
    assert len(results) == 3
    assert results[0].signal == RewardSignal.CORRECT
    assert results[1].signal == RewardSignal.PENALIZED


def test_directional_case_insensitive():
    fn = DirectionalRewardFn()
    r = fn.score("BULLISH view", "bullish")
    assert r.breakdown.get("direction") == 1.0


# ---------------------------------------------------------------------------
# LengthRewardFn
# ---------------------------------------------------------------------------


def test_length_in_range():
    fn = LengthRewardFn(min_tokens=5, max_tokens=10)
    r = fn.score("one two three four five six")
    assert r.score == 1.0
    assert r.signal == RewardSignal.CORRECT


def test_length_below_range():
    fn = LengthRewardFn(min_tokens=50, max_tokens=500)
    r = fn.score("short")
    assert 0.0 <= r.score < 1.0


def test_length_above_range():
    fn = LengthRewardFn(min_tokens=5, max_tokens=10)
    r = fn.score(" ".join(["tok"] * 100))
    assert r.score >= 0.0
    assert r.score < 1.0


def test_length_invalid_bounds():
    with pytest.raises(ValueError):
        LengthRewardFn(min_tokens=10, max_tokens=5)


def test_length_breakdown_tokens():
    fn = LengthRewardFn(min_tokens=1, max_tokens=100)
    r = fn.score("a b c")
    assert r.breakdown["tokens"] == 3.0


def test_length_score_floor():
    fn = LengthRewardFn(min_tokens=5, max_tokens=10)
    r = fn.score(" ".join(["tok"] * 10000))
    assert r.score >= 0.0


def test_length_default_construction():
    fn = LengthRewardFn()
    assert fn.min_tokens == 50
    assert fn.max_tokens == 500


# ---------------------------------------------------------------------------
# FormatRewardFn
# ---------------------------------------------------------------------------


def test_format_required_all_match():
    fn = FormatRewardFn(required_patterns=[r"<think>", r"</think>"])
    r = fn.score("<think>reasoning</think>")
    assert r.score == 1.0
    assert r.signal == RewardSignal.CORRECT


def test_format_required_partial_match():
    fn = FormatRewardFn(required_patterns=[r"<think>", r"</think>"])
    r = fn.score("<think>only open")
    assert r.score == 0.0


def test_format_forbidden_present():
    fn = FormatRewardFn(forbidden_patterns=[r"\bprofanity\b"])
    r = fn.score("this contains profanity")
    assert r.score == pytest.approx(-0.5)
    assert r.signal == RewardSignal.PENALIZED


def test_format_forbidden_absent():
    fn = FormatRewardFn(forbidden_patterns=[r"\bprofanity\b"])
    r = fn.score("clean text")
    assert r.score == 0.0


def test_format_forbidden_overrides_required():
    fn = FormatRewardFn(required_patterns=[r"hello"], forbidden_patterns=[r"badword"])
    r = fn.score("hello and badword")
    assert r.score == pytest.approx(-0.5)


def test_format_empty_both():
    fn = FormatRewardFn()
    r = fn.score("any text")
    assert r.score == 0.0


def test_format_breakdown_counts():
    fn = FormatRewardFn(required_patterns=[r"a", r"b"], forbidden_patterns=[r"xx"])
    r = fn.score("a b")
    assert r.breakdown["required_hits"] == 2.0
    assert r.breakdown["forbidden_hits"] == 0.0


# ---------------------------------------------------------------------------
# Registry / polymorphism
# ---------------------------------------------------------------------------


def test_registry_keys():
    assert set(REWARD_FUNCTION_REGISTRY.keys()) == {"directional", "length", "format"}


def test_registry_classes():
    assert REWARD_FUNCTION_REGISTRY["directional"] is DirectionalRewardFn
    assert REWARD_FUNCTION_REGISTRY["length"] is LengthRewardFn
    assert REWARD_FUNCTION_REGISTRY["format"] is FormatRewardFn


@pytest.mark.parametrize("name", ["directional", "length", "format"])
def test_registry_instantiable(name):
    cls = REWARD_FUNCTION_REGISTRY[name]
    inst = cls()
    assert isinstance(inst, BaseRewardFunction)
