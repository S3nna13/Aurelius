"""Unit tests for src/inference/multimodal_thinking_chain.py — 15 tests."""

from __future__ import annotations

import pytest

from src.inference.multimodal_thinking_chain import (
    ChainStep,
    MultimodalThinkingChain,
    MultimodalThinkingConfig,
    StepLimitError,
    StepType,
    ThinkingBudgetError,
    VisionStepLimitError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tokens(n: int, start: int = 1) -> list[int]:
    """Return a list of n sequential token IDs starting at `start`."""
    return list(range(start, start + n))


def default_chain(**kwargs) -> MultimodalThinkingChain:
    """Create a chain with default config, optionally overriding fields."""
    cfg = MultimodalThinkingConfig(**kwargs)
    return MultimodalThinkingChain(cfg)


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = MultimodalThinkingConfig()
    assert cfg.max_steps == 50
    assert cfg.max_thinking_tokens == 98304
    assert cfg.max_tokens_per_step == 4096
    assert cfg.vision_step_limit == 50
    assert cfg.allow_interleave is True


# ---------------------------------------------------------------------------
# 2. test_add_think_step
# ---------------------------------------------------------------------------


def test_add_think_step():
    chain = MultimodalThinkingChain()
    tokens = make_tokens(10)
    step = chain.add_step(StepType.THINK, tokens)

    assert isinstance(step, ChainStep)
    assert step.step_type is StepType.THINK
    assert step.token_count == 10
    assert chain.thinking_tokens_used() == 10
    assert chain.step_count() == 1


# ---------------------------------------------------------------------------
# 3. test_add_vision_step
# ---------------------------------------------------------------------------


def test_add_vision_step():
    chain = MultimodalThinkingChain()
    tokens = make_tokens(20)
    step = chain.add_step(StepType.VISION, tokens)

    assert step.step_type is StepType.VISION
    assert chain.vision_steps_used() == 1
    assert chain.thinking_tokens_used() == 0


# ---------------------------------------------------------------------------
# 4. test_add_tool_call_step
# ---------------------------------------------------------------------------


def test_add_tool_call_step():
    chain = MultimodalThinkingChain()
    tokens = make_tokens(5)
    step = chain.add_step(StepType.TOOL_CALL, tokens)

    assert step.step_type is StepType.TOOL_CALL
    assert step.token_count == 5
    assert chain.step_count() == 1
    assert chain.vision_steps_used() == 0
    assert chain.thinking_tokens_used() == 0


# ---------------------------------------------------------------------------
# 5. test_step_truncated
# ---------------------------------------------------------------------------


def test_step_truncated():
    chain = default_chain(max_tokens_per_step=8)
    long_tokens = make_tokens(20)
    step = chain.add_step(StepType.TEXT, long_tokens)

    assert step.token_count == 8
    assert len(step.tokens) == 8
    # Should be the first 8 tokens
    assert step.tokens == long_tokens[:8]


# ---------------------------------------------------------------------------
# 6. test_max_steps_raises
# ---------------------------------------------------------------------------


def test_max_steps_raises():
    chain = default_chain(max_steps=3)
    for _ in range(3):
        chain.add_step(StepType.TEXT, make_tokens(1))

    with pytest.raises(StepLimitError):
        chain.add_step(StepType.TEXT, make_tokens(1))


# ---------------------------------------------------------------------------
# 7. test_thinking_budget_exceeded_raises
# ---------------------------------------------------------------------------


def test_thinking_budget_exceeded_raises():
    chain = default_chain(max_thinking_tokens=20, max_steps=10)
    chain.add_step(StepType.THINK, make_tokens(15))

    with pytest.raises(ThinkingBudgetError):
        # 15 + 10 = 25 > 20
        chain.add_step(StepType.THINK, make_tokens(10))


# ---------------------------------------------------------------------------
# 8. test_vision_limit_raises
# ---------------------------------------------------------------------------


def test_vision_limit_raises():
    chain = default_chain(vision_step_limit=2, max_steps=10)
    chain.add_step(StepType.VISION, make_tokens(5))
    chain.add_step(StepType.VISION, make_tokens(5))

    with pytest.raises(VisionStepLimitError):
        chain.add_step(StepType.VISION, make_tokens(5))


# ---------------------------------------------------------------------------
# 9. test_get_flat_tokens
# ---------------------------------------------------------------------------


def test_get_flat_tokens():
    chain = MultimodalThinkingChain()
    tokens_a = [1, 2, 3]
    tokens_b = [4, 5]
    tokens_c = [6, 7, 8, 9]

    chain.add_step(StepType.THINK, tokens_a)
    chain.add_step(StepType.VISION, tokens_b)
    chain.add_step(StepType.TEXT, tokens_c)

    flat = chain.get_flat_tokens()
    assert flat == tokens_a + tokens_b + tokens_c


# ---------------------------------------------------------------------------
# 10. test_budget_remaining_correct
# ---------------------------------------------------------------------------


def test_budget_remaining_correct():
    chain = default_chain(max_steps=10, max_thinking_tokens=100, vision_step_limit=5)
    chain.add_step(StepType.THINK, make_tokens(30))
    chain.add_step(StepType.VISION, make_tokens(10))
    chain.add_step(StepType.TOOL_CALL, make_tokens(5))

    remaining = chain.budget_remaining()
    assert remaining["steps"] == 7  # 10 - 3
    assert remaining["thinking_tokens"] == 70  # 100 - 30
    assert remaining["vision_steps"] == 4  # 5 - 1


# ---------------------------------------------------------------------------
# 11. test_validate_interleave_true — think before tool, interleave=False → valid
# ---------------------------------------------------------------------------


def test_validate_interleave_true():
    chain = default_chain(allow_interleave=False)
    chain.add_step(StepType.THINK, make_tokens(5))
    chain.add_step(StepType.TOOL_CALL, make_tokens(3))

    assert chain.validate_interleave() is True


# ---------------------------------------------------------------------------
# 12. test_validate_interleave_false — tool before think, interleave=False → invalid
# ---------------------------------------------------------------------------


def test_validate_interleave_false():
    chain = default_chain(allow_interleave=False, max_steps=10)
    chain.add_step(StepType.TOOL_CALL, make_tokens(3))
    chain.add_step(StepType.THINK, make_tokens(5))

    assert chain.validate_interleave() is False


# ---------------------------------------------------------------------------
# 13. test_validate_interleave_always_true — allow_interleave=True → always True
# ---------------------------------------------------------------------------


def test_validate_interleave_always_true():
    chain = default_chain(allow_interleave=True, max_steps=10)
    chain.add_step(StepType.TOOL_CALL, make_tokens(3))
    chain.add_step(StepType.THINK, make_tokens(5))
    chain.add_step(StepType.TOOL_CALL, make_tokens(2))

    assert chain.validate_interleave() is True


# ---------------------------------------------------------------------------
# 14. test_to_summary_keys
# ---------------------------------------------------------------------------


def test_to_summary_keys():
    chain = MultimodalThinkingChain()
    chain.add_step(StepType.THINK, make_tokens(10))
    chain.add_step(StepType.VISION, make_tokens(5))
    chain.add_step(StepType.TEXT, make_tokens(3))

    summary = chain.to_summary()
    assert "total_steps" in summary
    assert "by_type" in summary
    assert "total_tokens" in summary
    assert "thinking_tokens" in summary

    assert summary["total_steps"] == 3
    assert summary["total_tokens"] == 18
    assert summary["thinking_tokens"] == 10
    assert summary["by_type"].get("think") == 1
    assert summary["by_type"].get("vision") == 1
    assert summary["by_type"].get("text") == 1


# ---------------------------------------------------------------------------
# 15. test_mixed_chain — add THINK + VISION + TOOL_CALL + TEXT, flat tokens correct
# ---------------------------------------------------------------------------


def test_mixed_chain():
    chain = default_chain(max_steps=10, max_thinking_tokens=500, vision_step_limit=5)

    think_tokens = make_tokens(15, start=1)
    vision_tokens = make_tokens(8, start=100)
    tool_tokens = make_tokens(6, start=200)
    text_tokens = make_tokens(4, start=300)

    chain.add_step(StepType.THINK, think_tokens)
    chain.add_step(StepType.VISION, vision_tokens)
    chain.add_step(StepType.TOOL_CALL, tool_tokens)
    chain.add_step(StepType.TEXT, text_tokens)

    flat = chain.get_flat_tokens()
    expected = think_tokens + vision_tokens + tool_tokens + text_tokens
    assert flat == expected
    assert len(flat) == 15 + 8 + 6 + 4
    assert chain.step_count() == 4
    assert chain.thinking_tokens_used() == 15
    assert chain.vision_steps_used() == 1
