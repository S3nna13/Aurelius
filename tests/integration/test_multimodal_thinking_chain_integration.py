"""Integration test for MultimodalThinkingChain — end-to-end chain building,
budget accounting, limit enforcement, and registry wiring.
"""

from __future__ import annotations

import pytest

from src.inference import DECODER_REGISTRY
from src.inference.multimodal_thinking_chain import (
    MultimodalThinkingChain,
    MultimodalThinkingConfig,
    StepType,
    VisionStepLimitError,
)

# ---------------------------------------------------------------------------
# Integration test: full chain exercise
# ---------------------------------------------------------------------------


def test_multimodal_thinking_chain_integration():
    """Build a constrained chain, verify accounting, limits, and registry."""

    # -----------------------------------------------------------------------
    # 1. Build chain with tight budget
    # -----------------------------------------------------------------------
    config = MultimodalThinkingConfig(
        max_steps=5,
        max_thinking_tokens=100,
        vision_step_limit=2,
        max_tokens_per_step=4096,
        allow_interleave=True,
    )
    chain = MultimodalThinkingChain(config)

    # Add THINK step with 50 tokens
    think_tokens = list(range(1, 51))  # 50 tokens
    chain.add_step(StepType.THINK, think_tokens)

    # Add VISION step with 10 tokens
    vision_tokens = list(range(100, 110))  # 10 tokens
    chain.add_step(StepType.VISION, vision_tokens)

    # Add TOOL_CALL step with 5 tokens
    tool_tokens = list(range(200, 205))  # 5 tokens
    chain.add_step(StepType.TOOL_CALL, tool_tokens)

    # Add TOOL_RESULT step with 3 tokens
    result_tokens = list(range(300, 303))  # 3 tokens
    chain.add_step(StepType.TOOL_RESULT, result_tokens)

    # -----------------------------------------------------------------------
    # 2. Verify step counts and token accounting
    # -----------------------------------------------------------------------
    assert chain.step_count() == 4
    assert chain.thinking_tokens_used() == 50
    assert chain.vision_steps_used() == 1

    # -----------------------------------------------------------------------
    # 3. Verify flat tokens length = 50 + 10 + 5 + 3 = 68
    # -----------------------------------------------------------------------
    flat = chain.get_flat_tokens()
    assert len(flat) == 68
    assert flat == think_tokens + vision_tokens + tool_tokens + result_tokens

    # -----------------------------------------------------------------------
    # 4. Verify budget_remaining
    # -----------------------------------------------------------------------
    remaining = chain.budget_remaining()
    assert remaining["steps"] == 1  # 5 - 4 = 1
    assert remaining["thinking_tokens"] == 50  # 100 - 50 = 50
    assert remaining["vision_steps"] == 1  # 2 - 1 = 1

    # -----------------------------------------------------------------------
    # 5. Add a second VISION step — still within limit
    # -----------------------------------------------------------------------
    chain.add_step(StepType.VISION, list(range(400, 405)))  # 5 tokens; uses last step slot
    assert chain.vision_steps_used() == 2
    assert chain.step_count() == 5

    # -----------------------------------------------------------------------
    # 6. Attempting a third VISION step must raise VisionStepLimitError
    #    (also max_steps=5 is now reached, but VisionStepLimitError is checked
    #    after StepLimitError — create a separate chain with room for steps)
    # -----------------------------------------------------------------------
    config2 = MultimodalThinkingConfig(
        max_steps=20,
        max_thinking_tokens=1000,
        vision_step_limit=2,
    )
    chain2 = MultimodalThinkingChain(config2)
    chain2.add_step(StepType.VISION, list(range(10)))
    chain2.add_step(StepType.VISION, list(range(10)))

    with pytest.raises(VisionStepLimitError):
        chain2.add_step(StepType.VISION, list(range(10)))

    # -----------------------------------------------------------------------
    # 7. Verify registry wiring
    # -----------------------------------------------------------------------
    assert "multimodal_thinking_chain" in DECODER_REGISTRY
    assert DECODER_REGISTRY["multimodal_thinking_chain"] is MultimodalThinkingChain

    # Instantiate from registry to prove it's callable
    chain_from_registry = DECODER_REGISTRY["multimodal_thinking_chain"]()
    assert isinstance(chain_from_registry, MultimodalThinkingChain)
    assert chain_from_registry.step_count() == 0
