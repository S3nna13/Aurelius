"""Integration tests: reasoning_level_controller wired into DECODER_REGISTRY.

Verifies that the reasoning-level entry is discoverable via the inference
package's DECODER_REGISTRY and behaves correctly when called through it.

GPT-OSS-120B (arXiv:2508.10925).
SWE-bench Verified: low=47.9%, medium=52.6%, high=62.4%.
"""
from __future__ import annotations

import src.inference as inference
from src.inference import DECODER_REGISTRY, LEVEL_CONFIGS


# ---------------------------------------------------------------------------
# 1. "reasoning_level" key is registered
# ---------------------------------------------------------------------------

def test_reasoning_level_key_in_decoder_registry() -> None:
    assert "reasoning_level" in DECODER_REGISTRY


# ---------------------------------------------------------------------------
# 2. Call parse_reasoning_level via registry entry — correct config returned
# ---------------------------------------------------------------------------

def test_registry_entry_returns_correct_config_for_medium() -> None:
    parse_fn = DECODER_REGISTRY["reasoning_level"]
    result = parse_fn("Reasoning: medium")
    assert result == dict(LEVEL_CONFIGS["medium"])


# ---------------------------------------------------------------------------
# 3. All three levels accessible via the same registry entry
# ---------------------------------------------------------------------------

def test_all_three_levels_via_registry() -> None:
    parse_fn = DECODER_REGISTRY["reasoning_level"]

    low_result = parse_fn("Reasoning: low")
    assert low_result["reasoning_level"] == "low"
    assert low_result["temperature"] == 0.3
    assert low_result["max_tokens"] == 512

    medium_result = parse_fn("Reasoning: medium")
    assert medium_result["reasoning_level"] == "medium"
    assert medium_result["temperature"] == 0.6
    assert medium_result["max_tokens"] == 2048

    high_result = parse_fn("Reasoning: high")
    assert high_result["reasoning_level"] == "high"
    assert high_result["temperature"] == 1.0
    assert high_result["max_tokens"] == 8192


# ---------------------------------------------------------------------------
# 4. Regression guard: pre-existing inference package symbols intact
# ---------------------------------------------------------------------------

def test_existing_inference_package_symbols_intact() -> None:
    # SCHEDULER_REGISTRY was present before this cycle.
    assert hasattr(inference, "SCHEDULER_REGISTRY")
    assert "continuous_batching" in inference.SCHEDULER_REGISTRY

    # ContinuousBatchingScheduler must still be importable.
    assert hasattr(inference, "ContinuousBatchingScheduler")

    # MultiSampleVoter and voting result type also intact.
    assert hasattr(inference, "MultiSampleVoter")
    assert hasattr(inference, "VoteResult")
