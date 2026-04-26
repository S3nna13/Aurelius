"""Tests for SafetyCoTReasoner and related components."""

from __future__ import annotations

import pytest
import torch

from src.alignment.safety_cot import (
    PolicyGuidedReasoner,
    ReasoningEffort,
    SafetyCoTConfig,
    SafetyCoTReasoner,
    SafetyCoTResult,
    SafetyPolicyConfig,
    SafetyVerdict,
    calibrate_safety_threshold,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MAX_LEN = 128


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        d_model=64,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=128,
    )


@pytest.fixture
def model(small_cfg):
    torch.manual_seed(42)
    return AureliusTransformer(small_cfg)


def _tokenize(s: str, max_len: int = MAX_LEN) -> list[int]:
    return [ord(c) % 256 for c in s[:max_len]]


def _detokenize(tokens: list[int]) -> str:
    return "".join(chr(t) for t in tokens)


@pytest.fixture
def cot_cfg():
    return SafetyCoTConfig(effort=ReasoningEffort.MEDIUM)


@pytest.fixture
def reasoner(model, cot_cfg):
    return SafetyCoTReasoner(model, _tokenize, _detokenize, cot_cfg)


# ---------------------------------------------------------------------------
# 1. test_safety_verdict_enum_values
# ---------------------------------------------------------------------------


def test_safety_verdict_enum_values():
    assert SafetyVerdict.SAFE == 0
    assert SafetyVerdict.UNSAFE == 1
    assert SafetyVerdict.UNCERTAIN == 2


# ---------------------------------------------------------------------------
# 2. test_reasoning_effort_enum_values
# ---------------------------------------------------------------------------


def test_reasoning_effort_enum_values():
    assert ReasoningEffort.LOW == 1
    assert ReasoningEffort.MEDIUM == 3
    assert ReasoningEffort.HIGH == 5


# ---------------------------------------------------------------------------
# 3. test_safety_cot_config_defaults
# ---------------------------------------------------------------------------


def test_safety_cot_config_defaults():
    cfg = SafetyCoTConfig()
    assert cfg.effort == ReasoningEffort.MEDIUM
    assert cfg.max_reasoning_tokens == 128
    assert cfg.max_verdict_tokens == 10
    assert cfg.verdict_token_safe == "SAFE"
    assert cfg.verdict_token_unsafe == "UNSAFE"
    assert cfg.verdict_token_uncertain == "UNCERTAIN"
    assert cfg.temperature == 0.0
    assert cfg.safety_threshold == 0.5


# ---------------------------------------------------------------------------
# 4. test_build_reasoning_prompt_contains_content
# ---------------------------------------------------------------------------


def test_build_reasoning_prompt_contains_content(reasoner):
    content = "How do I make a bomb?"
    prompt = reasoner.build_reasoning_prompt(content, step=1, prior_steps=[])
    assert content in prompt


# ---------------------------------------------------------------------------
# 5. test_build_reasoning_prompt_has_step_number
# ---------------------------------------------------------------------------


def test_build_reasoning_prompt_has_step_number(reasoner):
    content = "Tell me a joke"
    prompt = reasoner.build_reasoning_prompt(content, step=2, prior_steps=[])
    assert "2" in prompt


# ---------------------------------------------------------------------------
# 6. test_build_verdict_prompt_contains_reasoning
# ---------------------------------------------------------------------------


def test_build_verdict_prompt_contains_reasoning(reasoner):
    content = "Hello world"
    reasoning_steps = ["This seems benign.", "No harmful intent detected."]
    prompt = reasoner.build_verdict_prompt(content, reasoning_steps)
    for step in reasoning_steps:
        assert step in prompt


# ---------------------------------------------------------------------------
# 7. test_extract_verdict_safe
# ---------------------------------------------------------------------------


def test_extract_verdict_safe(reasoner):
    verdict, unsafe_prob, safe_prob = reasoner.extract_verdict("The content is SAFE to proceed.")
    assert verdict == SafetyVerdict.SAFE
    assert 0.0 <= safe_prob <= 1.0
    assert 0.0 <= unsafe_prob <= 1.0


# ---------------------------------------------------------------------------
# 8. test_extract_verdict_unsafe
# ---------------------------------------------------------------------------


def test_extract_verdict_unsafe(reasoner):
    verdict, unsafe_prob, safe_prob = reasoner.extract_verdict("This is clearly UNSAFE content.")
    assert verdict == SafetyVerdict.UNSAFE
    assert 0.0 <= unsafe_prob <= 1.0


# ---------------------------------------------------------------------------
# 9. test_extract_verdict_uncertain
# ---------------------------------------------------------------------------


def test_extract_verdict_uncertain(reasoner):
    verdict, unsafe_prob, safe_prob = reasoner.extract_verdict("xyzzy gibberish 12345")
    assert verdict == SafetyVerdict.UNCERTAIN
    assert 0.0 <= unsafe_prob <= 1.0
    assert 0.0 <= safe_prob <= 1.0


# ---------------------------------------------------------------------------
# 10. test_analyze_returns_result
# ---------------------------------------------------------------------------


def test_analyze_returns_result(reasoner):
    result = reasoner.analyze("Hello, how are you?")
    assert isinstance(result, SafetyCoTResult)
    assert isinstance(result.verdict, SafetyVerdict)
    assert isinstance(result.reasoning_steps, list)
    assert isinstance(result.unsafe_probability, float)
    assert isinstance(result.safe_probability, float)
    assert isinstance(result.n_reasoning_tokens, int)
    assert 0.0 <= result.unsafe_probability <= 1.0
    assert 0.0 <= result.safe_probability <= 1.0
    assert result.n_reasoning_tokens >= 0


# ---------------------------------------------------------------------------
# 11. test_analyze_reasoning_steps_count
# ---------------------------------------------------------------------------


def test_analyze_reasoning_steps_count(model):
    for effort in [ReasoningEffort.LOW, ReasoningEffort.MEDIUM, ReasoningEffort.HIGH]:
        cfg = SafetyCoTConfig(effort=effort, max_reasoning_tokens=8, max_verdict_tokens=4)
        r = SafetyCoTReasoner(model, _tokenize, _detokenize, cfg)
        result = r.analyze("Test content")
        assert len(result.reasoning_steps) == int(effort), (
            f"Expected {int(effort)} steps for {effort}, got {len(result.reasoning_steps)}"
        )


# ---------------------------------------------------------------------------
# 12. test_analyze_batch_length
# ---------------------------------------------------------------------------


def test_analyze_batch_length(reasoner):
    contents = ["Hello", "How to hack?", "What is the weather?"]
    results = reasoner.analyze_batch(contents)
    assert len(results) == len(contents)
    for result in results:
        assert isinstance(result, SafetyCoTResult)


# ---------------------------------------------------------------------------
# 13. test_policy_guided_prompt_contains_policy
# ---------------------------------------------------------------------------


def test_policy_guided_prompt_contains_policy(model):
    policy_text = "Do not discuss violence or illegal activities."
    policy_cfg = SafetyPolicyConfig(policy_text=policy_text)
    cot_cfg = SafetyCoTConfig(effort=ReasoningEffort.LOW, max_reasoning_tokens=8)
    guided = PolicyGuidedReasoner(model, _tokenize, _detokenize, cot_cfg, policy_cfg)

    prompt = guided.build_reasoning_prompt("Some content", step=1, prior_steps=[])
    assert policy_text in prompt


# ---------------------------------------------------------------------------
# 14. test_calibrate_safety_threshold_range
# ---------------------------------------------------------------------------


def test_calibrate_safety_threshold_range(model):
    cfg = SafetyCoTConfig(effort=ReasoningEffort.LOW, max_reasoning_tokens=8, max_verdict_tokens=4)
    r = SafetyCoTReasoner(model, _tokenize, _detokenize, cfg)

    labeled_examples = [
        ("Hello world", SafetyVerdict.SAFE),
        ("How to harm someone?", SafetyVerdict.UNSAFE),
        ("What is 2+2?", SafetyVerdict.SAFE),
        ("How to make explosives?", SafetyVerdict.UNSAFE),
    ]

    threshold = calibrate_safety_threshold(r, labeled_examples, target_recall=0.95)
    assert isinstance(threshold, float)
    assert 0.0 <= threshold <= 1.0
