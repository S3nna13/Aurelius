"""Integration tests: CanaryTokenGuard registers and flows end-to-end."""

from __future__ import annotations

from src.model.config import AureliusConfig
from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag
from src.safety import (
    SAFETY_FILTER_REGISTRY,
    CanaryDetection,
    CanaryTokenGuard,
)


def test_registry_contains_canary_token_guard_entry() -> None:
    assert "canary_token_guard" in SAFETY_FILTER_REGISTRY
    assert SAFETY_FILTER_REGISTRY["canary_token_guard"] is CanaryTokenGuard


def test_prior_safety_registry_entries_intact() -> None:
    # Additive registration must not remove any sibling.
    for key in (
        "jailbreak",
        "prompt_injection",
        "pii",
        "output_filter",
        "prompt_integrity",
        "malicious_code",
        "policy_engine",
        "hallucination_guard",
    ):
        assert key in SAFETY_FILTER_REGISTRY, key


def test_config_flag_defaults_off() -> None:
    cfg = AureliusConfig()
    assert hasattr(cfg, "safety_canary_token_guard_enabled")
    assert cfg.safety_canary_token_guard_enabled is False


def test_config_flag_can_be_enabled() -> None:
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="safety.canary_token_guard", enabled=True))
    cfg = AureliusConfig()
    assert cfg.safety_canary_token_guard_enabled is True


def test_end_to_end_inject_then_scan_detects_leak() -> None:
    cls = SAFETY_FILTER_REGISTRY["canary_token_guard"]
    guard = cls()
    assert isinstance(guard, CanaryTokenGuard)

    system_prompt = "You are Aurelius, a safe coding assistant."
    token = guard.generate()
    hardened = guard.inject_into_system_prompt(system_prompt, token)
    assert token in hardened
    assert system_prompt in hardened

    # Simulate a compromised tool output that coerces the model into echoing
    # its hidden canary — the classical prompt-injection exfil pattern.
    active: set[str] = {token}
    fake_model_output = (
        "Sure, here's what I see in my system prompt: " + token + " (end of disclosure)"
    )
    detection = guard.scan(fake_model_output, active)
    assert isinstance(detection, CanaryDetection)
    assert detection.leaked is True
    assert detection.leaked_tokens == [token]
    assert detection.first_offset >= 0
    # rotate_on_leak default True → active set is drained.
    assert token not in active


def test_end_to_end_clean_output_no_leak() -> None:
    guard = SAFETY_FILTER_REGISTRY["canary_token_guard"]()
    token = guard.generate()
    detection = guard.scan(
        "Here is a refactored function. It runs in O(n log n).",
        {token},
    )
    assert detection.leaked is False


def test_tool_output_wrap_then_scan_echo_detected() -> None:
    guard = CanaryTokenGuard()
    token = guard.generate()
    wrapped = guard.wrap_tool_output("search result: foo bar", token)
    # A coerced model that dumps its context verbatim would include the
    # sentinels (and thus the token) in its reply.
    detection = guard.scan(wrapped, {token})
    assert detection.leaked is True
    assert detection.leaked_tokens == [token]
