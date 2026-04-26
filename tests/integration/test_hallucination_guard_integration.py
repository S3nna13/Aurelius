"""Integration tests: HallucinationGuard wired into the safety registry and
constructable from AureliusConfig behind a feature flag that defaults OFF."""

from __future__ import annotations

from src.model.config import AureliusConfig
from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag
from src.safety import (
    HARM_CLASSIFIER_REGISTRY,
    SAFETY_FILTER_REGISTRY,
    HallucinationClaim,
    HallucinationEvidence,
    HallucinationGuard,
    HallucinationValidationResult,
)


def test_registry_contains_hallucination_guard_entry() -> None:
    assert "hallucination_guard" in SAFETY_FILTER_REGISTRY
    assert SAFETY_FILTER_REGISTRY["hallucination_guard"] is HallucinationGuard


def test_prior_registry_entries_intact() -> None:
    for key in (
        "jailbreak",
        "prompt_injection",
        "pii",
        "output_filter",
        "prompt_integrity",
        "malicious_code",
        "policy_engine",
    ):
        assert key in SAFETY_FILTER_REGISTRY, key
    for key in ("harm_taxonomy", "refusal", "constitutional"):
        assert key in HARM_CLASSIFIER_REGISTRY, key


def test_default_config_has_flag_off() -> None:
    cfg = AureliusConfig()
    assert hasattr(cfg, "safety_hallucination_guard_enabled")
    assert cfg.safety_hallucination_guard_enabled is False
    # Knob defaults are sane
    assert 0.0 < cfg.hallucination_similarity_threshold <= 1.0
    assert 0.0 <= cfg.hallucination_confidence_decay <= 1.0


def test_default_config_architecture_unchanged() -> None:
    """Adding the hallucination flag must not disturb any existing defaults."""
    cfg = AureliusConfig()
    assert cfg.d_model == 2048
    assert cfg.n_layers == 24
    assert cfg.vocab_size == 128_000
    assert cfg.enable_prm_training is False  # sibling flag still OFF


def test_end_to_end_via_registry_and_config() -> None:
    FEATURE_FLAG_REGISTRY.register(FeatureFlag(name="safety.hallucination_guard", enabled=True))
    cfg = AureliusConfig()
    assert cfg.safety_hallucination_guard_enabled is True
    cls = SAFETY_FILTER_REGISTRY["hallucination_guard"]
    guard = cls(
        similarity_threshold=cfg.hallucination_similarity_threshold,
        confidence_decay=cfg.hallucination_confidence_decay,
    )

    claims = [
        HallucinationClaim(
            text="Aurelius has 1395000000 parameters",
            source_model="aurelius",
            confidence=0.95,
        ),
        HallucinationClaim(
            text="The moon is made of green cheese",
            source_model="aurelius",
            confidence=0.95,
        ),
    ]
    evidence = [
        HallucinationEvidence(
            text="Aurelius model has 1395000000 parameters total",
            source="model_card",
            source_type="retrieved_doc",
        ),
    ]
    result = guard.validate(claims, evidence)
    assert isinstance(result, HallucinationValidationResult)
    assert len(result.validated_claims) == 1
    assert len(result.unvalidated_claims) == 1
    assert result.is_valid is False


def test_flag_off_is_no_op_path() -> None:
    """With the flag OFF we never instantiate the guard; verify the default
    config path does not require the guard to function."""
    FEATURE_FLAG_REGISTRY._flags.pop("safety.hallucination_guard", None)
    cfg = AureliusConfig()
    # Simulate the gate a caller would write.
    if cfg.safety_hallucination_guard_enabled:  # pragma: no cover - flag OFF
        SAFETY_FILTER_REGISTRY["hallucination_guard"]()
    # If we got here, no guard was constructed: behavior preserved.
    assert cfg.safety_hallucination_guard_enabled is False
