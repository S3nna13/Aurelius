"""Tests for src.runtime.surface_flags — feature-flag registry extraction."""

from __future__ import annotations

from src.runtime.feature_flags import FeatureFlag, FeatureFlagRegistry
from src.runtime.surface_flags import (
    SafetyFlags,
    EvalFlags,
    ServingFlags,
    AgentFlags,
    ChatFlags,
    InferenceFlags,
    TrainingFlags,
    DataFlags,
    RetrievalFlags,
    LongcontextFlags,
    AlignmentFlags,
    ModelFlags,
    SecurityFlags,
    register_all_surface_flags,
    FEATURE_FLAG_REGISTRY,
)


def test_surface_flag_groups_register_all_flags():
    registry = FeatureFlagRegistry()
    register_all_surface_flags(registry)
    assert len(registry._flags) >= 40, f"expected >=40 flags, got {len(registry._flags)}"


def test_all_flags_default_off():
    registry = FeatureFlagRegistry()
    register_all_surface_flags(registry)
    for name, flag in registry._flags.items():
        assert flag.enabled is False, f"flag {name} should default OFF"


def test_flag_names_are_namespaced():
    registry = FeatureFlagRegistry()
    register_all_surface_flags(registry)
    for name in registry._flags:
        assert "." in name, f"flag {name} should be namespaced as surface.flag"


def test_safety_flags_structure():
    flags = SafetyFlags()
    assert flags.hallucination_guard.name == "safety.hallucination_guard"
    assert flags.hallucination_guard.enabled is False


def test_eval_flags_structure():
    flags = EvalFlags()
    assert flags.taubench.name == "eval.taubench"
    assert flags.crescendo_probe.name == "eval.crescendo_probe"


def test_serving_flags_structure():
    flags = ServingFlags()
    assert flags.structured_output.name == "serving.structured_output"
    assert "type" in flags.structured_output.metadata


def test_agent_flags_structure():
    flags = AgentFlags()
    assert flags.budget_bounded_loop.name == "agent.budget_bounded_loop"
    assert "max_tool_invocations" in flags.budget_bounded_loop.metadata


def test_inference_flags_structure():
    flags = InferenceFlags()
    assert flags.sink_logit_bias.name == "inference.sink_logit_bias"
    assert "bonus" in flags.sink_logit_bias.metadata


def test_training_flags_structure():
    flags = TrainingFlags()
    assert flags.tool_call_supervision.name == "training.tool_call_supervision"
    assert flags.prm_training.name == "training.prm_training"


def test_env_override_enables_flag():
    import os
    os.environ["AURELIUS_FF_SAFETY_HALLUCINATION_GUARD"] = "1"
    try:
        registry = FeatureFlagRegistry()
        register_all_surface_flags(registry)
        assert registry.is_enabled("safety.hallucination_guard") is True
    finally:
        del os.environ["AURELIUS_FF_SAFETY_HALLUCINATION_GUARD"]


def test_config_delegates_to_registry():
    from src.model.config import AureliusConfig
    cfg = AureliusConfig()
    assert cfg.safety_hallucination_guard_enabled is False
    assert cfg.agent_budget_bounded_loop_enabled is False
    assert cfg.eval_crescendo_probe_enabled is False


def test_config_backward_compatible_shape():
    from src.model.config import AureliusConfig
    cfg = AureliusConfig()
    assert cfg.d_model == 2048
    assert cfg.n_heads == 16
    assert cfg.n_kv_heads == 8
    assert cfg.head_dim == 128
    assert cfg.vocab_size == 128_000
    assert cfg.max_seq_len == 8192


def test_config_post_init_still_validates():
    from src.model.config import AureliusConfig
    import pytest
    with pytest.raises(AssertionError):
        AureliusConfig(d_model=999, n_heads=16, head_dim=128)