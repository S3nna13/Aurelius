"""Tests for the capability contract module."""

from __future__ import annotations

import dataclasses

import pytest

from src.runtime.capability_contract import (
    AgentCapability,
    CapabilityContract,
    InferenceCapability,
    ModelCapability,
    RuntimeCapability,
    SafetyCapability,
)


class TestModelCapability:
    def test_defaults(self):
        cap = ModelCapability()
        assert cap.family == "aurelius"
        assert cap.variant == "decoder-only"
        assert cap.params_b > 0
        assert cap.context_length > 0
        assert cap.gqa_enabled is True

    def test_frozen(self):
        cap = ModelCapability()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cap.family = "other"


class TestInferenceCapability:
    def test_defaults(self):
        cap = InferenceCapability()
        assert "pytorch" in cap.backends
        assert cap.streaming is True
        assert cap.tool_use is True

    def test_frozen(self):
        cap = InferenceCapability()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cap.streaming = False


class TestAgentCapability:
    def test_defaults(self):
        cap = AgentCapability()
        assert cap.max_agents > 0
        assert cap.subagents_enabled is True
        assert cap.swarm_enabled is True
        assert cap.memory_layers > 0


class TestRuntimeCapability:
    def test_defaults(self):
        cap = RuntimeCapability()
        assert cap.python_version
        assert cap.os_name
        assert len(cap.surfaces) > 0


class TestSafetyCapability:
    def test_defaults(self):
        cap = SafetyCapability()
        assert cap.prompt_injection_scanner is True
        assert cap.secret_redaction is True
        assert cap.clawdrain_detection is True
        assert cap.admission_controller is True
        assert cap.prism_lifecycle_hooks > 0

    def test_frozen(self):
        cap = SafetyCapability()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cap.secret_redaction = False


class TestCapabilityContract:
    def test_defaults(self):
        contract = CapabilityContract()
        assert contract.version == "1.0.0"
        assert isinstance(contract.model, ModelCapability)
        assert isinstance(contract.inference, InferenceCapability)
        assert isinstance(contract.agent, AgentCapability)
        assert isinstance(contract.safety, SafetyCapability)
        assert isinstance(contract.runtime, RuntimeCapability)

    def test_to_dict(self):
        contract = CapabilityContract()
        d = contract.to_dict()
        assert isinstance(d, dict)
        assert "version" in d
        assert "model" in d
        assert d["model"]["family"] == "aurelius"
        assert d["inference"]["streaming"] is True
        assert d["safety"]["admission_controller"] is True

    def test_to_dict_is_json_serialisable(self):
        import json

        contract = CapabilityContract()
        s = json.dumps(contract.to_dict())
        assert '"aurelius"' in s

    def test_from_runtime(self):
        contract = CapabilityContract.from_runtime()
        assert isinstance(contract, CapabilityContract)
        assert contract.runtime.python_version
        assert contract.safety.admission_controller is True

    def test_frozen(self):
        contract = CapabilityContract()
        with pytest.raises(dataclasses.FrozenInstanceError):
            contract.version = "2.0.0"
