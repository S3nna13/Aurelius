"""Capability contract — versioned, frozen declaration of what a runtime can do.

Captures model, inference, agent, and runtime capabilities in a single
serialisable object.  Useful for health checks, client feature-negotiation,
and CI smoke-tests.
"""

from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import asdict, dataclass, field, is_dataclass, replace
from typing import Any

# ---------------------------------------------------------------------------
# Leaf dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelCapability:
    """What the model layer can do."""

    family: str = "aurelius"
    variant: str = "decoder-only"
    params_b: float = 1.395
    context_length: int = 4096
    vocab_size: int = 50257
    gqa_enabled: bool = True
    rope_enabled: bool = True
    swiglu_enabled: bool = True
    moe_enabled: bool = False
    mla_enabled: bool = False
    quantization: str = "fp32"


@dataclass(frozen=True)
class InferenceCapability:
    """What the inference layer supports."""

    backends: tuple[str, ...] = ("pytorch", "mock")
    streaming: bool = True
    tool_use: bool = True
    function_calling: bool = True
    speculative_decoding: bool = True
    kv_cache_compression: bool = True
    paged_kv_cache: bool = True


@dataclass(frozen=True)
class AgentCapability:
    """What the agent layer supports."""

    max_agents: int = 64
    subagents_enabled: bool = True
    background_jobs: bool = True
    checkpoints: bool = True
    approval_gates: bool = True
    skill_registry: bool = True
    memory_layers: int = 5
    tool_sandbox: bool = True
    swarm_enabled: bool = True
    plan_observe_reflect: bool = True


@dataclass(frozen=True)
class SafetyCapability:
    """What safety and security controls are available."""

    prompt_injection_scanner: bool = True
    pii_detection: bool = True
    secret_redaction: bool = True
    output_sanitization: bool = True
    clawdrain_detection: bool = True
    quantclaw_gating: bool = True
    prism_lifecycle_hooks: int = 10
    admission_controller: bool = True
    fail_closed_defaults: bool = True


@dataclass(frozen=True)
class RuntimeCapability:
    """What the runtime environment provides."""

    python_version: str = field(
        default_factory=lambda: ".".join(str(part) for part in sys.version_info[:3])
    )
    os_name: str = field(default_factory=platform.system)
    arch: str = field(default_factory=platform.machine)
    surfaces: tuple[str, ...] = ("cli", "api", "frontend")
    feature_flags: tuple[str, ...] = ()
    has_rust: bool = False
    has_node: bool = False


@dataclass(frozen=True)
class CapabilityContract:
    """Top-level versioned container."""

    version: str = "1.0.0"
    model: ModelCapability = field(default_factory=ModelCapability)
    inference: InferenceCapability = field(default_factory=InferenceCapability)
    agent: AgentCapability = field(default_factory=AgentCapability)
    safety: SafetyCapability = field(default_factory=SafetyCapability)
    runtime: RuntimeCapability = field(default_factory=RuntimeCapability)

    # -- serialisation helpers ------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        def _unfold(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                return {k: _unfold(v) for k, v in asdict(obj).items()}
            return obj

        return _unfold(self)

    @classmethod
    def from_runtime(cls) -> CapabilityContract:
        """Build a contract by probing the live runtime."""
        model = ModelCapability()
        inference = InferenceCapability()

        # Probe agent features
        agent = AgentCapability()
        try:
            importlib.import_module("agent.agent_mode_registry")
            agent = replace(agent, skill_registry=True)
        except Exception:
            pass
        try:
            importlib.import_module("agent.agent_swarm")
            agent = replace(agent, swarm_enabled=True)
        except Exception:
            pass

        # Probe safety controls. Missing optional modules disable only their
        # corresponding capability bit; the contract remains serialisable.
        safety = SafetyCapability(
            prompt_injection_scanner=_module_available("src.safety.prompt_injection_scanner"),
            pii_detection=_module_available("src.safety.pii_detector"),
            secret_redaction=_module_available("src.safety.output_sanitizer"),
            output_sanitization=_module_available("src.safety.output_sanitizer"),
            clawdrain_detection=_module_available("src.safety.clawdrain_detector"),
            quantclaw_gating=_module_available("src.safety.quantclaw_gate"),
            admission_controller=_module_available("src.safety.admission_controller"),
        )

        # Probe runtime
        has_rust = os.path.isdir("crates") if os.path.isdir(".") else False
        has_node = os.path.isdir("node_modules") if os.path.isdir(".") else False

        # Feature flags — only include enabled ones (conservative)
        from .feature_flags import FEATURE_FLAG_REGISTRY, RuntimeFlag  # noqa: PLC0415

        enabled_flags: tuple[str, ...] = tuple(
            flag.name.lower() for flag in RuntimeFlag if FEATURE_FLAG_REGISTRY.is_enabled(flag)
        )
        runtime = RuntimeCapability(
            has_rust=has_rust, has_node=has_node, feature_flags=enabled_flags
        )

        return cls(
            model=model,
            inference=inference,
            agent=agent,
            safety=safety,
            runtime=runtime,
        )


def _module_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception:
        return False
    return True
