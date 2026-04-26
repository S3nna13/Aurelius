"""Surface-specific feature-flag groups.

Extracts the ~50 boolean flags from AureliusConfig into per-surface
namespaced groups.  Each surface owns its own flag collection; the
backbone AureliusConfig retains only model-architecture fields.

Per v8 mandate: "AureliusConfig is the backbone runtime config for one
variant, not a dumping ground. Prefer family manifests, variant configs,
adapters, or downstream surface configs over adding booleans to the
backbone config."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .feature_flags import FEATURE_FLAG_REGISTRY, FeatureFlag, FeatureFlagRegistry


def _ff(name: str, enabled: bool = False, **kwargs: object) -> FeatureFlag:
    return FeatureFlag(name=name, enabled=enabled, metadata=dict(kwargs))


@dataclass
class SafetyFlags:
    hallucination_guard: FeatureFlag = field(
        default_factory=lambda: _ff("safety.hallucination_guard")
    )
    canary_token_guard: FeatureFlag = field(
        default_factory=lambda: _ff("safety.canary_token_guard")
    )
    skill_scanner: FeatureFlag = field(default_factory=lambda: _ff("safety.skill_scanner"))
    reward_hack_detector: FeatureFlag = field(
        default_factory=lambda: _ff("safety.reward_hack_detector")
    )
    rule_engine: FeatureFlag = field(default_factory=lambda: _ff("safety.rule_engine"))
    lexical_entropy_anomaly: FeatureFlag = field(
        default_factory=lambda: _ff("safety.lexical_entropy_anomaly")
    )
    knn_known_bad_guard: FeatureFlag = field(
        default_factory=lambda: _ff("safety.knn_known_bad_guard")
    )


@dataclass
class EvalFlags:
    taubench: FeatureFlag = field(default_factory=lambda: _ff("eval.taubench", domain="coding"))
    crescendo_probe: FeatureFlag = field(default_factory=lambda: _ff("eval.crescendo_probe"))
    many_shot_jailbreak: FeatureFlag = field(
        default_factory=lambda: _ff("eval.many_shot_jailbreak")
    )
    tree_of_attacks: FeatureFlag = field(default_factory=lambda: _ff("eval.tree_of_attacks"))
    vibe_code_reviewer: FeatureFlag = field(default_factory=lambda: _ff("eval.vibe_code_reviewer"))
    behavioral_audit: FeatureFlag = field(default_factory=lambda: _ff("eval.behavioral_audit"))
    shade_arena: FeatureFlag = field(default_factory=lambda: _ff("eval.shade_arena"))
    agent_red_team: FeatureFlag = field(default_factory=lambda: _ff("eval.agent_red_team"))
    synthetic_jailbreak: FeatureFlag = field(
        default_factory=lambda: _ff("eval.synthetic_jailbreak")
    )


@dataclass
class ServingFlags:
    structured_output: FeatureFlag = field(
        default_factory=lambda: _ff("serving.structured_output", type="json_schema")
    )
    circuit_breaker: FeatureFlag = field(default_factory=lambda: _ff("serving.circuit_breaker"))
    sse_stream_encoder: FeatureFlag = field(
        default_factory=lambda: _ff("serving.sse_stream_encoder")
    )
    sse_chat_stream: FeatureFlag = field(default_factory=lambda: _ff("serving.sse_chat_stream"))
    function_calling_api: FeatureFlag = field(
        default_factory=lambda: _ff("serving.function_calling_api")
    )


@dataclass
class AgentFlags:
    dispatch_task: FeatureFlag = field(default_factory=lambda: _ff("agent.dispatch_task"))
    budget_bounded_loop: FeatureFlag = field(
        default_factory=lambda: _ff("agent.budget_bounded_loop", max_tool_invocations=8)
    )
    tool_sandbox_denylist: FeatureFlag = field(
        default_factory=lambda: _ff("agent.tool_sandbox_denylist")
    )
    five_state_controller: FeatureFlag = field(
        default_factory=lambda: _ff("agent.five_state_controller")
    )
    web_browse_tool: FeatureFlag = field(default_factory=lambda: _ff("agent.web_browse_tool"))


@dataclass
class ChatFlags:
    threat_intel_persona: FeatureFlag = field(
        default_factory=lambda: _ff("chat.threat_intel_persona")
    )
    security_personas: FeatureFlag = field(default_factory=lambda: _ff("chat.security_personas"))
    token_budget_allocator: FeatureFlag = field(
        default_factory=lambda: _ff("chat.token_budget_allocator")
    )
    system_prompt_priority: FeatureFlag = field(
        default_factory=lambda: _ff("chat.system_prompt_priority")
    )


@dataclass
class InferenceFlags:
    sink_logit_bias: FeatureFlag = field(
        default_factory=lambda: _ff("inference.sink_logit_bias", bonus=1.0, last_n=4)
    )
    beam_verifier_selector: FeatureFlag = field(
        default_factory=lambda: _ff("inference.beam_verifier_selector")
    )


@dataclass
class TrainingFlags:
    tool_call_supervision: FeatureFlag = field(
        default_factory=lambda: _ff("training.tool_call_supervision")
    )
    prm_training: FeatureFlag = field(
        default_factory=lambda: _ff("training.prm_training", step_token_id=50256, aggregation="min")
    )


@dataclass
class DataFlags:
    cwe_synthesis: FeatureFlag = field(default_factory=lambda: _ff("data.cwe_synthesis"))
    ngram_deduplication: FeatureFlag = field(
        default_factory=lambda: _ff("data.ngram_deduplication")
    )


@dataclass
class RetrievalFlags:
    prf_query_expander: FeatureFlag = field(
        default_factory=lambda: _ff("retrieval.prf_query_expander")
    )
    citation_tracker: FeatureFlag = field(default_factory=lambda: _ff("retrieval.citation_tracker"))
    code_aware_embedder: FeatureFlag = field(
        default_factory=lambda: _ff("retrieval.code_aware_embedder")
    )


@dataclass
class LongcontextFlags:
    sliding_window_causal_mask: FeatureFlag = field(
        default_factory=lambda: _ff("longcontext.sliding_window_causal_mask", window_size=512)
    )
    compaction_trigger: FeatureFlag = field(
        default_factory=lambda: _ff("longcontext.compaction_trigger")
    )


@dataclass
class AlignmentFlags:
    adversarial_code_battle: FeatureFlag = field(
        default_factory=lambda: _ff("alignment.adversarial_code_battle")
    )
    constitution_dimensions: FeatureFlag = field(
        default_factory=lambda: _ff("alignment.constitution_dimensions")
    )


@dataclass
class ModelFlags:
    merging: FeatureFlag = field(default_factory=lambda: _ff("model.merging"))
    toolformer_data_gen: FeatureFlag = field(
        default_factory=lambda: _ff("model.toolformer_data_gen", utility_threshold=0.1)
    )


@dataclass
class SecurityFlags:
    soc_pipeline: FeatureFlag = field(default_factory=lambda: _ff("security.soc_pipeline"))


@dataclass
class ContextFlags:
    context_extension_strategy: str = "none"
    context_target_len: int = 8192


SURFACE_FLAG_GROUPS = {
    "safety": SafetyFlags,
    "eval": EvalFlags,
    "serving": ServingFlags,
    "agent": AgentFlags,
    "chat": ChatFlags,
    "inference": InferenceFlags,
    "training": TrainingFlags,
    "data": DataFlags,
    "retrieval": RetrievalFlags,
    "longcontext": LongcontextFlags,
    "alignment": AlignmentFlags,
    "model": ModelFlags,
    "security": SecurityFlags,
}


def register_all_surface_flags(registry: FeatureFlagRegistry | None = None) -> None:
    """Register all surface-specific feature flags into the given registry.

    This extracts the ~50 boolean flags that were in AureliusConfig into
    their per-surface groups, each namespaced as ``surface.flag_name``.
    """
    target = registry or FEATURE_FLAG_REGISTRY
    groups = {
        "safety": SafetyFlags(),
        "eval": EvalFlags(),
        "serving": ServingFlags(),
        "agent": AgentFlags(),
        "chat": ChatFlags(),
        "inference": InferenceFlags(),
        "training": TrainingFlags(),
        "data": DataFlags(),
        "retrieval": RetrievalFlags(),
        "longcontext": LongcontextFlags(),
        "alignment": AlignmentFlags(),
        "model": ModelFlags(),
        "security": SecurityFlags(),
    }
    for _surface_name, group in groups.items():
        for field_name, value in vars(group).items():
            if isinstance(value, FeatureFlag):
                target.register(value)


register_all_surface_flags()

__all__ = [
    "SafetyFlags",
    "EvalFlags",
    "ServingFlags",
    "AgentFlags",
    "ChatFlags",
    "InferenceFlags",
    "TrainingFlags",
    "DataFlags",
    "RetrievalFlags",
    "LongcontextFlags",
    "AlignmentFlags",
    "ModelFlags",
    "SecurityFlags",
    "ContextFlags",
    "SURFACE_FLAG_GROUPS",
    "register_all_surface_flags",
]
