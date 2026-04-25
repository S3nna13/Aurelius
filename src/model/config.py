"""Aurelius 1.3B model configuration.

Backbone runtime config for one variant — model architecture fields only.
Feature flags have been migrated to per-surface registries in
src/runtime/surface_flags.py and are looked up via FeatureFlagRegistry.
The legacy boolean fields remain as backward-compatible aliases that delegate
to the FeatureFlagRegistry, and will be removed in a future version.

Per v8 mandate: "AureliusConfig is the backbone runtime config for one variant,
not a dumping ground. Prefer family manifests, variant configs, adapters, or
downstream surface configs over adding booleans to the backbone config."
"""

from dataclasses import dataclass

from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY


@dataclass
class AureliusConfig:
    """Hyperparameters for the Aurelius 1.3B transformer.

    Architecture: decoder-only transformer with GQA, SwiGLU FFN, RoPE, RMSNorm.
    Target parameter count: ~1.3B.

    Feature flags are now in src/runtime/surface_flags.py via FeatureFlagRegistry.
    The boolean fields below are backward-compatible aliases that delegate to
    the registry. New flags should be added to the surface-specific groups,
    NOT as new booleans here.
    """

    # Model dimensions
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    d_ff: int = 5632

    # Vocabulary and sequence
    vocab_size: int = 128_000
    max_seq_len: int = 8192

    # RoPE
    rope_theta: float = 500_000.0

    # RoPE scaling for context extension
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0
    rope_original_max_seq_len: int = 8192

    # Normalization
    rms_norm_eps: float = 1e-6

    # Regularization
    dropout: float = 0.0

    # Embedding
    tie_embeddings: bool = True

    # Training efficiency
    use_gradient_checkpointing: bool = False

    # Context extension (moved from boolean-per-strategy to a single field)
    context_extension_strategy: str = "none"
    context_target_len: int = 8192

    # --- Backward-compatible feature-flag aliases -------------------------
    # These delegate to FeatureFlagRegistry. Do NOT add new ones here;
    # add them to src/runtime/surface_flags.py instead.

    @property
    def enable_toolformer_data_gen(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("model.toolformer_data_gen")

    @property
    def toolformer_utility_threshold(self) -> float:
        flag = FEATURE_FLAG_REGISTRY._flags.get("model.toolformer_data_gen")
        return flag.metadata.get("utility_threshold", 0.1) if flag else 0.1

    @property
    def enable_taubench_eval(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("eval.taubench")

    @property
    def taubench_domain(self) -> str:
        flag = FEATURE_FLAG_REGISTRY._flags.get("eval.taubench")
        return flag.metadata.get("domain", "coding") if flag else "coding"

    @property
    def enable_structured_output(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("serving.structured_output")

    @property
    def structured_output_type(self) -> str:
        flag = FEATURE_FLAG_REGISTRY._flags.get("serving.structured_output")
        return flag.metadata.get("type", "json_schema") if flag else "json_schema"

    @property
    def enable_prm_training(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("training.prm_training")

    @property
    def prm_step_token_id(self) -> int:
        flag = FEATURE_FLAG_REGISTRY._flags.get("training.prm_training")
        return flag.metadata.get("step_token_id", 50256) if flag else 50256

    @property
    def prm_aggregation(self) -> str:
        flag = FEATURE_FLAG_REGISTRY._flags.get("training.prm_training")
        return flag.metadata.get("aggregation", "min") if flag else "min"

    @property
    def safety_hallucination_guard_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("safety.hallucination_guard")

    @property
    def hallucination_similarity_threshold(self) -> float:
        return 0.5

    @property
    def hallucination_confidence_decay(self) -> float:
        return 0.5

    @property
    def alignment_adversarial_code_battle_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("alignment.adversarial_code_battle")

    @property
    def alignment_constitution_dimensions_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("alignment.constitution_dimensions")

    @property
    def safety_canary_token_guard_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("safety.canary_token_guard")

    @property
    def safety_skill_scanner_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("safety.skill_scanner")

    @property
    def data_cwe_synthesis_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("data.cwe_synthesis")

    @property
    def safety_reward_hack_detector_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("safety.reward_hack_detector")

    @property
    def agent_dispatch_task_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("agent.dispatch_task")

    @property
    def serving_circuit_breaker_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("serving.circuit_breaker")

    @property
    def eval_crescendo_probe_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("eval.crescendo_probe")

    @property
    def eval_many_shot_jailbreak_probe_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("eval.many_shot_jailbreak")

    @property
    def eval_tree_of_attacks_probe_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("eval.tree_of_attacks")

    @property
    def eval_vibe_code_reviewer_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("eval.vibe_code_reviewer")

    @property
    def eval_behavioral_audit_taxonomy_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("eval.behavioral_audit")

    @property
    def eval_shade_arena_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("eval.shade_arena")

    @property
    def chat_threat_intel_persona_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("chat.threat_intel_persona")

    @property
    def chat_security_personas_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("chat.security_personas")

    @property
    def agent_budget_bounded_loop_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("agent.budget_bounded_loop")

    @property
    def agent_budget_max_tool_invocations(self) -> int:
        flag = FEATURE_FLAG_REGISTRY._flags.get("agent.budget_bounded_loop")
        return flag.metadata.get("max_tool_invocations", 8) if flag else 8

    @property
    def training_tool_call_supervision_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("training.tool_call_supervision")

    @property
    def eval_synthetic_jailbreak_generator_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("eval.synthetic_jailbreak")

    @property
    def serving_sse_stream_encoder_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("serving.sse_stream_encoder")

    @property
    def serving_sse_chat_stream_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("serving.sse_chat_stream")

    @property
    def inference_sink_logit_bias_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("inference.sink_logit_bias")

    @property
    def inference_sink_logit_bonus(self) -> float:
        flag = FEATURE_FLAG_REGISTRY._flags.get("inference.sink_logit_bias")
        return flag.metadata.get("bonus", 1.0) if flag else 1.0

    @property
    def inference_sink_last_n_positions(self) -> int:
        flag = FEATURE_FLAG_REGISTRY._flags.get("inference.sink_logit_bias")
        return flag.metadata.get("last_n", 4) if flag else 4

    @property
    def agent_tool_sandbox_denylist_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("agent.tool_sandbox_denylist")

    @property
    def safety_rule_engine_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("safety.rule_engine")

    @property
    def longcontext_sliding_window_causal_mask_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("longcontext.sliding_window_causal_mask")

    @property
    def longcontext_sliding_window_size(self) -> int:
        flag = FEATURE_FLAG_REGISTRY._flags.get("longcontext.sliding_window_causal_mask")
        return flag.metadata.get("window_size", 512) if flag else 512

    @property
    def security_soc_pipeline_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("security.soc_pipeline")

    @property
    def retrieval_prf_query_expander_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("retrieval.prf_query_expander")

    @property
    def retrieval_citation_tracker_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("retrieval.citation_tracker")

    @property
    def retrieval_code_aware_embedder_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("retrieval.code_aware_embedder")

    @property
    def chat_token_budget_allocator_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("chat.token_budget_allocator")

    @property
    def chat_system_prompt_priority_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("chat.system_prompt_priority")

    @property
    def inference_beam_verifier_selector_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("inference.beam_verifier_selector")

    @property
    def data_ngram_deduplication_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("data.ngram_deduplication")

    @property
    def safety_lexical_entropy_anomaly_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("safety.lexical_entropy_anomaly")

    @property
    def safety_knn_known_bad_guard_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("safety.knn_known_bad_guard")

    @property
    def agent_five_state_controller_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("agent.five_state_controller")

    @property
    def longcontext_compaction_trigger_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("longcontext.compaction_trigger")

    @property
    def eval_agent_red_team_bench_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("eval.agent_red_team")

    @property
    def agent_web_browse_tool_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("agent.web_browse_tool")

    @property
    def serving_function_calling_api_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("serving.function_calling_api")

    @property
    def model_merging_enabled(self) -> bool:
        return FEATURE_FLAG_REGISTRY.is_enabled("model.merging")

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) "
            f"* head_dim ({self.head_dim})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )