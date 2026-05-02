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

from dataclasses import dataclass, field
from typing import Any

from src.runtime.feature_flags import FEATURE_FLAG_REGISTRY


@dataclass
class MoEConfig:
    enabled: bool = False
    num_experts: int = 8
    top_k: int = 2
    every_n_layers: int = 2
    d_model: int = 2048
    d_ff: int = 8192
    capacity_factor: float = 1.25
    jitter_noise: float = 0.0
    load_balance_alpha: float = 0.01
    z_loss_coeff: float = 0.001
    router_type: str = "topk"


@dataclass
class MoDConfig:
    enabled: bool = False
    capacity_factor: float = 0.5
    router_type: str = "learned"
    aux_loss_coeff: float = 0.01
    z_loss_coeff: float = 0.001


@dataclass
class ReMoDEConfig:
    enabled: bool = False
    mod_capacity: float = 0.5
    moe_num_experts: int = 8
    moe_top_k: int = 2
    d_model: int = 2048
    d_ff: int = 8192
    load_balance_alpha: float = 0.01
    mod_aux_loss_coeff: float = 0.01
    shared_experts: int = 1
    every_n_layers: int = 1


@dataclass
class HLMConfig:
    enabled: bool = False
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 8192
    layer_pattern: list[str] = field(default_factory=lambda: [])
    moe: MoEConfig = field(default_factory=MoEConfig)
    mod: MoDConfig = field(default_factory=MoDConfig)
    remode: ReMoDEConfig = field(default_factory=ReMoDEConfig)


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

    # MoE (Mixture of Experts)
    moe_enabled: bool = False
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_every_n_layers: int = 2
    moe_capacity_factor: float = 1.25
    moe_jitter_noise: float = 0.0
    moe_strategy: str = "topk"
    moe_z_loss_coeff: float = 0.001
    moe_expert_dropout: float = 0.0

    # MoD (Mixture of Depth)
    mod_enabled: bool = False
    mod_capacity_factor: float = 0.5
    mod_router_type: str = "learned"
    mod_aux_loss_coeff: float = 0.01

    # ReMoDE (Residual MoD + MoE)
    remode_enabled: bool = False
    remode_mod_capacity: float = 0.5
    remode_num_experts: int = 8
    remode_top_k: int = 2
    remode_shared_experts: int = 1
    remode_every_n_layers: int = 1

    # HLM (Hybrid Language Model)
    hlm_enabled: bool = False
    hlm_layer_pattern: str = ""

    # Multi-Token Prediction (MTP)
    mtp_enabled: bool = False
    mtp_n_predict: int = 2
    mtp_share_params: bool = True
    mtp_loss_weight: float = 0.3

    # Context extension (moved from boolean-per-strategy to a single field)
    context_extension_strategy: str = "none"
    context_target_len: int = 8192

    # Hybrid Attention (CSA/HCA) — DeepSeek-V4 style
    hybrid_attention_enabled: bool = False
    attention_compression_rate_csa: int = 4
    attention_compression_rate_hca: int = 128
    attention_num_indexer_heads: int = 64
    attention_indexer_head_dim: int = 128
    attention_top_k: int = 512
    attention_num_query_heads_csa: int = 64
    attention_num_query_heads_hca: int = 64
    attention_query_compression_dim: int = 1024
    attention_output_projection_groups: int = 8
    attention_intermediate_output_dim: int = 1024
    attention_sliding_window_size: int = 128
    attention_partial_rope_dim: int = 64
    attention_num_sink_tokens: int = 4

    # mHC (Manifold-Constrained Hyper-Connections)
    mhc_enabled: bool = False
    mhc_expansion_factor: int = 4
    mhc_sinkhorn_iterations: int = 20

    # MLA (Multi-head Latent Attention)
    mla_enabled: bool = False
    mla_kv_lrank: int = 512
    mla_q_lrank: int = 1536
    mla_rope_dim: int = 64

    # FP8 training
    fp8_training_enabled: bool = False
    fp8_activation_quant_tile: int = 128
    fp8_weight_quant_block: int = 128
    fp8_accumulation_promotion_interval: int = 128

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
        if hasattr(self, "_prm_step_token_id"):
            return self._prm_step_token_id  # type: ignore[return-value]
        flag = FEATURE_FLAG_REGISTRY._flags.get("training.prm_training")
        return flag.metadata.get("step_token_id", 50256) if flag else 50256

    @prm_step_token_id.setter
    def prm_step_token_id(self, value: int) -> None:
        self._prm_step_token_id = value

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

    @classmethod
    def aurelius_1_3b(cls) -> "AureliusConfig":
        """Default 1.3B dense config."""
        return cls()

    @classmethod
    def aurelius_2_7b(cls) -> "AureliusConfig":
        """2.7B dense: 32 layers, d_model=2560, GQA 20:5."""
        return cls(
            d_model=2560,
            n_layers=32,
            n_heads=20,
            n_kv_heads=5,
            head_dim=128,
            d_ff=7168,
            vocab_size=128000,
            max_seq_len=8192,
            rope_theta=500_000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_embeddings=True,
        )

    @classmethod
    def aurelius_3b(cls) -> "AureliusConfig":
        """3B dense: 32 layers, d_model=2560, GQA 20:5, wider FFN."""
        return cls(
            d_model=2560,
            n_layers=34,
            n_heads=20,
            n_kv_heads=5,
            head_dim=128,
            d_ff=8192,
            vocab_size=128000,
            max_seq_len=16384,
            rope_theta=500_000.0,
            rms_norm_eps=1e-6,
            dropout=0.1,
            tie_embeddings=True,
        )

    @classmethod
    def aurelius_3b_moe(cls) -> "AureliusConfig":
        """3B MoE: 32 layers, 8 experts, top-2, every 2 layers."""
        return cls(
            d_model=2048,
            n_layers=32,
            n_heads=16,
            n_kv_heads=8,
            head_dim=128,
            d_ff=5632,
            vocab_size=128000,
            max_seq_len=16384,
            rope_theta=500_000.0,
            rms_norm_eps=1e-6,
            dropout=0.1,
            tie_embeddings=True,
            moe_enabled=True,
            moe_num_experts=8,
            moe_top_k=2,
            moe_every_n_layers=2,
        )

    @classmethod
    def aurelius_7b(cls) -> "AureliusConfig":
        """7B dense: 36 layers, d_model=3584, GQA 28:4."""
        return cls(
            d_model=3584,
            n_layers=36,
            n_heads=28,
            n_kv_heads=4,
            head_dim=128,
            d_ff=10240,
            vocab_size=128000,
            max_seq_len=32768,
            rope_theta=500_000.0,
            rms_norm_eps=1e-6,
            dropout=0.1,
            tie_embeddings=True,
        )

    @classmethod
    def aurelius_7b_moe(cls) -> "AureliusConfig":
        """7B MoE: 32 layers, 16 experts, top-4."""
        return cls(
            d_model=2560,
            n_layers=32,
            n_heads=20,
            n_kv_heads=5,
            head_dim=128,
            d_ff=7168,
            vocab_size=128000,
            max_seq_len=32768,
            rope_theta=500_000.0,
            rms_norm_eps=1e-6,
            dropout=0.1,
            tie_embeddings=True,
            moe_enabled=True,
            moe_num_experts=16,
            moe_top_k=4,
            moe_every_n_layers=1,
        )

    @classmethod
    def aurelius_3b_original(cls) -> "AureliusConfig":
        """3.0B dense: 28 layers, d_model=3072, GQA 24:6 (original)."""
        return cls(
            d_model=3072,
            n_layers=28,
            n_heads=24,
            n_kv_heads=6,
            head_dim=128,
            d_ff=8192,
            vocab_size=128000,
            max_seq_len=4096,
            rope_theta=500_000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_embeddings=True,
        )

    @classmethod
    def aurelius_moe_5b(cls) -> "AureliusConfig":
        """5-6B MoE: 24 layers, d_model=2048, 8 experts, top-2."""
        return cls(
            d_model=2048,
            n_layers=24,
            n_heads=16,
            n_kv_heads=8,
            head_dim=128,
            d_ff=5632,
            vocab_size=128000,
            max_seq_len=8192,
            rope_theta=500_000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_embeddings=True,
            moe_enabled=True,
            moe_num_experts=8,
            moe_top_k=2,
            moe_every_n_layers=2,
            moe_capacity_factor=1.25,
        )

    @classmethod
    def aurelius_flash_284b(cls) -> "AureliusConfig":
        """284B Flash config: 43 layers, d_model=4096, CSA/HCA hybrid, 256 MoE experts."""
        return cls(
            d_model=4096,
            n_layers=43,
            n_heads=32,
            n_kv_heads=8,
            head_dim=128,
            d_ff=11264,
            vocab_size=128000,
            max_seq_len=1_000_000,
            rope_theta=500_000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_embeddings=True,
            moe_enabled=True,
            moe_num_experts=256,
            moe_top_k=6,
            moe_every_n_layers=1,
            moe_capacity_factor=1.25,
            moe_jitter_noise=0.0,
            hybrid_attention_enabled=True,
            attention_compression_rate_csa=4,
            attention_compression_rate_hca=128,
            attention_top_k=512,
            attention_num_query_heads_csa=64,
            attention_num_query_heads_hca=64,
            attention_query_compression_dim=1024,
            attention_output_projection_groups=8,
            attention_intermediate_output_dim=1024,
            attention_sliding_window_size=128,
            attention_partial_rope_dim=64,
            mhc_enabled=True,
            mhc_expansion_factor=4,
            mhc_sinkhorn_iterations=20,
        )

    @classmethod
    def aurelius_pro_1_6t(cls) -> "AureliusConfig":
        """1.6T Pro config: 61 layers, d_model=7168, CSA/HCA hybrid, 384 MoE experts."""
        return cls(
            d_model=7168,
            n_layers=61,
            n_heads=56,
            n_kv_heads=8,
            head_dim=128,
            d_ff=18432,
            vocab_size=128000,
            max_seq_len=1_000_000,
            rope_theta=500_000.0,
            rms_norm_eps=1e-6,
            dropout=0.0,
            tie_embeddings=True,
            moe_enabled=True,
            moe_num_experts=384,
            moe_top_k=6,
            moe_every_n_layers=1,
            moe_capacity_factor=1.25,
            moe_jitter_noise=0.0,
            hybrid_attention_enabled=True,
            attention_compression_rate_csa=4,
            attention_compression_rate_hca=128,
            attention_top_k=1024,
            attention_num_query_heads_csa=128,
            attention_num_query_heads_hca=128,
            attention_query_compression_dim=1536,
            attention_output_projection_groups=16,
            attention_intermediate_output_dim=1024,
            attention_sliding_window_size=128,
            attention_partial_rope_dim=64,
            mhc_enabled=True,
            mhc_expansion_factor=4,
            mhc_sinkhorn_iterations=20,
        )

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim, (  # noqa: S101
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) "
            f"* head_dim ({self.head_dim})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (  # noqa: S101
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
MOE_PRESETS: dict[str, dict[str, Any]] = {
    "small": {"num_experts": 4, "top_k": 2, "d_model": 512, "d_ff": 2048},
    "base": {"num_experts": 8, "top_k": 2, "d_model": 2048, "d_ff": 8192},
    "large": {"num_experts": 16, "top_k": 4, "d_model": 4096, "d_ff": 16384},
    "flash": {"num_experts": 256, "top_k": 6, "d_model": 4096, "d_ff": 16384},
    "pro": {"num_experts": 384, "top_k": 6, "d_model": 7168, "d_ff": 28672},
}

MOD_PRESETS: dict[str, dict[str, Any]] = {
    "light": {"capacity_factor": 0.25},
    "medium": {"capacity_factor": 0.5},
    "heavy": {"capacity_factor": 0.75},
}

HLM_PRESETS: dict[str, dict[str, Any]] = {
    "hybrid_moe": {
        "layer_pattern": ["dense", "dense", "dense", "moe"] * 6,
        "moe": MOE_PRESETS["base"],
    },
    "hybrid_mod": {
        "layer_pattern": ["mod", "dense"] * 12,
        "mod": MOD_PRESETS["medium"],
    },
    "remode": {
        "layer_pattern": ["remode"] * 24,
        "remode": {"mod_capacity": 0.5, "moe_num_experts": 8, "moe_top_k": 2},
    },
    "deepseek_v4": {
        "layer_pattern": ["dense", "dense", "remode"] * 8,
        "remode": {"mod_capacity": 0.3, "moe_num_experts": 128, "moe_top_k": 6},
    },
}
