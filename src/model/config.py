"""Aurelius 1.3B model configuration."""

from dataclasses import dataclass


@dataclass
class AureliusConfig:
    """Hyperparameters for the Aurelius 1.3B transformer.

    Architecture: decoder-only transformer with GQA, SwiGLU FFN, RoPE, RMSNorm.
    Target parameter count: ~1.3B.
    """

    # Model dimensions
    d_model: int = 2048
    n_layers: int = 24
    n_heads: int = 16       # query heads
    n_kv_heads: int = 8     # key/value heads (GQA ratio 2:1)
    head_dim: int = 128      # d_model // n_heads
    d_ff: int = 5632         # SwiGLU intermediate dim ≈ 2/3 × 4 × d_model

    # Vocabulary and sequence
    vocab_size: int = 128_000
    max_seq_len: int = 8192

    # RoPE
    rope_theta: float = 500_000.0

    # RoPE scaling for context extension
    rope_scaling_type: str = "none"   # "none" or "yarn"
    rope_scaling_factor: float = 1.0  # scale factor for YaRN (e.g., 4.0 for 4x extension)
    rope_original_max_seq_len: int = 8192  # original training context length

    # Normalization
    rms_norm_eps: float = 1e-6

    # Regularization
    dropout: float = 0.0

    # Embedding
    tie_embeddings: bool = True

    # Training efficiency
    use_gradient_checkpointing: bool = False

    # Toolformer data generation (default OFF)
    enable_toolformer_data_gen: bool = False
    toolformer_utility_threshold: float = 0.1

    # τ-bench evaluation (default OFF)
    enable_taubench_eval: bool = False
    taubench_domain: str = "coding"

    # Structured output / grammar-constrained decoding (serving, default OFF)
    enable_structured_output: bool = False
    structured_output_type: str = "json_schema"

    # Dynamic context window extension via NTK-aware / YaRN / LongRoPE (default OFF)
    context_extension_strategy: str = "none"   # "none", "auto", "linear", "ntk", "yarn", "longrope"
    context_target_len: int = 8192             # target inference context length

    # Process Reward Model (PRM) -- step-level reward scoring (default OFF)
    enable_prm_training: bool = False
    prm_step_token_id: int = 50256   # token ID marking step boundaries
    prm_aggregation: str = "min"     # "min" or "mean" for best-of-N selection

    # Hallucination guard: claim-vs-evidence validator (default OFF)
    safety_hallucination_guard_enabled: bool = False
    hallucination_similarity_threshold: float = 0.5
    hallucination_confidence_decay: float = 0.5

    # Adversarial code battle: red-vs-blue code patching loop (default OFF)
    alignment_adversarial_code_battle_enabled: bool = False

    # Runtime canary-token guard for prompt-injection defense (default OFF)
    safety_canary_token_guard_enabled: bool = False

    # Skill Markdown supply-chain scanner (default OFF)
    safety_skill_scanner_enabled: bool = False

    # CWE synthetic (vulnerable, secure) pair generator (default OFF)
    data_cwe_synthesis_enabled: bool = False

    # Reward-hacking trajectory detector (default OFF)
    safety_reward_hack_detector_enabled: bool = False

    # Parallel fan-out dispatch-task agent primitive (default OFF)
    agent_dispatch_task_enabled: bool = False

    # Serving-layer circuit breaker resilience primitive (default OFF)
    serving_circuit_breaker_enabled: bool = False

    # Crescendo multi-turn jailbreak probe (llm_red_team eval; default OFF)
    eval_crescendo_probe_enabled: bool = False

    # Behavioral-audit 40-dimension taxonomy (eval; default OFF)
    eval_behavioral_audit_taxonomy_enabled: bool = False

    # Threat-intelligence analyst persona for chat surface (default OFF)
    chat_threat_intel_persona_enabled: bool = False

    # Budget-bounded ReAct wrapper (tool-call cap; default OFF registry flag)
    agent_budget_bounded_loop_enabled: bool = False
    agent_budget_max_tool_invocations: int = 8

    # Tool-span supervision loss head (default OFF)
    training_tool_call_supervision_enabled: bool = False

    # Synthetic jailbreak probe generator for eval harnesses (default OFF)
    eval_synthetic_jailbreak_generator_enabled: bool = False

    # SSE wire encoder for streaming responses (default OFF)
    serving_sse_stream_encoder_enabled: bool = False

    # Sink-token logit bias helper for decoding (default OFF)
    inference_sink_logit_bias_enabled: bool = False
    inference_sink_logit_bonus: float = 1.0
    inference_sink_last_n_positions: int = 4

    def __post_init__(self) -> None:
        assert self.d_model == self.n_heads * self.head_dim, (
            f"d_model ({self.d_model}) must equal n_heads ({self.n_heads}) "
            f"* head_dim ({self.head_dim})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
