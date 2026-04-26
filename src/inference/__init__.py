"""Aurelius inference subsystem."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Continuous batching (Orca, OSDI 2022)
# ---------------------------------------------------------------------------
from src.inference.continuous_batching_scheduler import (
    BatchStep,
    ContinuousBatchingScheduler,
    InferenceRequest,
)

try:  # pragma: no cover - only triggered if a decoder registry exists elsewhere
    DECODER_REGISTRY  # type: ignore[name-defined]
except NameError:
    pass
else:  # pragma: no cover
    DECODER_REGISTRY["continuous_batching"] = ContinuousBatchingScheduler  # type: ignore[name-defined]

try:
    SCHEDULER_REGISTRY  # type: ignore[name-defined]
except NameError:
    SCHEDULER_REGISTRY = {}

SCHEDULER_REGISTRY["continuous_batching"] = ContinuousBatchingScheduler

from src.inference.json_mode_decoder import (
    JSONDecoderState,
    JSONMaskBuilder,
    is_valid_json_prefix,
)

# ---------------------------------------------------------------------------
# Multi-sample voting / self-consistency (Wang 2022, Chen 2024)
# ---------------------------------------------------------------------------
from src.inference.multi_sample_voting import (
    MultiSampleVoter,
    VoteResult,
)

__all__ = [
    "BatchStep",
    "ContinuousBatchingScheduler",
    "InferenceRequest",
    "JSONDecoderState",
    "JSONMaskBuilder",
    "MultiSampleVoter",
    "SCHEDULER_REGISTRY",
    "VoteResult",
    "is_valid_json_prefix",
]

from src.inference.sink_logit_bias import (  # noqa: E402
    SinkLogitBiasApplier,
    apply_sink_token_logit_bias,
)

LOGIT_BIAS_REGISTRY: dict[str, type] = {
    "sink_tokens": SinkLogitBiasApplier,
}

__all__ += [
    "SinkLogitBiasApplier",
    "apply_sink_token_logit_bias",
    "LOGIT_BIAS_REGISTRY",
]

from src.inference.beam_verifier_selector import BeamVerifierSelector  # noqa: E402

BEAM_VERIFIER_SELECTION_REGISTRY: dict[str, type] = {
    "argmax": BeamVerifierSelector,
}

__all__ += ["BeamVerifierSelector", "BEAM_VERIFIER_SELECTION_REGISTRY"]

# ---------------------------------------------------------------------------
# Reasoning Level Controller — GPT-OSS-120B (arXiv:2508.10925)
# Maps system-prompt "Reasoning: low/medium/high" to generation hyperparams.
# SWE-bench Verified: low=47.9%, medium=52.6%, high=62.4%.
# ---------------------------------------------------------------------------
from src.inference.reasoning_level_controller import (  # noqa: E402
    LEVEL_CONFIGS,
    apply_reasoning_level,
    parse_reasoning_level,
)

DECODER_REGISTRY: dict[str, object] = {
    "reasoning_level": parse_reasoning_level,
}

__all__ += [
    "DECODER_REGISTRY",
    "LEVEL_CONFIGS",
    "apply_reasoning_level",
    "parse_reasoning_level",
]

# ---------------------------------------------------------------------------
# Multimodal Thinking Chain — Kimi K2.6-style interleaved think/tool/vision
# ---------------------------------------------------------------------------
from src.inference.multimodal_thinking_chain import (  # noqa: E402
    ChainStep,
    MultimodalThinkingChain,
    MultimodalThinkingConfig,
    StepLimitError,
    StepType,
    ThinkingBudgetError,
    VisionStepLimitError,
)

DECODER_REGISTRY["multimodal_thinking_chain"] = MultimodalThinkingChain

__all__ += [
    "ChainStep",
    "MultimodalThinkingChain",
    "MultimodalThinkingConfig",
    "StepLimitError",
    "StepType",
    "ThinkingBudgetError",
    "VisionStepLimitError",
]

# ---------------------------------------------------------------------------
# Soft Thinking — differentiable probabilistic token embedding mixing (2025)
# ---------------------------------------------------------------------------
from src.inference.soft_thinking import (  # noqa: E402
    SoftThinkingConfig,
    SoftThinkingMixer,
)

DECODER_REGISTRY["soft_thinking"] = SoftThinkingMixer

__all__ += [
    "SoftThinkingConfig",
    "SoftThinkingMixer",
]

# ---------------------------------------------------------------------------
# Wait Token Forcer — S1 paper (2025) sequence-manipulation approach
# ---------------------------------------------------------------------------
from src.inference.wait_token_forcer import (  # noqa: E402
    WaitTokenForcer,
    WaitTokenForcerConfig,
)

DECODER_REGISTRY["wait_token_forcer"] = WaitTokenForcer

__all__ += [
    "WaitTokenForcer",
    "WaitTokenForcerConfig",
]

# ---------------------------------------------------------------------------
# CoCoNut — Chain of Continuous Thought (Hao et al. 2024)
# Latent-space multi-step reasoning: hidden states are fed back directly,
# bypassing the embedding lookup to avoid premature token commitment.
# ---------------------------------------------------------------------------
from src.inference.coconut import (  # noqa: E402
    CoCoNut,
    CoCoNutConfig,
    ContinuousReasoningStep,
)

DECODER_REGISTRY["coconut"] = CoCoNut

__all__ += [
    "CoCoNut",
    "CoCoNutConfig",
    "ContinuousReasoningStep",
]

# ---------------------------------------------------------------------------
# Eagle3 — confidence-gated speculative decoding (Cycle 130-A, 2025)
# ---------------------------------------------------------------------------
from src.inference.eagle3_decoding import (  # noqa: E402
    ConfidenceHead,
    Eagle3Config,
    Eagle3Decoder,
    Eagle3Drafter,
    Eagle3Verifier,
)

DECODER_REGISTRY["eagle3"] = Eagle3Decoder

__all__ += [
    "ConfidenceHead",
    "Eagle3Config",
    "Eagle3Decoder",
    "Eagle3Drafter",
    "Eagle3Verifier",
]

# ---------------------------------------------------------------------------
# Radix Cache — prefix-sharing KV cache via radix tree (SGLang RadixAttention, 2024)
# Nodes represent shared prefixes; leaves are per-request extensions.
# Reduces redundant KV computation for requests with common prefixes.
# ---------------------------------------------------------------------------
from src.inference.radix_cache import (  # noqa: E402
    CacheBlock,
    RadixCache,
    RadixCacheConfig,
    RadixNode,
)

DECODER_REGISTRY["radix_cache"] = RadixCache

__all__ += [
    "CacheBlock",
    "RadixCache",
    "RadixCacheConfig",
    "RadixNode",
]

# ---------------------------------------------------------------------------
# Hydra Speculative Decoding — multi-head parallel draft (Ankner et al. 2024)
# N draft heads attached to the target model's hidden states; all heads run
# in a single forward pass, eliminating a separate draft model entirely.
# ---------------------------------------------------------------------------
from src.inference.hydra_speculative import (  # noqa: E402
    HydraConfig,
    HydraHead,
    HydraSpeculative,
)

DECODER_REGISTRY["hydra"] = HydraSpeculative

__all__ += [
    "HydraConfig",
    "HydraHead",
    "HydraSpeculative",
]

# ---------------------------------------------------------------------------
# Chunked Prefill Scheduler — Sarathi-Serve / vLLM 2024 interleaved scheduling
# Splits long prompts into fixed-size chunks and interleaves prefill chunks
# with decode steps, bounding decode latency while keeping utilization high.
# ---------------------------------------------------------------------------
from src.inference.chunk_prefill_scheduler import (  # noqa: E402
    BatchSlot,
    ChunkPrefillConfig,
    ChunkPrefillScheduler,
    Request as ChunkPrefillRequest,
    RequestState as ChunkPrefillRequestState,
)

DECODER_REGISTRY["chunk_prefill"] = ChunkPrefillScheduler

__all__ += [
    "BatchSlot",
    "ChunkPrefillConfig",
    "ChunkPrefillRequest",
    "ChunkPrefillRequestState",
    "ChunkPrefillScheduler",
]

# ---------------------------------------------------------------------------
# Reward-Guided Search — ARGS / value-guided beam search for reasoning (2025)
# Scores candidates as (1-λ)*log_prob + λ*value_score with length penalty.
# Used in MCTS reasoning, PRM-guided beam search, and math/code decoding.
# ---------------------------------------------------------------------------
from src.inference.reward_guided_search import (  # noqa: E402
    RewardGuidedSearch,
    SearchBeam,
    SearchConfig,
)

DECODER_REGISTRY["reward_guided_search"] = RewardGuidedSearch

__all__ += [
    "RewardGuidedSearch",
    "SearchBeam",
    "SearchConfig",
]

# ---------------------------------------------------------------------------
# Min-P Sampler — dynamic probability-floor adaptive sampling (Nguyen 2024)
# p_min = min_p * max(p_tokens) adapts the threshold to the sharpness of
# the distribution, unlike the fixed-cutoff of standard nucleus sampling.
# ---------------------------------------------------------------------------
from src.inference.minp_sampler import (  # noqa: E402
    MinPConfig,
    MinPSampler,
)

DECODER_REGISTRY["minp_sampler"] = MinPSampler

__all__ += [
    "MinPConfig",
    "MinPSampler",
]

# ---------------------------------------------------------------------------
# Skeleton-of-Thought — plan-then-parallel-expand decoding (Bao et al. 2023)
# Generates a structured skeleton (list of N points) then expands each point
# independently, enabling batched parallel expansion for lower latency.
# ---------------------------------------------------------------------------
from src.inference.skeleton_of_thought import (  # noqa: E402
    SkeletonOfThoughtDecoder,
    SkeletonParser,
    SkeletonPoint,
    SoTConfig,
    SoTResult,
)

DECODER_REGISTRY["skeleton_of_thought"] = SkeletonOfThoughtDecoder

__all__ += [
    "SkeletonOfThoughtDecoder",
    "SkeletonParser",
    "SkeletonPoint",
    "SoTConfig",
    "SoTResult",
]

# ---------------------------------------------------------------------------
# Tree of Thought — BFS/DFS reasoning tree search (Yao et al. 2023)
# Explores a tree of intermediate reasoning steps with value-based pruning.
# BFS keeps the top-b frontier per level; DFS backtracks below threshold.
# ---------------------------------------------------------------------------
from src.inference.tree_of_thought import (  # noqa: E402
    ThoughtNode,
    ToTConfig,
    ToTTree,
    TreeOfThoughtDecoder,
)

# DECODER_REGISTRY["tree_of_thought"] is set inside tree_of_thought.py

__all__ += [
    "ThoughtNode",
    "ToTConfig",
    "ToTTree",
    "TreeOfThoughtDecoder",
]

# ---------------------------------------------------------------------------
# Adaptive KV Eviction — attention-score-based dynamic eviction
# (H2O / PyramidKV / SnapKV, Cycle 137-B)
# ---------------------------------------------------------------------------
from src.inference.adaptive_kv_eviction import (  # noqa: E402
    AdaptiveKVConfig,
    AdaptiveKVEvictionManager,
    KVCacheState,
)

DECODER_REGISTRY["adaptive_kv_eviction"] = AdaptiveKVEvictionManager

__all__ += [
    "AdaptiveKVConfig",
    "AdaptiveKVEvictionManager",
    "KVCacheState",
]

# ---------------------------------------------------------------------------
# KV Cache Eviction — policy-agnostic eviction primitives (Cycle 132)
# Pure-stdlib bookkeeping layer: LRU / LFU / FIFO / WEIGHTED / SINK_PRESERVING.
# ---------------------------------------------------------------------------
from src.inference.kv_cache_eviction import (  # noqa: E402
    EVICTION_POLICY_REGISTRY,
    CacheEntry,
    EvictionDecision,
    EvictionEngine,
    EvictionError,
    EvictionPolicy,
    select_victims,
)

__all__ += [
    "EVICTION_POLICY_REGISTRY",
    "CacheEntry",
    "EvictionDecision",
    "EvictionEngine",
    "EvictionError",
    "EvictionPolicy",
    "select_victims",
]

# ---------------------------------------------------------------------------
# Speculative Sampler — draft-verify cycle with acceptance criterion
# ---------------------------------------------------------------------------
from src.inference.speculative_sampler import (  # noqa: E402
    DraftToken,
    SpeculativeConfig,
    SpeculativeSampler,
    VerificationResult,
)

DECODER_REGISTRY["speculative_sampler"] = SpeculativeSampler

__all__ += [
    "DraftToken",
    "SpeculativeConfig",
    "SpeculativeSampler",
    "VerificationResult",
]

# ---------------------------------------------------------------------------
# Continuous Batching V2 — priority queues, preemption, memory budgeting
# ---------------------------------------------------------------------------
from src.inference.continuous_batching_v2 import (  # noqa: E402
    BatchRequest,
    BatchingConfig,
    ContinuousBatcherV2,
    RequestPriority,
)

SCHEDULER_REGISTRY["continuous_batching_v2"] = ContinuousBatcherV2

__all__ += [
    "BatchRequest",
    "BatchingConfig",
    "ContinuousBatcherV2",
    "RequestPriority",
]

# ---------------------------------------------------------------------------
# Prefix Caching — LRU KV cache keyed by prefix hash (Cycle 204)
# ---------------------------------------------------------------------------
from src.inference.prefix_caching import (  # noqa: E402
    CachedPrefix,
    PrefixCache,
    PrefixCacheConfig,
    compute_prefix_hash,
)

DECODER_REGISTRY["prefix_caching"] = PrefixCache

__all__ += [
    "CachedPrefix",
    "PrefixCache",
    "PrefixCacheConfig",
    "compute_prefix_hash",
]

# ---------------------------------------------------------------------------
# Request Coalescing — deduplicate concurrent identical requests (Cycle 204)
# ---------------------------------------------------------------------------
from src.inference.request_coalescing import (  # noqa: E402
    CoalescingConfig,
    CoalescingSlot,
    RequestCoalescer,
)

DECODER_REGISTRY["request_coalescing"] = RequestCoalescer

__all__ += [
    "CoalescingConfig",
    "CoalescingSlot",
    "RequestCoalescer",
]
