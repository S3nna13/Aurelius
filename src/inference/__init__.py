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
