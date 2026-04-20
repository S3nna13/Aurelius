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
