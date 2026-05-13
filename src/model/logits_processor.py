"""Centralized logit transformation processors.

This module provides shared implementations for common sampling-time
logit modifications:

- Temperature scaling
- Repetition penalty
- Top-p (nucleus) filtering

All samplers across ``src.inference`` should delegate to these classes
instead of duplicating the math.  This guarantees consistent behaviour
and makes it easier to audit, test, and improve sampling quality.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Re-export from the existing canonical implementation to avoid
# circular imports.  The actual class definitions live in
# ``src.inference.logit_processors``; importing here allows
# ``from src.model.logits_processor import TemperatureScaling`` etc.
# without creating a second source-of-truth.
from src.inference.logit_processors import (  # type: ignore
    LogitProcessor,
    RepetitionPenalty,
    TemperatureScaling,
    TopPLogitsProcessor,
)

__all__ = [
    "LogitProcessor",
    "TemperatureScaling",
    "RepetitionPenalty",
    "TopPLogitsProcessor",
]


class UnifiedLogitsProcessor(LogitProcessor):
    """Apply a configurable pipeline of standard processors in order.

    The order is important: 1) temperature, 2) repetition penalty, 3) top-p.
    This matches the conventional sampling pipeline used across Aurelius.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> None:
        self.processors: list[LogitProcessor] = []
        if temperature != 1.0:
            self.processors.append(TemperatureScaling(temperature))
        if repetition_penalty != 1.0:
            self.processors.append(RepetitionPenalty(repetition_penalty))
        if top_p < 1.0:
            self.processors.append(TopPLogitsProcessor(top_p))

    def __call__(self, logits: Tensor, input_ids: Tensor | None = None) -> Tensor:
        # If input_ids is None, replace with empty tensor for processors that need it
        if input_ids is None:
            input_ids = torch.tensor([], dtype=torch.long, device=logits.device)
        for processor in self.processors:
            logits = processor(logits, input_ids)
        return logits


def apply_standard_sampling_logits(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    input_ids: Tensor | None = None,
) -> Tensor:
    """Convenience function: apply the unified pipeline in one call."""
    processor = UnifiedLogitsProcessor(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    return processor(logits, input_ids)
