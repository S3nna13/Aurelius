"""Unified sampling utilities.

All inference modules should import and use :func:`sample_token` rather than
re-implementing temperature/top-p/repetition logic.  The function applies
a deterministic, well-tested pipeline via
:class:`~src.model.logits_processor.UnifiedLogitsProcessor`.
"""

from __future__ import annotations

import torch
from torch import Tensor

from src.model.logits_processor import (  # type: ignore
    UnifiedLogitsProcessor,
)


def sample_token(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    input_ids: Tensor | None = None,
) -> int:
    """Sample a single next token from logits using the standard pipeline.

    Parameters
    ----------
    logits: Tensor
        Unnormalized logits of shape ``[vocab_size]``.
    temperature: float
        Softmax temperature; values < 1 sharpen, > 1 flatten.
    top_p: float
        Nucleus (top-p) probability mass threshold in (0, 1].
    repetition_penalty: float
        Values > 1 penalise previously-seen tokens; < 1 encourage repetition.
    input_ids: Tensor | None
        Previously-generated token ids (1D). Required when ``repetition_penalty != 1.0``.
        Ignored otherwise.

    Returns
    -------
    int
        Sampled token ID.
    """
    proc = UnifiedLogitsProcessor(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    filtered = proc(logits, input_ids)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sample_tokens(
    logits: Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    input_ids: Tensor | None = None,
    num_samples: int = 1,
) -> Tensor:
    """Sample multiple token IDs from logits."""
    proc = UnifiedLogitsProcessor(
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    filtered = proc(logits, input_ids)
    probs = torch.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=num_samples)

