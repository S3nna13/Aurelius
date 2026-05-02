"""Attention-sink style logit bonus on selected vocabulary ids.

Xiao et al. ("Efficient Streaming Language Models with Attention Sinks",
arXiv:2309.17453) show that keeping **sink** attention mass on a few dedicated
tokens stabilises streaming.  At decoding time, a lightweight heuristic is to
add a scalar bonus to sink token logits on the **most recent** time steps so
probability mass can pool there without architectural changes.

This module only manipulates logits tensors — it does not import attention
kernels or modify frozen model code.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def apply_sink_token_logit_bias(
    logits: torch.Tensor,
    sink_token_ids: Sequence[int],
    *,
    last_n_positions: int,
    bonus: float,
) -> torch.Tensor:
    """Return a new tensor with ``bonus`` added to sink ids on the last ``N`` steps.

    Args:
        logits: ``[B, T, V]`` decoder logits (any dtype that supports ``+``).
        sink_token_ids: Vocabulary positions to boost (duplicates allowed —
            bonus is applied once per listed id per position).
        last_n_positions: How many final time indices receive the bonus.
        bonus: Scalar added in the logits' dtype after promotion.

    Raises:
        ValueError: on shape / range errors (no silent clamping beyond
            ``last_n_positions`` capped to ``T``).
    """
    if logits.dim() != 3:
        raise ValueError(f"logits must be [B,T,V], got {tuple(logits.shape)}")
    b, t, v = logits.shape
    if last_n_positions < 0:
        raise ValueError("last_n_positions must be >= 0")
    if last_n_positions == 0:
        return logits.clone()
    if last_n_positions > t:
        raise ValueError(f"last_n_positions ({last_n_positions}) exceeds sequence length ({t})")
    if not sink_token_ids:
        raise ValueError("sink_token_ids must be non-empty")

    out = logits.clone()
    start = t - last_n_positions
    bonus_t = torch.tensor(bonus, device=logits.device, dtype=logits.dtype)
    ids = torch.tensor(list(sink_token_ids), device=logits.device, dtype=torch.long)
    if (ids < 0).any() or (ids >= v).any():
        bad = ids[(ids < 0) | (ids >= v)]
        raise ValueError(f"sink_token_ids out of range for vocab={v}: {bad}")

    out[:, start:, ids] = out[:, start:, ids] + bonus_t
    return out


class SinkLogitBiasApplier:
    """Callable holder for fixed hyper-parameters (registry-friendly)."""

    def __init__(
        self,
        sink_token_ids: Sequence[int],
        *,
        last_n_positions: int = 4,
        bonus: float = 1.0,
    ) -> None:
        if not sink_token_ids:
            raise ValueError("sink_token_ids must be non-empty")
        self.sink_token_ids = tuple(int(x) for x in sink_token_ids)
        self.last_n_positions = int(last_n_positions)
        self.bonus = float(bonus)

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return apply_sink_token_logit_bias(
            logits,
            self.sink_token_ids,
            last_n_positions=self.last_n_positions,
            bonus=self.bonus,
        )


__all__ = [
    "SinkLogitBiasApplier",
    "apply_sink_token_logit_bias",
]
