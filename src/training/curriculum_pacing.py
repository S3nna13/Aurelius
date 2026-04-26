"""Pacing functions and difficulty-adaptive schedulers for curriculum learning."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class PacingConfig:
    """Configuration for curriculum pacing.

    Attributes:
        pacing_fn:       Which pacing schedule to use ("linear" | "exponential" | "step").
        start_fraction:  Fraction of (easiest) data available at step 0.
        end_fraction:    Fraction of data available at or after n_steps.
        n_steps:         Total number of training steps for the pacing schedule.
        step_size:       Number of steps between each increment (for "step" pacing).
    """

    pacing_fn: str = "linear"
    start_fraction: float = 0.2
    end_fraction: float = 1.0
    n_steps: int = 1000
    step_size: int = 100


# ---------------------------------------------------------------------------
# Pacing functions
# ---------------------------------------------------------------------------


def linear_pacing(step: int, n_steps: int, start: float, end: float) -> float:
    """Linearly interpolate the data fraction from *start* to *end*.

    Returns a value clamped to [start, end].
    """
    if n_steps <= 0:
        return end
    fraction = start + (end - start) * (step / n_steps)
    return float(max(start, min(end, fraction)))


def exponential_pacing(step: int, n_steps: int, start: float, end: float) -> float:
    """Exponentially grow the data fraction from *start* to *end*.

    Formula: start * (end / start) ** (step / n_steps)

    Clamped to [start, end].
    """
    if n_steps <= 0:
        return end
    if start <= 0:
        raise ValueError("start must be > 0 for exponential pacing")
    fraction = start * (end / start) ** (step / n_steps)
    return float(max(start, min(end, fraction)))


def step_pacing(step: int, step_size: int, start: float, end: float, n_steps: int) -> float:
    """Staircase pacing: fraction increases by a fixed increment every step_size steps.

    Number of increments = ceil(n_steps / step_size).
    The fraction at the final stair equals end.
    Clamped to [start, end].
    """
    if step_size <= 0:
        raise ValueError("step_size must be > 0")
    if n_steps <= 0:
        return end
    n_stairs = max(1, math.ceil(n_steps / step_size))
    stair = min(step // step_size, n_stairs)
    increment = (end - start) / n_stairs
    fraction = start + stair * increment
    return float(max(start, min(end, fraction)))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def get_pacing_fraction(step: int, config: PacingConfig) -> float:
    """Return the data fraction for step according to config."""
    fn = config.pacing_fn
    if fn == "linear":
        return linear_pacing(step, config.n_steps, config.start_fraction, config.end_fraction)
    elif fn == "exponential":
        return exponential_pacing(step, config.n_steps, config.start_fraction, config.end_fraction)
    elif fn == "step":
        return step_pacing(
            step, config.step_size, config.start_fraction, config.end_fraction, config.n_steps
        )
    else:
        raise ValueError(f"Unknown pacing_fn: {fn!r}. Choose 'linear', 'exponential', or 'step'.")


# ---------------------------------------------------------------------------
# DifficultyScorer
# ---------------------------------------------------------------------------


class DifficultyScorer:
    """Score samples by difficulty using per-sample cross-entropy loss.

    The model is used in inference mode only (torch.no_grad).
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def score(self, input_ids_list: list[Tensor]) -> Tensor:
        """Compute per-sample CE loss for each input in input_ids_list.

        Each element of the list is a 1-D or 2-D (1, seq_len) LongTensor.
        Returns a 1-D float Tensor of shape (N,) with one difficulty score
        per sample -- higher means harder.
        """
        scores: list[float] = []
        self.model.eval()
        with torch.no_grad():
            for ids in input_ids_list:
                # Ensure shape (1, seq_len)
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                seq_len = ids.shape[1]
                if seq_len < 2:
                    scores.append(0.0)
                    continue
                # Use shifted labels: predict tokens 1..T from 0..T-1
                _loss, logits, _pkv = self.model(ids)
                # logits: (1, seq_len, vocab_size)
                # Shift: input[:-1] -> predict input[1:]
                shift_logits = logits[:, :-1, :].contiguous()  # (1, S-1, V)
                shift_labels = ids[:, 1:].contiguous()  # (1, S-1)
                loss_val = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="mean",
                )
                scores.append(loss_val.item())
        return torch.tensor(scores, dtype=torch.float32)


# ---------------------------------------------------------------------------
# CurriculumSampler
# ---------------------------------------------------------------------------


class CurriculumSampler:
    """Curriculum-aware data sampler.

    At each training step, only the easiest fraction of data (ranked by
    difficulty score) is eligible for sampling.  The fraction grows over time
    according to the pacing schedule.
    """

    def __init__(
        self,
        samples: list,
        difficulties: Tensor,
        config: PacingConfig,
    ) -> None:
        if len(samples) != len(difficulties):
            raise ValueError(
                f"samples and difficulties must have the same length, "
                f"got {len(samples)} vs {len(difficulties)}"
            )
        self.samples = list(samples)
        self.difficulties = difficulties.float().clone()
        self.config = config

    def _sorted_order(self) -> list[int]:
        return torch.argsort(self.difficulties).tolist()

    def get_batch(self, step: int, batch_size: int) -> list:
        """Return a batch of batch_size samples from the easiest data fraction.

        The eligible pool is the easiest fraction * N samples.  Samples are
        drawn randomly (with replacement if the pool is smaller than batch_size).
        """
        fraction = get_pacing_fraction(step, self.config)
        n_total = len(self.samples)
        pool_size = max(1, int(math.ceil(fraction * n_total)))

        sorted_indices = self._sorted_order()
        pool_indices = sorted_indices[:pool_size]

        # Random selection from the pool
        if batch_size <= pool_size:
            chosen = torch.randperm(pool_size)[:batch_size].tolist()
            return [self.samples[pool_indices[i]] for i in chosen]
        else:
            # With replacement when batch_size > pool_size
            chosen = torch.randint(0, pool_size, (batch_size,)).tolist()
            return [self.samples[pool_indices[i]] for i in chosen]

    def update_difficulties(self, new_scores: Tensor) -> None:
        """Replace stored difficulty scores with new_scores.

        new_scores must have the same length as the original sample list.
        """
        if len(new_scores) != len(self.samples):
            raise ValueError(
                f"new_scores length {len(new_scores)} does not match "
                f"samples length {len(self.samples)}"
            )
        self.difficulties = new_scores.float().clone()
