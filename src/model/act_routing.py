"""Adaptive Computation Time: learned per-token halting with ponder cost regularization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ACTConfig:
    """Configuration for Adaptive Computation Time routing.

    Attributes:
        max_steps: Maximum number of computation steps per token.
        halt_threshold: Cumulative halting probability at which a token is considered done.
        ponder_cost: Penalty coefficient per computation step (ponder cost weight).
        epsilon: Minimum halting probability injected at each step to ensure progress.
    """

    max_steps: int = 8
    halt_threshold: float = 0.99
    ponder_cost: float = 0.01
    epsilon: float = 1e-2


# ---------------------------------------------------------------------------
# Halting Unit
# ---------------------------------------------------------------------------


class HaltingUnit(nn.Module):
    """Learned per-token halting probability predictor.

    Projects each token's hidden state to a scalar halting probability via a
    linear layer followed by sigmoid activation.

    Args:
        d_model: Hidden state dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, x: Tensor) -> Tensor:
        """Compute per-token halting probabilities.

        Args:
            x: ``(B, T, d_model)`` hidden states.

        Returns:
            ``(B, T)`` halting probabilities in (0, 1).
        """
        # (B, T, d_model) -> (B, T, 1) -> (B, T)
        return torch.sigmoid(self.linear(x)).squeeze(-1)


# ---------------------------------------------------------------------------
# ACT State Update
# ---------------------------------------------------------------------------


def compute_act_state(
    h_t: Tensor,
    cumulative_halt: Tensor,
    halted: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute adjusted halting weights and determine newly-halted tokens.

    For tokens that are already halted their contribution is zeroed.  For tokens
    whose cumulative halting probability *would* exceed the threshold, the
    remainder (1 - cumulative) is used so that weights sum to at most 1.

    Args:
        h_t: Per-token halting probabilities at the current step, shape ``(B, T)``.
        cumulative_halt: Cumulative halting probability accumulated so far, ``(B, T)``.
        halted: Boolean mask of tokens that already halted in a prior step, ``(B, T)``.

    Returns:
        Tuple of:
            adjusted_h_t: ``(B, T)`` adjusted halting weights for this step.
            new_halted: ``(B, T)`` bool — tokens that halt at this step (not yet
                        halted, but cumulative now reaches threshold).
    """
    # Tokens that were already halted contribute nothing this step.
    h_t = h_t.masked_fill(halted, 0.0)

    # Would the cumulative halt exceed the threshold?
    new_cumulative = cumulative_halt + h_t
    exceeds = new_cumulative >= 1.0  # using 1.0 as the hard cap (remainder trick)

    # Use remainder for tokens that would exceed
    remainder = (1.0 - cumulative_halt).clamp(min=0.0)
    adjusted_h_t = torch.where(exceeds & ~halted, remainder, h_t)

    # A token is newly halted if it's not already halted AND it now reaches/exceeds threshold
    new_halted = exceeds & ~halted

    return adjusted_h_t, new_halted


# ---------------------------------------------------------------------------
# Ponder Cost
# ---------------------------------------------------------------------------


def compute_ponder_cost(n_steps_per_token: Tensor) -> Tensor:
    """Compute the mean number of steps taken across all tokens.

    Args:
        n_steps_per_token: ``(B, T)`` integer tensor of steps taken per token.

    Returns:
        Scalar mean ponder cost.
    """
    return n_steps_per_token.float().mean()


# ---------------------------------------------------------------------------
# ACT Wrapper
# ---------------------------------------------------------------------------


class ACTWrapper(nn.Module):
    """Wraps a computation block with Adaptive Computation Time halting.

    At each step the block is applied to the current hidden state.  A learned
    halting unit produces per-token halting probabilities.  Outputs are
    accumulated as a weighted sum weighted by the per-step halting weights.
    Iteration stops once all tokens have halted or ``max_steps`` is reached.

    Args:
        block: An ``nn.Module`` mapping ``(B, T, d_model)`` -> ``(B, T, d_model)``.
        d_model: Hidden state dimension.
        config: ``ACTConfig`` instance controlling halting behaviour.
    """

    def __init__(self, block: nn.Module, d_model: int, config: ACTConfig) -> None:
        super().__init__()
        self.block = block
        self.config = config
        self.halting_unit = HaltingUnit(d_model)

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        """Run ACT over the wrapped block.

        Args:
            x: ``(B, T, d_model)`` input hidden states.

        Returns:
            Tuple of:
                accumulated_output: ``(B, T, d_model)`` weighted output.
                info: dict with keys:
                    ``"ponder_cost"`` (float),
                    ``"mean_steps"`` (float),
                    ``"halted_at_step"`` (``(B, T)`` int Tensor, step index where each
                    token halted; 0-indexed, equals ``max_steps - 1`` for tokens
                    that never formally halted).
        """
        B, T, D = x.shape
        config = self.config

        # Running state
        cumulative_halt = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        halted = torch.zeros(B, T, device=x.device, dtype=torch.bool)
        accumulated = torch.zeros_like(x)
        n_steps_per_token = torch.zeros(B, T, device=x.device, dtype=torch.long)
        halted_at_step = torch.full((B, T), config.max_steps - 1, device=x.device, dtype=torch.long)

        current = x

        for step in range(config.max_steps):
            # Apply computation block
            current = self.block(current)

            # Get raw halting probabilities, inject epsilon floor
            h_t = self.halting_unit(current)  # (B, T)
            h_t = h_t + config.epsilon
            h_t = h_t.clamp(max=1.0)

            # Adjust for already-halted tokens and threshold
            adjusted_h_t, newly_halted = compute_act_state(h_t, cumulative_halt, halted)

            # Accumulate weighted output
            # adjusted_h_t: (B, T) -> (B, T, 1) for broadcasting
            accumulated = accumulated + adjusted_h_t.unsqueeze(-1) * current

            # Update cumulative halt
            cumulative_halt = cumulative_halt + adjusted_h_t

            # Track steps per token (count this step for non-halted tokens)
            n_steps_per_token = n_steps_per_token + (~halted).long()

            # Record halt step for newly halted tokens
            halted_at_step = torch.where(newly_halted & ~halted, torch.full_like(halted_at_step, step), halted_at_step)

            # Update halted mask
            halted = halted | newly_halted

            # Early stop if all tokens halted
            if halted.all():
                break

        ponder_cost_tensor = compute_ponder_cost(n_steps_per_token)

        info = {
            "ponder_cost": ponder_cost_tensor.item(),
            "mean_steps": n_steps_per_token.float().mean().item(),
            "halted_at_step": halted_at_step,
        }

        return accumulated, info


# ---------------------------------------------------------------------------
# ACT Loss
# ---------------------------------------------------------------------------


def compute_act_loss(
    task_loss: Tensor,
    ponder_cost: Tensor,
    ponder_weight: float,
) -> Tensor:
    """Combine task loss with ponder cost regularization.

    Args:
        task_loss: Scalar task loss (e.g. cross-entropy).
        ponder_cost: Scalar ponder cost (mean steps).
        ponder_weight: Weight applied to the ponder cost term.

    Returns:
        Scalar total loss = task_loss + ponder_weight * ponder_cost.
    """
    return task_loss + ponder_weight * ponder_cost
