"""Adaptive Computation Time (ACT) for Aurelius transformer.

Implements per-token halting probabilities and dynamic layer allocation using
the ACT algorithm (Graves 2016). Unlike early_exit_v2.py (patience-based) and
dynamic_depth.py (skip routing), this module computes a weighted sum of hidden
states across computation steps, gated by learned halting probabilities.

Components:
    ACTConfig             — dataclass controlling halting threshold and ponder cost
    HaltingUnit           — learns per-token halting probability via Linear + Sigmoid
    act_forward           — ACT loop: weighted accumulation with remainder budget
    ponder_loss           — regularisation loss penalising excessive pondering
    ACTTransformerLayer   — wraps a list of sub-layers with ACT computation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ACTConfig:
    """Configuration for Adaptive Computation Time.

    Attributes:
        threshold:    Cumulative halting probability above which a token stops.
                      Tokens halt when sum(p_t) >= threshold. Range (0, 1].
        max_steps:    Maximum number of computation steps (layers) to run,
                      regardless of the halting probability.
        epsilon:      Small value added to the final halting step so that the
                      remainder term is always non-negative and well defined.
        ponder_cost:  Coefficient scaling the ponder regularisation loss.
    """

    threshold: float = 0.99
    max_steps: int = 10
    epsilon: float = 0.01
    ponder_cost: float = 0.01


# ---------------------------------------------------------------------------
# Halting Unit
# ---------------------------------------------------------------------------

class HaltingUnit(nn.Module):
    """Learns per-token halting probability.

    A single linear projection followed by a sigmoid maps each token's hidden
    state to a scalar halting probability in (0, 1).

    Args:
        d_model: Hidden dimension of the transformer.

    Input:
        hidden: (B, T, D) — hidden states for all tokens.

    Output:
        (B, T, 1) — halting probability for each token in (0, 1).
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Compute per-token halting probabilities.

        Args:
            hidden: (B, T, D) hidden states.

        Returns:
            (B, T, 1) halting probabilities in (0, 1).
        """
        return torch.sigmoid(self.linear(hidden))   # (B, T, 1)


# ---------------------------------------------------------------------------
# ACT forward loop
# ---------------------------------------------------------------------------

def act_forward(
    hidden_states: torch.Tensor,
    layers: Sequence[nn.Module],
    halting_unit: HaltingUnit,
    config: ACTConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adaptive Computation Time forward pass.

    Iterates over `layers` (up to config.max_steps), accumulating a weighted
    sum of hidden states.  At each step t:
      - p_t = halting_unit(h_t)           per-token halt prob (B, T, 1)
      - remainder_t = 1 - sum_{s<t}(p_s)  leftover budget before this step
      - weight_t = min(p_t, remainder_t)  actual weight used this step

    A token halts when its cumulative halting probability reaches
    config.threshold (or on the final allowed step).

    The ponder cost (mean number of steps taken across all tokens) is returned
    as a scalar for use in the ponder_loss regulariser.

    Args:
        hidden_states: (B, T, D) initial hidden states.
        layers:        Sequence of nn.Module layers to iterate over.
                       Each layer must accept a single (B, T, D) tensor and
                       return a (B, T, D) tensor (or a tuple whose first
                       element is (B, T, D)).
        halting_unit:  HaltingUnit module producing per-token halt probs.
        config:        ACTConfig controlling threshold and max_steps.

    Returns:
        output:      (B, T, D) weighted sum of hidden states across steps.
        ponder_cost: scalar tensor — mean number of effective steps taken.
    """
    B, T, D = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype

    # Accumulated weighted output
    output = torch.zeros(B, T, D, device=device, dtype=dtype)
    # Cumulative halting probability per token: (B, T, 1)
    cumulative_halt = torch.zeros(B, T, 1, device=device, dtype=dtype)
    # Remaining budget per token: (B, T, 1)
    remainder = torch.ones(B, T, 1, device=device, dtype=dtype)
    # Accumulated ponder (number of effective steps) per token: (B, T, 1)
    ponder_steps = torch.zeros(B, T, 1, device=device, dtype=dtype)

    h = hidden_states
    n_steps = min(len(layers), config.max_steps)

    for step_idx in range(n_steps):
        layer = layers[step_idx]

        # Run this computation step
        layer_out = layer(h)
        if isinstance(layer_out, tuple):
            h = layer_out[0]
        else:
            h = layer_out   # (B, T, D)

        # Compute halting probability for tokens that haven't halted yet
        p = halting_unit(h)                         # (B, T, 1), in (0, 1)

        is_last_step = (step_idx == n_steps - 1)

        if is_last_step:
            # On the final step, use up the full remaining budget
            weight = remainder
            # Mark all tokens as halted
            halt_mask = torch.ones_like(remainder, dtype=torch.bool)
        else:
            # Tokens that would exceed threshold: use the remainder as weight
            # Tokens that haven't reached threshold: use p as weight
            new_cumulative = cumulative_halt + p
            halt_mask = new_cumulative >= config.threshold  # (B, T, 1)

            # For halting tokens: weight = remainder (leftover budget)
            # For non-halting tokens: weight = p
            weight = torch.where(halt_mask, remainder, p)

        # Accumulate weighted hidden state
        output = output + weight * h                # (B, T, D)

        # Accumulate ponder steps (each token gets +1 per step, weighted)
        ponder_steps = ponder_steps + weight

        # Update remainder and cumulative halt for non-halted tokens
        cumulative_halt = cumulative_halt + torch.where(halt_mask, remainder, p)
        remainder = torch.clamp(1.0 - cumulative_halt, min=0.0)

        # If all tokens have halted, we can stop early
        if halt_mask.all():
            break

    # ponder_cost: mean effective steps across all (B, T) positions
    ponder_cost = ponder_steps.mean()

    return output, ponder_cost


# ---------------------------------------------------------------------------
# Ponder loss
# ---------------------------------------------------------------------------

def ponder_loss(ponder_cost: torch.Tensor, target_ponder: float = 1.0) -> torch.Tensor:
    """Regularisation loss penalising deviation from a target ponder cost.

    Encourages the model to take approximately `target_ponder` steps on
    average, penalising both under- and over-computation equally.

    Args:
        ponder_cost:   Scalar tensor — mean steps taken (from act_forward).
        target_ponder: Desired average number of computation steps.

    Returns:
        Scalar tensor — mean squared error between ponder_cost and target.
    """
    return ((ponder_cost - target_ponder) ** 2).mean()


# ---------------------------------------------------------------------------
# ACTTransformerLayer
# ---------------------------------------------------------------------------

class ACTTransformerLayer(nn.Module):
    """Wraps a list of sub-layers with Adaptive Computation Time.

    Instead of running each sub-layer exactly once in sequence, ACT decides
    per-token how many computation steps to take (up to config.max_steps),
    accumulating a weighted combination of the outputs.

    Args:
        layers: nn.ModuleList of sub-layers (each (B,T,D) → (B,T,D)).
        d_model: Hidden dimension.
        config:  ACTConfig controlling the halting behaviour.
    """

    def __init__(
        self,
        layers: nn.ModuleList,
        d_model: int,
        config: ACTConfig,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.halting_unit = HaltingUnit(d_model)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run ACT over the wrapped sub-layers.

        Args:
            x:       (B, T, D) input hidden states.
            **kwargs: Ignored; present for API compatibility.

        Returns:
            output:      (B, T, D) ACT-weighted output.
            ponder_cost: scalar tensor — mean computation steps taken.
        """
        output, ponder_cost = act_forward(
            hidden_states=x,
            layers=self.layers,
            halting_unit=self.halting_unit,
            config=self.config,
        )
        return output, ponder_cost
