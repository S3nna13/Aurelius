"""Adaptive Computation Time (Graves, 2016) — learn when to halt per token."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ACTConfig:
    """Configuration for Adaptive Computation Time."""

    max_steps: int = 8  # maximum pondering steps
    epsilon: float = 0.01  # halting threshold (halt when cumulative prob >= 1-epsilon)
    ponder_cost: float = 0.001  # regularization coefficient
    d_model: int = 64


class HaltingUnit(nn.Module):
    """Computes per-token halting probability at each ACT step.

    Args:
        d_model: Hidden dimension size.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, 1)

    def forward(self, hidden: Tensor) -> Tensor:
        """Compute halting probabilities.

        Args:
            hidden: Hidden states of shape (B, T, D).

        Returns:
            Halting probabilities of shape (B, T), values in (0, 1).
        """
        return torch.sigmoid(self.linear(hidden)).squeeze(-1)  # (B, T)


class ACTLayer(nn.Module):
    """Wraps a TransformerBlock-like layer with Adaptive Computation Time.

    The base layer must accept and return tensors of shape (B, T, D).

    Args:
        layer:  The base computation layer (e.g. nn.Linear or TransformerBlock).
        config: ACTConfig controlling halting behaviour.
    """

    def __init__(self, layer: nn.Module, config: ACTConfig) -> None:
        super().__init__()
        self.layer = layer
        self.halting_unit = HaltingUnit(config.d_model)
        self.config = config

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Run the ACT loop over the base layer.

        Args:
            x: Input hidden states of shape (B, T, D).

        Returns:
            accumulated:   Weighted accumulation of layer outputs, shape (B, T, D).
            n_updates:     Number of steps each position ran, shape (B, T).
            remainders:    1 - sum of halting probs (for ponder cost), shape (B, T).
        """
        B, T, D = x.shape
        epsilon = self.config.epsilon
        max_steps = self.config.max_steps

        accumulated = torch.zeros_like(x)
        halting_probs = torch.zeros(B, T, device=x.device, dtype=x.dtype)
        n_updates = torch.zeros(B, T, device=x.device, dtype=x.dtype)

        for _step in range(max_steps):
            # Compute halting probability for this step
            h = self.halting_unit(x)  # (B, T)

            # Which positions are still running (haven't halted yet)?
            still_running = halting_probs < (1.0 - epsilon)  # (B, T) bool

            if not still_running.any():
                break

            # Where would adding h push cumulative past the threshold?
            will_halt = still_running & ((halting_probs + h) > (1.0 - epsilon))

            # For positions about to halt, use the remainder; otherwise use h
            p = torch.where(will_halt, 1.0 - halting_probs, h * still_running.float())

            halting_probs = halting_probs + p
            n_updates = n_updates + still_running.float()

            # Run the base layer
            new_x = self.layer(x)  # (B, T, D)

            # Accumulate weighted output
            accumulated = accumulated + p.unsqueeze(-1) * new_x

        remainders = 1.0 - halting_probs

        return accumulated, n_updates, remainders


def act_ponder_cost(n_updates: Tensor, remainders: Tensor, ponder_cost: float) -> Tensor:
    """ACT regularization loss: penalize excessive pondering.

    Args:
        n_updates:   Number of steps run per position, shape (B, T).
        remainders:  Remainder halting probabilities, shape (B, T).
        ponder_cost: Scalar regularization coefficient.

    Returns:
        Scalar ponder loss tensor.
    """
    return ponder_cost * torch.mean(n_updates + remainders)


class ACTTransformer(nn.Module):
    """Transformer model with ACT applied at each layer.

    For demonstration, proxy layers are nn.Linear(d_model, d_model) which
    preserve (B, T, D) tensor shapes.

    Args:
        d_model:  Hidden dimension size.
        n_layers: Number of ACT-wrapped layers.
        config:   ACTConfig controlling ACT behaviour.
    """

    def __init__(self, d_model: int, n_layers: int, config: ACTConfig) -> None:
        super().__init__()
        self.config = config
        self.n_layers = n_layers

        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.act_layers = nn.ModuleList([ACTLayer(self.layers[i], config) for i in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Run input through all ACT layers and sum ponder costs.

        Args:
            x: Pre-embedded input of shape (B, T, D).

        Returns:
            output:            Processed tensor of shape (B, T, D).
            total_ponder_cost: Scalar sum of ponder costs across all layers.
        """
        total_ponder_cost = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for act_layer in self.act_layers:
            x, n_updates, remainders = act_layer(x)
            total_ponder_cost = total_ponder_cost + act_ponder_cost(
                n_updates, remainders, self.config.ponder_cost
            )

        x = self.norm(x)
        output = self.head(x)
        return output, total_ponder_cost

    def mean_ponder_steps(self, x: Tensor) -> float:
        """Return the mean number of ACT steps across all layers and positions.

        Args:
            x: Pre-embedded input of shape (B, T, D).

        Returns:
            Mean number of pondering steps as a Python float.
        """
        all_n_updates: list[Tensor] = []
        with torch.no_grad():
            for act_layer in self.act_layers:
                _, n_updates, _ = act_layer(x)
                all_n_updates.append(n_updates)
        stacked = torch.stack(all_n_updates, dim=0)  # (n_layers, B, T)
        return float(stacked.mean().item())


def act_efficiency_stats(n_updates: Tensor, max_steps: int) -> dict[str, float]:
    """Compute efficiency statistics for an ACT forward pass.

    Args:
        n_updates: Number of steps run per position, shape (B, T).
        max_steps: Maximum allowed steps (from ACTConfig).

    Returns:
        Dictionary with keys:
            "mean_steps":       Mean steps taken across all positions.
            "max_steps_used":   Maximum steps used across any position.
            "early_halt_rate":  Fraction of positions that halted before max_steps.
    """
    mean_steps = float(n_updates.mean().item())
    max_steps_used = float(n_updates.max().item())
    early_halt_rate = float((n_updates < max_steps).float().mean().item())

    return {
        "mean_steps": mean_steps,
        "max_steps_used": max_steps_used,
        "early_halt_rate": early_halt_rate,
    }
