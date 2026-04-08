"""Adaptive Computation Time (ACT) for transformers.

Implements ACT from "Adaptive Computation Time for Recurrent Neural Networks"
(Graves, 2016). Each token learns how many pondering steps to take before
halting, allowing the model to allocate more compute to harder tokens and
less to easier ones.

Key idea: a learned halting gate reads the current hidden state and outputs
a probability in (0, 1). Steps accumulate until the cumulative halting
probability exceeds a threshold (default 0.99). The output is a weighted
sum of per-step hidden states, with weights equal to the halting probabilities
assigned at each step. The final step's weight is adjusted to ensure all
weights sum exactly to 1 (the "remainder" trick).

A scalar ponder cost (epsilon * mean ponder steps) is returned alongside the
output so the caller can add it to the training loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class HaltingUnit(nn.Module):
    """Computes per-token halting probability from the current hidden state.

    A simple linear projection followed by sigmoid, producing a scalar
    halting probability in (0, 1) for each token. The bias is initialised
    to -1 so the gate starts biased toward *continuing* rather than halting.

    Args:
        d_model: Hidden feature dimension.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, 1, bias=True)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Return halting probabilities.

        Args:
            hidden: ``(batch, seq_len, d_model)``

        Returns:
            Halting probabilities of shape ``(batch, seq_len, 1)`` in (0, 1).
        """
        return torch.sigmoid(self.gate(hidden))


class ACTLayer(nn.Module):
    """Adaptive Computation Time wrapper around a transformer block.

    Applies the same ``block`` repeatedly (up to ``max_steps`` times),
    accumulating a weighted combination of per-step hidden states.  The
    halting unit decides, at each step, how much probability mass to
    assign to stopping.  The last step before the cumulative halt
    probability would exceed ``threshold`` receives the *remainder*
    weight so that all per-token weights sum to exactly 1.

    Args:
        block: Any callable ``x -> x`` (e.g. a ``TransformerBlock``).
        d_model: Hidden feature dimension.
        max_steps: Maximum number of pondering steps (default 8).
        threshold: Cumulative halting probability threshold (default 0.99).
        epsilon: Ponder cost regularisation coefficient (default 0.01).
    """

    def __init__(
        self,
        block: nn.Module,
        d_model: int,
        max_steps: int = 8,
        threshold: float = 0.99,
        epsilon: float = 0.01,
    ) -> None:
        super().__init__()
        self.block = block
        self.halting_unit = HaltingUnit(d_model)
        self.max_steps = max_steps
        self.threshold = threshold
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the ACT pondering loop.

        Args:
            x: Input tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tuple ``(output, ponder_cost)`` where:

            - ``output``: ``(batch, seq_len, d_model)`` — weighted combination
              of hidden states across pondering steps.
            - ``ponder_cost``: scalar — ``epsilon * mean(N + R)`` over all
              tokens, where *N* is the number of full steps taken and *R* is
              the remainder probability at the final step.
        """
        batch, seq_len, d_model = x.shape
        device = x.device
        dtype = x.dtype

        # Cumulative halting probability accumulated so far.
        # Shape: (batch, seq_len, 1)
        cumulative_halt = torch.zeros(batch, seq_len, 1, device=device, dtype=dtype)

        # Weighted combination of hidden states.
        output_accum = torch.zeros_like(x)

        # Ponder time N + R: counts full steps + remainder per token.
        ponder_time = torch.zeros(batch, seq_len, 1, device=device, dtype=dtype)

        # Boolean mask: True for tokens still pondering (not yet halted).
        # We keep all tokens alive for gradient purposes; weights naturally
        # go to zero for halted tokens.
        still_pondering = torch.ones(batch, seq_len, 1, dtype=torch.bool, device=device)

        for step in range(self.max_steps):
            # Run the block on the current hidden state.
            block_out = self.block(x)  # (batch, seq_len, d_model)

            # Halting probability for this step: (batch, seq_len, 1)
            h = self.halting_unit(x)

            # Determine whether adding h would push cumulative_halt over threshold.
            # Tokens that would exceed threshold get the remainder weight instead.
            would_exceed = (cumulative_halt + h) >= self.threshold  # (B, S, 1)

            # Compute remainder for tokens about to halt.
            remainder = 1.0 - cumulative_halt  # weight to use if halting this step

            # Effective weight for this step:
            #   - still pondering tokens that would exceed threshold -> remainder
            #   - still pondering tokens below threshold              -> h
            #   - already halted tokens                              -> 0
            weight = torch.where(
                still_pondering & would_exceed,
                remainder,
                torch.where(still_pondering, h, torch.zeros_like(h)),
            )

            # Accumulate weighted hidden states.
            output_accum = output_accum + weight * block_out

            # Ponder time: each still-pondering token adds 1 step; remainder
            # tokens add their remainder probability as the fractional part.
            step_ponder = torch.where(
                still_pondering & would_exceed,
                remainder,  # last step: add remainder (fractional)
                torch.where(still_pondering, torch.ones_like(h), torch.zeros_like(h)),
            )
            ponder_time = ponder_time + step_ponder

            # Update cumulative halting probability (clamped to 1).
            cumulative_halt = torch.clamp(cumulative_halt + weight, max=1.0)

            # Mark tokens as halted if they've exceeded the threshold.
            still_pondering = still_pondering & ~would_exceed

            # Early exit if all tokens have halted (purely an optimisation;
            # correctness does not depend on this).
            if not still_pondering.any():
                break

        # After the loop, any token still pondering (hit max_steps without
        # exceeding the threshold) must receive its remaining probability mass
        # so that all per-token weights sum exactly to 1.
        if still_pondering.any():
            remainder = 1.0 - cumulative_halt  # (batch, seq_len, 1)
            # Run the block one final time for these tokens.
            block_out = self.block(x)
            weight = torch.where(still_pondering, remainder, torch.zeros_like(remainder))
            output_accum = output_accum + weight * block_out
            ponder_time = ponder_time + torch.where(
                still_pondering, remainder, torch.zeros_like(remainder)
            )

        # Ponder cost: epsilon * mean over batch and seq of ponder_time.
        ponder_cost = self.epsilon * ponder_time.mean()

        return output_accum, ponder_cost

    def mean_ponder_steps(self, x: torch.Tensor) -> float:
        """Return average pondering steps for monitoring (no grad tracking).

        Args:
            x: ``(batch, seq_len, d_model)``

        Returns:
            Average number of pondering steps as a Python float.
        """
        with torch.no_grad():
            _, ponder_cost = self.forward(x)
            # Recover mean ponder_time from ponder_cost / epsilon.
            return (ponder_cost / self.epsilon).item()


class ACTTransformer(nn.Module):
    """Stack of ACTLayers for demonstration and testing.

    In production use, ``ACTLayer`` would wrap individual
    ``TransformerBlock`` instances.  Here each layer wraps a simple
    ``nn.Linear`` to keep the module self-contained and easily testable.

    Args:
        d_model: Hidden feature dimension.
        n_layers: Number of ACT layers (default 2).
        max_steps: Maximum pondering steps per layer (default 4).
    """

    def __init__(self, d_model: int, n_layers: int = 2, max_steps: int = 4) -> None:
        super().__init__()
        blocks = [nn.Linear(d_model, d_model) for _ in range(n_layers)]
        self.act_layers = nn.ModuleList(
            [ACTLayer(block, d_model, max_steps=max_steps) for block in blocks]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply all ACT layers, accumulating ponder cost.

        Args:
            x: ``(batch, seq_len, d_model)``

        Returns:
            Tuple ``(output, total_ponder_cost)`` where ``total_ponder_cost``
            is the sum of ponder costs across all layers.
        """
        total_ponder = torch.tensor(0.0, device=x.device)
        for layer in self.act_layers:
            x, ponder = layer(x)
            total_ponder = total_ponder + ponder
        return x, total_ponder
