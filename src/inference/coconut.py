"""CoCoNut: Chain of Continuous Thought (Hao et al. 2024).

Instead of decoding discrete tokens during reasoning, the model operates in
continuous latent space — the hidden state at each "reasoning step" is fed
directly back as input, bypassing the embedding lookup. This allows richer
intermediate representations and avoids commitment to specific tokens during
thinking.

Reference: Hao et al., "Training Large Language Models to Reason in a
Continuous Latent Space" (2024). https://arxiv.org/abs/2412.06769
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CoCoNutConfig:
    d_model: int = 2048
    n_continuous_steps: int = 8        # reasoning steps in latent space
    continuous_step_hidden: int | None = None  # if None, defaults to d_model
    dropout: float = 0.0
    use_layer_norm: bool = True


# ---------------------------------------------------------------------------
# Single residual latent reasoning step
# ---------------------------------------------------------------------------

class ContinuousReasoningStep(nn.Module):
    """Single residual latent reasoning step.

    Computes: h' = LN(Linear(h) + h)

    When use_ln=False the layer norm is replaced by an identity, giving:
        h' = Linear(h) + h
    """

    def __init__(
        self,
        d_model: int,
        hidden: int,
        dropout: float,
        use_ln: bool,
    ) -> None:
        super().__init__()
        # Project through hidden dimension then back to d_model
        if hidden == d_model:
            self.proj = nn.Linear(d_model, d_model)
        else:
            self.proj = nn.Sequential(
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, d_model),
            )
        self.norm = nn.LayerNorm(d_model) if use_ln else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, h: Tensor) -> Tensor:
        """Apply one residual latent step.

        Args:
            h: [..., d_model] — any leading batch/sequence dimensions

        Returns:
            h_next: [..., d_model] — same shape as input
        """
        return self.norm(self.dropout(self.proj(h)) + h)


# ---------------------------------------------------------------------------
# Chain of Continuous Thought reasoner
# ---------------------------------------------------------------------------

class CoCoNut(nn.Module):
    """Chain of Continuous Thought reasoner (CoCoNut).

    Applies n_continuous_steps of ContinuousReasoningStep to an initial
    hidden state, keeping the computation entirely in latent space.
    """

    def __init__(self, config: CoCoNutConfig) -> None:
        super().__init__()
        self.config = config

        hidden = config.continuous_step_hidden
        if hidden is None:
            hidden = config.d_model

        self.steps = nn.ModuleList(
            [
                ContinuousReasoningStep(
                    d_model=config.d_model,
                    hidden=hidden,
                    dropout=config.dropout,
                    use_ln=config.use_layer_norm,
                )
                for _ in range(config.n_continuous_steps)
            ]
        )

    # ------------------------------------------------------------------
    # Core reasoning methods
    # ------------------------------------------------------------------

    def reason(self, h: Tensor) -> Tensor:
        """Apply all continuous reasoning steps.

        Args:
            h: [B, d_model] or [B, T, d_model] — initial hidden state

        Returns:
            final hidden state after all steps, same shape as input
        """
        for step in self.steps:
            h = step(h)
        return h

    def reason_with_trace(self, h: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Apply all continuous reasoning steps, collecting intermediate states.

        Args:
            h: [B, d_model] or [B, T, d_model] — initial hidden state

        Returns:
            (final_h, trace) where trace is a list of n_continuous_steps
            tensors [h_1, h_2, ..., h_n], each the same shape as input.
            The final entry of trace equals final_h.
        """
        trace: list[Tensor] = []
        for step in self.steps:
            h = step(h)
            trace.append(h)
        return h, trace

    def forward(self, h: Tensor) -> Tensor:
        """Alias for reason()."""
        return self.reason(h)
