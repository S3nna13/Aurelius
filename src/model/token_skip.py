"""Token Skip: Adaptive computation via per-token layer skipping.

Easy tokens (high-confidence predictions at intermediate layers) can skip
remaining layers, reducing average compute. A lightweight confidence gate
decides at each layer whether each token should continue processing.

Inspired by: Schuster et al. 2022 (Confident Adaptive Language Modeling)
             Elbayad et al. 2020 (Depth-Adaptive Transformer)
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class ConfidenceGate(nn.Module):
    """Lightweight gate that decides whether tokens should exit early.

    A Linear(d_model, 1) → sigmoid maps each token's hidden state to a
    scalar confidence score. Tokens whose score exceeds `threshold` are
    considered "confident" and can skip subsequent layers.
    """

    def __init__(self, d_model: int, threshold: float = 0.5) -> None:
        super().__init__()
        self.gate = nn.Linear(d_model, 1)
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        """Compute per-token confidence scores.

        Args:
            x: (B, T, d_model) hidden states.

        Returns:
            (B, T) confidence scores in [0, 1].
        """
        # (B, T, 1) → squeeze to (B, T)
        return torch.sigmoid(self.gate(x)).squeeze(-1)

    def exit_mask(self, x: Tensor) -> Tensor:
        """Compute boolean mask indicating which tokens should exit.

        Args:
            x: (B, T, d_model) hidden states.

        Returns:
            (B, T) bool tensor — True means the token should exit (skip
            remaining layers).
        """
        return self.forward(x) > self.threshold


class SkippableLayer(nn.Module):
    """Wraps a transformer layer so that already-exited tokens can be skipped.

    For simplicity the implementation computes the full layer output for *all*
    tokens and then blends: positions where `skip_mask` is True keep their
    original hidden state while positions where it is False receive the layer's
    output.  After blending a new exit mask is computed from the updated hidden
    states and returned alongside the output.
    """

    def __init__(self, layer: nn.Module, gate: ConfidenceGate) -> None:
        super().__init__()
        self.layer = layer
        self.gate = gate

    def forward(
        self,
        x: Tensor,
        skip_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run the layer with optional token skipping.

        Args:
            x: (B, T, d_model) input hidden states.
            skip_mask: (B, T) bool tensor.  True = this token has already
                exited and should NOT be updated by this layer.  If None,
                all tokens are processed.

        Returns:
            output: (B, T, d_model) updated hidden states.
            new_skip_mask: (B, T) bool tensor from gate applied to output.
        """
        layer_out = self.layer(x)

        if skip_mask is None:
            out = layer_out
        else:
            # Expand mask to (B, T, 1) for broadcasting over d_model.
            mask_3d = skip_mask.unsqueeze(-1)  # (B, T, 1)
            out = torch.where(mask_3d, x, layer_out)

        new_skip_mask = self.gate.exit_mask(out)
        return out, new_skip_mask


class TokenSkipModel(nn.Module):
    """Transformer model with per-token adaptive layer skipping.

    Each layer is paired with a ConfidenceGate.  At each layer, tokens that
    have already been flagged as "confident" are carried forward unchanged.
    Once a token exits it never re-enters computation.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(d_model, d_model), nn.GELU()) for _ in range(n_layers)]
        )
        self.gates = nn.ModuleList(
            [ConfidenceGate(d_model, threshold) for _ in range(n_layers)]
        )
        self.skippable_layers = [
            SkippableLayer(layer, gate)
            for layer, gate in zip(self.layers, self.gates)
        ]

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:
        """Run the model with adaptive token skipping.

        Args:
            x: (B, T, d_model) input tensor.

        Returns:
            output: (B, T, d_model) final hidden states.
            stats: dict with keys:
                'exit_fractions'      — list[float] of length n_layers, fraction
                                        of tokens that exit at each layer.
                'mean_layers_computed' — float, average number of layers each
                                        token actually traverses.
        """
        B, T, _ = x.shape

        # cumulative_skip[b, t] == True means token t in batch b has exited.
        cumulative_skip = torch.zeros(B, T, dtype=torch.bool, device=x.device)

        exit_fractions: List[float] = []
        # layers_computed[b, t] accumulates how many layers each token ran through.
        layers_computed = torch.zeros(B, T, device=x.device)

        for skippable in self.skippable_layers:
            x, new_skip_mask = skippable(x, cumulative_skip)

            # Count this layer for tokens that were NOT already skipped.
            active = ~cumulative_skip  # (B, T) bool
            layers_computed += active.float()

            # Fraction of *all* tokens that exit after this layer.
            cumulative_skip = cumulative_skip | new_skip_mask
            exit_fractions.append(cumulative_skip.float().mean().item())

        stats: Dict[str, Any] = {
            "exit_fractions": exit_fractions,
            "mean_layers_computed": layers_computed.mean().item(),
        }
        return x, stats


class SkipRateLoss(nn.Module):
    """Auxiliary loss encouraging a desired average skip rate.

    Computes the MSE between the mean of the per-layer exit fractions and
    `target_skip_rate`.
    """

    def __init__(self, target_skip_rate: float = 0.5) -> None:
        super().__init__()
        self.target_skip_rate = target_skip_rate

    def forward(self, exit_fractions: List[float]) -> Tensor:
        """Compute the skip-rate auxiliary loss.

        Args:
            exit_fractions: list of per-layer exit fractions (floats in [0,1]).

        Returns:
            Scalar tensor: MSE(mean(exit_fractions), target_skip_rate).
        """
        mean_exit = torch.tensor(exit_fractions).mean()
        target = torch.tensor(self.target_skip_rate)
        return nn.functional.mse_loss(mean_exit, target)
