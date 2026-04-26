"""IA³ — Infused Adapter by Inhibiting and Amplifying Inner Activations.

Liu et al., 2022 (arXiv:2205.05638).

IA³ multiplies learned vectors into key, value, and FFN activations.
This implementation focuses on FFN down_proj scaling — the most parameter-
efficient path to behaviour adaptation.

Only the injected scale vectors are trainable; all base model weights are frozen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class IA3Config:
    """Configuration for IA³ injection."""

    target_modules: list[str] = field(default_factory=lambda: ["k_proj", "v_proj", "down_proj"])
    init_ia3_weights: bool = True  # init scaling vectors to ones
    trainable_only: bool = True  # only ia3 vectors trainable


# ---------------------------------------------------------------------------
# Core layer
# ---------------------------------------------------------------------------


class IA3Layer(nn.Module):
    """Scales activations with a learned vector of ones-initialized weights.

    Args:
        d: Dimensionality of the scaling vector (must match last dim of input).
    """

    def __init__(self, d: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d))

    def forward(self, x: Tensor) -> Tensor:
        """Multiply last dimension of x by self.scale.

        Args:
            x: Tensor of shape (..., d).

        Returns:
            Tensor of same shape with last dim scaled.
        """
        return x * self.scale


# ---------------------------------------------------------------------------
# Scaled linear wrapper
# ---------------------------------------------------------------------------


class IA3ScaledLinear(nn.Module):
    """Linear layer with IA³ scaling applied to its output.

    Wraps an existing nn.Linear and applies an IA3Layer to the output.

    Args:
        linear:    The underlying nn.Linear.
        ia3_layer: The IA3Layer that scales the output.
    """

    def __init__(self, linear: nn.Linear, ia3_layer: IA3Layer) -> None:
        super().__init__()
        self.linear = linear
        self.ia3_layer = ia3_layer

    def forward(self, x: Tensor) -> Tensor:
        return self.ia3_layer(self.linear(x))


# ---------------------------------------------------------------------------
# Parameter utilities
# ---------------------------------------------------------------------------


def count_trainable_parameters(model: nn.Module) -> int:
    """Count parameters where requires_grad=True."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Count all parameters regardless of requires_grad."""
    return sum(p.numel() for p in model.parameters())


def get_ia3_parameters(ia3_layers: dict[str, IA3Layer]) -> list[nn.Parameter]:
    """Return list of all IA³ scale parameters."""
    return [layer.scale for layer in ia3_layers.values()]


# ---------------------------------------------------------------------------
# Injection
# ---------------------------------------------------------------------------


def inject_ia3_layers(model: nn.Module, cfg: IA3Config) -> dict[str, IA3Layer]:
    """Inject IA3Layer scaling into FFN down_proj activations across all transformer layers.

    For each layer in model.layers:
      - Wraps layer.ffn.down_proj with an IA3ScaledLinear that applies IA3Layer
        scaling to the output before returning.

    Freezes all parameters except the injected IA3 scales.

    Returns:
        Dict mapping "layer_{i}.down_proj" -> IA3Layer instance.
    """
    ia3_map: dict[str, IA3Layer] = {}

    for i, layer in enumerate(model.layers):
        linear: nn.Linear = layer.ffn.down_proj
        # down_proj: (d_ff -> d_model); output dimension is d_model
        out_features = linear.out_features
        ia3_layer = IA3Layer(out_features)
        scaled = IA3ScaledLinear(linear, ia3_layer)
        layer.ffn.down_proj = scaled
        ia3_map[f"layer_{i}.down_proj"] = ia3_layer

    # Freeze everything, then re-enable only IA3 scales
    for p in model.parameters():
        p.requires_grad_(False)

    for ia3_layer in ia3_map.values():
        ia3_layer.scale.requires_grad_(True)

    return ia3_map


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_ia3_weights(ia3_layers: dict[str, IA3Layer], path: str) -> None:
    """Save IA³ scales as {name: tensor} dict via torch.save."""
    state = {name: layer.scale.data for name, layer in ia3_layers.items()}
    torch.save(state, path)


def load_ia3_weights(ia3_layers: dict[str, IA3Layer], path: str) -> None:
    """Load IA³ scales from path and copy into existing ia3_layers."""
    state: dict[str, Tensor] = torch.load(path, weights_only=True)
    for name, tensor in state.items():
        ia3_layers[name].scale.data.copy_(tensor)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class IA3Trainer:
    """Fine-tuning trainer using IA³ adapters."""

    def __init__(
        self,
        model: nn.Module,
        ia3_layers: dict[str, IA3Layer],
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.ia3_layers = ia3_layers
        self.optimizer = optimizer

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict[str, float]:
        """Forward pass, cross-entropy loss, backward pass, optimiser step.

        Args:
            input_ids: (batch, seq_len) token indices.
            labels:    (batch, seq_len) target token ids; -100 is ignored.

        Returns:
            Dict with keys "loss" (float) and "n_ia3_params" (int cast to float).
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Model returns (loss, logits, pkv) — loss is None unless labels passed
        loss_val, logits, _ = self.model(input_ids)

        # Compute CE ourselves with ignore_index=-100
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        loss.backward()
        self.optimizer.step()

        n_ia3 = sum(p.numel() for p in get_ia3_parameters(self.ia3_layers))
        return {"loss": loss.item(), "n_ia3_params": float(n_ia3)}

    def merge_into_model(self) -> None:
        """Bake the IA³ scales into each IA3ScaledLinear's underlying weight.

        For down_proj: weight shape is (d_model, d_ff), scale shape is (d_model,).
        new_weight = linear.weight * ia3_layer.scale.unsqueeze(1)
        """
        for layer in self.model.layers:
            ffn_proj = layer.ffn.down_proj
            if isinstance(ffn_proj, IA3ScaledLinear):
                scale = ffn_proj.ia3_layer.scale.detach()
                # weight: (d_model, d_ff), scale: (d_model,)
                ffn_proj.linear.weight.data.mul_(scale.unsqueeze(1))
                # Reset scale to ones so forward() is a no-op
                ffn_proj.ia3_layer.scale.data.fill_(1.0)
