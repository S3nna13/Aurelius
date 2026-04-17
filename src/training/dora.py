"""DoRA (Weight-Decomposed Low-Rank Adaptation) — Liu et al., 2024.

DoRA decomposes a pre-trained weight W into magnitude m and direction V = W/‖W‖_col,
then adapts direction via LoRA:
    W' = m ⊙ (V + ΔV) / ‖V + ΔV‖_col
where ΔV = B @ A (low-rank), and m is a learnable vector of shape (1, out_features),
initialized to the column norms of W.

Reference: Liu et al. (2024) "DoRA: Weight-Decomposed Low-Rank Adaptation"
           https://arxiv.org/abs/2402.09353
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _col_norms(weight: torch.Tensor) -> torch.Tensor:
    """Compute per-output-feature (column) L2 norms of a weight matrix.

    Args:
        weight: Shape (out_features, in_features).

    Returns:
        Tensor of shape (1, out_features).
    """
    # weight: (out, in) — norm over the in dimension
    return weight.norm(p=2, dim=1, keepdim=True).T  # (1, out_features)


def _normalize_cols(weight: torch.Tensor) -> torch.Tensor:
    """Return weight with each output column normalized to unit L2 norm.

    Args:
        weight: Shape (out_features, in_features).

    Returns:
        Tensor of same shape with unit column norms.
    """
    norms = weight.norm(p=2, dim=1, keepdim=True)  # (out, 1)
    return weight / (norms + 1e-12)


class DoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear using DoRA.

    The original weight is frozen. During forward:
        W_adapted = weight_0 + B @ A          # direction update
        W_dir     = W_adapted / ‖W_adapted‖_col  # unit-norm directions
        W'        = m * W_dir                 # scale by magnitude
        output    = x @ W'.T + bias

    At initialization B=0, so W_adapted = weight_0 and W' = m * V = weight_0,
    meaning the output is identical to the original linear layer.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        # --- Frozen base weight (no grad) ---
        # Initialized to random normal; replaced by from_linear for real use.
        weight_init = torch.empty(out_features, in_features)
        nn.init.kaiming_uniform_(weight_init, a=math.sqrt(5))
        self.register_buffer("weight_0", weight_init)

        # --- Learnable magnitude vector: shape (1, out_features) ---
        col_norms = _col_norms(weight_init)  # (1, out_features)
        self.m = nn.Parameter(col_norms.clone())

        # --- LoRA matrices ---
        # A: (rank, in_features)  — initialized ~ N(0, 1/sqrt(rank))
        # B: (out_features, rank) — initialized to zeros (so ΔV=0 at init)
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features).normal_(std=1.0 / math.sqrt(rank))
        )
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # --- Optional bias ---
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute DoRA-adapted linear transformation.

        Args:
            x: Input tensor of shape (..., in_features).

        Returns:
            Output tensor of shape (..., out_features).
        """
        # W_adapted = weight_0 + B @ A   shape: (out, in)
        w_adapted = self.weight_0 + self.lora_B @ self.lora_A

        # Normalize each output row to unit norm  — shape: (out, in)
        w_dir = _normalize_cols(w_adapted)

        # Scale by magnitude: m is (1, out) — broadcast over (out, in)
        w_prime = self.m.T * w_dir  # (out, in)

        return F.linear(x, w_prime, self.bias)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 4) -> "DoRALinear":
        """Construct a DoRALinear from an existing nn.Linear.

        The original weight and bias are copied; the weight is frozen.

        Args:
            linear: Source nn.Linear layer.
            rank:   LoRA rank.

        Returns:
            Initialized DoRALinear with the same weight/bias.
        """
        has_bias = linear.bias is not None
        dora = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            rank=rank,
            bias=has_bias,
        )
        # Copy weights (no grad)
        dora.weight_0 = linear.weight.detach().clone()
        # Re-initialize m to the actual column norms of this weight
        dora.m = nn.Parameter(_col_norms(dora.weight_0).clone())
        # Copy bias
        if has_bias:
            dora.bias = nn.Parameter(linear.bias.detach().clone())
        return dora

    def merge(self) -> nn.Linear:
        """Return a plain nn.Linear with the adapted weight W' baked in.

        Returns:
            nn.Linear whose weight = m * normalize(weight_0 + B @ A)
            and bias copied from self.bias.
        """
        with torch.no_grad():
            w_adapted = self.weight_0 + self.lora_B @ self.lora_A
            w_dir = _normalize_cols(w_adapted)
            w_prime = self.m.T * w_dir  # (out, in)

        has_bias = self.bias is not None
        merged = nn.Linear(self.in_features, self.out_features, bias=has_bias)
        merged.weight = nn.Parameter(w_prime.clone())
        if has_bias:
            merged.bias = nn.Parameter(self.bias.detach().clone())
        return merged


class DoRAModel(nn.Module):
    """Wraps any model and replaces specified Linear layers with DoRALinear.

    Only the magnitude vector m, LoRA A, and LoRA B matrices are trained.
    The base weight (weight_0) is frozen.

    Example::

        model = MyTransformer(...)
        dora_model = DoRAModel(model, target_modules=["q_proj", "v_proj"], rank=8)
        optimizer = torch.optim.AdamW(dora_model.trainable_parameters(), lr=1e-4)
    """

    def __init__(
        self,
        model: nn.Module,
        target_modules: List[str],
        rank: int = 4,
    ) -> None:
        super().__init__()
        self.model = model
        self.target_modules = list(target_modules)
        self.rank = rank
        self.replace_layers()

    def replace_layers(self) -> None:
        """Walk named modules and replace matching Linear layers with DoRALinear."""
        # Collect replacements first to avoid mutating during iteration
        replacements: list[tuple[nn.Module, str, DoRALinear]] = []

        for full_name, module in self.model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            # Check if the module's local name (last component) matches any target
            local_name = full_name.split(".")[-1]
            if local_name not in self.target_modules:
                continue

            dora_layer = DoRALinear.from_linear(module, rank=self.rank)

            # Find the parent module and attribute name
            parts = full_name.split(".")
            parent = self.model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            replacements.append((parent, parts[-1], dora_layer))

        for parent, attr, dora_layer in replacements:
            setattr(parent, attr, dora_layer)

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Return only the trainable DoRA parameters (m, lora_A, lora_B).

        The frozen base weights (weight_0) are excluded.

        Returns:
            List of nn.Parameter objects with requires_grad=True.
        """
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, *args, **kwargs):
        """Delegate forward pass to the wrapped model."""
        return self.model(*args, **kwargs)
