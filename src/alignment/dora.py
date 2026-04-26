"""DoRA: Weight-Decomposed Low-Rank Adaptation (arXiv:2402.09353).

Decomposes weight W = m * (V / ||V||_col) where V = W0 + scale * B @ A.
The magnitude vector m is learned separately from the directional LoRA update.

Critical: .detach() on the norm denominator to prevent gradient explosion (Section 4.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear implementing DoRA adaptation.

    The original weight W0 is frozen. Only A, B (LoRA matrices) and m
    (magnitude vector) are trainable.

    Args:
        weight: The original weight tensor (out_features, in_features). Will be cloned and frozen.
        rank: LoRA rank.
        alpha: LoRA scaling factor. scale = alpha / rank.
    """

    def __init__(self, weight: torch.Tensor, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        out_features, in_features = weight.shape

        # Frozen original weight
        self.register_buffer("W", weight.clone().detach())

        self.rank = rank
        self.scale = alpha / rank

        # LoRA matrices — A initialized with kaiming uniform, B with zeros
        self.A = nn.Parameter(torch.empty(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5**0.5)

        # Magnitude vector — initialized to per-row (column of W^T) norms of W0
        # m has shape (out_features, 1) for broadcasting
        m_init = weight.norm(p=2, dim=1, keepdim=True).detach()
        self.m = nn.Parameter(m_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Current adapted weight (direction matrix)
        V_prime = self.W + self.scale * (self.B @ self.A)

        # CRITICAL: .detach() prevents gradients from flowing through the norm denominator.
        # Without this, the learning signal for m mixes with the direction gradient — see paper Section 4.3.  # noqa: E501
        V_prime_norm = V_prime.norm(p=2, dim=1, keepdim=True).detach()

        # Magnitude rescaling factor: (out_features, 1)
        norm_scale = self.m / V_prime_norm

        # Compute base and LoRA contributions separately
        base_out = F.linear(x, self.W)  # (*, out_features)
        lora_out = F.linear(x, self.scale * (self.B @ self.A))  # (*, out_features)

        # DoRA formula: output = (m / ||V'||) * V' @ x^T
        #             = norm_scale * (W + sBA) @ x^T
        #             = norm_scale * (base_out + lora_out)
        # norm_scale squeezed to (out_features,) broadcasts over (*, out_features)
        ns = norm_scale.squeeze(-1)  # (out_features,)
        return ns * base_out + ns * lora_out

    def merge_weights(self) -> torch.Tensor:
        """Return the merged DoRA weight = (m / ||V'||_col) * V'.

        This is the weight that would produce identical outputs to forward()
        but without any runtime overhead.
        """
        V_prime = self.W + self.scale * (self.B @ self.A)
        V_prime_norm = V_prime.norm(p=2, dim=1, keepdim=True)
        return (self.m / V_prime_norm) * V_prime


def apply_dora_to_model(
    model: nn.Module,
    target_modules: list[str],
    rank: int,
    alpha: float = 1.0,
) -> dict[str, DoRALinear]:
    """Replace named linear layers with DoRALinear in-place.

    Args:
        model: The model to modify.
        target_modules: List of parameter name prefixes to replace
                        (e.g. ['layers.0.attn.q_proj', 'layers.1.attn.q_proj']).
                        Matches the parent module path (without '.weight').
        rank: LoRA rank for all replacements.
        alpha: LoRA alpha scaling for all replacements.

    Returns:
        Dict mapping module path -> DoRALinear for the replaced modules.
    """
    replaced: dict[str, DoRALinear] = {}

    for target in target_modules:
        parts = target.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        leaf_name = parts[-1]
        leaf_module = getattr(parent, leaf_name)

        if not isinstance(leaf_module, nn.Linear):
            raise ValueError(f"{target} is not an nn.Linear")

        dora = DoRALinear(leaf_module.weight, rank=rank, alpha=alpha)
        setattr(parent, leaf_name, dora)
        replaced[target] = dora

    return replaced
