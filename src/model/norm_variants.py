"""norm_variants.py — Additional normalization layer variants for Aurelius.

Provides GroupNorm1D, DynamicTanh (DyT), QKNorm, ScaleNorm, and a
replace_norms utility to swap RMSNorm instances in an existing model.

References:
  - GroupNorm: Wu & He, 2018
  - DyT: Liu et al., 2025
  - QK-Norm: Henry et al. / used in many modern LLMs
  - ScaleNorm: Nguyen & Salazar, 2019
"""

import math

import torch
import torch.nn as nn


class GroupNorm1D(nn.Module):
    """Group Normalization adapted for 1-D sequence models.

    Accepts inputs of shape (B, T, C), reshapes to apply nn.GroupNorm
    along the channel axis, then restores the original shape.

    Args:
        num_channels: int — d_model / number of channels (C).
        num_groups:   int — number of groups; must divide num_channels.
        eps:          float — small constant for numerical stability.
    """

    def __init__(self, num_channels: int, num_groups: int = 8, eps: float = 1e-5):
        super().__init__()
        assert num_channels % num_groups == 0, (  # noqa: S101
            f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
        )
        self.gn = nn.GroupNorm(num_groups, num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, T, C)"""
        B, T, C = x.shape
        # GroupNorm expects (N, C, *) — treat B*T as batch dimension
        x = x.reshape(B * T, C)
        x = self.gn(x)
        return x.reshape(B, T, C)


class DynamicTanh(nn.Module):
    """DyT (Dynamic Tanh) normalization — Liu et al. 2025.

    Replaces LayerNorm / RMSNorm with a parameter-efficient alternative:

        DyT(x) = γ * tanh(α * x) + β

    where α is a learnable *scalar* (shared across all channels), and
    γ, β are learnable per-channel scale / shift vectors (analagous to
    the affine parameters in LayerNorm).

    No running statistics, no division — just a smooth bounded map.

    Args:
        d_model:    int   — feature dimension.
        init_alpha: float — initial value for the scalar α (default 0.5).
    """

    def __init__(self, d_model: int, init_alpha: float = 0.5):
        super().__init__()
        # α is a single scalar, not per-channel
        self.alpha = nn.Parameter(torch.tensor(init_alpha))
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) → (..., d_model)"""
        return self.gamma * torch.tanh(self.alpha * x) + self.beta


class QKNorm(nn.Module):
    """QK-Norm: independently normalize Q and K before the attention dot product.

    Prevents attention logit explosion with sequence length by keeping the
    pre-softmax scores on a well-conditioned scale.

    Per head:
        Q_norm = RMSNorm(Q) * scale_q   # scale_q shape: (n_heads, 1, 1)
        K_norm = RMSNorm(K) * scale_k   # scale_k shape: (n_kv_heads, 1, 1)

    Supports Grouped Query Attention (GQA) where n_kv_heads ≠ n_heads by
    keeping separate per-head-type scale parameters.

    Args:
        n_heads:   int   — number of query heads.
        head_dim:  int   — dimension per head.
        eps:       float — small constant for RMS normalization.
    """

    def __init__(self, n_heads: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        init_val = math.sqrt(head_dim)
        # Per-head scales — shape broadcastable over (B, heads, T, head_dim)
        self.scale_q = nn.Parameter(torch.full((n_heads, 1, 1), init_val))
        self.scale_k = nn.Parameter(torch.full((n_heads, 1, 1), init_val))
        self.eps = eps

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """RMS-normalize along the last dimension.

        x / sqrt(mean(x^2) + eps)
        """
        rms = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x.float() / rms).to(x.dtype)

    def forward(
        self,
        q: torch.Tensor,  # (B, n_heads,    T, head_dim)
        k: torch.Tensor,  # (B, n_kv_heads, T, head_dim)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (normalized_q, normalized_k) with the same shapes as inputs."""
        n_kv_heads = k.shape[1]

        # Normalize
        q_norm = self.normalize(q)  # (B, n_heads,    T, head_dim)
        k_norm = self.normalize(k)  # (B, n_kv_heads, T, head_dim)

        # Apply per-head scales.
        # scale_q: (n_heads, 1, 1) — broadcasts over (B, n_heads, T, head_dim)
        q_out = q_norm * self.scale_q

        # For GQA, scale_k was initialised with n_heads entries.
        # Slice to n_kv_heads so shapes match without error.
        scale_k = self.scale_k[:n_kv_heads]  # (n_kv_heads, 1, 1)
        k_out = k_norm * scale_k

        return q_out, k_out


class ScaleNorm(nn.Module):
    """ScaleNorm (Nguyen & Salazar 2019).

    Normalises each token vector by its L2 norm, then scales by a single
    learnable scalar g:

        ScaleNorm(x) = g * x / ||x||_2

    Simpler than LayerNorm: only one trainable parameter.

    Args:
        d_model: int   — used to set the initial value of g = sqrt(d_model).
        eps:     float — small constant added before division.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(math.sqrt(d_model)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_model) → (..., d_model)"""
        norm = x.float().pow(2).sum(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (self.g * x.float() / norm).to(x.dtype)


def replace_norms(model: nn.Module, norm_type: str = "dyt") -> int:
    """Replace every RMSNorm instance in *model* with the requested norm type.

    Performs an in-place walk of model.named_modules(), locates RMSNorm
    layers, and uses setattr on the parent module to swap them out.

    Args:
        model:     nn.Module — the model to modify in-place.
        norm_type: str       — one of "dyt", "scalenorm", "groupnorm".

    Returns:
        int — the number of modules that were replaced.

    Raises:
        ValueError — if norm_type is unrecognised.
    """
    from src.model.rms_norm import RMSNorm  # local import to avoid circular deps

    valid = {"dyt", "scalenorm", "groupnorm"}
    if norm_type not in valid:
        raise ValueError(f"norm_type must be one of {valid}, got {norm_type!r}")

    # Build a mapping: dotted_path → (parent_module, child_attr_name, RMSNorm)
    replacements = []
    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            # Derive parent path and attribute name
            parts = name.rsplit(".", 1)
            if len(parts) == 1:
                parent = model
                attr = parts[0]
            else:
                parent_name, attr = parts
                parent = model
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
            replacements.append((parent, attr, module))

    for parent, attr, rms in replacements:
        d_model = rms.weight.shape[0]
        if norm_type == "dyt":
            new_norm = DynamicTanh(d_model)
        elif norm_type == "scalenorm":
            new_norm = ScaleNorm(d_model)
        else:  # groupnorm
            # Choose the largest divisor of d_model that is <= 32 and >= 1
            num_groups = 1
            for g in [32, 16, 8, 4, 2, 1]:
                if d_model % g == 0:
                    num_groups = g
                    break
            new_norm = GroupNorm1D(d_model, num_groups)
        setattr(parent, attr, new_norm)

    return len(replacements)
