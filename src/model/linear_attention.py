"""Linear attention with kernel feature maps.

Implements O(T) linear attention as an efficient alternative to O(T²) standard
attention, using kernel feature maps to approximate the softmax kernel.

Supported feature maps:
  - ELU+1  (always positive, simple, no extra parameters)
  - ReLU+ε (always non-negative, very cheap)
  - Random Fourier Features (approximates RBF kernel, requires stored ω/b)

References:
  Katharopoulos et al. (2020) "Transformers are RNNs"
  Rahimi & Recht (2007) "Random Features for Large-Scale Kernel Machines"
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LinearAttnConfig:
    d_model: int = 64
    n_heads: int = 2
    head_dim: int = 32
    feature_map: str = "elu"      # "elu" | "relu" | "random_fourier"
    n_features: int = 64          # used only for random_fourier
    causal: bool = True
    eps: float = 1e-6


# ---------------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------------

def elu_feature_map(x: Tensor) -> Tensor:
    """ELU+1 feature map: ELU(x) + 1 (always positive)."""
    return F.elu(x) + 1.0


def relu_feature_map(x: Tensor) -> Tensor:
    """ReLU feature map: ReLU(x) + epsilon for numerical stability."""
    return F.relu(x) + 1e-6


def random_fourier_features(
    x: Tensor,      # (..., D)
    omega: Tensor,  # (D, n_features)
    bias: Tensor,   # (n_features,)
) -> Tensor:
    """Random Fourier Features: cos(x @ omega + bias) * sqrt(2 / n_features).

    Returns (..., n_features).
    """
    # x: (..., D), omega: (D, n_features) -> projection: (..., n_features)
    projection = x @ omega + bias
    scale = math.sqrt(2.0 / omega.shape[1])
    return torch.cos(projection) * scale


# ---------------------------------------------------------------------------
# Core attention functions
# ---------------------------------------------------------------------------

def linear_attention_causal(
    q: Tensor,      # (B, H, T, D_k) — feature-mapped queries
    k: Tensor,      # (B, H, T, D_k) — feature-mapped keys
    v: Tensor,      # (B, H, T, D_v)
    eps: float = 1e-6,
) -> Tensor:
    """Causal linear attention via sequential scan.

    Maintains running state:
      S = sum_{i<=t}(k_i^T v_i)  shape (B, H, D_k, D_v)
      z = sum_{i<=t}(k_i)        shape (B, H, D_k)

    At each step t:
      S_t = S_{t-1} + outer(k_t, v_t)
      z_t = z_{t-1} + k_t
      y_t = q_t @ S_t / (q_t @ z_t + eps)

    Returns (B, H, T, D_v).
    """
    B, H, T, D_k = q.shape
    D_v = v.shape[-1]

    S = q.new_zeros(B, H, D_k, D_v)   # running KV state
    z = q.new_zeros(B, H, D_k)         # running key normaliser

    outputs = []
    for t in range(T):
        k_t = k[:, :, t, :]    # (B, H, D_k)
        v_t = v[:, :, t, :]    # (B, H, D_v)
        q_t = q[:, :, t, :]    # (B, H, D_k)

        # Outer product update: (B, H, D_k, 1) * (B, H, 1, D_v) -> (B, H, D_k, D_v)
        S = S + torch.einsum("bhd,bhe->bhde", k_t, v_t)
        z = z + k_t

        # Numerator: q_t @ S  -> (B, H, D_v)
        num = torch.einsum("bhd,bhde->bhe", q_t, S)
        # Denominator: q_t · z  -> (B, H, 1)
        den = (torch.einsum("bhd,bhd->bh", q_t, z) + eps).unsqueeze(-1)

        outputs.append((num / den).unsqueeze(2))   # (B, H, 1, D_v)

    return torch.cat(outputs, dim=2)   # (B, H, T, D_v)


def linear_attention_noncausal(
    q: Tensor,      # (B, H, T, D_k)
    k: Tensor,      # (B, H, T, D_k)
    v: Tensor,      # (B, H, T, D_v)
    eps: float = 1e-6,
) -> Tensor:
    """Non-causal linear attention: O(T*D²) instead of O(T²).

    S = sum_i k_i^T v_i   shape (B, H, D_k, D_v)
    z = sum_i k_i          shape (B, H, D_k)
    y_t = q_t @ S / (q_t @ z + eps)

    Returns (B, H, T, D_v).
    """
    # S: (B, H, D_k, D_v)
    S = torch.einsum("bhtd,bhte->bhde", k, v)
    # z: (B, H, D_k)
    z = k.sum(dim=2)

    # Numerator: (B, H, T, D_k) @ (B, H, D_k, D_v) -> (B, H, T, D_v)
    num = torch.einsum("bhtd,bhde->bhte", q, S)
    # Denominator: (B, H, T, D_k) * (B, H, 1, D_k) -> (B, H, T, 1)
    den = (torch.einsum("bhtd,bhd->bht", q, z) + eps).unsqueeze(-1)

    return num / den   # (B, H, T, D_v)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    """Multi-head linear attention with configurable kernel feature map."""

    def __init__(self, cfg: LinearAttnConfig) -> None:
        super().__init__()
        self.cfg = cfg

        inner_dim = cfg.n_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, inner_dim, bias=False)
        self.out_proj = nn.Linear(inner_dim, cfg.d_model, bias=False)

        if cfg.feature_map == "random_fourier":
            # Use n_features == head_dim so output dimensionality is unchanged
            n_feat = cfg.head_dim  # simplification: keep D constant
            omega_data = torch.randn(cfg.head_dim, n_feat) * 0.01
            bias_data = torch.rand(n_feat) * 2 * math.pi
            self.omega = nn.Parameter(omega_data, requires_grad=False)
            self.bias = nn.Parameter(bias_data, requires_grad=False)
        else:
            self.omega = None
            self.bias = None

    def apply_feature_map(self, x: Tensor) -> Tensor:
        """Apply configured feature map to (..., head_dim) tensor."""
        fm = self.cfg.feature_map
        if fm == "elu":
            return elu_feature_map(x)
        elif fm == "relu":
            return relu_feature_map(x)
        elif fm == "random_fourier":
            return random_fourier_features(x, self.omega, self.bias)
        else:
            raise ValueError(f"Unknown feature_map: {fm!r}")

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D) -> (B, T, D)."""
        B, T, D = x.shape
        H = self.cfg.n_heads
        hd = self.cfg.head_dim

        # Project and reshape to (B, H, T, head_dim)
        def _proj_reshape(proj: nn.Linear) -> Tensor:
            out = proj(x)                      # (B, T, H*hd)
            out = out.view(B, T, H, hd)        # (B, T, H, hd)
            return out.permute(0, 2, 1, 3)     # (B, H, T, hd)

        q = _proj_reshape(self.q_proj)
        k = _proj_reshape(self.k_proj)
        v = _proj_reshape(self.v_proj)

        # Apply feature maps to Q and K
        q = self.apply_feature_map(q)   # (B, H, T, hd)
        k = self.apply_feature_map(k)   # (B, H, T, hd)

        # Attention
        if self.cfg.causal:
            y = linear_attention_causal(q, k, v, eps=self.cfg.eps)
        else:
            y = linear_attention_noncausal(q, k, v, eps=self.cfg.eps)

        # y: (B, H, T, hd) -> (B, T, H*hd)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, H * hd)
        return self.out_proj(y)   # (B, T, D)


# ---------------------------------------------------------------------------
# Complexity helper
# ---------------------------------------------------------------------------

def compute_linear_attention_complexity(T: int, D: int, H: int) -> dict[str, int]:
    """Return ops counts for linear vs standard attention.

    linear_ops  = T * D * D * H   (O(T·D²·H) for the sequential scan)
    standard_ops = T * T * D * H  (O(T²·D·H) for full softmax attention)
    """
    return {
        "linear_ops": T * D * D * H,
        "standard_ops": T * T * D * H,
    }


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class LinearAttentionBlock(nn.Module):
    """Pre-norm transformer block using linear attention."""

    def __init__(self, cfg: LinearAttnConfig) -> None:
        super().__init__()
        self.attn = LinearAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pre-norm residual: x + attn(norm1(x)), x + ffn(norm2(x))."""
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
