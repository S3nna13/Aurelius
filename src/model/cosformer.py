"""cosFormer: Rethinking Softmax in Attention (Qin et al., arXiv:2202.08791).

Implements cosFormer, which replaces the softmax kernel with a cosine-based
decomposition to achieve linear-complexity O(T) attention.

The key idea: replace exp(QK^T/sqrt(d)) with a non-negative kernel
  phi(q)^T phi(k)  where  phi(x) = [relu(x), relu(-x)]
combined with a cosine position reweighting so adjacent tokens interact more
strongly than distant ones.

Non-causal forward pass (associativity trick):
  O = (Q' @ (K'^T @ V)) / (Q' @ K'.sum(dim=1, keepdim=True)^T + eps)

Causal forward pass: prefix-sum recurrence over time steps.

Classes
-------
CosformerKernel        — feature map phi(x) = cat([relu(x), relu(-x)])
CosformerAttention     — single-head cosformer attention
CosformerLayer         — multi-head cosformer (projects to heads, merges)
CosformerBlock         — CosformerLayer + FFN + RMSNorm pre-norm

References
----------
Qin et al. (2022) "cosFormer: Rethinking Softmax in Attention"
https://arxiv.org/abs/2202.08791
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Feature map
# ---------------------------------------------------------------------------

class CosformerKernel(nn.Module):
    """Positive-definite feature map phi(x) = [relu(x), relu(-x)].

    For input shape (..., d), output shape is (..., 2*d).  All output values
    are >= 0 by construction, which is required for the linear attention
    associativity trick to be numerically stable.
    """

    def forward(self, x: Tensor) -> Tensor:  # (..., d) -> (..., 2d)
        return torch.cat([F.relu(x), F.relu(-x)], dim=-1)


# ---------------------------------------------------------------------------
# Helper: cosine position reweighting
# ---------------------------------------------------------------------------

def _cos_sin_position_bias(T: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Return cos and sin position weights of shape (T, 1).

    cos_w[t] = cos(pi * t / (2 * T))
    sin_w[t] = sin(pi * t / (2 * T))

    Used to modulate Q (cos) and K (cos) so that the inner product
    <phi(q_m) * cos_m, phi(k_n) * cos_n> ∝ cos(pi*(m-n)/(2T)) for the
    dominant term, matching the paper's cosine reweighting.
    """
    t = torch.arange(T, device=device, dtype=dtype)  # (T,)
    theta = math.pi * t / (2.0 * max(T, 1))
    cos_w = torch.cos(theta).unsqueeze(-1)  # (T, 1)
    sin_w = torch.sin(theta).unsqueeze(-1)  # (T, 1)
    return cos_w, sin_w


# ---------------------------------------------------------------------------
# Single-head attention
# ---------------------------------------------------------------------------

class CosformerAttention(nn.Module):
    """Single-head cosFormer attention.

    Parameters
    ----------
    d_head : int
        Dimension of each head's Q/K/V vectors.
    eps : float
        Small constant for denominator normalisation.
    """

    def __init__(self, d_head: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.d_head = d_head
        self.eps = eps
        self.kernel = CosformerKernel()

    def forward(self, q: Tensor, k: Tensor, v: Tensor, causal: bool = False) -> Tensor:
        """Compute cosFormer attention.

        Parameters
        ----------
        q, k, v : Tensor of shape (B, T, d_head)
        causal  : bool — if True use prefix-sum recurrence (O(T) causal).

        Returns
        -------
        Tensor of shape (B, T, d_head)
        """
        B, T, _ = q.shape
        device, dtype = q.device, q.dtype

        # 1. Apply feature map: (B, T, 2*d_head)
        q_feat = self.kernel(q)  # (B, T, 2d)
        k_feat = self.kernel(k)  # (B, T, 2d)

        # 2. Cosine position reweighting
        cos_w, sin_w = _cos_sin_position_bias(T, device, dtype)
        # Modulate with cos for Q and K (produces cos(m-n) cross-term via
        # cos(m)*cos(n) + sin(m)*sin(n) = cos(m-n))
        q_cos = q_feat * cos_w  # (B, T, 2d)
        q_sin = q_feat * sin_w  # (B, T, 2d)
        k_cos = k_feat * cos_w  # (B, T, 2d)
        k_sin = k_feat * sin_w  # (B, T, 2d)

        # Full modulated Q' and K' each of shape (B, T, 4d) so that
        # <Q'_m, K'_n> includes cos(m-n) terms from the paper.
        q_prime = torch.cat([q_cos, q_sin], dim=-1)  # (B, T, 4d)
        k_prime = torch.cat([k_cos, k_sin], dim=-1)  # (B, T, 4d)

        if causal:
            return self._causal_forward(q_prime, k_prime, v)
        else:
            return self._noncausal_forward(q_prime, k_prime, v)

    # ------------------------------------------------------------------
    # Non-causal (full bidirectional)
    # ------------------------------------------------------------------

    def _noncausal_forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """O(T*(4d)^2) → effectively O(T*d*d) using associativity.

        O = (Q' @ (K'^T @ V)) / (Q' @ sum_k(K') + eps)
        """
        # kv : (B, 4d, d_head)  — aggregate across time first
        kv = torch.einsum("bti,btj->bij", k, v)   # (B, 4d, d_v)
        # numerator: (B, T, d_head)
        out_num = torch.einsum("bti,bij->btj", q, kv)

        # denominator: sum of K' over time -> (B, 1, 4d), then contract with Q'
        k_sum = k.sum(dim=1, keepdim=True)               # (B, 1, 4d)
        denom = (q * k_sum).sum(dim=-1, keepdim=True)    # (B, T, 1)
        denom = denom.clamp(min=self.eps)

        return out_num / denom  # (B, T, d_head)

    # ------------------------------------------------------------------
    # Causal (prefix-sum recurrence)
    # ------------------------------------------------------------------

    def _causal_forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Causal linear attention via O(T) prefix-sum recurrence.

        S_t = S_{t-1} + k_t^T v_t     (accumulated KV state, (4d, d_v))
        z_t = z_{t-1} + k_t            (accumulated key sum, (4d,))
        o_t = q_t @ S_t / (q_t . z_t)
        """
        B, T, feat_dim = q.shape
        d_v = v.shape[-1]
        device, dtype = q.device, q.dtype

        S = torch.zeros(B, feat_dim, d_v, device=device, dtype=dtype)  # (B, 4d, d_v)
        z = torch.zeros(B, feat_dim, device=device, dtype=dtype)        # (B, 4d)
        outputs = []

        for t in range(T):
            kt = k[:, t, :]  # (B, 4d)
            vt = v[:, t, :]  # (B, d_v)
            qt = q[:, t, :]  # (B, 4d)

            # Update state
            S = S + torch.einsum("bi,bj->bij", kt, vt)  # (B, 4d, d_v)
            z = z + kt                                    # (B, 4d)

            # Compute output for step t
            num = torch.einsum("bi,bij->bj", qt, S)      # (B, d_v)
            den = (qt * z).sum(dim=-1, keepdim=True)     # (B, 1)
            den = den.clamp(min=self.eps)
            outputs.append((num / den).unsqueeze(1))     # (B, 1, d_v)

        return torch.cat(outputs, dim=1)  # (B, T, d_v)


# ---------------------------------------------------------------------------
# Multi-head layer
# ---------------------------------------------------------------------------

class CosformerLayer(nn.Module):
    """Multi-head cosFormer attention layer.

    Projects d_model → (n_heads, d_head) for Q, K, V then merges.

    Parameters
    ----------
    d_model : int
    n_heads : int
    eps     : float
    """

    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-6) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.attn_heads = nn.ModuleList(
            [CosformerAttention(self.d_head, eps=eps) for _ in range(n_heads)]
        )

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        """
        Parameters
        ----------
        x : (B, T, d_model)

        Returns
        -------
        Tensor of shape (B, T, d_model)
        """
        B, T, _ = x.shape

        q = self.q_proj(x)  # (B, T, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: (B, T, n_heads, d_head) -> list of (B, T, d_head)
        q_heads = q.view(B, T, self.n_heads, self.d_head).unbind(dim=2)
        k_heads = k.view(B, T, self.n_heads, self.d_head).unbind(dim=2)
        v_heads = v.view(B, T, self.n_heads, self.d_head).unbind(dim=2)

        head_outputs = [
            self.attn_heads[h](q_heads[h], k_heads[h], v_heads[h], causal=causal)
            for h in range(self.n_heads)
        ]

        # Merge heads: (B, T, d_model)
        out = torch.cat(head_outputs, dim=-1)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# RMSNorm (local, no external deps)
# ---------------------------------------------------------------------------

class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


# ---------------------------------------------------------------------------
# Full block
# ---------------------------------------------------------------------------

class CosformerBlock(nn.Module):
    """CosformerLayer + FFN + RMSNorm (pre-norm style).

    Architecture (pre-norm):
        x = x + CosformerLayer(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Parameters
    ----------
    d_model : int
    n_heads : int
    d_ff    : int   — inner FFN dimension
    eps     : float
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(d_model, eps=eps)
        self.attn = CosformerLayer(d_model, n_heads, eps=eps)
        self.norm2 = _RMSNorm(d_model, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: Tensor, causal: bool = False) -> Tensor:
        x = x + self.attn(self.norm1(x), causal=causal)
        x = x + self.ffn(self.norm2(x))
        return x
