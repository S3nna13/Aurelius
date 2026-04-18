"""Linear Attention variants v3: RetNet-style Retention and RWKV-style recurrent attention.

Implements:
  - LinearAttention      : O(T) causal linear attention with ELU or RBF feature maps
  - RetentionHead        : single retention head with gamma decay
  - MultiScaleRetention  : multi-head retention (RetNet)
  - RWKVTimeMix          : RWKV WKV time-mixing layer
  - LinearTransformerBlock: block combining any of the above + FFN
  - LinearAttentionConfig : dataclass of hyperparameters

References:
  Sun et al. (2023) "Retentive Network: A Successor to Transformer for LLMs"
  Peng et al. (2023) "RWKV: Reinventing RNNs for the Transformer Era"
  Katharopoulos et al. (2020) "Transformers are RNNs"
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LinearAttentionConfig:
    d_model: int = 32
    n_heads: int = 4
    n_layers: int = 2
    attention_type: str = "retention"   # "retention" | "linear" | "rwkv"
    gamma_min: float = 0.9
    gamma_max: float = 0.999
    ffn_mult: int = 4
    eps: float = 1e-6


# ---------------------------------------------------------------------------
# Feature maps for LinearAttention
# ---------------------------------------------------------------------------

def _elu_feature_map(x: Tensor) -> Tensor:
    """ELU+1: always strictly positive."""
    return F.elu(x) + 1.0


def _rbf_feature_map(x: Tensor, omega: Tensor, bias: Tensor) -> Tensor:
    """Random Fourier Features approximating RBF kernel.

    x: (..., d)  omega: (d, n_feat)  bias: (n_feat,)  -> (..., n_feat)
    """
    proj = x @ omega + bias
    scale = math.sqrt(2.0 / omega.shape[1])
    return torch.cos(proj) * scale


# ---------------------------------------------------------------------------
# LinearAttention
# ---------------------------------------------------------------------------

class LinearAttention(nn.Module):
    """Multi-head causal linear attention.

    Input q/k/v: [B, T, n_heads, d_head]
    Output:       [B, T, n_heads, d_head]

    State update: S_t = S_{t-1} + phi(k_t)^T v_t
                  z_t = z_{t-1} + phi(k_t)
    Output:       out_t = phi(q_t) S_t / (phi(q_t) . z_t + eps)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        feature_map: str = "elu",
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.feature_map = feature_map
        self.eps = eps

        if feature_map == "rbf":
            # Random Fourier Features — fixed non-trainable projection
            n_feat = self.d_head
            omega_data = torch.randn(self.d_head, n_feat)
            bias_data = torch.rand(n_feat) * 2.0 * math.pi
            self.register_buffer("_omega", omega_data)
            self.register_buffer("_bias", bias_data)
        else:
            self._omega: Optional[Tensor] = None
            self._bias: Optional[Tensor] = None

    def _phi(self, x: Tensor) -> Tensor:
        """Apply feature map to [..., d_head] tensor."""
        if self.feature_map == "elu":
            return _elu_feature_map(x)
        elif self.feature_map == "rbf":
            return _rbf_feature_map(x, self._omega, self._bias)
        else:
            raise ValueError(f"Unknown feature_map: {self.feature_map!r}")

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Causal linear attention via sequential O(T) recurrent computation.

        Args:
            q, k, v: [B, T, n_heads, d_head]
        Returns:
            [B, T, n_heads, d_head]
        """
        B, T, H, D = q.shape
        assert H == self.n_heads and D == self.d_head

        phi_q = self._phi(q)   # [B, T, H, D]
        phi_k = self._phi(k)   # [B, T, H, D]

        # S: [B, H, D, D]  — KV outer-product state
        # z: [B, H, D]     — key normaliser
        S = q.new_zeros(B, H, D, D)
        z = q.new_zeros(B, H, D)

        outputs = []
        for t in range(T):
            k_t = phi_k[:, t, :, :]   # [B, H, D]
            v_t = v[:, t, :, :]       # [B, H, D]
            q_t = phi_q[:, t, :, :]   # [B, H, D]

            # Outer product: k_t^T v_t  -> [B, H, D, D]
            S = S + torch.einsum("bhd,bhe->bhde", k_t, v_t)
            z = z + k_t

            # out_t = phi(q_t) S_t / (phi(q_t) . z_t + eps)
            num = torch.einsum("bhd,bhde->bhe", q_t, S)           # [B, H, D]
            den = (torch.einsum("bhd,bhd->bh", q_t, z) + self.eps).unsqueeze(-1)  # [B, H, 1]
            outputs.append((num / den).unsqueeze(1))               # [B, 1, H, D]

        return torch.cat(outputs, dim=1)   # [B, T, H, D]

    def parallel_forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Parallel prefix-sum producing the same result as forward().

        Uses cumulative outer-product sums — equivalent result without the
        Python time-step loop.

        Args:
            q, k, v: [B, T, n_heads, d_head]
        Returns:
            [B, T, n_heads, d_head]
        """
        B, T, H, D = q.shape
        phi_q = self._phi(q)   # [B, T, H, D]
        phi_k = self._phi(k)   # [B, T, H, D]

        # outer: [B, T, H, D, D]
        outer = torch.einsum("bthd,bthe->bthde", phi_k, v)

        # Cumulative prefix sum along T axis
        prefix_S = torch.cumsum(outer, dim=1)   # [B, T, H, D, D]
        prefix_z = torch.cumsum(phi_k, dim=1)   # [B, T, H, D]

        # num: phi_q . S_t  -> [B, T, H, D]
        num = torch.einsum("bthd,bthde->bthe", phi_q, prefix_S)
        # den: phi_q . z_t  -> [B, T, H]
        den = (torch.einsum("bthd,bthd->bth", phi_q, prefix_z) + self.eps).unsqueeze(-1)

        return num / den   # [B, T, H, D]


# ---------------------------------------------------------------------------
# RetentionHead
# ---------------------------------------------------------------------------

class RetentionHead(nn.Module):
    """Single retention head with exponential decay factor gamma.

    Retention recurrence:
      s_t = gamma * s_{t-1} + k_t^T v_t
      out_t = q_t s_t
    """

    def __init__(self, d_head: int, gamma: float) -> None:
        super().__init__()
        self.d_head = d_head
        self.gamma = gamma

    def forward_recurrent(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Step-by-step recurrent retention.

        Args:
            q, k, v: [B, T, d_head]
            state: optional [B, d_head, d_head] initial state (default zeros)
        Returns:
            (out, new_state)  — out: [B, T, d_head], new_state: [B, d_head, d_head]
        """
        B, T, D = q.shape
        if state is None:
            state = q.new_zeros(B, D, D)

        outputs = []
        for t in range(T):
            k_t = k[:, t, :]   # [B, D]
            v_t = v[:, t, :]   # [B, D]
            q_t = q[:, t, :]   # [B, D]

            # s_t = gamma * s_{t-1} + k_t^T v_t
            outer = torch.einsum("bd,be->bde", k_t, v_t)  # [B, D, D]
            state = self.gamma * state + outer

            # out_t = q_t s_t
            out_t = torch.einsum("bd,bde->be", q_t, state)  # [B, D]
            outputs.append(out_t.unsqueeze(1))

        out = torch.cat(outputs, dim=1)   # [B, T, D]
        return out, state

    def forward_parallel(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """Parallel retention using causal decay mask.

        D_{ij} = gamma^(i-j) if i >= j else 0

        Args:
            q, k, v: [B, T, d_head]
        Returns:
            [B, T, d_head]
        """
        B, T, D = q.shape
        device = q.device
        dtype = q.dtype

        # Build causal decay mask: [T, T]
        idx = torch.arange(T, device=device, dtype=dtype)
        diff = idx.unsqueeze(1) - idx.unsqueeze(0)   # [T, T]  row - col
        mask = (diff >= 0).to(dtype)
        decay = (self.gamma ** diff.clamp(min=0)) * mask   # [T, T]

        # QK^T: [B, T, T]
        qk = torch.bmm(q, k.transpose(1, 2))

        scale = math.sqrt(D)
        attn = (qk * decay.unsqueeze(0)) / scale   # [B, T, T]

        return torch.bmm(attn, v)   # [B, T, D]


# ---------------------------------------------------------------------------
# MultiScaleRetention
# ---------------------------------------------------------------------------

class MultiScaleRetention(nn.Module):
    """Multi-head retention (RetNet).

    Each head has a different gamma: gamma_i = 1 - 2^(-5 - i).
    Output is normalised with GroupNorm and projected.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Per-head decay factors: gamma_i = 1 - 2^(-5 - i)
        gammas = [1.0 - 2.0 ** (-5.0 - i) for i in range(n_heads)]
        self.heads = nn.ModuleList(
            [RetentionHead(self.d_head, g) for g in gammas]
        )

        # Projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # GroupNorm for output normalisation across heads
        self.group_norm = nn.GroupNorm(n_heads, d_model)

    def _split_heads(self, x: Tensor) -> Tensor:
        """[B, T, d_model] -> [B, T, n_heads, d_head]."""
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.d_head)

    def forward(self, x: Tensor, use_recurrent: bool = False) -> Tensor:
        """Forward pass over full sequence.

        Args:
            x: [B, T, d_model]
            use_recurrent: use recurrent scan (slower) vs parallel (faster)
        Returns:
            [B, T, d_model]
        """
        B, T, _ = x.shape
        q = self._split_heads(self.W_q(x))   # [B, T, H, D]
        k = self._split_heads(self.W_k(x))   # [B, T, H, D]
        v = self._split_heads(self.W_v(x))   # [B, T, H, D]

        head_outs = []
        for h, head in enumerate(self.heads):
            q_h = q[:, :, h, :]   # [B, T, d_head]
            k_h = k[:, :, h, :]
            v_h = v[:, :, h, :]
            if use_recurrent:
                out_h, _ = head.forward_recurrent(q_h, k_h, v_h)
            else:
                out_h = head.forward_parallel(q_h, k_h, v_h)
            head_outs.append(out_h)   # [B, T, d_head]

        # Concatenate heads: [B, T, d_model]
        y = torch.cat(head_outs, dim=-1)

        # GroupNorm expects [B, C, *] — reshape: [B, d_model, T]
        y = y.transpose(1, 2)
        y = self.group_norm(y)
        y = y.transpose(1, 2)   # [B, T, d_model]

        return self.W_o(y)

    def forward_recurrent(
        self,
        x: Tensor,
        states: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Autoregressive recurrent forward (one or more time steps).

        Args:
            x: [B, T, d_model]
            states: list of n_heads tensors [B, d_head, d_head], or None
        Returns:
            (out [B, T, d_model], new_states list)
        """
        if states is None:
            states = [None] * self.n_heads  # type: ignore[list-item]

        q = self._split_heads(self.W_q(x))   # [B, T, H, D]
        k = self._split_heads(self.W_k(x))   # [B, T, H, D]
        v = self._split_heads(self.W_v(x))   # [B, T, H, D]

        head_outs: List[Tensor] = []
        new_states: List[Tensor] = []
        for h, head in enumerate(self.heads):
            q_h = q[:, :, h, :]
            k_h = k[:, :, h, :]
            v_h = v[:, :, h, :]
            out_h, new_s = head.forward_recurrent(q_h, k_h, v_h, state=states[h])
            head_outs.append(out_h)
            new_states.append(new_s)

        y = torch.cat(head_outs, dim=-1)   # [B, T, d_model]
        y = y.transpose(1, 2)
        y = self.group_norm(y)
        y = y.transpose(1, 2)

        return self.W_o(y), new_states


# ---------------------------------------------------------------------------
# RWKVTimeMix
# ---------------------------------------------------------------------------

class RWKVTimeMix(nn.Module):
    """RWKV-style time-mixing layer.

    WKV recurrence (numerically-stable sequential scan):
        wkv_t = (sum_{i<t} exp(-(t-1-i)*w + k_i) * v_i + exp(u + k_t) * v_t)
                / (sum_{i<t} exp(-(t-1-i)*w + k_i) + exp(u + k_t))
    """

    def __init__(self, d_model: int, layer_id: int = 0) -> None:
        super().__init__()
        self.d_model = d_model
        self.layer_id = layer_id

        # Learnable interpolation coefficients (time-mixing)
        self.time_mix_k = nn.Parameter(torch.ones(d_model))
        self.time_mix_v = nn.Parameter(torch.ones(d_model))
        self.time_mix_r = nn.Parameter(torch.ones(d_model))

        # Linear projections
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # time_decay: negative values for stability (per-channel)
        self.time_decay = nn.Parameter(-torch.ones(d_model))

        # time_first: bonus for current token
        self.time_first = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: Tensor) -> Tensor:
        """RWKV time-mixing forward pass.

        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        B, T, D = x.shape

        # Shift x by one step: x_prev[t] = x[t-1] (zero-padded at t=0)
        x_prev = F.pad(x[:, :-1, :], (0, 0, 1, 0))   # [B, T, D]

        # Time-mixed inputs for k, v, r
        xk = x * self.time_mix_k + x_prev * (1.0 - self.time_mix_k)
        xv = x * self.time_mix_v + x_prev * (1.0 - self.time_mix_v)
        xr = x * self.time_mix_r + x_prev * (1.0 - self.time_mix_r)

        k = self.W_k(xk)                       # [B, T, D]
        v = self.W_v(xv)                       # [B, T, D]
        r = torch.sigmoid(self.W_r(xr))        # [B, T, D] gate

        # WKV computation via sequential scan with log-sum-exp trick.
        # Maintain running log-normaliser (p) and scaled numerator sum (q).
        w = self.time_decay   # [D]  (negative encouraged)
        u = self.time_first   # [D]

        p = torch.full((B, D), float('-inf'), device=x.device, dtype=x.dtype)
        q = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        wkv_out = []
        for t in range(T):
            k_t = k[:, t, :]   # [B, D]
            v_t = v[:, t, :]   # [B, D]

            # Current-token contribution (bonus u)
            new_p = torch.maximum(p, u + k_t)
            scale_old = torch.exp(torch.clamp(p - new_p, min=-30.0))
            scale_cur = torch.exp(torch.clamp(u + k_t - new_p, min=-30.0))

            num_t = scale_old * q + scale_cur * v_t
            den_t = scale_old + scale_cur

            wkv = num_t / (den_t + 1e-9)   # [B, D]
            wkv_out.append(wkv.unsqueeze(1))

            # Update running state: decay by w, add k_t / v_t contribution
            new_p2 = torch.maximum(p + w, k_t)
            scale_prev = torch.exp(torch.clamp(p + w - new_p2, min=-30.0))
            scale_new = torch.exp(torch.clamp(k_t - new_p2, min=-30.0))
            q = scale_prev * q + scale_new * v_t
            p = new_p2

        wkv = torch.cat(wkv_out, dim=1)   # [B, T, D]

        # Gated output
        y = r * wkv
        return self.W_o(y)


# ---------------------------------------------------------------------------
# LinearTransformerBlock
# ---------------------------------------------------------------------------

class LinearTransformerBlock(nn.Module):
    """Pre-norm transformer block with configurable linear attention type.

    attention_type:
      "linear"    — LinearAttention with ELU feature map + explicit Q/K/V/O
      "retention" — MultiScaleRetention
      "rwkv"      — RWKVTimeMix
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_type: str = "retention",
        layer_id: int = 0,
        ffn_mult: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.attention_type = attention_type
        self.d_model = d_model
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        if attention_type == "linear":
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            self.W_v = nn.Linear(d_model, d_model, bias=False)
            self.W_o = nn.Linear(d_model, d_model, bias=False)
            self._attn = LinearAttention(d_model, n_heads, feature_map="elu", eps=eps)
        elif attention_type == "retention":
            self._attn = MultiScaleRetention(d_model, n_heads)
        elif attention_type == "rwkv":
            self._attn = RWKVTimeMix(d_model, layer_id=layer_id)
        else:
            raise ValueError(f"Unknown attention_type: {attention_type!r}")

        ffn_hidden = ffn_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Pre-norm residual forward.

        Args:
            x: [B, T, d_model]
        Returns:
            [B, T, d_model]
        """
        h = self.norm1(x)

        if self.attention_type == "linear":
            B, T, D = h.shape
            H = self.n_heads
            dh = D // H

            def _split(proj: nn.Linear) -> Tensor:
                return proj(h).view(B, T, H, dh)

            q = _split(self.W_q)
            k = _split(self.W_k)
            v = _split(self.W_v)
            attn_out = self._attn(q, k, v).view(B, T, D)
            attn_out = self.W_o(attn_out)
        elif self.attention_type == "retention":
            attn_out = self._attn(h)
        else:  # rwkv
            attn_out = self._attn(h)

        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
