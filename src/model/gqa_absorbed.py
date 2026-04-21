"""GQA with Absorbed KV Projection (2025).

In standard GQA, W_k and W_v are [n_kv_heads, head_dim, d_model] — each token
generates n_kv_heads key/value vectors.  The "absorbed" variant precomputes
W_qk = W_q @ W_k^T for each query-head group so attention can be computed
directly in a lower-dimensional space, reducing KV bandwidth.

Reference: Absorbed-GQA derivation (2025).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class GQAAbsorbedConfig:
    """Configuration for GQA with optional absorbed KV projection.

    Attributes:
        d_model:      Model (embedding) dimension.
        n_heads:      Number of query heads.
        n_kv_heads:   Number of key/value heads.  ``n_heads`` must be
                      divisible by ``n_kv_heads``.
        head_dim:     Per-head feature dimension.
        dropout:      Attention dropout probability (applied during training).
        use_absorbed: When *True* the forward pass uses the absorbed KV path;
                      otherwise the standard GQA expand-and-repeat path is
                      used.
    """

    d_model: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 128
    dropout: float = 0.0
    use_absorbed: bool = False

    def __post_init__(self) -> None:
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by "
                f"n_kv_heads ({self.n_kv_heads})."
            )

    @property
    def groups_per_kv(self) -> int:
        """Number of Q heads that share each KV head."""
        return self.n_heads // self.n_kv_heads


# ---------------------------------------------------------------------------
# GQAAbsorbedAttention
# ---------------------------------------------------------------------------


class GQAAbsorbedAttention(nn.Module):
    """Grouped Query Attention with optional absorbed KV projection.

    When ``cfg.use_absorbed`` is *False* (default) the module behaves like
    standard GQA: KV heads are expanded via ``repeat_interleave`` to match
    the number of Q heads.

    When ``cfg.use_absorbed`` is *True* (or after calling :meth:`absorb`) the
    module pre-multiplies each group's query weights with the corresponding KV
    weight matrices so that attention is computed entirely in the
    n_kv_heads × head_dim space, saving KV bandwidth.

    Both paths produce numerically equivalent outputs (to floating-point
    tolerance) for the same set of projection weights.

    Args:
        cfg: :class:`GQAAbsorbedConfig` instance.
    """

    def __init__(self, cfg: GQAAbsorbedConfig) -> None:
        super().__init__()
        self.cfg = cfg

        q_dim = cfg.n_heads * cfg.head_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim

        self.q_proj = nn.Linear(cfg.d_model, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.out_proj = nn.Linear(q_dim, cfg.d_model, bias=False)

        self.attn_drop = nn.Dropout(cfg.dropout)
        self._scale = math.sqrt(cfg.head_dim)

        # Buffers populated by absorb() — shape (n_kv_heads, groups_per_kv, head_dim, head_dim)
        self.register_buffer("_absorbed_qk", None, persistent=False)
        self.register_buffer("_absorbed_qv", None, persistent=False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def absorb(self) -> None:
        """Precompute and cache the absorbed projection matrices.

        For each KV head *g* and each of its ``groups_per_kv`` Q heads *h*::

            absorbed_qk[g, h] = W_q[g*groups + h] @ W_k[g].T   (head_dim × head_dim)
            absorbed_qv[g, h] = W_q[g*groups + h] @ W_v[g].T   (head_dim × head_dim)

        where W_q is shaped (n_heads, head_dim, d_model) and W_k / W_v are
        (n_kv_heads, head_dim, d_model).

        After calling this method the module stores the results as non-persistent
        buffers named ``_absorbed_qk`` and ``_absorbed_qv``.
        """
        cfg = self.cfg
        G = cfg.groups_per_kv

        # (n_heads, head_dim, d_model)
        Wq = self.q_proj.weight.view(cfg.n_heads, cfg.head_dim, cfg.d_model)
        # (n_kv_heads, head_dim, d_model)
        Wk = self.k_proj.weight.view(cfg.n_kv_heads, cfg.head_dim, cfg.d_model)
        Wv = self.v_proj.weight.view(cfg.n_kv_heads, cfg.head_dim, cfg.d_model)

        # Reshape Wq → (n_kv_heads, groups_per_kv, head_dim, d_model)
        Wq_grouped = Wq.view(cfg.n_kv_heads, G, cfg.head_dim, cfg.d_model)

        # absorbed_qk[g, h] = Wq_grouped[g, h] @ Wk[g].T  → (head_dim, head_dim)
        # einsum: (nkv, G, D, d) x (nkv, D, d) → (nkv, G, D, D)
        absorbed_qk = torch.einsum("nghd,gkd->nghk", Wq_grouped, Wk)
        absorbed_qv = torch.einsum("nghd,gkd->nghk", Wq_grouped, Wv)

        self._absorbed_qk = absorbed_qk.detach()
        self._absorbed_qv = absorbed_qv.detach()

    def kv_heads_ratio(self) -> float:
        """KV bandwidth reduction ratio: n_kv_heads / n_heads."""
        return self.cfg.n_kv_heads / self.cfg.n_heads

    # ------------------------------------------------------------------
    # Attention paths
    # ------------------------------------------------------------------

    def _standard_gqa(self, x: Tensor) -> Tensor:
        """Standard GQA forward: expand KV heads via repeat_interleave.

        Args:
            x: ``(B, T, d_model)``

        Returns:
            ``(B, T, d_model)``
        """
        cfg = self.cfg
        B, T, _ = x.shape

        # Project & reshape to (B, H, T, head_dim)
        q = self.q_proj(x).view(B, T, cfg.n_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim).transpose(1, 2)

        # Expand KV heads: (B, n_kv_heads, T, D) → (B, n_heads, T, D)
        if cfg.groups_per_kv > 1:
            k = k.repeat_interleave(cfg.groups_per_kv, dim=1)
            v = v.repeat_interleave(cfg.groups_per_kv, dim=1)

        # Scaled dot-product
        attn = torch.matmul(q, k.transpose(-2, -1)) / self._scale  # (B, H, T, T)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = out.transpose(1, 2).contiguous().view(B, T, cfg.n_heads * cfg.head_dim)
        return self.out_proj(out)

    def _absorbed_gqa(self, x: Tensor) -> Tensor:
        """Absorbed GQA forward: attention computed in reduced KV space.

        For each KV head group *g* with Q-head slice *[g*G : (g+1)*G]*:

        1. Project x → keys  ``k_g = x @ W_k[g].T``  shape ``(B, T, head_dim)``
        2. Project x → values ``v_g = x @ W_v[g].T``  shape ``(B, T, head_dim)``
        3. For each Q head *h* in group *g*:
               ``q_h  = x @ W_q[g*G + h].T``              — shape ``(B, T, head_dim)``
               ``attn = softmax(q_h @ k_g.T / scale)``     — shape ``(B, T, T)``
               ``out_h = attn @ v_g``                       — shape ``(B, T, head_dim)``
        4. Concatenate all heads → out_proj

        This produces numerically identical results to :meth:`_standard_gqa`.

        Args:
            x: ``(B, T, d_model)``

        Returns:
            ``(B, T, d_model)``
        """
        cfg = self.cfg
        B, T, _ = x.shape
        G = cfg.groups_per_kv

        # Full Q projection: (B, T, n_heads*head_dim)
        q_full = self.q_proj(x)
        # Reshape: (B, T, n_kv_heads, G, head_dim)
        q_grouped = q_full.view(B, T, cfg.n_kv_heads, G, cfg.head_dim)

        # K/V projections: (B, T, n_kv_heads, head_dim)
        k_kv = self.k_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim)
        v_kv = self.v_proj(x).view(B, T, cfg.n_kv_heads, cfg.head_dim)

        # Compute attention for each KV group
        # q_grouped: (B, T, nkv, G, D)
        # k_kv:      (B, T, nkv, D)  →  (B, nkv, T, D) for matmul
        k_kv_t = k_kv.permute(0, 2, 1, 3)  # (B, nkv, T, D)
        v_kv_t = v_kv.permute(0, 2, 1, 3)  # (B, nkv, T, D)

        # q_grouped: (B, nkv, G, T, D) — permute (0,2,3,1,4)→(B,nkv,G,T,D)
        q_perm = q_grouped.permute(0, 2, 3, 1, 4)  # (B, nkv, G, T, D)

        # Scores: (B, nkv, G, T, T)
        # q_perm @ k^T: (B, nkv, G, T, D) x (B, nkv, 1, D, T) → (B, nkv, G, T, T)
        scores = torch.matmul(q_perm, k_kv_t.unsqueeze(2).transpose(-2, -1)) / self._scale

        attn = F.softmax(scores, dim=-1)  # (B, nkv, G, T, T)
        attn = self.attn_drop(attn)

        # Weighted values: (B, nkv, G, T, T) x (B, nkv, 1, T, D) → (B, nkv, G, T, D)
        out_grouped = torch.matmul(attn, v_kv_t.unsqueeze(2))  # (B, nkv, G, T, D)

        # Merge back to (B, T, n_heads*head_dim)
        # (B, nkv, G, T, D) → (B, T, nkv, G, D) → (B, T, n_heads*head_dim)
        out = out_grouped.permute(0, 3, 1, 2, 4).contiguous()
        out = out.view(B, T, cfg.n_heads * cfg.head_dim)

        return self.out_proj(out)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Dispatch to absorbed or standard GQA based on ``cfg.use_absorbed``.

        Args:
            x: Input tensor of shape ``(B, T, d_model)``.

        Returns:
            Output tensor of shape ``(B, T, d_model)``.
        """
        if self.cfg.use_absorbed:
            return self._absorbed_gqa(x)
        return self._standard_gqa(x)
