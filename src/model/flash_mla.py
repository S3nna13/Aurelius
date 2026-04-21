"""FlashMLA -- Fused Absorbed-KV Multi-head Latent Attention (DeepSeek 2025).

Key insight: Instead of projecting K,V from low-rank back to full head_dim
before attention, absorb the up-projections into Q's attention computation.
The absorbed query for head h is:

    Q_abs_h = x @ W_q_h^T @ W_k_h   (shape [B, T, kv_lrank])

Scores for head h: Q_abs_h @ c^T  where c is the KV-compressed representation.
This is mathematically identical to the standard path's Q[h] @ K[h]^T because:

    Q[h] @ K[h]^T  =  (x @ W_q_h^T) @ (c @ W_k_h^T)^T
                    =  x @ W_q_h^T @ W_k_h @ c^T
                    =  (x @ (W_q_h^T @ W_k_h)) @ c^T
                    =  Q_abs_h @ c^T

Cache benefit: at inference we store c (kv_lrank floats/token) instead of
K and V separately (2 * n_heads * head_dim floats/token).  The up-projections
are absorbed into the pre-multiplied per-head query matrices.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class FlashMLAConfig:
    d_model: int = 2048
    n_heads: int = 16
    head_dim: int = 128
    kv_lrank: int = 512       # low-rank dim for KV compression
    q_lrank: int = 1536       # low-rank dim for Q (optional, informational)
    rope_dim: int = 64        # dimensions for RoPE (decoupled, reserved for future use)
    dropout: float = 0.0


class FlashMLAAttention(nn.Module):
    """FlashMLA attention with optional absorbed-projection inference path.

    Standard path (use_absorbed=False):
        1. c = kv_down(x)                                # [B, T, kv_lrank]
        2. K = k_up(c)    reshaped to heads              # [B, n_heads, T, head_dim]
        3. V = v_up(c)    reshaped to heads              # [B, n_heads, T, head_dim]
        4. Q = q_proj(x)  reshaped to heads              # [B, n_heads, T, head_dim]
        5. attn(Q, K, V) -> out_proj

    Absorbed path (use_absorbed=True, after absorb_projections()):
        absorbed_qk[h] = W_q_h^T @ W_k_h               # [d_model, kv_lrank] per head
        Stored as absorbed_qk: [n_heads, d_model, kv_lrank]

        1. c = kv_down(x)                               # [B, T, kv_lrank]
        2. Q_abs[h] = x @ absorbed_qk[h]               # [B, T, kv_lrank]  per head
        3. scores[h] = Q_abs[h] @ c^T * scale          # [B, T, T]  identical to standard
        4. V = v_up(c)  reshaped                        # [B, n_heads, T, head_dim]
        5. out = softmax(scores) @ V  -> out_proj

    Both paths produce the same numerical output (within floating-point tolerance).
    """

    def __init__(self, cfg: FlashMLAConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.kv_lrank = cfg.kv_lrank
        self.scale = cfg.head_dim ** -0.5

        # Core projections
        self.kv_down = nn.Linear(cfg.d_model, cfg.kv_lrank, bias=False)
        self.k_up = nn.Linear(cfg.kv_lrank, cfg.n_heads * cfg.head_dim, bias=False)
        self.v_up = nn.Linear(cfg.kv_lrank, cfg.n_heads * cfg.head_dim, bias=False)
        self.q_proj = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.out_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

        # Absorbed projection buffer (populated by absorb_projections()).
        # Shape: [n_heads, d_model, kv_lrank]  -- per-head fused Q+K matrix.
        # None until absorb_projections() is called.
        self.register_buffer("absorbed_qk", None, persistent=False)

        self.attn_dropout = nn.Dropout(cfg.dropout)

    # ------------------------------------------------------------------
    # KV compression
    # ------------------------------------------------------------------

    def _compress_kv(self, x: Tensor) -> Tensor:
        """Compress x into the KV low-rank space.

        Args:
            x: [B, T, d_model]

        Returns:
            compressed_kv: [B, T, kv_lrank]
        """
        return self.kv_down(x)

    # ------------------------------------------------------------------
    # Projection absorption
    # ------------------------------------------------------------------

    def absorb_projections(self) -> None:
        """Pre-multiply each head's W_q_h with W_k_h and store as absorbed_qk.

        For head h:
            W_q_h: [head_dim, d_model]   (slice of q_proj.weight)
            W_k_h: [head_dim, kv_lrank]  (slice of k_up.weight)
            absorbed_qk[h] = W_q_h^T @ W_k_h  =  [d_model, kv_lrank]

        Stored as absorbed_qk: [n_heads, d_model, kv_lrank].
        After this call, use_absorbed=True in forward() uses cached matrices.
        Calling this method multiple times is safe (idempotent).
        """
        with torch.no_grad():
            # q_proj.weight: [n_heads*head_dim, d_model]
            # k_up.weight:   [n_heads*head_dim, kv_lrank]
            Wq = self.q_proj.weight.view(self.n_heads, self.head_dim, self.cfg.d_model)
            Wk = self.k_up.weight.view(self.n_heads, self.head_dim, self.kv_lrank)
            # Per-head: absorbed[h] = W_q_h^T @ W_k_h
            # Wq[h]: [head_dim, d_model] -> Wq[h].T: [d_model, head_dim]
            # Wk[h]: [head_dim, kv_lrank]
            # [d_model, head_dim] @ [head_dim, kv_lrank] = [d_model, kv_lrank]
            absorbed = torch.einsum("hid,hik->hdk", Wq, Wk)
            # absorbed: [n_heads, d_model, kv_lrank]
        self.absorbed_qk = absorbed.detach()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor, use_absorbed: bool = False) -> Tensor:
        """Compute FlashMLA attention.

        Args:
            x: [B, T, d_model]
            use_absorbed: If True, use absorbed QK matrices (requires prior
                          call to absorb_projections()). Both paths produce
                          the same output to floating-point tolerance.

        Returns:
            output: [B, T, d_model]
        """
        B, T, _ = x.shape
        c = self._compress_kv(x)  # [B, T, kv_lrank]

        if use_absorbed:
            return self._forward_absorbed(x, c, B, T)
        return self._forward_standard(x, c, B, T)

    def _forward_standard(self, x: Tensor, c: Tensor, B: int, T: int) -> Tensor:
        """Standard MLA: expand K,V from compressed c then compute attention."""
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_up(c).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_up(c).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.cfg.dropout if self.training else 0.0,
            is_causal=True,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)

    def _forward_absorbed(self, x: Tensor, c: Tensor, B: int, T: int) -> Tensor:
        """Absorbed path: each head queries directly in KV-compressed space.

        Mathematical identity (per head h):
            Q[h] @ K[h]^T = (x @ W_q_h^T) @ (c @ W_k_h^T)^T
                           = x @ (W_q_h^T @ W_k_h) @ c^T
                           = Q_abs_h @ c^T

        Where Q_abs_h = x @ absorbed_qk[h], shape [B, T, kv_lrank].
        The scale for the dot product is the same head_dim^(-0.5) as standard
        (the absorbed scores have the same magnitude as the standard ones since
        we use W_q_h^T @ W_k_h -- not a re-scaled version).
        """
        if self.absorbed_qk is None:
            raise RuntimeError(
                "absorb_projections() must be called before use_absorbed=True."
            )

        # absorbed_qk: [n_heads, d_model, kv_lrank]
        # x:           [B, T, d_model]
        # Q_abs:       [B, n_heads, T, kv_lrank]
        Q_abs = torch.einsum("btd,hdk->bhtk", x, self.absorbed_qk)

        # c as K in compressed space: [B, T, kv_lrank] -> [B, 1, T, kv_lrank]
        # broadcast across heads
        c_heads = c.unsqueeze(1)  # [B, 1, T, kv_lrank]

        # Attention scores: [B, n_heads, T, T]
        # scale is the same as standard (head_dim-based) because:
        #   Var(Q_abs[h]) = Var(x @ W_q_h^T @ W_k_h) which is comparable to
        #   Var(Q[h]) * kv_lrank ... so we preserve the same scale as standard.
        scores = torch.matmul(Q_abs, c_heads.transpose(-2, -1)) * self.scale

        # Causal mask
        causal_mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)

        if self.training and self.cfg.dropout > 0.0:
            attn_weights = self.attn_dropout(attn_weights)

        # V still needs full expansion
        V = self.v_up(c).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        out = torch.matmul(attn_weights, V)  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out_proj(out)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def kv_cache_size_ratio(self) -> float:
        """Return kv_lrank / (n_heads * head_dim).

        FlashMLA caches kv_lrank floats per token vs n_heads * head_dim for
        standard MHA (per K or V).  A ratio < 1.0 means genuine compression.
        """
        return self.kv_lrank / (self.n_heads * self.head_dim)
