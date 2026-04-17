"""Token Merging (ToMe) adapted for LLMs.

Implements bipartite soft matching to reduce the number of tokens in each
transformer layer by merging the most-similar token pairs.

Reference:
    Bolya et al. (2022). "Token Merging: Your ViT But Faster".
    arXiv:2210.09461  https://arxiv.org/abs/2210.09461
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. BipartiteSoftMatching
# ---------------------------------------------------------------------------


class BipartiteSoftMatching:
    """Compute which token pairs to merge via bipartite soft matching.

    The sequence is split into two sets:
        A — tokens at even positions (0, 2, 4, …)
        B — tokens at odd  positions (1, 3, 5, …)

    For each token in A its cosine similarity with every token in B is
    computed.  The top-*r* (A, B) pairs by similarity score are selected for
    merging.

    Args:
        r: Number of token pairs to merge per forward pass.  Clamped to
           ``min(r, len(A))``, i.e. ``min(r, T // 2)``.
    """

    def __init__(self, r: int = 8) -> None:
        if r < 0:
            raise ValueError(f"r must be >= 0, got {r}")
        self.r = r

    # ------------------------------------------------------------------
    def match(self, metric: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select the top-r token pairs to merge.

        Args:
            metric: Key vectors of shape (B, T, d) used as the similarity
                    signal.  Cosine similarity is computed along the *d* axis.

        Returns:
            src: (B, r_eff) — indices of tokens in set A that will be merged.
            dst: (B, r_eff) — indices of tokens in set B that each src token
                 is merged into.

        ``r_eff = min(r, T // 2)``.  When r == 0 both tensors are empty
        (shape ``(B, 0)``).
        """
        B, T, d = metric.shape

        # Split into A (even) and B (odd)
        a_idx = torch.arange(0, T - T % 2, 2, device=metric.device)  # (|A|,)
        b_idx = torch.arange(1, T,          2, device=metric.device)  # (|B|,)

        n_a = a_idx.shape[0]
        n_b = b_idx.shape[0]

        r_eff = min(self.r, min(n_a, n_b))

        if r_eff == 0 or n_a == 0 or n_b == 0:
            empty = torch.zeros(B, 0, dtype=torch.long, device=metric.device)
            return empty, empty

        # Gather the metric vectors for each set
        a_vecs = metric[:, a_idx, :]  # (B, n_a, d)
        b_vecs = metric[:, b_idx, :]  # (B, n_b, d)

        # Normalise for cosine similarity
        a_norm = F.normalize(a_vecs, dim=-1)   # (B, n_a, d)
        b_norm = F.normalize(b_vecs, dim=-1)   # (B, n_b, d)

        # Similarity matrix: (B, n_a, n_b)
        sim = torch.bmm(a_norm, b_norm.transpose(1, 2))

        # For each token in A, pick the most similar token in B
        best_sim, best_b_local = sim.max(dim=2)   # (B, n_a)

        # Pick the top-r_eff pairs by similarity score
        _, top_a_local = best_sim.topk(r_eff, dim=1)  # (B, r_eff)

        # Convert local A / B indices back to sequence positions
        a_idx_expanded = a_idx.unsqueeze(0).expand(B, -1)  # (B, n_a)
        b_idx_expanded = b_idx.unsqueeze(0).expand(B, -1)  # (B, n_b)

        src = a_idx_expanded.gather(1, top_a_local)          # (B, r_eff)
        dst_local = best_b_local.gather(1, top_a_local)       # (B, r_eff)
        dst = b_idx_expanded.gather(1, dst_local)             # (B, r_eff)

        return src, dst


# ---------------------------------------------------------------------------
# 2. ToMeMerger
# ---------------------------------------------------------------------------


class ToMeMerger:
    """Merge and unmerge tokens according to src/dst index tensors.

    Merge: for each selected pair (src_i, dst_i), replace both tokens with a
    size-weighted average, then remove the src tokens — leaving dst positions
    with the merged value.  Unselected tokens pass through unchanged.

    Unmerge: scatter merged dst tokens back to both the src and dst positions
    in the original (longer) sequence.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def merge(
        x: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        size: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge r token pairs.

        Args:
            x:    (B, T, d) token representations.
            src:  (B, r) indices of tokens in set A.
            dst:  (B, r) indices of tokens in set B (merge target).
            size: (B, T) token size / weight tensor (how many original tokens
                  each current token represents).  If ``None`` defaults to
                  all-ones.

        Returns:
            x_merged:    (B, T - r, d)
            size_merged: (B, T - r)
        """
        B, T, d = x.shape
        r = src.shape[1]
        device = x.device

        if size is None:
            size = torch.ones(B, T, device=device, dtype=x.dtype)

        if r == 0:
            return x, size

        # Gather sizes for weighting (no in-place ops on differentiable tensors)
        src_size = size.gather(1, src)           # (B, r)
        dst_size = size.gather(1, dst)           # (B, r)
        total    = src_size + dst_size           # (B, r)

        # Weighted merge value
        src_vals = x.gather(2, src.unsqueeze(-1).expand(-1, -1, d))  # (B, r, d)
        dst_vals = x.gather(2, dst.unsqueeze(-1).expand(-1, -1, d))  # (B, r, d)
        merged = (
            src_vals * src_size.unsqueeze(-1) + dst_vals * dst_size.unsqueeze(-1)
        ) / total.unsqueeze(-1)  # (B, r, d)

        # Write merged value to dst positions via out-of-place scatter
        x_updated = x.scatter(
            2,
            dst.unsqueeze(-1).expand(-1, -1, d),
            merged,
        )  # (B, T, d) — new tensor, no in-place modification

        sz_updated = size.scatter(1, dst, total)  # (B, T) — new tensor

        # Build keep mask: True for positions NOT in src
        keep = torch.ones(B, T, dtype=torch.bool, device=device)
        keep.scatter_(1, src, False)  # src/dst are integer indices — safe in-place on non-differentiable bool mask

        n_keep = T - r
        x_merged  = x_updated[keep].view(B, n_keep, d)
        sz_merged = sz_updated[keep].view(B, n_keep)

        return x_merged, sz_merged

    # ------------------------------------------------------------------
    @staticmethod
    def unmerge(
        x_merged: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        T_orig: int,
    ) -> torch.Tensor:
        """Restore the original sequence length by copying merged tokens.

        For each merged pair (src_i, dst_i), the value at dst_i is copied back
        to both dst_i and src_i in the output.

        Args:
            x_merged: (B, T - r, d)
            src:      (B, r)  — original src positions in [0, T_orig).
            dst:      (B, r)  — original dst positions in [0, T_orig).
            T_orig:   original sequence length T.

        Returns:
            (B, T_orig, d)
        """
        B, T_reduced, d = x_merged.shape
        r = src.shape[1]
        device = x_merged.device

        # Build a mapping: for each original position, which position in the
        # compressed sequence does it come from?
        # kept positions map 1-to-1 to compressed indices;
        # src positions borrow from the corresponding dst position in the
        # compressed sequence.

        # Step 1: build per-original-position index into x_merged.
        # We create a "lookup" tensor of shape (B, T_orig) that tells us
        # which row of x_merged to read.

        # Compressed index for each kept (non-src) original position: 0 … T_reduced-1
        keep_mask = torch.ones(B, T_orig, dtype=torch.bool, device=device)
        keep_mask.scatter_(1, src, False)

        # Assign compressed indices to kept positions
        compressed_idx = torch.cumsum(keep_mask.long(), dim=1) - 1  # (B, T_orig)
        # For src positions, use the dst position's compressed index.
        # First, find the compressed index for each dst position.
        dst_compressed = compressed_idx.gather(1, dst)  # (B, r)
        # Overwrite src positions' lookup with dst compressed index
        lookup = compressed_idx.scatter(1, src, dst_compressed)  # (B, T_orig) — out-of-place

        # Step 2: gather from x_merged using lookup
        out = x_merged.gather(
            1,
            lookup.unsqueeze(-1).expand(-1, -1, d),  # (B, T_orig, d)
        )
        return out


# ---------------------------------------------------------------------------
# 3. ToMeAttention
# ---------------------------------------------------------------------------


class ToMeAttention(nn.Module):
    """Multi-head self-attention with ToMe token reduction.

    The simpler "merge x before QKV" strategy is used:
        1. Run BipartiteSoftMatching on (normalised) input x to get src/dst.
        2. Merge x → x_reduced  (shape B, T-r, d_model).
        3. Run standard scaled-dot-product attention on the reduced sequence.
        4. Unmerge the output back to shape B, T, d_model.

    Args:
        d_model: Embedding / hidden dimension.
        n_heads: Number of attention heads.
        r:       Tokens to reduce per layer.
    """

    def __init__(self, d_model: int, n_heads: int, r: int = 8) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.r        = r

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

        self.matcher = BipartiteSoftMatching(r=r)
        self.merger  = ToMeMerger()

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        B, T, _ = x.shape

        # 1. Compute merge indices from the raw input
        src, dst = self.matcher.match(x)  # (B, r_eff) each

        # 2. Merge x before projections
        x_reduced, _ = self.merger.merge(x, src, dst)  # (B, T-r_eff, d_model)
        T_red = x_reduced.shape[1]

        # 3. QKV projection on reduced sequence
        qkv = self.qkv_proj(x_reduced)                 # (B, T_red, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)                  # each (B, T_red, d_model)

        # Reshape to multi-head form
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, T_red, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)

        # 4. Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T_red, T_red)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, v)                     # (B, H, T_red, head_dim)

        # Merge heads
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_red, self.d_model)

        # Output projection
        out_reduced = self.out_proj(attn_out)  # (B, T_red, d_model)

        # 5. Unmerge back to original sequence length
        out = self.merger.unmerge(out_reduced, src, dst, T_orig=T)  # (B, T, d_model)
        return out


# ---------------------------------------------------------------------------
# 4. ToMeBlock
# ---------------------------------------------------------------------------


class ToMeBlock(nn.Module):
    """Single transformer block: ToMeAttention + FFN + RMSNorm residuals.

    Pre-norm architecture:
        x = x + ToMeAttention(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Args:
        d_model: Model dimension.
        n_heads: Number of attention heads.
        d_ff:    FFN intermediate dimension.
        r:       Tokens to merge per layer.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        r: int = 8,
    ) -> None:
        super().__init__()
        self.attn_norm = _RMSNorm(d_model)
        self.ffn_norm  = _RMSNorm(d_model)
        self.attn      = ToMeAttention(d_model, n_heads, r=r)
        self.ffn       = _FFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------


class _RMSNorm(nn.Module):
    """Minimal RMSNorm for use within this module."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x * rms * self.weight


class _FFN(nn.Module):
    """Simple two-layer feed-forward network with GELU activation."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "BipartiteSoftMatching",
    "ToMeMerger",
    "ToMeAttention",
    "ToMeBlock",
]
