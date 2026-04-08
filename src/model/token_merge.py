"""Token Merging (ToMe) — Bolya et al. 2022.

Reduce sequence length by merging similar tokens before attention, then unmerge
after. Reduces attention complexity while preserving information.

Reference: "Token Merging: Your ViT but Faster" (https://arxiv.org/abs/2210.09461)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


def bipartite_soft_matching(
    metric: torch.Tensor,  # (B, N, C) — tokens to compare (e.g., keys)
    r: int,                # number of token pairs to merge
) -> tuple[Callable, Callable]:
    """Find r token pairs to merge using bipartite graph matching.

    Algorithm:
    1. Split tokens into two sets A (even indices) and B (odd indices).
    2. Compute cosine similarity between A and B: (|A|, |B|) matrix.
    3. For each token in A, find its most similar token in B.
    4. Select top-r pairs by similarity score, ensuring unique B selections
       (each B token can be merged at most once).
    5. Return (merge_fn, unmerge_fn) closures.

    merge_fn(x: tensor) -> merged_x: (B, N-r, C)
    unmerge_fn(x: tensor) -> unmerged_x: (B, N, C)

    Merging: average the merged pair's values.
    Unmerging: copy merged token value back to both positions.

    Args:
        metric: (B, N, C) tensor of token representations used for similarity.
        r: number of token pairs to merge.

    Returns:
        (merge_fn, unmerge_fn) — callable closures operating on (B, N, C) tensors.
    """
    B, N, C = metric.shape

    # Need at least 2 tokens (1 in A, 1 in B)
    half = N // 2
    r = min(r, half)

    if r <= 0:
        def merge_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        def unmerge_fn(x: torch.Tensor) -> torch.Tensor:
            return x

        return merge_fn, unmerge_fn

    # Split into two sets by even/odd indices
    a_idx = torch.arange(0, N, 2, device=metric.device)   # even indices: |A| = ceil(N/2)
    b_idx = torch.arange(1, N, 2, device=metric.device)   # odd indices:  |B| = floor(N/2)

    a = metric[:, a_idx, :]  # (B, |A|, C)
    b = metric[:, b_idx, :]  # (B, |B|, C)

    # Cosine similarity
    a_norm = F.normalize(a, dim=-1)  # (B, |A|, C)
    b_norm = F.normalize(b, dim=-1)  # (B, |B|, C)
    scores = torch.bmm(a_norm, b_norm.transpose(1, 2))  # (B, |A|, |B|)

    # For each token in A, find its best match in B
    best_scores, best_b = scores.max(dim=2)  # (B, |A|)

    # Select top-r pairs by score, but we must ensure each B is used at most once.
    # Strategy: greedily select top-r from (A, B) pairs where each B appears once.
    # We sort A tokens by score descending; keep the first occurrence of each B index.
    # This is equivalent to the original ToMe paper's approach.
    #
    # For batch processing, we do per-batch greedy selection.
    # Since we need a fixed r per batch for vectorised merge/unmerge, we use
    # the same r for all batches (the minimum r that is valid across all batches).
    #
    # Implementation: sort flat (score, b_index) pairs per batch and select greedily.

    # Precompute selected (A index, B index) pairs for each batch element.
    # Store as tensors of shape (B, r).
    sel_a_list = []
    sel_b_list = []

    for batch_i in range(B):
        scores_i = best_scores[batch_i]   # (|A|,)
        match_b_i = best_b[batch_i]       # (|A|,)

        # Sort A by descending score
        sorted_order = torch.argsort(scores_i, descending=True)
        used_b = set()
        chosen_a = []
        chosen_b = []
        for a_local in sorted_order.tolist():
            b_local = match_b_i[a_local].item()
            if b_local not in used_b:
                used_b.add(b_local)
                chosen_a.append(a_local)
                chosen_b.append(b_local)
                if len(chosen_a) == r:
                    break

        # If we couldn't find r unique pairs (shouldn't happen when r <= half),
        # pad with the first chosen pair (safe fallback).
        while len(chosen_a) < r:
            chosen_a.append(chosen_a[0])
            chosen_b.append(chosen_b[0])

        sel_a_list.append(torch.tensor(chosen_a, device=metric.device))
        sel_b_list.append(torch.tensor(chosen_b, device=metric.device))

    # sel_a: (B, r) — local indices into A set
    # sel_b: (B, r) — local indices into B set
    sel_a = torch.stack(sel_a_list, dim=0)  # (B, r)
    sel_b = torch.stack(sel_b_list, dim=0)  # (B, r)

    # Convert to global sequence indices
    # a_idx[sel_a]: (B, r), b_idx[sel_b]: (B, r)
    global_a = a_idx[sel_a]  # (B, r) — global positions in [0, N)
    global_b = b_idx[sel_b]  # (B, r) — global positions in [0, N)

    # Build per-batch keep masks (which B positions are removed).
    # Shape: (B, N). True = position is kept.
    # Removing global_b positions (one unique B per row guaranteed by construction).
    remove_mask = torch.zeros(B, N, dtype=torch.bool, device=metric.device)
    remove_mask.scatter_(1, global_b, True)   # mark removed positions
    keep_mask = ~remove_mask                  # (B, N)

    def merge_fn(x: torch.Tensor) -> torch.Tensor:
        """Merge top-r token pairs; return (B, N-r, C)."""
        B2, N2, C2 = x.shape

        # Merged value = average of A and B token values
        x_a = x.gather(1, global_a.unsqueeze(-1).expand(-1, -1, C2))  # (B, r, C)
        x_b = x.gather(1, global_b.unsqueeze(-1).expand(-1, -1, C2))  # (B, r, C)
        merged_val = (x_a + x_b) / 2.0  # (B, r, C)

        # Update A positions with merged values; B positions will be dropped
        x_out = x.clone()
        x_out.scatter_(1, global_a.unsqueeze(-1).expand(-1, -1, C2), merged_val)

        # Compact: remove B positions
        # keep_mask: (B, N) — True = keep; uniform across batches is NOT guaranteed,
        # so we handle per batch using masked_select + reshape.
        n_keep = N2 - r
        # Use boolean indexing per batch with a vectorised approach:
        # expand mask to (B, N, C) and select
        keep_3d = keep_mask.unsqueeze(-1).expand(-1, -1, C2)  # (B, N, C)
        # masked_select returns 1D; reshape to (B, n_keep, C)
        return x_out[keep_3d].view(B2, n_keep, C2)

    def unmerge_fn(x: torch.Tensor) -> torch.Tensor:
        """Restore merged sequence back to original length (B, N, C)."""
        B2, N_merged, C2 = x.shape
        N_orig = N  # captured from outer scope

        # Allocate output
        out = torch.zeros(B2, N_orig, C2, device=x.device, dtype=x.dtype)

        # Scatter compact values back into kept positions
        keep_3d = keep_mask.unsqueeze(-1).expand(-1, -1, C2)  # (B, N_orig, C)
        out[keep_3d] = x.reshape(-1)  # fill in order

        # Copy merged A values into the removed B positions
        x_a_vals = out.gather(1, global_a.unsqueeze(-1).expand(-1, -1, C2))  # (B, r, C)
        out.scatter_(1, global_b.unsqueeze(-1).expand(-1, -1, C2), x_a_vals)

        return out

    return merge_fn, unmerge_fn


class TokenMergeAttention(nn.Module):
    """Attention layer with token merging.

    Wraps GroupedQueryAttention (or any attention module) and adds:
    1. Pre-merge: reduce sequence length by r.
    2. Attention on reduced sequence.
    3. Post-unmerge: restore full sequence length.

    Args:
        attention: the underlying attention module.
        r: tokens to merge (0 = disabled).
        trace_source: if True, track which tokens were merged for analysis.
    """

    def __init__(
        self,
        attention: nn.Module,
        r: int = 8,
        trace_source: bool = False,
    ) -> None:
        super().__init__()
        self.attention = attention
        self.r = r
        self.trace_source = trace_source
        self._merge_history: list = []

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask=None,
        past_kv=None,
    ) -> tuple[torch.Tensor, tuple]:
        """Forward pass with optional token merging.

        1. Compute keys (for similarity metric) — use a quick projection.
        2. Build merge/unmerge functions via bipartite_soft_matching.
        3. Merge x (and freqs_cis for the merged positions).
        4. Run attention on merged sequence.
        5. Unmerge output back to original length.
        6. Return (output, kv_cache).

        Note: When r=0 or seq_len is too small to merge, skip merging entirely.
        """
        B, N, C = x.shape

        # Skip merging if disabled or sequence too short (need at least 4 tokens)
        effective_r = self.r
        if effective_r <= 0 or N < 4:
            return self.attention(x, freqs_cis, mask, past_kv)

        # Use key projection as the similarity metric
        with torch.no_grad():
            metric = self.attention.k_proj(x)  # (B, N, n_kv_heads * head_dim)

        merge_fn, unmerge_fn = bipartite_soft_matching(metric, effective_r)

        # Merge input tokens
        x_merged = merge_fn(x)  # (B, N-r, C)
        N_merged = x_merged.shape[1]

        # Slice freqs_cis to match the merged sequence length
        freqs_merged = freqs_cis[:N_merged]

        if self.trace_source:
            self._merge_history.append({
                "before": N,
                "after": N_merged,
            })

        # Run attention on merged sequence
        out_merged, kv_cache = self.attention(x_merged, freqs_merged, mask, past_kv)

        # Unmerge back to original length
        out = unmerge_fn(out_merged)  # (B, N, C)

        return out, kv_cache


def apply_token_merging(
    model: nn.Module,
    r: int = 8,
    layer_indices: list[int] | None = None,
) -> None:
    """Replace attention layers in a model with TokenMergeAttention.

    Args:
        model: AureliusTransformer (must have a .layers ModuleList of blocks
               each with an .attn attribute).
        r: tokens to merge per layer.
        layer_indices: which layers to apply ToMe to (None = all).

    Modifies model in-place.
    """
    layers = model.layers  # nn.ModuleList of TransformerBlock
    for i, block in enumerate(layers):
        if layer_indices is not None and i not in layer_indices:
            continue
        if hasattr(block, "attn"):
            block.attn = TokenMergeAttention(block.attn, r=r)


class ToMeStats:
    """Track token merging statistics."""

    def __init__(self) -> None:
        self.total_tokens_before: int = 0
        self.total_tokens_after: int = 0

    def record(self, before: int, after: int) -> None:
        """Record a single merging event."""
        self.total_tokens_before += before
        self.total_tokens_after += after

    def compression_ratio(self) -> float:
        """Return compression ratio.

        1.0 = no compression, 0.5 = half the tokens.
        """
        if self.total_tokens_before == 0:
            return 1.0
        return self.total_tokens_after / self.total_tokens_before
