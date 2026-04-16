"""Token Merging v2 (ToMe-style): reduce sequence length by merging similar tokens.

API surface
-----------
MergeConfig                   — configuration dataclass
compute_token_similarity      — pairwise similarity matrix (T, T)
select_tokens_to_merge        — pick (src, dst) index pairs to merge
merge_tokens                  — collapse src into dst, return shorter sequence
unmerge_tokens                — scatter merged tokens back to original length
TokenMerger                   — convenience class wrapping the full pipeline
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MergeConfig:
    """Configuration for token merging.

    Attributes:
        merge_ratio:       fraction of tokens to remove (0 < ratio <= 1).
        similarity_metric: pairwise similarity used for matching.
                           One of "cosine", "dot", "l2".
        merge_mode:        how to combine src and dst embeddings.
                           One of "mean", "weighted".
        n_keep_start:      always keep the first n tokens (e.g. CLS / BOS).
    """
    merge_ratio: float = 0.5
    similarity_metric: str = "cosine"  # "cosine" | "dot" | "l2"
    merge_mode: str = "mean"           # "mean" | "weighted"
    n_keep_start: int = 1


# ---------------------------------------------------------------------------
# compute_token_similarity
# ---------------------------------------------------------------------------

def compute_token_similarity(tokens: Tensor, metric: str = "cosine") -> Tensor:
    """Compute pairwise token similarity.

    Args:
        tokens: (T, d) token embeddings.
        metric: one of "cosine", "dot", "l2".

    Returns:
        sim: (T, T) symmetric similarity matrix.
             cosine — normalised dot product in [-1, 1], diagonal ≈ 1.
             dot    — raw dot product.
             l2     — negative squared Euclidean distance; diagonal ≈ 0.
    """
    if metric == "cosine":
        normed = F.normalize(tokens, p=2, dim=-1)   # (T, d)
        return normed @ normed.T                    # (T, T)
    elif metric == "dot":
        return tokens @ tokens.T                   # (T, T)
    elif metric == "l2":
        # -||a - b||^2 = 2 a·b - ||a||^2 - ||b||^2
        dots = tokens @ tokens.T                   # (T, T)
        sq_norms = (tokens * tokens).sum(dim=-1)   # (T,)
        return 2 * dots - sq_norms.unsqueeze(1) - sq_norms.unsqueeze(0)
    else:
        raise ValueError(
            f"Unknown similarity_metric {metric!r}. Choose 'cosine', 'dot', or 'l2'."
        )


# ---------------------------------------------------------------------------
# select_tokens_to_merge
# ---------------------------------------------------------------------------

def select_tokens_to_merge(
    similarity: Tensor,
    merge_ratio: float,
    n_keep_start: int = 1,
) -> Tuple[Tensor, Tensor]:
    """Select token pairs to merge based on pairwise similarity.

    Strategy:
      - Work only with the upper triangle (i < j) to avoid duplicate pairs.
      - Tokens in [0, n_keep_start) are never selected as *src* (the token
        that gets removed).  They can be a *dst* (the token that absorbs).
      - For each candidate pair (i, j) with i < j:
          dst = i  (lower index stays), src = j  (higher index is removed).
      - Select the top-k pairs by similarity score where k = int(T * merge_ratio).
      - Greedy deduplication: once a token appears in src_indices it is not
        reused, ensuring every src appears exactly once.

    Args:
        similarity: (T, T) pairwise similarity.
        merge_ratio: fraction of T to merge; k = int(T * merge_ratio).
        n_keep_start: number of leading tokens protected from being src.

    Returns:
        src_indices: (k,) indices of tokens to remove.
        dst_indices: (k,) indices of tokens to merge into.
    """
    T = similarity.shape[0]
    k = int(T * merge_ratio)
    k = max(0, min(k, T - n_keep_start - 1))  # can't merge everything

    if k == 0:
        empty = torch.zeros(0, dtype=torch.long, device=similarity.device)
        return empty, empty

    # Build candidate (i, j) pairs with i < j and j >= n_keep_start
    # (j is the src / token-to-remove)
    rows, cols = torch.triu_indices(T, T, offset=1, device=similarity.device)
    # rows[m] < cols[m] by construction; src = cols, dst = rows
    # Filter: src (cols) must not be in the protected prefix
    valid = cols >= n_keep_start
    rows, cols = rows[valid], cols[valid]

    if rows.numel() == 0:
        empty = torch.zeros(0, dtype=torch.long, device=similarity.device)
        return empty, empty

    pair_scores = similarity[rows, cols]  # (num_valid_pairs,)

    # Sort by descending score; greedily pick k pairs without src repetition
    order = pair_scores.argsort(descending=True)
    selected_src: list[int] = []
    selected_dst: list[int] = []
    seen_src: set[int] = set()

    for idx in order.tolist():
        if len(selected_src) >= k:
            break
        src_tok = int(cols[idx].item())
        dst_tok = int(rows[idx].item())
        if src_tok not in seen_src:
            seen_src.add(src_tok)
            selected_src.append(src_tok)
            selected_dst.append(dst_tok)

    device = similarity.device
    src_indices = torch.tensor(selected_src, dtype=torch.long, device=device)
    dst_indices = torch.tensor(selected_dst, dtype=torch.long, device=device)
    return src_indices, dst_indices


# ---------------------------------------------------------------------------
# merge_tokens
# ---------------------------------------------------------------------------

def merge_tokens(
    tokens: Tensor,
    src_indices: Tensor,
    dst_indices: Tensor,
    mode: str = "mean",
) -> Tensor:
    """Merge src tokens into dst tokens and return the reduced sequence.

    Args:
        tokens:      (T, d) token embeddings.
        src_indices: (k,)  indices of tokens to remove.
        dst_indices: (k,)  indices of tokens to merge into.
        mode:        "mean" or "weighted" (both use simple average for now).

    Returns:
        merged: (T - k, d) token sequence with src tokens removed.
    """
    T, d = tokens.shape
    k = src_indices.shape[0]

    if k == 0:
        return tokens.clone()

    if mode not in ("mean", "weighted"):
        raise ValueError(f"Unknown merge_mode {mode!r}. Choose 'mean' or 'weighted'.")

    out = tokens.clone()

    # Gather src and dst embeddings
    src_vecs = tokens[src_indices]  # (k, d)
    dst_vecs = tokens[dst_indices]  # (k, d)

    # Both "mean" and "weighted" use simple average (weighted == mean by spec)
    merged_vecs = (dst_vecs + src_vecs) * 0.5  # (k, d)

    # Write back to dst positions
    out[dst_indices] = merged_vecs

    # Remove src positions — build keep mask
    keep_mask = torch.ones(T, dtype=torch.bool, device=tokens.device)
    keep_mask[src_indices] = False

    return out[keep_mask]  # (T - k, d)


# ---------------------------------------------------------------------------
# unmerge_tokens
# ---------------------------------------------------------------------------

def unmerge_tokens(
    merged: Tensor,
    src_indices: Tensor,
    dst_indices: Tensor,
    original_len: int,
) -> Tensor:
    """Reconstruct a sequence of length original_len from the merged sequence.

    Each position in src_indices copies the value from its corresponding
    dst position in the merged sequence.

    Args:
        merged:       (T - k, d) merged token sequence.
        src_indices:  (k,) original indices of removed tokens.
        dst_indices:  (k,) original indices of their merge destinations.
        original_len: T, the original sequence length.

    Returns:
        reconstructed: (original_len, d)
    """
    T_merged, d = merged.shape
    k = src_indices.shape[0]
    device = merged.device

    output = torch.zeros(original_len, d, dtype=merged.dtype, device=device)

    # Build mapping from original position → merged position
    # The merged sequence consists of original tokens with src_indices removed,
    # in their original order.
    keep_mask = torch.ones(original_len, dtype=torch.bool, device=device)
    if k > 0:
        keep_mask[src_indices] = False

    # Scatter kept tokens back
    keep_positions = keep_mask.nonzero(as_tuple=True)[0]  # (T_merged,)
    output[keep_positions] = merged  # place merged tokens at kept positions

    # Fill src positions with value from their dst position in output
    if k > 0:
        output[src_indices] = output[dst_indices]

    return output


# ---------------------------------------------------------------------------
# TokenMerger
# ---------------------------------------------------------------------------

class TokenMerger:
    """Full merge/unmerge pipeline driven by a MergeConfig.

    Usage::

        merger = TokenMerger(config)
        merged_tokens, info = merger.merge(tokens)
        # ... run through model layers ...
        restored = merger.unmerge(merged_tokens, info)
    """

    def __init__(self, config: MergeConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def merge(self, tokens: Tensor) -> Tuple[Tensor, dict]:
        """Merge similar tokens to produce a shorter sequence.

        Args:
            tokens: (T, d) token embeddings.

        Returns:
            merged_tokens: (T', d) where T' = T - k.
            merge_info:    dict with keys:
                             "src_indices"  (k,)
                             "dst_indices"  (k,)
                             "original_len" int
        """
        cfg = self.config
        sim = compute_token_similarity(tokens, metric=cfg.similarity_metric)
        src_idx, dst_idx = select_tokens_to_merge(
            sim, merge_ratio=cfg.merge_ratio, n_keep_start=cfg.n_keep_start
        )
        merged = merge_tokens(tokens, src_idx, dst_idx, mode=cfg.merge_mode)
        info = {
            "src_indices": src_idx,
            "dst_indices": dst_idx,
            "original_len": tokens.shape[0],
        }
        return merged, info

    # ------------------------------------------------------------------
    def unmerge(self, merged: Tensor, merge_info: dict) -> Tensor:
        """Reconstruct the original-length sequence.

        Args:
            merged:     (T', d) merged token embeddings.
            merge_info: dict returned by :meth:`merge`.

        Returns:
            (original_len, d) reconstructed tensor.
        """
        return unmerge_tokens(
            merged,
            merge_info["src_indices"],
            merge_info["dst_indices"],
            merge_info["original_len"],
        )

    # ------------------------------------------------------------------
    def compression_ratio(self, original_len: int) -> float:
        """Fraction of tokens kept after merging.

        Returns:
            tokens_kept / original_len  in (0, 1].
        """
        k = int(original_len * self.config.merge_ratio)
        k = max(0, min(k, original_len - self.config.n_keep_start - 1))
        tokens_kept = original_len - k
        return float(tokens_kept) / float(original_len)
