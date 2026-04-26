"""Token Merging (ToMe): reduce redundant tokens via bipartite matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ToMeConfig:
    r: int = 4  # number of token pairs to merge per layer
    merge_mode: str = "mean"  # "mean" | "weighted"
    similarity: str = "cosine"  # "cosine" | "dot"


def bipartite_soft_matching(
    x: Tensor,
    r: int,
    similarity: str = "cosine",
) -> tuple[Tensor, Tensor, Tensor]:
    """Bipartite token matching.

    Split tokens into src (odd indices) and dst (even indices).
    For each src token, find its most similar dst token.
    Select top-r src-dst pairs to merge.

    Args:
        x: (B, T, d) token representations
        r: number of pairs to merge (capped at min(T//2, r))
        similarity: "cosine" or "dot"

    Returns:
        merge_src_idx: (B, r') — source token indices to merge (r' = actual merges)
        merge_dst_idx: (B, r') — destination token indices
        unmerge_map:   (B, T)  — maps original positions to new positions
    """
    B, T, d = x.shape

    # Split into dst (even indices) and src (odd indices)
    dst = x[:, 0::2, :]  # (B, T_dst, d)  where T_dst = ceil(T/2)
    src = x[:, 1::2, :]  # (B, T_src, d)  where T_src = floor(T/2)

    dst.shape[1]
    T_src = src.shape[1]

    # Cap r
    r_actual = min(r, T_src)

    if r_actual == 0 or T_src == 0:
        empty_src = torch.zeros(B, 0, dtype=torch.long, device=x.device)
        empty_dst = torch.zeros(B, 0, dtype=torch.long, device=x.device)
        # unmerge_map: identity — each position maps to itself
        unmerge_map = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        return empty_src, empty_dst, unmerge_map

    # Compute similarity between each src and all dst tokens
    if similarity == "cosine":
        src_norm = F.normalize(src, p=2, dim=-1)  # (B, T_src, d)
        dst_norm = F.normalize(dst, p=2, dim=-1)  # (B, T_dst, d)
        scores = torch.bmm(src_norm, dst_norm.transpose(1, 2))  # (B, T_src, T_dst)
    elif similarity == "dot":
        scores = torch.bmm(src, dst.transpose(1, 2))  # (B, T_src, T_dst)
    else:
        raise ValueError(f"Unknown similarity: {similarity!r}. Choose 'cosine' or 'dot'.")

    # For each src token, find its best matching dst token
    best_dst_scores, best_dst_local = scores.max(dim=-1)  # (B, T_src)

    # Select top-r_actual src tokens by their best score
    _, top_src_local = best_dst_scores.topk(r_actual, dim=-1)  # (B, r_actual)

    # Convert local src indices → original token indices (src tokens are at odd positions)
    # src token i → original token index 2*i + 1
    merge_src_idx = top_src_local * 2 + 1  # (B, r_actual)

    # For each selected src, get its matched dst local index, then convert to original
    # dst token i → original token index 2*i
    top_src_local_exp = top_src_local  # (B, r_actual)
    matched_dst_local = best_dst_local.gather(1, top_src_local_exp)  # (B, r_actual)
    merge_dst_idx = matched_dst_local * 2  # (B, r_actual)

    # Build unmerge_map: maps original token positions to positions in the merged sequence.
    # Tokens that are merged (src positions) get mapped to their dst position.
    # The merged sequence is formed by removing the src tokens from the sequence.
    # unmerge_map[b, t] = index in merged sequence where token t's value lives.
    #
    # Strategy: mark which positions are being merged away (src positions selected).
    # After removing them, the keep positions are renumbered sequentially.
    # Merged-away positions point to the renumbered position of their dst.

    # Build a mask of positions being merged away
    merged_away = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    merged_away.scatter_(1, merge_src_idx, True)

    # keep_mask: positions that remain in the merged sequence
    keep_mask = ~merged_away  # (B, T)

    # Compute new indices for kept positions via cumsum
    new_pos = keep_mask.long().cumsum(dim=1) - 1  # (B, T), 0-indexed new positions for kept

    # unmerge_map: for kept positions, their new position; for merged-away, their dst's new pos
    unmerge_map = new_pos.clone()

    # For merged-away positions, point to their dst token's new position
    # dst original indices for each selected merge pair
    # merge_dst_idx: (B, r_actual) — original position of dst token
    dst_new_pos = new_pos.gather(1, merge_dst_idx)  # (B, r_actual) — new pos of dst
    # Scatter into unmerge_map at src positions
    unmerge_map.scatter_(1, merge_src_idx, dst_new_pos)

    return merge_src_idx, merge_dst_idx, unmerge_map


def merge_tokens(
    x: Tensor,
    src_idx: Tensor,
    dst_idx: Tensor,
    mode: str = "mean",
) -> tuple[Tensor, Tensor]:
    """Merge src tokens into their matched dst tokens.

    Args:
        x: (B, T, d) token representations
        src_idx: (B, r) source token indices to remove/merge
        dst_idx: (B, r) destination token indices to merge into
        mode: "mean" or "weighted"

    Returns:
        merged: (B, T-r, d) — reduced token sequence
        size: (B, T-r) — number of original tokens represented in each merged token
    """
    B, T, d = x.shape
    r = src_idx.shape[1]

    if r == 0:
        size = torch.ones(B, T, dtype=x.dtype, device=x.device)
        return x, size

    x_out = x.clone()
    size = torch.ones(B, T, dtype=x.dtype, device=x.device)

    # Gather src and dst values
    src_idx_exp = src_idx.unsqueeze(-1).expand(-1, -1, d)  # (B, r, d)
    dst_idx_exp = dst_idx.unsqueeze(-1).expand(-1, -1, d)  # (B, r, d)

    src_vals = x.gather(1, src_idx_exp)  # (B, r, d)
    dst_vals = x.gather(1, dst_idx_exp)  # (B, r, d)

    if mode == "mean":
        merged_vals = (src_vals + dst_vals) * 0.5
    elif mode == "weighted":
        # Weighted by token count; initially both are 1, so same as mean
        merged_vals = (src_vals + dst_vals) * 0.5
    else:
        raise ValueError(f"Unknown merge_mode: {mode!r}. Choose 'mean' or 'weighted'.")

    # Write merged values back to dst positions
    x_out.scatter_(1, dst_idx_exp, merged_vals)

    # Update sizes: dst tokens now represent 2 original tokens
    dst_size_delta = torch.ones(B, r, dtype=x.dtype, device=x.device)
    size.scatter_add_(1, dst_idx, dst_size_delta)

    # Remove src positions: build keep mask
    keep_mask = torch.ones(B, T, dtype=torch.bool, device=x.device)
    keep_mask.scatter_(1, src_idx, False)

    # Gather kept tokens (must be same count per batch item)
    T_new = T - r
    # nonzero approach: gather D-expanded indices
    keep_indices = keep_mask.nonzero(as_tuple=False)  # (B*T_new, 2)
    col_indices = keep_indices[:, 1].view(B, T_new)  # (B, T_new)

    col_idx_exp = col_indices.unsqueeze(-1).expand(-1, -1, d)
    merged = x_out.gather(1, col_idx_exp)  # (B, T_new, d)

    size_merged = size.gather(1, col_indices)  # (B, T_new)

    return merged, size_merged


def unmerge_tokens(
    x: Tensor,
    src_idx: Tensor,
    dst_idx: Tensor,
    size: Tensor,
    T_orig: int,
) -> Tensor:
    """Restore merged tokens back to original sequence length T_orig.

    Each src position gets the value of its merged dst token.

    Args:
        x: (B, T-r, d) merged token sequence
        src_idx: (B, r) original source token indices
        dst_idx: (B, r) original destination token indices
        size: (B, T-r) token sizes (not used for value recovery, kept for API consistency)
        T_orig: original sequence length

    Returns:
        (B, T_orig, d) reconstructed sequence
    """
    B, T_new, d = x.shape
    r = src_idx.shape[1]
    device = x.device

    if r == 0:
        return x

    # We need to reconstruct T_orig tokens from T_new = T_orig - r tokens.
    # The merged sequence has the dst tokens (with merged values).
    # We need to:
    #   1. Place T_new merged tokens back at their original positions (keep positions)
    #   2. Fill src positions with their matched dst token's value

    # Build the keep_mask to find which original positions are in the merged sequence
    keep_mask = torch.ones(B, T_orig, dtype=torch.bool, device=device)
    keep_mask.scatter_(1, src_idx, False)

    # Get the original positions of kept tokens
    keep_positions = keep_mask.nonzero(as_tuple=False)  # (B*T_new, 2)
    col_positions = keep_positions[:, 1].view(B, T_new)  # (B, T_new)

    # Build output buffer
    output = torch.zeros(B, T_orig, d, dtype=x.dtype, device=device)

    # Place merged tokens back at their kept positions
    col_pos_exp = col_positions.unsqueeze(-1).expand(-1, -1, d)
    output.scatter_(1, col_pos_exp, x)

    # Now fill src positions with the value of their matched dst token
    # First, gather the dst values from the output (already placed)
    dst_idx_exp = dst_idx.unsqueeze(-1).expand(-1, -1, d)  # (B, r, d)
    dst_vals = output.gather(1, dst_idx_exp)  # (B, r, d)

    # Scatter dst values to src positions
    src_idx_exp = src_idx.unsqueeze(-1).expand(-1, -1, d)  # (B, r, d)
    output.scatter_(1, src_idx_exp, dst_vals)

    return output


class ToMeLayer(nn.Module):
    """Wraps a transformer layer with token merging/unmerging."""

    def __init__(self, layer: nn.Module, config: ToMeConfig) -> None:
        super().__init__()
        self.layer = layer
        self.config = config

    def forward(self, x: Tensor) -> Tensor:
        """Merge -> layer forward -> unmerge."""
        B, T, d = x.shape
        r = self.config.r

        src_idx, dst_idx, unmerge_map = bipartite_soft_matching(
            x, r=r, similarity=self.config.similarity
        )

        x_merged, size = merge_tokens(x, src_idx, dst_idx, mode=self.config.merge_mode)

        # Run the wrapped layer on merged sequence
        out = self.layer(x_merged)

        # If layer returns a tuple (e.g. TransformerBlock returns (hidden, kv)), unpack
        if isinstance(out, tuple):
            hidden = out[0]
            hidden_unmerged = unmerge_tokens(hidden, src_idx, dst_idx, size, T)
            return (hidden_unmerged,) + out[1:]

        # Plain tensor output
        return unmerge_tokens(out, src_idx, dst_idx, size, T)


class ToMeWrapper(nn.Module):
    """Apply ToMe to every layer of a model."""

    def __init__(self, model: nn.Module, config: ToMeConfig) -> None:
        """Wraps model.layers with ToMeLayer. Keep model.embed, model.norm, model.lm_head."""
        super().__init__()
        self.embed = model.embed
        self.norm = model.norm
        self.lm_head = model.lm_head
        self.config = config

        # Store the original model reference for accessing freqs_cis and model config
        self._model = model

        # Wrap each layer with ToMeLayer
        self.layers = nn.ModuleList([ToMeLayer(layer, config) for layer in model.layers])

        # Copy freqs_cis buffer if present
        if hasattr(model, "freqs_cis"):
            self.register_buffer("freqs_cis", model.freqs_cis, persistent=False)
        else:
            self.freqs_cis = None

        # Store n_layers for compression ratio calculation
        self.n_layers = len(self.layers)

    def forward(self, input_ids: Tensor) -> tuple[Tensor, Tensor, None]:
        """Returns (loss, logits, None) — same API as base model."""
        B, S = input_ids.shape

        x = self.embed(input_ids)  # (B, S, d_model)

        # Get freqs_cis if available (for compatibility with TransformerBlock)
        if self.freqs_cis is not None:
            freqs_cis = self.freqs_cis[:S]
        else:
            freqs_cis = None

        # Forward through ToMe-wrapped layers
        # Each ToMeLayer.forward expects (x: Tensor) → Tensor
        # The internal layer may be a TransformerBlock expecting (x, freqs_cis, mask, past_kv)
        # We need to handle this: call the internal layer directly in ToMeLayer
        # But ToMeLayer calls self.layer(x_merged) with only x.
        # For AureliusTransformer's TransformerBlock, we need to pass freqs_cis.
        # We handle this by running the layers manually here.

        for i, tome_layer in enumerate(self.layers):
            r = self.config.r
            src_idx, dst_idx, unmerge_map = bipartite_soft_matching(
                x, r=r, similarity=self.config.similarity
            )

            x_merged, size = merge_tokens(x, src_idx, dst_idx, mode=self.config.merge_mode)

            T_merged = x_merged.shape[1]
            T_orig = x.shape[1]

            # Call the underlying (unwrapped) transformer layer
            inner_layer = tome_layer.layer
            if freqs_cis is not None:
                # TransformerBlock-style: forward(x, freqs_cis, mask, past_kv)
                freqs_merged = freqs_cis[:T_merged]
                out, _kv = inner_layer(x_merged, freqs_merged, None, None)
            else:
                out = inner_layer(x_merged)
                if isinstance(out, tuple):
                    out = out[0]

            x = unmerge_tokens(out, src_idx, dst_idx, size, T_orig)

        x = self.norm(x)
        logits = self.lm_head(x)  # (B, S, vocab_size)

        loss = None
        return loss, logits, None

    def get_compression_ratio(self) -> float:
        """Estimate compression: (T - r*n_layers) / T for typical T."""
        # Use a typical sequence length of 512 for estimation
        T_typical = 512
        r = self.config.r
        n_layers = self.n_layers
        # Each layer reduces by r tokens, but cap at sequence length
        T_reduced = max(1, T_typical - r * n_layers)
        ratio = T_reduced / T_typical
        # Clamp to (0, 1]
        return float(max(1e-6, min(1.0, ratio)))
