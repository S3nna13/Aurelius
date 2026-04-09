"""Token merging (ToMe): reduce token count by merging similar tokens for inference speedup."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ToMeConfig:
    """Configuration for Token Merging (ToMe)."""

    r: int = 8
    """Number of token pairs to merge per layer."""

    merge_mode: str = "mean"
    """Merge strategy: 'mean' (average) | 'weighted' (weighted by similarity score)."""

    similarity_threshold: float = 0.0
    """Minimum cosine similarity to merge. 0.0 = always merge top-r pairs."""


def compute_token_similarity(x: Tensor) -> Tensor:
    """Compute pairwise cosine similarity between adjacent token pairs.

    Args:
        x: (B, T, D) hidden states.

    Returns:
        (B, T-1) cosine similarities; similarity[b, i] = cosim(x[b,i], x[b,i+1]).
    """
    # x: (B, T, D)
    left = x[:, :-1, :]   # (B, T-1, D)
    right = x[:, 1:, :]   # (B, T-1, D)

    # Normalise along the D dimension
    left_norm = F.normalize(left, p=2, dim=-1)   # (B, T-1, D)
    right_norm = F.normalize(right, p=2, dim=-1)  # (B, T-1, D)

    # Element-wise dot product summed over D → (B, T-1)
    similarity = (left_norm * right_norm).sum(dim=-1)
    return similarity


def bipartite_matching(
    similarity: Tensor,
    r: int,
    threshold: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Select top-r adjacent pairs to merge based on cosine similarity.

    Args:
        similarity: (B, T-1) cosine similarities.
        r: maximum number of pairs to merge.
        threshold: minimum similarity required to merge a pair.

    Returns:
        merge_indices: (B, r_actual) — the i+1 token index of each selected pair.
        keep_indices:  (B, T - r_actual) — indices of tokens NOT merged away.
    """
    B, T_minus_1 = similarity.shape
    T = T_minus_1 + 1

    # Apply threshold: candidates must exceed threshold
    valid_mask = similarity > threshold  # (B, T-1)

    # Clamp r to max candidates per batch item (handled per-sample below)
    # We need consistent r_actual across the batch; use the minimum that works.
    # Strategy: sort descending, take top-r, then mask out those below threshold.
    sorted_vals, sorted_idx = similarity.sort(dim=-1, descending=True)  # (B, T-1)

    # How many valid candidates per batch item?
    valid_counts = valid_mask.sum(dim=-1)  # (B,)
    r_actual = min(r, int(valid_counts.min().item()), T_minus_1)
    r_actual = max(r_actual, 0)

    if r_actual == 0:
        # Nothing to merge
        keep_indices = torch.arange(T, device=similarity.device).unsqueeze(0).expand(B, -1)
        merge_indices = torch.zeros(B, 0, dtype=torch.long, device=similarity.device)
        return merge_indices, keep_indices

    # Top-r_actual adjacent-pair indices (the LEFT token of each pair)
    # sorted_idx[:, :r_actual] → indices into 0..T-2, representing pair (i, i+1)
    pair_left_idx = sorted_idx[:, :r_actual]  # (B, r_actual)

    # The token we "merge into" its predecessor is token i+1
    merge_indices = pair_left_idx + 1  # (B, r_actual)  — the right token of each pair

    # Build keep mask: all tokens except those at merge_indices
    keep_mask = torch.ones(B, T, dtype=torch.bool, device=similarity.device)
    # Scatter False at merge positions
    keep_mask.scatter_(1, merge_indices, False)

    # keep_indices: gather the True positions
    # Since r_actual is constant across batch, we can do this cleanly
    keep_indices = keep_mask.nonzero(as_tuple=False)  # (B*(T-r_actual), 2)
    keep_indices = keep_indices[:, 1].view(B, T - r_actual)  # (B, T-r_actual)

    return merge_indices, keep_indices


def merge_tokens(
    x: Tensor,
    merge_indices: Tensor,
    keep_indices: Tensor,
    mode: str = "mean",
) -> Tensor:
    """Merge tokens indicated by merge_indices into their predecessor (merge_index - 1).

    Args:
        x: (B, T, D)
        merge_indices: (B, r_actual)
        keep_indices:  (B, T_new)
        mode: "mean" — average the two tokens; "weighted" — keep the kept token unchanged.

    Returns:
        (B, T_new, D) merged hidden states.
    """
    B, T, D = x.shape
    r_actual = merge_indices.shape[1]

    if r_actual == 0:
        # Nothing to merge; just gather kept tokens
        idx = keep_indices.unsqueeze(-1).expand(-1, -1, D)
        return x.gather(1, idx)

    # Work on a mutable copy so we can accumulate into kept positions
    x_out = x.clone()

    if mode == "mean":
        # For each merge pair: average x[b, m-1] and x[b, m] and store at m-1
        pred_indices = merge_indices - 1  # (B, r_actual) — predecessor positions

        # Gather predecessor and merge token values
        pred_idx_exp = pred_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, r, D)
        merge_idx_exp = merge_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, r, D)

        pred_vals = x.gather(1, pred_idx_exp)    # (B, r, D)
        merge_vals = x.gather(1, merge_idx_exp)  # (B, r, D)
        averaged = (pred_vals + merge_vals) / 2.0  # (B, r, D)

        # Write averaged values back to predecessor positions
        x_out.scatter_(1, pred_idx_exp, averaged)

    # mode="weighted": kept tokens stay unchanged (asymmetric merge — merged token discarded)

    # Gather only kept tokens
    idx = keep_indices.unsqueeze(-1).expand(-1, -1, D)
    return x_out.gather(1, idx)


def unmerge_tokens(
    merged: Tensor,
    merge_indices: Tensor,
    keep_indices: Tensor,
    T_orig: int,
) -> Tensor:
    """Reconstruct the original-length sequence by duplicating merged tokens back.

    For each merge position m, the kept token at position m-1 is duplicated to position m.

    Args:
        merged: (B, T_new, D)
        merge_indices: (B, r_actual)
        keep_indices:  (B, T_new)
        T_orig: original sequence length.

    Returns:
        (B, T_orig, D) approximate reconstruction.
    """
    B, T_new, D = merged.shape
    device = merged.device

    # Start with a zeroed output buffer
    output = torch.zeros(B, T_orig, D, dtype=merged.dtype, device=device)

    # Place kept tokens back at their original positions
    idx = keep_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, T_new, D)
    output.scatter_(1, idx, merged)

    if merge_indices.shape[1] > 0:
        # For each merged-away position m, duplicate the value from the kept predecessor m-1.
        # The predecessor m-1 is in keep_indices; find its position in the merged sequence.
        pred_indices = merge_indices - 1  # (B, r_actual) — original predecessor positions

        # For each predecessor original index, find its position in keep_indices
        # keep_indices is sorted (built via nonzero), so we can searchsorted
        # But a simple gather from output also works since we already placed kept tokens.
        pred_idx_exp = pred_indices.unsqueeze(-1).expand(-1, -1, D)  # (B, r, D)
        pred_vals = output.gather(1, pred_idx_exp)  # (B, r, D)

        merge_idx_exp = merge_indices.unsqueeze(-1).expand(-1, -1, D)
        output.scatter_(1, merge_idx_exp, pred_vals)

    return output


class ToMeLayer(nn.Module):
    """Applies Token Merging to reduce the sequence length of hidden states."""

    def __init__(self, config: ToMeConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, x: Tensor) -> tuple[Tensor, dict]:
        """Apply ToMe to a hidden state tensor.

        Args:
            x: (B, T, D)

        Returns:
            merged_x: (B, T_new, D)
            merge_info: dict with keys:
                "merge_indices":     (B, r_actual) LongTensor
                "keep_indices":      (B, T_new) LongTensor
                "compression_ratio": float — T_new / T_orig
        """
        B, T, D = x.shape

        similarity = compute_token_similarity(x)  # (B, T-1)
        merge_indices, keep_indices = bipartite_matching(
            similarity,
            r=self.config.r,
            threshold=self.config.similarity_threshold,
        )
        merged_x = merge_tokens(x, merge_indices, keep_indices, mode=self.config.merge_mode)

        T_new = merged_x.shape[1]
        compression_ratio = T_new / T

        merge_info = {
            "merge_indices": merge_indices,
            "keep_indices": keep_indices,
            "compression_ratio": compression_ratio,
        }
        return merged_x, merge_info


def apply_tome_to_hidden_states(
    hidden_states: list[Tensor],
    config: ToMeConfig,
) -> tuple[list[Tensor], list[dict]]:
    """Apply ToMe independently to each hidden state tensor in a list.

    Args:
        hidden_states: list of (B, T_i, D) tensors.
        config: ToMeConfig controlling merge behaviour.

    Returns:
        merged_states: list of (B, T_new_i, D) tensors.
        merge_infos:   list of merge_info dicts (one per input tensor).
    """
    layer = ToMeLayer(config)
    merged_states: list[Tensor] = []
    merge_infos: list[dict] = []

    for hs in hidden_states:
        merged, info = layer(hs)
        merged_states.append(merged)
        merge_infos.append(info)

    return merged_states, merge_infos
