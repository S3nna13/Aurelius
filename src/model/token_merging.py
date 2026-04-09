"""Token Merging (ToMe-inspired) — reduce sequence length during inference.

Merge similar adjacent tokens to reduce compute per layer, then unmerge to
restore the original sequence length after processing.

Reference: "Token Merging: Your ViT but Faster" (Bolya et al. 2022)
           https://arxiv.org/abs/2210.09461
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ToMeConfig:
    """Configuration for Token Merging.

    Attributes:
        r: Number of tokens to merge per layer.
        merge_mode: How to combine merged token pairs: "mean" | "max" | "weighted".
        similarity_metric: How to measure token similarity: "cosine" | "dot" | "l2".
        layer_start: Only apply merging starting from this layer index.
    """

    r: int = 8
    merge_mode: str = "mean"
    similarity_metric: str = "cosine"
    layer_start: int = 0


# ---------------------------------------------------------------------------
# Similarity computation
# ---------------------------------------------------------------------------

def compute_token_similarity(x: torch.Tensor, metric: str = "cosine") -> torch.Tensor:
    """Compute pairwise similarity matrix between tokens.

    Args:
        x: Token representations of shape (B, T, D).
        metric: One of "cosine", "dot", "l2".

    Returns:
        Symmetric similarity matrix of shape (B, T, T).
        - cosine: normalised dot products in [-1, 1].
        - dot: raw dot products.
        - l2: negative pairwise squared L2 distances (higher = more similar).
    """
    if metric == "cosine":
        x_norm = F.normalize(x, dim=-1)  # (B, T, D)
        return torch.bmm(x_norm, x_norm.transpose(1, 2))  # (B, T, T)

    if metric == "dot":
        return torch.bmm(x, x.transpose(1, 2))  # (B, T, T)

    if metric == "l2":
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        # Negate so that higher value = more similar (closer).
        sq_norm = (x * x).sum(dim=-1, keepdim=True)  # (B, T, 1)
        dot = torch.bmm(x, x.transpose(1, 2))         # (B, T, T)
        sq_dist = sq_norm + sq_norm.transpose(1, 2) - 2.0 * dot
        return -sq_dist  # (B, T, T)

    raise ValueError(f"Unknown similarity metric: {metric!r}. Choose 'cosine', 'dot', or 'l2'.")


# ---------------------------------------------------------------------------
# Pair selection (adjacent only — preserves locality)
# ---------------------------------------------------------------------------

def find_merge_pairs(
    similarity: torch.Tensor,
    r: int,
) -> list[tuple[int, int]]:
    """Find up to r non-overlapping adjacent token pairs with highest similarity.

    Only adjacent pairs (i, i+1) are considered so that the merge operation
    cannot reorder tokens and locality is preserved.

    Args:
        similarity: Pairwise similarity matrix of shape (B=1, T, T).
        r: Maximum number of pairs to return.

    Returns:
        List of (i, i+1) index pairs sorted by similarity descending.
        At most r pairs, possibly fewer when T < 2r.
    """
    # Work with batch index 0 (caller must pass B=1 for inference use-case)
    sim = similarity[0]  # (T, T)
    T = sim.shape[0]

    # Gather scores for adjacent pairs: (i, i+1) for i in [0, T-2]
    if T < 2:
        return []

    # Max pairs we can produce without index overlap: floor((T-1)/2)
    max_pairs = (T - 1) // 2
    effective_r = min(r, max_pairs)
    if effective_r <= 0:
        return []

    # Extract scores for all adjacent pairs
    adj_scores: list[tuple[float, int, int]] = []
    for i in range(T - 1):
        score = sim[i, i + 1].item()
        adj_scores.append((score, i, i + 1))

    # Sort by score descending
    adj_scores.sort(key=lambda t: t[0], reverse=True)

    # Greedily pick non-overlapping pairs (each token index used at most once)
    used: set[int] = set()
    pairs: list[tuple[int, int]] = []
    for score, i, j in adj_scores:
        if i not in used and j not in used:
            pairs.append((i, j))
            used.add(i)
            used.add(j)
            if len(pairs) == effective_r:
                break

    return pairs


# ---------------------------------------------------------------------------
# Merge / unmerge
# ---------------------------------------------------------------------------

def merge_tokens(
    x: torch.Tensor,
    pairs: list[tuple[int, int]],
    mode: str = "mean",
) -> torch.Tensor:
    """Merge token pairs in x.

    For each pair (i, i+1), replace both tokens with a single merged token and
    remove the higher-index token from the sequence.

    Pairs are processed in reverse index order so that indices of earlier pairs
    remain valid after removal.

    Args:
        x: Token tensor of shape (B, T, D).
        pairs: List of (i, i+1) pairs to merge.
        mode: "mean" | "max" | "weighted" (weighted = equal weight, same as mean).

    Returns:
        Merged tensor of shape (B, T - len(pairs), D).
    """
    if not pairs:
        return x

    x = x.clone()

    # Sort pairs so we process highest first → lower indices unaffected
    pairs_desc = sorted(pairs, key=lambda p: p[0], reverse=True)

    for i, j in pairs_desc:
        # i < j always (adjacent)
        a = x[:, i, :]  # (B, D)
        b = x[:, j, :]  # (B, D)

        if mode == "mean" or mode == "weighted":
            merged = (a + b) * 0.5
        elif mode == "max":
            merged = torch.max(a, b)
        else:
            raise ValueError(f"Unknown merge_mode: {mode!r}. Choose 'mean', 'max', or 'weighted'.")

        # Write merged value back to position i
        x[:, i, :] = merged

        # Remove position j from the sequence
        x = torch.cat([x[:, :j, :], x[:, j + 1:, :]], dim=1)

    return x


def unmerge_tokens(
    merged: torch.Tensor,
    pairs: list[tuple[int, int]],
    original_len: int,
) -> torch.Tensor:
    """Restore original sequence length by duplicating merged tokens.

    The inverse of merge_tokens: re-inserts the merged token at the position
    of the removed token (j), duplicating it from position i.

    Pairs must be the same list returned by find_merge_pairs.  They are
    re-inserted in forward order (lowest index first) so that position
    arithmetic mirrors how merge_tokens removed them.

    Args:
        merged: Tensor of shape (B, T_merged, D).
        pairs: Original (i, j) pairs passed to merge_tokens.
        original_len: T before merging.

    Returns:
        Tensor of shape (B, original_len, D).
    """
    if not pairs:
        return merged

    x = merged.clone()

    # Process pairs in ascending order of i (mirror of reverse removal)
    pairs_asc = sorted(pairs, key=lambda p: p[0])

    for i, j in pairs_asc:
        # At position i in the current (shorter) tensor, the merged token lives.
        # We need to insert a copy at index j.
        token = x[:, i, :].unsqueeze(1)  # (B, 1, D)
        x = torch.cat([x[:, :j, :], token, x[:, j:, :]], dim=1)

    return x


# ---------------------------------------------------------------------------
# ToMeLayer — wraps any TransformerBlock with merge/unmerge
# ---------------------------------------------------------------------------

class ToMeLayer(nn.Module):
    """Wraps a TransformerBlock with token merging around the forward call.

    Args:
        base_layer: The original TransformerBlock (or any nn.Module).
        config: ToMeConfig controlling merge behaviour.
        layer_idx: Index of this layer in the stack (used for layer_start check).
    """

    def __init__(
        self,
        base_layer: nn.Module,
        config: ToMeConfig,
        layer_idx: int = 0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        past_kv=None,
        **kwargs,
    ):
        """Run base_layer with optional token merging.

        Accepts the same positional arguments as TransformerBlock.forward
        so that AureliusTransformer can call layer(x, freqs_cis, mask, past_kv)
        without changes.

        If layer_idx < config.layer_start, skip merging entirely.

        TransformerBlock returns (hidden, kv_cache) — this method unpacks,
        processes, then reconstructs the tuple transparently.

        Args:
            x: Input tensor of shape (B, T, D).
            freqs_cis: RoPE frequencies tensor, trimmed to merged length if merging.
            mask: Optional attention mask.
            past_kv: Optional KV cache tuple.
            **kwargs: Any additional keyword arguments forwarded to base_layer.

        Returns:
            Same type as base_layer output (tensor or tuple).
        """
        skip_merge = (
            self.layer_idx < self.config.layer_start
            or self.config.r <= 0
            or x.shape[1] < 2
        )

        if skip_merge:
            return self.base_layer(x, freqs_cis, mask, past_kv, **kwargs)

        # Compute similarity and find pairs using only batch dim 0 (B=1 expected)
        with torch.no_grad():
            sim = compute_token_similarity(x, metric=self.config.similarity_metric)

        # Use only batch 0 for pair selection (inference B=1 convention)
        sim_b1 = sim[:1]  # (1, T, T)
        pairs = find_merge_pairs(sim_b1, r=self.config.r)

        original_len = x.shape[1]

        # Merge input tokens
        x_merged = merge_tokens(x, pairs, mode=self.config.merge_mode)

        # Trim freqs_cis to merged sequence length
        freqs_merged = freqs_cis
        if freqs_cis is not None:
            freqs_merged = freqs_cis[: x_merged.shape[1]]

        # Run base layer on merged sequence
        out = self.base_layer(x_merged, freqs_merged, mask, past_kv, **kwargs)

        # Handle tuple output: (hidden, kv_cache) from TransformerBlock
        if isinstance(out, tuple):
            hidden = out[0]
            rest = out[1:]
            hidden_unmerged = unmerge_tokens(hidden, pairs, original_len)
            return (hidden_unmerged,) + rest

        # Plain tensor output
        return unmerge_tokens(out, pairs, original_len)


# ---------------------------------------------------------------------------
# apply_tome — wrap model layers in-place
# ---------------------------------------------------------------------------

def apply_tome(model: nn.Module, config: ToMeConfig) -> nn.Module:
    """Wrap each layer in model.layers with ToMeLayer in-place.

    Args:
        model: AureliusTransformer with a .layers ModuleList.
        config: ToMeConfig controlling merge behaviour.

    Returns:
        The same model (modified in-place) for convenience.
    """
    for i, layer in enumerate(model.layers):
        model.layers[i] = ToMeLayer(layer, config=config, layer_idx=i)
    return model


# ---------------------------------------------------------------------------
# Speedup estimation
# ---------------------------------------------------------------------------

def estimate_speedup(
    original_seq_len: int,
    r: int,
    n_layers: int,
) -> float:
    """Estimate theoretical speedup from token merging.

    Each merging layer reduces the sequence by r tokens.  Attention cost scales
    quadratically with sequence length, so the speedup is approximated as the
    ratio of the original quadratic cost to the average reduced cost.

    Uses a simple linear average: avg_seq = original - r * n_layers / 2
    (assuming uniform reduction across layers), then clamps to [1.0, 4.0].

    Args:
        original_seq_len: Original sequence length T.
        r: Tokens merged per layer.
        n_layers: Total number of layers that perform merging.

    Returns:
        Estimated speedup factor in [1.0, 4.0].
    """
    if original_seq_len <= 0:
        return 1.0

    # Average sequence length: assume linear reduction each layer
    # After layer k: seq_len - r*k.  Average over n_layers layers:
    # avg = original - r * (n_layers - 1) / 2
    # But a simpler conservative estimate used in the ToMe paper:
    # avg_seq_len = original_seq_len - r * effective_layers
    effective_layers = min(n_layers, original_seq_len // max(r, 1))
    avg_seq_len = max(1, original_seq_len - r * effective_layers)

    speedup = original_seq_len / avg_seq_len
    return float(max(1.0, min(4.0, speedup)))
