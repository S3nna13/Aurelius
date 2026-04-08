"""Chunked cross-entropy loss for memory-efficient large-vocabulary training.

Computes cross-entropy in chunks along the token dimension to avoid
materializing the full (B*S, V) logit matrix at once.

Memory: O(chunk_size * V) instead of O(B * S * V)
For B=4, S=2048, V=128K, chunk=128: 128*128K = 16M params vs 1B params (~64x reduction)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def chunked_cross_entropy(
    logits: torch.Tensor,       # (B, S, V) or (N, V) — model logits
    labels: torch.Tensor,       # (B, S) or (N,) — target token ids
    chunk_size: int = 128,      # tokens processed at once
    ignore_index: int = -100,   # label value to ignore (padding)
    reduction: str = "mean",    # "mean" or "sum"
) -> torch.Tensor:
    """Compute cross-entropy loss in chunks to save memory.

    Mathematically equivalent to F.cross_entropy(logits.view(-1, V), labels.view(-1))
    but uses O(chunk_size * V) memory instead of O(B * S * V).

    Args:
        logits: Model logits, shape (B, S, V) or (N, V)
        labels: Target token ids, shape (B, S) or (N,)
        chunk_size: Number of tokens to process per chunk
        ignore_index: Labels with this value are excluded from loss
        reduction: "mean" (default) or "sum"

    Returns:
        Scalar loss tensor (gradients flow through logits)
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

    V = logits.shape[-1]
    # Flatten to (N, V) and (N,)
    logits_flat = logits.view(-1, V)
    labels_flat = labels.view(-1)

    N = logits_flat.shape[0]

    total_loss = logits_flat.new_zeros(())
    n_tokens = 0

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        logits_chunk = logits_flat[start:end]
        labels_chunk = labels_flat[start:end]

        # Count non-ignored tokens in this chunk
        mask = labels_chunk != ignore_index
        n_tokens += int(mask.sum().item())

        # Use reduction="sum" so we can accumulate across chunks
        chunk_loss = F.cross_entropy(
            logits_chunk,
            labels_chunk,
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_loss = total_loss + chunk_loss

    if reduction == "mean":
        if n_tokens == 0:
            # All tokens ignored — return zero loss (no NaN)
            return total_loss * 0.0
        return total_loss / n_tokens
    else:
        return total_loss


class ChunkedCrossEntropyLoss(torch.nn.Module):
    """Module wrapper for chunked cross-entropy.

    Drop-in replacement for nn.CrossEntropyLoss.
    """

    def __init__(
        self,
        chunk_size: int = 128,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return chunked_cross_entropy(
            logits, labels, self.chunk_size, self.ignore_index, self.reduction
        )

    def memory_usage_bytes(
        self, batch: int, seq_len: int, vocab: int, dtype_bytes: int = 2
    ) -> dict[str, int]:
        """Estimate peak memory usage for full vs chunked computation."""
        full = batch * seq_len * vocab * dtype_bytes
        chunked = self.chunk_size * vocab * dtype_bytes
        return {"full_bytes": full, "chunked_bytes": chunked, "ratio": full // max(1, chunked)}
