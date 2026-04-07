"""Standalone perplexity evaluator for AureliusTransformer.

Computes perplexity = exp(mean negative log-likelihood per token) on
arbitrary token sequences. Works directly with the model, no external deps.
"""
from __future__ import annotations

import math
import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class PerplexityResult:
    perplexity: float
    avg_nll: float
    total_tokens: int
    n_sequences: int

    def __repr__(self) -> str:
        return (
            f"PerplexityResult(ppl={self.perplexity:.2f}, "
            f"nll={self.avg_nll:.4f}, "
            f"tokens={self.total_tokens:,})"
        )


@torch.no_grad()
def compute_perplexity(
    model: nn.Module,
    token_sequences: list[list[int]] | list[torch.Tensor],
    stride: int | None = None,
    device: str | torch.device = "cpu",
) -> PerplexityResult:
    """Compute perplexity of a list of token sequences.

    For sequences longer than max_seq_len, uses a sliding window with stride
    to compute an unbiased estimate (Jurafsky & Martin sliding window PPL).

    Args:
        model: AureliusTransformer (any model with forward returning (loss, logits, _)).
        token_sequences: List of token id lists or tensors.
        stride: Sliding window stride for long sequences.
                Defaults to max_seq_len // 2.
        device: Device to run on.

    Returns:
        PerplexityResult with per-token NLL and overall PPL.
    """
    model.eval()
    model_device = next(model.parameters()).device
    max_seq_len = model.config.max_seq_len

    if stride is None:
        stride = max_seq_len // 2

    total_nll = 0.0
    total_tokens = 0
    n_sequences = 0

    for seq in token_sequences:
        if isinstance(seq, torch.Tensor):
            ids = seq.long()
        else:
            ids = torch.tensor(seq, dtype=torch.long)

        ids = ids.unsqueeze(0)  # (1, seq_len)
        seq_len = ids.shape[1]
        n_sequences += 1

        if seq_len <= max_seq_len:
            # Short sequence: single forward pass
            ids_dev = ids.to(model_device)
            _, logits, _ = model(ids_dev)
            # NLL: logits[:, :-1] predicts ids[:, 1:]
            if seq_len > 1:
                nll = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                    ids_dev[:, 1:].contiguous().view(-1),
                    reduction="sum",
                )
                total_nll += nll.item()
                total_tokens += seq_len - 1
        else:
            # Long sequence: sliding window
            pos = 0
            while pos < seq_len - 1:
                end = min(pos + max_seq_len, seq_len)
                window = ids[:, pos:end].to(model_device)

                _, logits, _ = model(window)
                window_len = end - pos

                # On first window, skip first token (no context)
                # On subsequent windows, skip stride tokens (already counted)
                target_start = 0 if pos == 0 else stride

                if window_len - 1 > target_start:
                    nll = F.cross_entropy(
                        logits[:, target_start:window_len-1].contiguous().view(-1, logits.shape[-1]),
                        window[:, target_start+1:window_len].contiguous().view(-1),
                        reduction="sum",
                    )
                    total_nll += nll.item()
                    total_tokens += window_len - 1 - target_start

                pos += stride
                if end == seq_len:
                    break

    if total_tokens == 0:
        return PerplexityResult(float("inf"), float("inf"), 0, n_sequences)

    avg_nll = total_nll / total_tokens
    ppl = math.exp(min(avg_nll, 20))  # cap at exp(20) ~485M to avoid overflow

    logger.info(
        "Perplexity: %.2f (nll=%.4f, tokens=%d, sequences=%d)",
        ppl, avg_nll, total_tokens, n_sequences,
    )

    return PerplexityResult(
        perplexity=ppl,
        avg_nll=avg_nll,
        total_tokens=total_tokens,
        n_sequences=n_sequences,
    )


def perplexity_on_dataset(
    model: nn.Module,
    dataloader,
    device: str | torch.device = "cpu",
) -> PerplexityResult:
    """Compute perplexity on a DataLoader yielding {"input_ids": Tensor} batches.

    Args:
        model: AureliusTransformer.
        dataloader: DataLoader yielding {"input_ids": Tensor} or (input_ids, labels) tuples.
        device: Compute device.

    Returns:
        PerplexityResult.
    """
    sequences = []
    for batch in dataloader:
        if isinstance(batch, dict):
            ids = batch["input_ids"]
        else:
            ids = batch[0]
        # Add each sequence in the batch
        for i in range(ids.shape[0]):
            sequences.append(ids[i].tolist())

    return compute_perplexity(model, sequences, device=device)
