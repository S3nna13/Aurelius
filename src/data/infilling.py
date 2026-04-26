"""Text infilling with span masking and fill-in-the-middle (FIM) training.

This module extends FIM to operate on tokenized sequences (integer tensors)
rather than raw strings.  It provides:

- ``InfillingConfig``     — hyper-parameters for span masking and FIM
- ``sample_span_lengths`` — Poisson-based span selection
- ``apply_span_mask``     — corrupt tokens and produce MLM-style labels
- ``fim_transform``       — token-level PSM reordering
- ``InfillingDataset``    — Dataset that applies the above transforms
- ``InfillingTrainer``    — thin training wrapper

Note: the string-level ``fim_transform`` in ``src/data/fim_transform.py``
is intentionally *not* re-used here because it operates on raw text.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

if TYPE_CHECKING:
    import torch.optim


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InfillingConfig:
    """Hyper-parameters for span masking and FIM training."""

    mask_prob: float = 0.15  # fraction of tokens to mask in total
    mean_span_length: float = 3.0  # mean length of each masked span
    fim_prob: float = 0.5  # probability of FIM vs span-mask per sample

    # Special token ids (must already be in the vocabulary)
    prefix_token: int = 1
    suffix_token: int = 2
    middle_token: int = 3


# ---------------------------------------------------------------------------
# Span sampling
# ---------------------------------------------------------------------------


def sample_span_lengths(
    seq_len: int,
    mask_prob: float,
    mean_span_length: float,
    rng: random.Random | None = None,
) -> list[tuple[int, int]]:
    """Sample non-overlapping masked spans using a Poisson process.

    Parameters
    ----------
    seq_len:
        Length of the input sequence.
    mask_prob:
        Expected fraction of tokens to mask.
    mean_span_length:
        Mean length of each span (geometric distribution parameter).
    rng:
        Optional seeded ``random.Random``.  A fresh RNG is created when *None*.

    Returns
    -------
    list of (start, end) pairs (end is exclusive) that are non-overlapping
    and entirely within ``[0, seq_len)``.
    """
    if rng is None:
        rng = random.Random()

    # Number of spans ~ Poisson(expected_masked / mean_span_length)
    expected_masked = seq_len * mask_prob
    lam = max(expected_masked / mean_span_length, 1e-9)
    # Sample from Poisson using sum of exponentials (simple approach for small lambda)
    import math

    # Knuth's algorithm for Poisson sampling
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        p *= rng.random()
        k += 1
    n_spans = max(k - 1, 0)

    if n_spans == 0 or seq_len == 0:
        return []

    # p(length = k) = (1 - q) * q^(k-1), q = 1 - 1/mean_span_length
    # Geometric sampling: length = ceil(log(u) / log(q))
    q = 1.0 - (1.0 / mean_span_length)
    q = min(max(q, 1e-9), 1.0 - 1e-9)

    spans: list[tuple[int, int]] = []
    occupied: set[int] = set()

    for _ in range(n_spans):
        # Sample span length from geometric(1/mean_span_length), min 1
        u = rng.random()
        if u == 0.0:
            u = 1e-12
        length = max(1, int(math.ceil(math.log(u) / math.log(q))))
        length = min(length, seq_len)

        # Try a random start position; skip if overlaps or out of bounds
        max_start = seq_len - length
        if max_start < 0:
            continue

        # Try up to 10 candidate positions
        for _ in range(10):
            start = rng.randint(0, max_start)
            end = start + length
            candidate = set(range(start, end))
            if candidate.isdisjoint(occupied):
                spans.append((start, end))
                occupied |= candidate
                break
        # If we couldn't place after retries, skip this span

    spans.sort(key=lambda s: s[0])
    return spans


# ---------------------------------------------------------------------------
# Span masking
# ---------------------------------------------------------------------------


def apply_span_mask(
    input_ids: Tensor,
    spans: list[tuple[int, int]],
) -> tuple[Tensor, Tensor]:
    """Apply span masking: replace masked tokens with 0, build label tensor.

    Parameters
    ----------
    input_ids:
        1-D integer tensor of shape ``(seq_len,)``.
    spans:
        List of ``(start, end)`` pairs (end exclusive) from
        ``sample_span_lengths``.

    Returns
    -------
    masked_ids:
        Copy of *input_ids* with masked positions set to ``0``.
    labels:
        Same shape as *input_ids*; ``-100`` at non-masked positions,
        original token id at masked positions.
    """
    masked_ids = input_ids.clone()
    labels = torch.full_like(input_ids, fill_value=-100)

    for start, end in spans:
        labels[start:end] = input_ids[start:end]
        masked_ids[start:end] = 0

    return masked_ids, labels


# ---------------------------------------------------------------------------
# Token-level FIM transform
# ---------------------------------------------------------------------------


def fim_transform(
    input_ids: Tensor,
    prefix_token: int,
    suffix_token: int,
    middle_token: int,
    rng: random.Random | None = None,
) -> Tensor:
    """Rearrange *input_ids* into PSM (Prefix-Suffix-Middle) FIM format.

    The sequence is split at a random point into prefix and suffix; the
    middle is taken as the rest.  Specifically:

    - prefix  = input_ids[:split1]
    - middle  = input_ids[split1:split2]
    - suffix  = input_ids[split2:]

    The output is::

        [prefix_token, *prefix, suffix_token, *suffix, middle_token, *middle]

    Parameters
    ----------
    input_ids:
        1-D integer tensor of shape ``(seq_len,)``.
    prefix_token, suffix_token, middle_token:
        Integer token ids for the special separator tokens.
    rng:
        Optional seeded ``random.Random``.

    Returns
    -------
    Tensor of shape ``(seq_len + 3,)`` — original tokens plus the three
    inserted special tokens.
    """
    if rng is None:
        rng = random.Random()

    seq_len = input_ids.shape[0]

    # Need at least 2 tokens to split into non-empty prefix/suffix
    if seq_len < 2:
        # Degenerate case: everything is the prefix, empty middle/suffix
        prefix = input_ids
        middle = input_ids.new_empty(0)
        suffix = input_ids.new_empty(0)
    else:
        # Two independent split points
        i, j = sorted(rng.sample(range(1, seq_len), min(2, seq_len - 1)))
        if i == j:
            i = max(1, j - 1)
        prefix = input_ids[:i]
        middle = input_ids[i:j]
        suffix = input_ids[j:]

    input_ids.new_tensor([prefix_token, suffix_token, middle_token])

    result = torch.cat(
        [
            input_ids.new_tensor([prefix_token]),
            prefix,
            input_ids.new_tensor([suffix_token]),
            suffix,
            input_ids.new_tensor([middle_token]),
            middle,
        ]
    )
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class InfillingDataset(Dataset):
    """Dataset that applies span masking or FIM to pre-tokenized samples.

    Parameters
    ----------
    samples:
        List of 1-D integer tensors (tokenized sequences).
    config:
        ``InfillingConfig`` controlling transform probabilities and tokens.
    """

    def __init__(
        self,
        samples: list[Tensor],
        config: InfillingConfig | None = None,
    ) -> None:
        self.samples = samples
        self.config = config or InfillingConfig()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Return ``(input_ids, labels)`` after applying an infilling transform.

        With probability ``fim_prob`` the sample is FIM-transformed;
        otherwise span masking is applied.

        For FIM samples the labels are set to ``-100`` everywhere (the task
        is auto-regressive next-token prediction, so no explicit MLM labels).
        For span-masked samples the labels follow ``apply_span_mask``.
        """
        ids = self.samples[idx]
        cfg = self.config
        rng = random.Random(idx)  # deterministic per index

        if rng.random() < cfg.fim_prob:
            # FIM transform: auto-regressive objective, no masked positions
            transformed = fim_transform(
                ids,
                cfg.prefix_token,
                cfg.suffix_token,
                cfg.middle_token,
                rng=rng,
            )
            labels = torch.full_like(transformed, fill_value=-100)
            return transformed, labels
        else:
            # Span masking
            spans = sample_span_lengths(
                len(ids),
                cfg.mask_prob,
                cfg.mean_span_length,
                rng=rng,
            )
            masked_ids, labels = apply_span_mask(ids, spans)
            return masked_ids, labels


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class InfillingTrainer:
    """Trains a model on the masked-LM infilling objective.

    Parameters
    ----------
    model:
        ``AureliusTransformer`` (or any model with the same API:
        ``loss, logits, pkv = model(input_ids)``).
    config:
        ``InfillingConfig`` controlling span masking.
    optimizer:
        Any ``torch.optim.Optimizer``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: InfillingConfig | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.model = model
        self.config = config or InfillingConfig()
        self.optimizer = optimizer

    def train_step(self, input_ids: Tensor) -> dict:
        """Perform one gradient-update step on *input_ids*.

        Span masking is applied internally.  Cross-entropy loss is computed
        only over the masked positions (where labels != -100).

        Parameters
        ----------
        input_ids:
            2-D tensor of shape ``(batch, seq_len)``.

        Returns
        -------
        dict with keys:
            ``"loss"``      — scalar float (Python float)
            ``"n_masked"``  — total number of masked tokens in the batch (int)
        """
        cfg = self.config
        B, S = input_ids.shape

        # Build masked inputs and labels for each item in the batch
        masked_list: list[Tensor] = []
        labels_list: list[Tensor] = []
        rng = random.Random()

        for b in range(B):
            spans = sample_span_lengths(S, cfg.mask_prob, cfg.mean_span_length, rng=rng)
            m_ids, lbl = apply_span_mask(input_ids[b], spans)
            masked_list.append(m_ids)
            labels_list.append(lbl)

        masked_input = torch.stack(masked_list, dim=0)  # (B, S)
        labels = torch.stack(labels_list, dim=0)  # (B, S)

        n_masked = int((labels != -100).sum().item())

        # Ensure at least one masked token so the loss is meaningful
        if n_masked == 0:
            # Force-mask the first token of every row
            labels[:, 0] = input_ids[:, 0]
            masked_input[:, 0] = 0
            n_masked = B

        # Forward pass — model returns (loss, logits, pkv)
        # We pass labels so the model can compute loss internally if it wants,
        # but we recompute it ourselves to restrict to masked positions.
        _loss, logits, _pkv = self.model(masked_input)

        # Compute masked-LM loss manually
        # logits: (B, S, vocab_size)  labels: (B, S)
        vocab_size = logits.shape[-1]
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )

        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item(), "n_masked": n_masked}
