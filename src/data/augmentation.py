"""Token-level augmentation transforms for 1D token ID tensors.

Three transforms that operate directly on token IDs without
any tokenizer dependency. The caller passes token IDs directly.

Also provides a functional API (AugmentationConfig, token_masking,
token_replacement, token_deletion, token_insertion, adjacent_swap,
span_masking) plus TokenAugmenter and AugmentedDataset helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class RandomTokenMask:
    """Replace random tokens with mask_id or a random token from [0, vocab_size).

    Args:
        p: Probability of masking each token.
        mask_id: Token ID to use for masking. If None, use a random token.
        vocab_size: Required when mask_id is None.
        seed: Optional seed for reproducibility.
    """

    p: float = 0.15
    mask_id: int | None = None
    vocab_size: int | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.mask_id is None and self.vocab_size is None:
            raise ValueError("vocab_size is required when mask_id is None")

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply masking. Input: 1D (S,) or 2D (B, S) tensor. Returns same shape."""
        output = input_ids.clone()

        gen = torch.Generator()
        if self.seed is not None:
            gen.manual_seed(self.seed)

        mask = torch.rand(input_ids.shape, generator=gen) < self.p

        if self.mask_id is not None:
            output[mask] = self.mask_id
        else:
            assert self.vocab_size is not None  # noqa: S101
            output[mask] = torch.randint(0, self.vocab_size, (mask.sum().item(),), generator=gen)

        return output


@dataclass
class TokenDropout:
    """Delete random tokens from the sequence.

    Args:
        p: Probability of dropping each token.
    """

    p: float = 0.1

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply dropout. Input must be 1D (S,). Returns shortened 1D tensor.
        Never drops all tokens -- guarantees at least 1 token remains."""
        keep_mask = torch.rand(len(input_ids)) >= self.p

        if keep_mask.sum() == 0:
            keep_mask[0] = True

        return input_ids[keep_mask]


@dataclass
class SpanCorruption:
    """T5-style span corruption: replace random spans with sentinel tokens.

    Randomly selects spans of mean_span_length, replaces each with a single
    sentinel token ID (sentinel_start, sentinel_start+1, ...).

    Args:
        p: Fraction of tokens to corrupt.
        mean_span_length: Average length of corrupted spans.
        sentinel_start: First sentinel token ID (e.g., vocab_size - 100).
    """

    p: float = 0.15
    mean_span_length: int = 3
    sentinel_start: int = 32000

    def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply span corruption. Input: 1D (S,). Returns 1D tensor (shorter).
        If no spans selected, returns input unchanged."""
        S = len(input_ids)
        if S == 0:
            return input_ids.clone()

        n_corrupt = max(1, int(S * self.p))
        n_spans = max(1, n_corrupt // self.mean_span_length)

        # Build a boolean mask of which positions are corrupted
        corrupt_mask = torch.zeros(S, dtype=torch.bool)
        for i in range(n_spans):
            start = torch.randint(0, S, ()).item()
            length = max(
                1,
                int(torch.poisson(torch.tensor(float(self.mean_span_length))).item()),
            )
            end = min(S, start + length)
            corrupt_mask[start:end] = True

        # Rebuild sequence: copy non-corrupt tokens, insert sentinels for spans
        result: list[int] = []
        in_span = False
        span_idx = 0
        for i in range(S):
            if corrupt_mask[i]:
                if not in_span:
                    result.append(self.sentinel_start + span_idx)
                    span_idx += 1
                    in_span = True
            else:
                in_span = False
                result.append(input_ids[i].item())

        if not result:
            return input_ids[:1].clone()

        return torch.tensor(result, dtype=input_ids.dtype)


# ---------------------------------------------------------------------------
# Functional augmentation API
# ---------------------------------------------------------------------------


@dataclass
class AugmentationConfig:
    """Configuration for TokenAugmenter."""

    mask_prob: float = 0.15
    replace_prob: float = 0.1
    delete_prob: float = 0.05
    insert_prob: float = 0.05
    swap_prob: float = 0.05
    vocab_size: int = 256
    mask_token_id: int = 1
    seed: int | None = None


def _make_generator(seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def token_masking(
    input_ids: Tensor,
    mask_prob: float,
    mask_token_id: int,
    vocab_size: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """MLM-style masking: replace mask_prob fraction of tokens with mask_token_id.

    Returns (masked_ids (B, T), mask (B, T) bool — True where masked).
    """
    output = input_ids.clone()
    mask = torch.rand(input_ids.shape, generator=generator) < mask_prob
    output[mask] = mask_token_id
    return output, mask


def token_replacement(
    input_ids: Tensor,
    replace_prob: float,
    vocab_size: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly replace replace_prob fraction of tokens with random vocab tokens.

    Returns augmented (B, T) tensor.
    """
    output = input_ids.clone()
    mask = torch.rand(input_ids.shape, generator=generator) < replace_prob
    n = int(mask.sum().item())
    if n > 0:
        output[mask] = torch.randint(
            0, vocab_size, (n,), generator=generator, dtype=input_ids.dtype
        )
    return output


def token_deletion(
    input_ids: Tensor,
    delete_prob: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly delete delete_prob fraction of tokens.

    Expects (1, T) or (T,). Returns (1, T') where T' <= T.
    """
    squeeze = input_ids.dim() == 1
    ids = input_ids.squeeze(0) if not squeeze else input_ids
    keep = torch.rand(ids.shape, generator=generator) >= delete_prob
    if keep.sum() == 0:
        keep[0] = True
    result = ids[keep].unsqueeze(0)
    return result


def token_insertion(
    input_ids: Tensor,
    insert_prob: float,
    vocab_size: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly insert random tokens at insert_prob fraction of positions.

    Expects (1, T). Returns (1, T') where T' >= T.
    """
    ids = input_ids.squeeze(0)
    T = ids.shape[0]
    insert_flags = torch.rand(T, generator=generator) < insert_prob
    tokens: list[int] = []
    for i in range(T):
        if insert_flags[i]:
            rand_tok = int(torch.randint(0, vocab_size, (1,), generator=generator).item())
            tokens.append(rand_tok)
        tokens.append(int(ids[i].item()))
    result = torch.tensor(tokens, dtype=ids.dtype).unsqueeze(0)
    return result


def adjacent_swap(
    input_ids: Tensor,
    swap_prob: float,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Randomly swap adjacent token pairs at swap_prob fraction of positions.

    Returns (B, T) — same length, some adjacent pairs swapped.
    Go left-to-right; when swapping i and i+1, skip i+1 to avoid double-swap.
    """
    output = input_ids.clone()
    B, T = output.shape
    for b in range(B):
        i = 0
        while i < T - 1:
            if torch.rand(1, generator=generator).item() < swap_prob:
                output[b, i], output[b, i + 1] = output[b, i + 1].clone(), output[b, i].clone()
                i += 2  # skip next to avoid double-swap
            else:
                i += 1
    return output


def span_masking(
    input_ids: Tensor,
    span_mask_prob: float,
    mean_span_len: int,
    mask_token_id: int,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """T5-style span masking: mask contiguous spans.

    Returns (masked_ids (B, T), mask (B, T) bool).
    """
    B, T = input_ids.shape
    output = input_ids.clone()
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    n_to_mask = max(1, int(T * span_mask_prob))
    n_spans = max(1, n_to_mask // max(1, mean_span_len))

    for b in range(B):
        for _ in range(n_spans):
            start = int(torch.randint(0, T, (1,), generator=generator).item())
            # sample span length from Poisson
            span_len = max(
                1,
                int(torch.poisson(torch.tensor(float(mean_span_len)), generator=generator).item()),
            )
            end = min(T, start + span_len)
            mask[b, start:end] = True

    output[mask] = mask_token_id
    return output, mask


class TokenAugmenter:
    """Applies a configurable sequence of augmentations."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config
        self._gen: torch.Generator | None = _make_generator(config.seed)

    def augment(self, input_ids: Tensor) -> Tensor:
        """Apply all enabled augmentations in sequence. Returns augmented (B, T*)."""
        cfg = self.config
        ids = input_ids

        # masking
        if cfg.mask_prob > 0:
            ids, _ = token_masking(ids, cfg.mask_prob, cfg.mask_token_id, cfg.vocab_size, self._gen)
        # replacement
        if cfg.replace_prob > 0:
            ids = token_replacement(ids, cfg.replace_prob, cfg.vocab_size, self._gen)

        # swap (works on 2D)
        if cfg.swap_prob > 0:
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
                ids = adjacent_swap(ids, cfg.swap_prob, self._gen)
                ids = ids.squeeze(0)
            else:
                ids = adjacent_swap(ids, cfg.swap_prob, self._gen)

        # deletion (batch size 1 only)
        if cfg.delete_prob > 0:
            if ids.dim() == 2 and ids.shape[0] == 1:
                ids = token_deletion(ids, cfg.delete_prob, self._gen)
            elif ids.dim() == 1:
                ids = token_deletion(ids.unsqueeze(0), cfg.delete_prob, self._gen).squeeze(0)

        # insertion (batch size 1 only)
        if cfg.insert_prob > 0:
            if ids.dim() == 2 and ids.shape[0] == 1:
                ids = token_insertion(ids, cfg.insert_prob, cfg.vocab_size, self._gen)
            elif ids.dim() == 1:
                ids = token_insertion(
                    ids.unsqueeze(0), cfg.insert_prob, cfg.vocab_size, self._gen
                ).squeeze(0)

        return ids

    def augment_with_labels(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return (augmented_ids, original_ids) — for reconstruction training."""
        return self.augment(input_ids), input_ids.clone()


class AugmentedDataset:
    """Wraps a list of token sequences with on-the-fly augmentation."""

    def __init__(self, sequences: list[Tensor], augmenter: TokenAugmenter) -> None:
        self._sequences = sequences
        self._augmenter = augmenter

    def __len__(self) -> int:
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Returns (augmented, original) for sequence at idx."""
        original = self._sequences[idx]
        augmented = self._augmenter.augment(original)
        return augmented, original

    def get_batch(self, indices: list[int]) -> tuple[Tensor, Tensor]:
        """Collate a batch from multiple indices. Pad to max length.

        Returns (augmented_batch (B, T_max), original_batch (B, T_max)).
        """
        pairs = [self[i] for i in indices]
        aug_list = [p[0].squeeze(0) if p[0].dim() > 1 else p[0] for p in pairs]
        ori_list = [p[1].squeeze(0) if p[1].dim() > 1 else p[1] for p in pairs]

        T_aug = max(t.shape[0] for t in aug_list)
        T_ori = max(t.shape[0] for t in ori_list)

        B = len(indices)
        aug_batch = torch.zeros(B, T_aug, dtype=aug_list[0].dtype)
        ori_batch = torch.zeros(B, T_ori, dtype=ori_list[0].dtype)

        for i, (a, o) in enumerate(zip(aug_list, ori_list)):
            aug_batch[i, : a.shape[0]] = a
            ori_batch[i, : o.shape[0]] = o

        return aug_batch, ori_batch
