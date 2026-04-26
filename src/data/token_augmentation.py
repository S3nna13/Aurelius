"""
Token-level data augmentation for LLM training.

Augmentation techniques that operate on tokenized sequences (integers),
not raw text. Useful for training LLMs on limited data.
"""

import random
from dataclasses import dataclass

import torch


@dataclass
class AugmentationConfig:
    random_deletion_prob: float = 0.1  # probability of deleting a token
    random_swap_prob: float = 0.1  # probability of swapping adjacent tokens
    random_insertion_prob: float = 0.0  # probability of inserting random token
    mask_token_id: int = 0  # token id used for masking (like <mask>)
    mask_prob: float = 0.15  # MLM-style masking probability
    vocab_size: int = 128  # needed for random token insertion
    keep_special_tokens: bool = True  # don't augment first/last token (BOS/EOS)
    seed: int | None = None


def random_deletion(
    tokens: list[int],
    prob: float = 0.1,
    keep_special_tokens: bool = True,
    rng: random.Random | None = None,
) -> list[int]:
    """Delete each token with probability prob. Never deletes first/last if keep_special."""
    if rng is None:
        rng = random.Random()

    if len(tokens) == 0:
        return tokens[:]

    if keep_special_tokens and len(tokens) <= 2:
        return tokens[:]

    result = []
    for i, tok in enumerate(tokens):
        # Always keep first and last if keep_special_tokens
        if keep_special_tokens and (i == 0 or i == len(tokens) - 1):
            result.append(tok)
        elif rng.random() >= prob:
            result.append(tok)

    return result


def random_swap(
    tokens: list[int],
    prob: float = 0.1,
    keep_special_tokens: bool = True,
    rng: random.Random | None = None,
) -> list[int]:
    """Swap each token with its right neighbor with probability prob."""
    if rng is None:
        rng = random.Random()

    result = tokens[:]
    n = len(result)
    if n < 2:
        return result

    # Determine the valid range for swapping
    start = 1 if keep_special_tokens else 0
    end = n - 2 if keep_special_tokens else n - 1  # last swappable index (i, i+1)

    i = start
    while i <= end:
        if rng.random() < prob:
            result[i], result[i + 1] = result[i + 1], result[i]
            i += 2  # skip next to avoid double-swap
        else:
            i += 1

    return result


def random_insertion(
    tokens: list[int],
    prob: float = 0.1,
    vocab_size: int = 128,
    keep_special_tokens: bool = True,
    rng: random.Random | None = None,
) -> list[int]:
    """Insert a random token after each position with probability prob."""
    if rng is None:
        rng = random.Random()

    if len(tokens) == 0:
        return tokens[:]

    result = []
    n = len(tokens)

    for i, tok in enumerate(tokens):
        result.append(tok)
        # Don't insert after the last token (which would be before EOS)
        # Don't insert after BOS (position 0) if keep_special_tokens
        if keep_special_tokens and (i == 0 or i == n - 1):
            continue
        if rng.random() < prob:
            new_tok = rng.randint(0, vocab_size - 1)
            result.append(new_tok)

    return result


def mlm_masking(
    tokens: list[int],
    mask_token_id: int,
    mask_prob: float = 0.15,
    vocab_size: int = 128,
    keep_special_tokens: bool = True,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
    """
    BERT-style masked language modeling.
    80% replace with mask_token_id
    10% replace with random token
    10% keep original
    Returns: (masked_tokens, labels) where labels[i] = original if masked, else -100
    """
    if rng is None:
        rng = random.Random()

    masked = tokens[:]
    labels = [-100] * len(tokens)

    for i, tok in enumerate(tokens):
        if keep_special_tokens and (i == 0 or i == len(tokens) - 1):
            continue
        if rng.random() < mask_prob:
            labels[i] = tok
            r = rng.random()
            if r < 0.8:
                masked[i] = mask_token_id
            elif r < 0.9:
                masked[i] = rng.randint(0, vocab_size - 1)
            # else keep original (10%)

    return masked, labels


def span_masking(
    tokens: list[int],
    mask_token_id: int,
    avg_span_length: float = 3.0,
    mask_ratio: float = 0.15,
    keep_special_tokens: bool = True,
    rng: random.Random | None = None,
) -> tuple[list[int], list[int]]:
    """
    T5-style span masking: mask contiguous spans of tokens.
    Returns: (masked, labels) of same length as input.
    labels[i] = original token if masked, else -100.
    """
    if rng is None:
        rng = random.Random()

    n = len(tokens)
    masked = tokens[:]
    labels = [-100] * n

    if n == 0:
        return masked, labels

    start = 1 if keep_special_tokens else 0
    end = n - 1 if keep_special_tokens else n  # exclusive

    eligible = end - start
    if eligible <= 0:
        return masked, labels

    num_to_mask = max(1, int(eligible * mask_ratio))
    masked_count = 0

    # Generate spans using geometric-like sampling
    i = start
    while masked_count < num_to_mask and i < end:
        # Sample span length from geometric distribution approximation
        span_len = max(1, int(rng.expovariate(1.0 / avg_span_length)))
        span_len = min(span_len, num_to_mask - masked_count, end - i)

        for j in range(i, i + span_len):
            labels[j] = tokens[j]
            masked[j] = mask_token_id
            masked_count += 1

        i += span_len
        # Skip some tokens before the next span
        gap = max(1, int(rng.expovariate(1.0 / avg_span_length)))
        i += gap

    return masked, labels


def token_cutout(
    tokens: list[int],
    cutout_len: int = 5,
    mask_token_id: int = 0,
    rng: random.Random | None = None,
) -> list[int]:
    """Replace a contiguous random span of length cutout_len with mask_token_id."""
    if rng is None:
        rng = random.Random()

    n = len(tokens)
    if n == 0 or cutout_len <= 0:
        return tokens[:]

    cutout_len = min(cutout_len, n)
    start = rng.randint(0, n - cutout_len)
    result = tokens[:]
    for i in range(start, start + cutout_len):
        result[i] = mask_token_id

    return result


def mixup_sequences(
    tokens_a: list[int],
    tokens_b: list[int],
    alpha: float = 0.5,
) -> list[int]:
    """
    Token-level mixup: interleave tokens from two sequences.
    Returns sequence of length max(len(a), len(b)).
    At each position, choose from a with prob alpha, else b.
    """
    rng = random.Random()
    len_a = len(tokens_a)
    len_b = len(tokens_b)
    out_len = max(len_a, len_b)

    result = []
    for i in range(out_len):
        has_a = i < len_a
        has_b = i < len_b

        if has_a and has_b:
            if rng.random() < alpha:
                result.append(tokens_a[i])
            else:
                result.append(tokens_b[i])
        elif has_a:
            result.append(tokens_a[i])
        else:
            result.append(tokens_b[i])

    return result


class TokenAugmentor:
    """Applies a configurable sequence of token augmentations."""

    def __init__(self, config: AugmentationConfig):
        self.config = config
        if config.seed is not None:
            self._rng = random.Random(config.seed)
        else:
            self._rng = random.Random()

    def augment(self, tokens: list[int]) -> list[int]:
        """Apply all enabled augmentations in order."""
        cfg = self.config
        result = tokens[:]

        if cfg.random_deletion_prob > 0.0:
            result = random_deletion(
                result,
                prob=cfg.random_deletion_prob,
                keep_special_tokens=cfg.keep_special_tokens,
                rng=self._rng,
            )

        if cfg.random_swap_prob > 0.0:
            result = random_swap(
                result,
                prob=cfg.random_swap_prob,
                keep_special_tokens=cfg.keep_special_tokens,
                rng=self._rng,
            )

        if cfg.random_insertion_prob > 0.0:
            result = random_insertion(
                result,
                prob=cfg.random_insertion_prob,
                vocab_size=cfg.vocab_size,
                keep_special_tokens=cfg.keep_special_tokens,
                rng=self._rng,
            )

        return result

    def augment_batch(self, batch: list[list[int]]) -> list[list[int]]:
        """Augment a batch of token sequences."""
        return [self.augment(tokens) for tokens in batch]

    def augment_tensor(self, tokens: torch.Tensor) -> torch.Tensor:
        """Augment a 1D or 2D (B, T) tensor of token ids."""
        if tokens.dim() == 1:
            token_list = tokens.tolist()
            augmented = self.augment(token_list)
            return torch.tensor(augmented, dtype=tokens.dtype)
        elif tokens.dim() == 2:
            results = []
            for row in tokens:
                token_list = row.tolist()
                augmented = self.augment(token_list)
                results.append(torch.tensor(augmented, dtype=tokens.dtype))
            # Pad to same length if needed (sequences may differ after deletion/insertion)
            max_len = max(r.shape[0] for r in results)
            padded = []
            for r in results:
                if r.shape[0] < max_len:
                    pad = torch.zeros(max_len - r.shape[0], dtype=tokens.dtype)
                    r = torch.cat([r, pad])
                padded.append(r)
            return torch.stack(padded)
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {tokens.dim()}D")

    def mlm_augment(self, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (masked_input, labels) tensors for MLM training."""
        masked, labels = mlm_masking(
            tokens,
            mask_token_id=self.config.mask_token_id,
            mask_prob=self.config.mask_prob,
            vocab_size=self.config.vocab_size,
            keep_special_tokens=self.config.keep_special_tokens,
            rng=self._rng,
        )
        return torch.tensor(masked, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
