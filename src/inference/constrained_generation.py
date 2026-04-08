"""Constrained generation: prefix/suffix constraints, vocabulary masks, and regex-guided decoding."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class ConstraintConfig:
    """Configuration for constrained generation."""

    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0  # penalize already-seen tokens (1.0 = off)


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """Apply repetition penalty to logits.

    For each token id in generated_ids, divides logits[id] by penalty
    if logit > 0, or multiplies if logit < 0 (standard formula).

    Args:
        logits: shape (V,)
        generated_ids: list of previously generated token ids
        penalty: penalty factor (>1.0 reduces probability of seen tokens)

    Returns:
        Modified logits of shape (V,).
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    logits = logits.clone()
    for token_id in set(generated_ids):
        if token_id < logits.size(0):
            if logits[token_id] > 0:
                logits[token_id] = logits[token_id] / penalty
            else:
                logits[token_id] = logits[token_id] * penalty
    return logits


def build_vocab_mask(allowed_tokens: list[int], vocab_size: int) -> torch.Tensor:
    """Build a boolean vocabulary mask.

    Args:
        allowed_tokens: list of token ids that are allowed
        vocab_size: total vocabulary size V

    Returns:
        Boolean tensor of shape (V,), True for allowed tokens.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for tok in allowed_tokens:
        if 0 <= tok < vocab_size:
            mask[tok] = True
    return mask


def apply_vocab_mask(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply vocabulary mask to logits by setting disallowed logits to -inf.

    Args:
        logits: shape (V,)
        mask: boolean tensor shape (V,), True = allowed

    Returns:
        Logits of shape (V,) with disallowed positions set to -inf.
    """
    logits = logits.clone()
    logits[~mask] = float("-inf")
    return logits


def _sample_top_p(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    """Sample from logits using temperature and top-p (nucleus) sampling.

    Args:
        logits: shape (V,)
        top_p: nucleus probability threshold
        temperature: sampling temperature

    Returns:
        Sampled token id.
    """
    # Apply temperature
    if temperature != 1.0 and temperature > 0.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=0)
        # Remove tokens with cumulative prob above top_p
        sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
        sorted_probs[sorted_indices_to_remove] = 0.0
        sorted_probs = sorted_probs / sorted_probs.sum()
        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
        token_id = sorted_indices[sampled_index].item()
    else:
        token_id = torch.multinomial(probs, num_samples=1).item()

    return int(token_id)


class PrefixConstrainedDecoder:
    """Forces generation to start with a specific token sequence.

    For the first len(required_prefix) steps, the required prefix tokens
    are forced. After that, sampling proceeds freely with top_p / temperature /
    repetition_penalty.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        required_prefix: list[int],
        config: ConstraintConfig,
    ) -> None:
        self.model = model
        self.required_prefix = required_prefix
        self.config = config

    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate tokens with a forced prefix.

        Args:
            input_ids: shape (1, T) prompt token ids

        Returns:
            Full sequence tensor of shape (1, T + max_new_tokens).
        """
        device = input_ids.device
        tokens: list[int] = input_ids[0].tolist()
        generated: list[int] = []

        with torch.no_grad():
            for step in range(self.config.max_new_tokens):
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                output = self.model(ids)
                logits = output[1][0, -1, :]  # (V,)

                if step < len(self.required_prefix):
                    # Force the required prefix token
                    next_token = self.required_prefix[step]
                else:
                    # Free sampling
                    logits = apply_repetition_penalty(
                        logits, generated, self.config.repetition_penalty
                    )
                    next_token = _sample_top_p(
                        logits, self.config.top_p, self.config.temperature
                    )

                tokens.append(next_token)
                generated.append(next_token)

        return torch.tensor([tokens], dtype=torch.long, device=device)


class RegexConstrainedDecoder:
    """Guides generation so output matches a regex pattern (simplified: character-class masks).

    At each step, computes allowed tokens based on which single printable ASCII
    characters would keep the current text compatible with the regex, then applies
    the mask before sampling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        pattern: str,
        tokenizer_decode: Callable[[list[int]], str],
        config: ConstraintConfig,
    ) -> None:
        self.model = model
        self.pattern = pattern
        self.tokenizer_decode = tokenizer_decode
        self.config = config

    def _get_allowed_tokens(self, current_text: str, vocab_size: int) -> torch.Tensor:
        """Compute a boolean mask of allowed tokens given the current text.

        Tries extending current_text with each printable ASCII character and
        checks if the extended text is still compatible with the regex pattern
        (i.e., could lead to a full match). Falls back to all-True mask if
        nothing is allowed.

        Args:
            current_text: text generated so far (not including prompt)
            vocab_size: total vocabulary size

        Returns:
            Boolean tensor of shape (vocab_size,).
        """
        mask = torch.zeros(vocab_size, dtype=torch.bool)
        lookahead_pattern = self.pattern + ".*"

        for token_id in range(vocab_size):
            char = self.tokenizer_decode([token_id])
            extended = current_text + char
            if re.match(lookahead_pattern, extended, re.DOTALL):
                mask[token_id] = True

        # Fallback: allow everything if no tokens pass
        if not mask.any():
            mask = torch.ones(vocab_size, dtype=torch.bool)

        return mask

    def generate(self, input_ids: torch.Tensor, prompt_text: str = "") -> torch.Tensor:
        """Generate tokens constrained by the regex pattern.

        Args:
            input_ids: shape (1, T) prompt token ids
            prompt_text: decoded text of the prompt (used as starting context)

        Returns:
            Full sequence tensor of shape (1, T + max_new_tokens).
        """
        device = input_ids.device
        tokens: list[int] = input_ids[0].tolist()
        generated: list[int] = []
        current_text = prompt_text

        vocab_size = self.model.config.vocab_size

        with torch.no_grad():
            for _ in range(self.config.max_new_tokens):
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                output = self.model(ids)
                logits = output[1][0, -1, :]  # (V,)

                # Compute allowed tokens and apply mask
                allowed_mask = self._get_allowed_tokens(current_text, vocab_size)
                logits = apply_vocab_mask(logits, allowed_mask)

                # Apply repetition penalty
                logits = apply_repetition_penalty(
                    logits, generated, self.config.repetition_penalty
                )

                next_token = _sample_top_p(
                    logits, self.config.top_p, self.config.temperature
                )

                tokens.append(next_token)
                generated.append(next_token)
                current_text += self.tokenizer_decode([next_token])

        return torch.tensor([tokens], dtype=torch.long, device=device)


class BannedTokensDecoder:
    """Generates text while preventing specific tokens from appearing in output.

    At each decoding step the banned token logits are set to -inf before sampling.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        banned_tokens: list[int],
        config: ConstraintConfig,
    ) -> None:
        self.model = model
        self.banned_tokens = banned_tokens
        self.config = config

    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate tokens with banned tokens excluded at each step.

        Args:
            input_ids: shape (1, T) prompt token ids

        Returns:
            Full sequence tensor of shape (1, T + max_new_tokens).
        """
        device = input_ids.device
        tokens: list[int] = input_ids[0].tolist()
        generated: list[int] = []

        vocab_size = self.model.config.vocab_size
        # Build a fixed ban mask (True = allowed, i.e. NOT banned)
        ban_mask = torch.ones(vocab_size, dtype=torch.bool, device=device)
        for tok in self.banned_tokens:
            if 0 <= tok < vocab_size:
                ban_mask[tok] = False

        with torch.no_grad():
            for _ in range(self.config.max_new_tokens):
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                output = self.model(ids)
                logits = output[1][0, -1, :]  # (V,)

                # Apply ban mask
                logits = apply_vocab_mask(logits, ban_mask)

                # Apply repetition penalty
                logits = apply_repetition_penalty(
                    logits, generated, self.config.repetition_penalty
                )

                next_token = _sample_top_p(
                    logits, self.config.top_p, self.config.temperature
                )

                tokens.append(next_token)
                generated.append(next_token)

        return torch.tensor([tokens], dtype=torch.long, device=device)
