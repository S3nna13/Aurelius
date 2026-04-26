"""Constrained decoding — enforces structural constraints during generation.

Provides token-level constraints (allowed/banned tokens), minimum length
enforcement, prefix forcing, and higher-level greedy/sampling decoders that
apply all of these constraints in a unified pipeline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# ConstraintConfig
# ---------------------------------------------------------------------------


@dataclass
class ConstraintConfig:
    """Configuration for constrained decoding."""

    allowed_tokens: list[int] | None = None
    banned_tokens: list[int] | None = None
    min_new_tokens: int = 0
    max_new_tokens: int = 100
    force_eos_token: int | None = None
    prefix_tokens: list[int] | None = None


# ---------------------------------------------------------------------------
# apply_token_constraints
# ---------------------------------------------------------------------------


def apply_token_constraints(
    logits: Tensor,
    allowed: list[int] | None = None,
    banned: list[int] | None = None,
) -> Tensor:
    """Apply token-level logit constraints.

    Banned tokens are set to -inf.  If an allowed list is provided, all tokens
    NOT in the allowed list are set to -inf (applied after banned suppression).

    Args:
        logits: Shape ``(vocab_size,)`` or ``(B, vocab_size)``.
        allowed: Optional list of token ids that are permitted.
        banned: Optional list of token ids that are forbidden.

    Returns:
        Modified logits tensor of the same shape.
    """
    logits = logits.clone()
    neg_inf = float("-inf")

    if banned:
        for tok in banned:
            if logits.dim() == 1:
                if 0 <= tok < logits.shape[-1]:
                    logits[tok] = neg_inf
            else:
                logits[:, tok] = neg_inf

    if allowed is not None:
        vocab_size = logits.shape[-1]
        # Build a mask: True for tokens NOT in allowed list
        mask = torch.ones(vocab_size, dtype=torch.bool, device=logits.device)
        for tok in allowed:
            if 0 <= tok < vocab_size:
                mask[tok] = False
        if logits.dim() == 1:
            logits[mask] = neg_inf
        else:
            logits[:, mask] = neg_inf

    return logits


# ---------------------------------------------------------------------------
# apply_min_length_constraint
# ---------------------------------------------------------------------------


def apply_min_length_constraint(
    logits: Tensor,
    current_len: int,
    min_len: int,
    eos_token_id: int,
) -> Tensor:
    """Suppress EOS before the minimum length is reached.

    Args:
        logits: Shape ``(vocab_size,)`` or ``(B, vocab_size)``.
        current_len: Number of tokens generated so far (not counting prompt).
        min_len: Minimum number of new tokens required.
        eos_token_id: Token id for end-of-sequence.

    Returns:
        Modified logits tensor.
    """
    if current_len < min_len:
        logits = logits.clone()
        if logits.dim() == 1:
            logits[eos_token_id] = float("-inf")
        else:
            logits[:, eos_token_id] = float("-inf")
    return logits


# ---------------------------------------------------------------------------
# force_prefix
# ---------------------------------------------------------------------------


def force_prefix(
    generated: list[int],
    prefix: list[int],
    step: int,
) -> int | None:
    """Return the forced token for the current decoding step.

    If ``step < len(prefix)``, the token at ``prefix[step]`` must be produced.
    Otherwise returns ``None`` (no forcing).

    Args:
        generated: Tokens generated so far (not used directly, kept for API
                   symmetry with callers that track state externally).
        prefix: The sequence of tokens that must appear at the start of output.
        step: Zero-based index of the current generation step.

    Returns:
        The forced token id, or ``None`` if beyond the prefix.
    """
    if step < len(prefix):
        return prefix[step]
    return None


# ---------------------------------------------------------------------------
# LogitProcessor
# ---------------------------------------------------------------------------


class LogitProcessor:
    """Applies all constraints from a :class:`ConstraintConfig` to logits.

    Constraints are applied in order:
    1. Prefix forcing — if still within the prefix, set all logits to -inf
       except the forced token.
    2. Allowed / banned token masking.
    3. Minimum-length EOS suppression.
    """

    def __init__(self, config: ConstraintConfig) -> None:
        self.config = config

    def __call__(self, logits: Tensor, generated_ids: list[int]) -> Tensor:
        """Apply all constraints and return modified ``(vocab_size,)`` logits.

        Args:
            logits: Raw model logits, shape ``(vocab_size,)``.
            generated_ids: Tokens generated so far (excludes the prompt).

        Returns:
            Constrained logits tensor with shape ``(vocab_size,)``.
        """
        logits = logits.clone()
        step = len(generated_ids)

        # 1. Prefix forcing
        if self.config.prefix_tokens is not None:
            forced = force_prefix(generated_ids, self.config.prefix_tokens, step)
            if forced is not None:
                mask = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
                if 0 <= forced < logits.shape[-1]:
                    mask[forced] = False
                logits[mask] = float("-inf")
                return logits  # prefix overrides everything else

        # 2. Allowed / banned token constraints
        logits = apply_token_constraints(
            logits,
            allowed=self.config.allowed_tokens,
            banned=self.config.banned_tokens,
        )

        # 3. Minimum-length EOS suppression
        if self.config.force_eos_token is not None:
            logits = apply_min_length_constraint(
                logits,
                current_len=step,
                min_len=self.config.min_new_tokens,
                eos_token_id=self.config.force_eos_token,
            )

        return logits


# ---------------------------------------------------------------------------
# ConstrainedGreedyDecoder
# ---------------------------------------------------------------------------


class ConstrainedGreedyDecoder:
    """Greedy decoder that applies :class:`ConstraintConfig` at each step.

    Args:
        model_fn: Callable that accepts ``(input_ids: Tensor)`` of shape
                  ``(1, seq_len)`` and returns logits of shape
                  ``(1, seq_len, vocab_size)``.
        config: Constraint configuration.
        eos_token_id: Token id that signals end of sequence.
    """

    def __init__(
        self,
        model_fn: Callable,
        config: ConstraintConfig,
        eos_token_id: int = 2,
    ) -> None:
        self.model_fn = model_fn
        self.config = config
        self.eos_token_id = eos_token_id
        self._processor = LogitProcessor(config)

    def decode(self, input_ids: Tensor) -> Tensor:
        """Greedily decode from ``input_ids`` with constraints applied.

        Args:
            input_ids: Prompt token ids, shape ``(seq_len,)`` or
                       ``(1, seq_len)``.

        Returns:
            Generated token ids including the original prompt, shape
            ``(T_out,)`` where ``T_out = prompt_len + n_generated``.
        """
        if input_ids.dim() == 1:
            current_ids = input_ids.unsqueeze(0)  # (1, T)
        else:
            current_ids = input_ids.clone()

        generated: list[int] = []

        for _ in range(self.config.max_new_tokens):
            logits_3d = self.model_fn(current_ids)  # (1, T, V)
            step_logits = logits_3d[0, -1, :]  # (V,)

            step_logits = self._processor(step_logits, generated)
            next_token = int(torch.argmax(step_logits).item())
            generated.append(next_token)

            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=current_ids.device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

            # Stop at EOS
            if next_token == self.eos_token_id:
                break

        input_ids.shape[-1]
        all_ids = current_ids[0]  # (prompt_len + n_generated,)
        return all_ids


# ---------------------------------------------------------------------------
# ConstrainedSampler
# ---------------------------------------------------------------------------


class ConstrainedSampler:
    """Multinomial sampler with :class:`ConstraintConfig` constraints applied.

    Args:
        model_fn: Same signature as :class:`ConstrainedGreedyDecoder`.
        config: Constraint configuration.
        eos_token_id: Token id that signals end of sequence.
        temperature: Sampling temperature; higher = more random.
    """

    def __init__(
        self,
        model_fn: Callable,
        config: ConstraintConfig,
        eos_token_id: int = 2,
        temperature: float = 1.0,
    ) -> None:
        self.model_fn = model_fn
        self.config = config
        self.eos_token_id = eos_token_id
        self.temperature = temperature
        self._processor = LogitProcessor(config)

    def _sample_one(self, input_ids: Tensor) -> Tensor:
        """Generate a single sampled sequence."""
        if input_ids.dim() == 1:
            current_ids = input_ids.unsqueeze(0)
        else:
            current_ids = input_ids.clone()

        generated: list[int] = []

        for _ in range(self.config.max_new_tokens):
            logits_3d = self.model_fn(current_ids)
            step_logits = logits_3d[0, -1, :]  # (V,)

            step_logits = self._processor(step_logits, generated)

            # Temperature scaling then sample
            if self.temperature != 1.0:
                step_logits = step_logits / self.temperature

            probs = torch.softmax(step_logits, dim=-1)
            next_token = int(torch.multinomial(probs, num_samples=1).item())
            generated.append(next_token)

            next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=current_ids.device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

            if next_token == self.eos_token_id:
                break

        return current_ids[0]

    def sample(self, input_ids: Tensor, n_samples: int = 1) -> list[Tensor]:
        """Draw ``n_samples`` independently sampled sequences.

        Args:
            input_ids: Prompt token ids, shape ``(seq_len,)`` or
                       ``(1, seq_len)``.
            n_samples: Number of sequences to generate.

        Returns:
            List of ``n_samples`` tensors, each of shape ``(T_out,)``.
        """
        return [self._sample_one(input_ids) for _ in range(n_samples)]


# ---------------------------------------------------------------------------
# compute_constraint_satisfaction
# ---------------------------------------------------------------------------


def compute_constraint_satisfaction(
    sequence: Tensor,
    config: ConstraintConfig,
) -> dict[str, bool]:
    """Check whether a generated sequence satisfies the configured constraints.

    Checks performed:
    - ``"no_banned_tokens"``: True if none of the banned tokens appear in the
      sequence.
    - ``"all_allowed"``: True if every token in the sequence is in the allowed
      list (or if no allowed list is specified).
    - ``"min_length_met"``: True if the sequence length is at least
      ``config.min_new_tokens``.

    Args:
        sequence: 1-D tensor of token ids (may include prompt tokens).
        config: The constraint configuration to check against.

    Returns:
        Dictionary mapping constraint names to boolean satisfaction values.
    """
    token_set = set(sequence.tolist())

    # no_banned_tokens
    if config.banned_tokens:
        no_banned = not any(t in token_set for t in config.banned_tokens)
    else:
        no_banned = True

    # all_allowed
    if config.allowed_tokens is not None:
        allowed_set = set(config.allowed_tokens)
        all_allowed = all(t in allowed_set for t in token_set)
    else:
        all_allowed = True

    # min_length_met
    min_length_met = len(sequence) >= config.min_new_tokens

    return {
        "no_banned_tokens": no_banned,
        "all_allowed": all_allowed,
        "min_length_met": min_length_met,
    }
