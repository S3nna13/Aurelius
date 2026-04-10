"""Lexically constrained decoding via constraint automaton.

Ensures required token sequences (phrases) appear in generated output.
This is distinct from format_enforcer.py (JSON/format constraints) and
logit_processors.py (statistical transformations) — here we use an explicit
constraint automaton that tracks progress through required token sequences
and steers the model's logit distribution to satisfy them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# ConstraintConfig
# ---------------------------------------------------------------------------

@dataclass
class ConstraintConfig:
    """Configuration for lexically constrained decoding."""

    constraints: list[list[int]]  # each inner list is a required token sequence
    bank_strategy: str = "ordered"  # "ordered" | "any"


# ---------------------------------------------------------------------------
# ConstraintState
# ---------------------------------------------------------------------------

class ConstraintState:
    """Tracks progress through a single required token sequence."""

    def __init__(self, tokens: list[int]) -> None:
        self._tokens = list(tokens)
        self._pointer: int = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def advance(self, token: int) -> bool:
        """Advance pointer if token matches next expected token.

        Returns True if the constraint just completed (pointer reached end).
        Returns False otherwise (no match or already completed).
        """
        if self.completed:
            return False
        if token == self._tokens[self._pointer]:
            self._pointer += 1
            return self.completed
        return False

    def reset(self) -> None:
        """Reset progress to the beginning of the constraint sequence."""
        self._pointer = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def completed(self) -> bool:
        """True when all tokens in the sequence have been matched."""
        return self._pointer >= len(self._tokens)

    @property
    def next_tokens(self) -> list[int]:
        """Tokens that would advance this constraint.

        Returns an empty list if the constraint is already completed or
        the token sequence is empty.
        """
        if self.completed or not self._tokens:
            return []
        return [self._tokens[self._pointer]]


# ---------------------------------------------------------------------------
# ConstraintBank
# ---------------------------------------------------------------------------

class ConstraintBank:
    """Manages multiple ConstraintState objects."""

    def __init__(self, constraints: list[list[int]], strategy: str = "ordered") -> None:
        self._strategy = strategy
        self._states: list[ConstraintState] = [
            ConstraintState(seq) for seq in constraints
        ]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def advance(self, token: int) -> None:
        """Update all constraint states with the given token."""
        for state in self._states:
            state.advance(token)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def all_completed(self) -> bool:
        """True when every constraint has been satisfied."""
        return all(s.completed for s in self._states)

    def get_active_next_tokens(self) -> set[int]:
        """Return the union of next_tokens from all uncompleted constraints."""
        result: set[int] = set()
        for state in self._states:
            if not state.completed:
                result.update(state.next_tokens)
        return result


# ---------------------------------------------------------------------------
# apply_constraint_mask
# ---------------------------------------------------------------------------

def apply_constraint_mask(
    logits: Tensor,
    bank: ConstraintBank,
    penalty: float = -1e9,
) -> Tensor:
    """Apply constraint-based logit adjustments.

    If any constraint is currently active (has next tokens it needs to see),
    the logits are modified so that:
    - Required next tokens keep their original logit values (+ 0).
    - All other tokens receive a large penalty so they are suppressed.

    If no constraint is active (all completed or bank is empty), logits are
    returned unchanged.

    Args:
        logits: Tensor of shape (vocab_size,) or (B, vocab_size).
        bank: The ConstraintBank tracking all constraint states.
        penalty: Value added to non-required tokens when a constraint is active.

    Returns:
        Tensor of the same shape as logits.
    """
    active_tokens = bank.get_active_next_tokens()
    if not active_tokens:
        # No active constraints — leave logits untouched.
        return logits

    vocab_size = logits.shape[-1]
    bias = torch.full((vocab_size,), penalty, dtype=logits.dtype, device=logits.device)
    for tok in active_tokens:
        if 0 <= tok < vocab_size:
            bias[tok] = 0.0  # no penalty for required tokens

    if logits.dim() == 1:
        return logits + bias
    # (B, V)
    return logits + bias.unsqueeze(0)


# ---------------------------------------------------------------------------
# ConstrainedDecoder
# ---------------------------------------------------------------------------

class ConstrainedDecoder:
    """Generates sequences that satisfy all lexical constraints via greedy decoding."""

    def __init__(self, model, config: ConstraintConfig) -> None:
        self.model = model
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _model_forward(self, input_ids: Tensor) -> Tensor:
        """Run model forward pass and return last-step logits, shape (vocab_size,)."""
        out = self.model(input_ids)
        # Model returns (loss, logits, pkv) plain tuple.
        if isinstance(out, tuple):
            logits = out[1]
        else:
            logits = out
        # logits shape: (B, T, V) → take last position of first batch item.
        if logits.dim() == 3:
            logits = logits[0, -1, :]
        elif logits.dim() == 2:
            logits = logits[0, -1] if logits.shape[0] == 1 else logits[-1]
        return logits  # (V,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int,
    ) -> tuple[Tensor, bool]:
        """Generate tokens with constraint logit masking.

        Args:
            input_ids: Prompt token IDs, shape (1, seq_len) or (seq_len,).
            max_new_tokens: Number of tokens to generate.

        Returns:
            (generated_ids, all_constraints_met) where generated_ids has
            shape (1, max_new_tokens).
        """
        # Normalise input to (1, seq_len).
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        bank = ConstraintBank(self.config.constraints, self.config.bank_strategy)
        generated: list[int] = []

        current_ids = input_ids.clone()

        for _ in range(max_new_tokens):
            logits = self._model_forward(current_ids)  # (V,)
            # Apply constraint mask so required tokens are strongly preferred.
            logits = apply_constraint_mask(logits, bank)
            next_token = int(torch.argmax(logits).item())
            generated.append(next_token)
            bank.advance(next_token)

            next_id_tensor = torch.tensor([[next_token]], dtype=torch.long)
            current_ids = torch.cat([current_ids, next_id_tensor], dim=1)

        generated_ids = torch.tensor([generated], dtype=torch.long)  # (1, max_new_tokens)
        return generated_ids, bank.all_completed
