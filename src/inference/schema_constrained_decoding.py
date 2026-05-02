"""Schema-constrained decoding via Finite State Machines.

Constrains token generation to conform to a schema (e.g., JSON-like structure)
by masking invalid tokens at each decoding step. Uses a pure PyTorch FSM
with no external grammar/parsing dependencies.

Distinct from constrained_decoding.py (beam/prefix trie) and grammar_constrained.py
(general CFG). This module targets JSON-schema-like structural constraints.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

# ---------------------------------------------------------------------------
# FSMState
# ---------------------------------------------------------------------------


class FSMState:
    """Represents a state in the finite state machine."""

    def __init__(self, state_id: int, accepting: bool = False) -> None:
        self.state_id: int = state_id
        self.accepting: bool = accepting
        self.transitions: dict[str, int] = {}

    def add_transition(self, category: str, next_state: int) -> None:
        """Register a transition from this state on the given token category."""
        self.transitions[category] = next_state


# ---------------------------------------------------------------------------
# TokenCategory
# ---------------------------------------------------------------------------


class TokenCategory:
    """Classifies tokens into structural categories for JSON-like schemas.

    Uses token_id % 5 to deterministically assign each token to one of:
        0 -> 'open_brace'
        1 -> 'close_brace'
        2 -> 'string'
        3 -> 'colon'
        4 -> 'number'
    """

    _CATEGORIES = ("open_brace", "close_brace", "string", "colon", "number")

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        # Precompute mapping: token_id -> category string
        self._token_to_cat: list[str] = [self._CATEGORIES[i % 5] for i in range(vocab_size)]
        # Precompute reverse mapping: category -> list of token ids
        self._cat_to_tokens: dict[str, list[int]] = {c: [] for c in self._CATEGORIES}
        for tid, cat in enumerate(self._token_to_cat):
            self._cat_to_tokens[cat].append(tid)

    def category(self, token_id: int) -> str:
        """Return the category string for the given token id."""
        return self._token_to_cat[token_id]

    def tokens_for_category(self, cat: str) -> list[int]:
        """Return all token ids belonging to the given category."""
        return list(self._cat_to_tokens[cat])


# ---------------------------------------------------------------------------
# JsonSchemeFSM
# ---------------------------------------------------------------------------


class JsonSchemeFSM:
    """Pre-built FSM for simple ``{"key": value}`` JSON structure.

    State transitions:
        State 0 (start)       -- open_brace  --> State 1
        State 1 (in_object)   -- string      --> State 2
        State 2 (after_key)   -- colon       --> State 3
        State 3 (after_colon) -- string/number --> State 4
        State 4 (after_value) -- close_brace --> State 5 (accepting)
        State 5 (done)        -- accepting, no transitions
    """

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.categorizer = TokenCategory(vocab_size)

        # Build states
        s0 = FSMState(0, accepting=False)
        s1 = FSMState(1, accepting=False)
        s2 = FSMState(2, accepting=False)
        s3 = FSMState(3, accepting=False)
        s4 = FSMState(4, accepting=False)
        s5 = FSMState(5, accepting=True)

        s0.add_transition("open_brace", 1)
        s1.add_transition("string", 2)
        s2.add_transition("colon", 3)
        s3.add_transition("string", 4)
        s3.add_transition("number", 4)
        s4.add_transition("close_brace", 5)
        # s5 has no transitions (done/accepting)

        self.states: dict[int, FSMState] = {0: s0, 1: s1, 2: s2, 3: s3, 4: s4, 5: s5}
        self.current_state: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    def valid_categories(self) -> list[str]:
        """Return valid next token categories from the current state."""
        state = self.states[self.current_state]
        return list(state.transitions.keys())

    def valid_token_mask(self) -> Tensor:
        """Return a bool tensor of shape ``(vocab_size,)`` marking valid tokens."""
        mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        for cat in self.valid_categories():
            for tid in self.categorizer.tokens_for_category(cat):
                mask[tid] = True
        return mask

    def transition(self, token_id: int) -> bool:
        """Attempt to transition using the given token id.

        Returns True if the transition succeeded, False if the token is
        not valid in the current state (state is NOT changed in that case).
        """
        cat = self.categorizer.category(token_id)
        state = self.states[self.current_state]
        if cat not in state.transitions:
            return False
        self.current_state = state.transitions[cat]
        return True

    def reset(self) -> None:
        """Reset FSM to the start state."""
        self.current_state = 0

    def is_complete(self) -> bool:
        """Return True if the current state is an accepting state."""
        return self.states[self.current_state].accepting


# ---------------------------------------------------------------------------
# FSMConstrainedDecoder
# ---------------------------------------------------------------------------


class FSMConstrainedDecoder:
    """Applies FSM constraints during greedy decoding.

    Args:
        model_fn: Callable ``(input_ids: LongTensor(1,T)) -> logits: (1,T,V)``.
        fsm: A :class:`JsonSchemeFSM` instance.
        vocab_size: Vocabulary size.
    """

    def __init__(
        self,
        model_fn: Callable[[Tensor], Tensor],
        fsm: JsonSchemeFSM,
        vocab_size: int,
    ) -> None:
        self.model_fn = model_fn
        self.fsm = fsm
        self.vocab_size = vocab_size

    def apply_fsm_mask(self, logits: Tensor) -> Tensor:
        """Mask logits for tokens that are invalid in the current FSM state.

        Args:
            logits: Shape ``(V,)`` — last-position logits.

        Returns:
            Masked logits of the same shape, with invalid tokens set to ``-inf``.
        """
        mask = self.fsm.valid_token_mask()  # (V,) bool
        masked = logits.clone()
        masked[~mask] = float("-inf")
        return masked

    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
    ) -> tuple[Tensor, bool]:
        """Greedy generation with FSM token masking.

        Args:
            prompt_ids: 1-D ``LongTensor`` of prompt token ids.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            ``(generated_tokens, is_valid)`` where ``generated_tokens`` is a
            1-D tensor and ``is_valid`` indicates whether the FSM is in an
            accepting state after generation.
        """
        self.fsm.reset()
        # Build running context: shape (1, T)
        context = prompt_ids.unsqueeze(0) if prompt_ids.dim() == 1 else prompt_ids
        generated: list[int] = []

        for _ in range(max_new_tokens):
            # Get logits from model
            logits_3d = self.model_fn(context)  # (1, T, V)
            last_logits = logits_3d[0, -1, :]  # (V,)

            # Apply FSM mask
            masked_logits = self.apply_fsm_mask(last_logits)

            # Greedy selection
            next_token = int(torch.argmax(masked_logits).item())
            generated.append(next_token)

            # Advance FSM
            success = self.fsm.transition(next_token)
            if not success:
                # Should not happen if masking is correct, but guard anyway
                break

            # Append token to context
            next_tensor = torch.tensor([[next_token]], dtype=torch.long)
            context = torch.cat([context, next_tensor], dim=1)

            # Stop early if FSM has reached an accepting state
            if self.fsm.is_complete():
                break

        generated_tokens = torch.tensor(generated, dtype=torch.long)
        return generated_tokens, self.fsm.is_complete()


# ---------------------------------------------------------------------------
# ConstraintSatisfactionChecker
# ---------------------------------------------------------------------------


class ConstraintSatisfactionChecker:
    """Verifies that a generated sequence satisfies schema constraints.

    Args:
        fsm: A :class:`JsonSchemeFSM` instance (will be reset before checking).
    """

    def __init__(self, fsm: JsonSchemeFSM) -> None:
        self.fsm = fsm

    def check_sequence(self, token_ids: Tensor) -> dict[str, Any]:
        """Replay tokens through the FSM and report constraint satisfaction.

        Args:
            token_ids: 1-D ``LongTensor`` of token ids to verify.

        Returns:
            A dict with keys:
                ``'valid'`` (bool): True if all transitions succeeded and FSM
                    is in an accepting state after the full sequence.
                ``'n_valid_transitions'`` (int): Number of successful transitions.
                ``'failed_at'`` (Optional[int]): Index of first failed transition,
                    or None if all succeeded.
        """
        self.fsm.reset()
        n_valid = 0
        failed_at: int | None = None

        ids = token_ids.tolist() if isinstance(token_ids, Tensor) else list(token_ids)

        for pos, tid in enumerate(ids):
            success = self.fsm.transition(tid)
            if success:
                n_valid += 1
            else:
                failed_at = pos
                break

        valid = (failed_at is None) and self.fsm.is_complete()
        return {
            "valid": valid,
            "n_valid_transitions": n_valid,
            "failed_at": failed_at,
        }

    def violation_positions(self, token_ids: Tensor) -> list[int]:
        """Return positions where the FSM transition failed.

        Args:
            token_ids: 1-D ``LongTensor`` of token ids to check.

        Returns:
            List of integer positions (0-indexed) where a transition failed.
        """
        self.fsm.reset()
        violations: list[int] = []

        ids = token_ids.tolist() if isinstance(token_ids, Tensor) else list(token_ids)

        for pos, tid in enumerate(ids):
            success = self.fsm.transition(tid)
            if not success:
                violations.append(pos)
                # FSM state is unchanged; reset to allow continued checking
                # by re-trying from the current (stuck) state.
                # To check ALL positions we reset and replay up to this point.
                # Simple approach: record violation and stop (FSM is stuck).
                break

        return violations
