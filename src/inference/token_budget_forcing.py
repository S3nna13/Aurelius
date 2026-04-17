"""Token Budget Forcing for inference-time reasoning control.

Implements budget forcing as described in "Thinking LLMs" (Anthropic 2024):
during the thinking phase a bias is added to the answer-start token logit
when the budget is exhausted, steering the model to conclude its chain-of-thought
and begin generating the answer.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import LongTensor, Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Configuration for token budget forcing.

    Args:
        thinking_budget:      Maximum number of thinking-phase tokens before
                              forcing a transition to the answer phase.
        answer_start_token_id: Token ID that marks the start of the answer
                              (e.g. a special </think> or <answer> token).
        transition_bias:      Logit bias added to the answer_start_token_id
                              when the budget is exhausted.  Default 100.0.
        truncate_thinking:    If True, discard thinking tokens from the
                              returned sequence.  Default False.
    """
    thinking_budget: int
    answer_start_token_id: int
    transition_bias: float = 100.0
    truncate_thinking: bool = False


# ---------------------------------------------------------------------------
# Budget tracker
# ---------------------------------------------------------------------------

class BudgetTracker:
    """Tracks how many thinking tokens have been generated.

    The tracker starts in *thinking phase*.  Each call to :meth:`step`
    increments the internal counter if still in the thinking phase.  When the
    ``answer_start_token_id`` is observed the tracker transitions to *answer
    phase* and stops counting.

    Args:
        config: :class:`BudgetConfig` instance.
    """

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        self._thinking_count: int = 0
        self._in_thinking_phase: bool = True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def thinking_count(self) -> int:
        """Number of thinking tokens generated so far."""
        return self._thinking_count

    @property
    def is_in_thinking_phase(self) -> bool:
        """True while still in the thinking phase."""
        return self._in_thinking_phase

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------

    def step(self, token_id: int) -> bool:
        """Process the most recently generated token.

        If in the thinking phase, increments the counter.  Transitions to the
        answer phase when ``token_id`` matches ``answer_start_token_id``.

        Returns:
            True when the thinking budget is exhausted
            (``thinking_count >= thinking_budget``), False otherwise.
        """
        if self._in_thinking_phase:
            if token_id == self._config.answer_start_token_id:
                self._in_thinking_phase = False
            else:
                self._thinking_count += 1

        return self._thinking_count >= self._config.thinking_budget

    def reset(self) -> None:
        """Reset the tracker to its initial state."""
        self._thinking_count = 0
        self._in_thinking_phase = True


# ---------------------------------------------------------------------------
# Logits processor
# ---------------------------------------------------------------------------

class BudgetForcingLogitsProcessor:
    """Logits processor that steers generation toward the answer phase.

    When the thinking budget would be exhausted at the *next* step, a large
    positive bias is added to the ``answer_start_token_id`` logit so that the
    model transitions to answering.  This processor does **not** call
    :meth:`BudgetTracker.step`; the decoder is responsible for that.

    Args:
        config:  :class:`BudgetConfig`.
        tracker: :class:`BudgetTracker` associated with the current sequence.
    """

    def __init__(self, config: BudgetConfig, tracker: BudgetTracker) -> None:
        self._config = config
        self._tracker = tracker

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        """Apply budget forcing bias.

        Args:
            input_ids: Token IDs generated so far — shape ``(1, T)`` or
                       ``(B, T)``.  Not used directly; provided for API
                       compatibility.
            scores:    Raw logits — shape ``(B, V)`` or ``(V,)``.

        Returns:
            ``scores`` with the ``answer_start_token_id`` logit boosted by
            ``transition_bias`` if the budget is currently exhausted,
            otherwise ``scores`` unchanged.
        """
        # Budget is considered exhausted when the count already meets or
        # exceeds the limit (same condition as tracker.step returning True).
        budget_exhausted = (
            self._tracker.is_in_thinking_phase
            and self._tracker.thinking_count >= self._config.thinking_budget
        )

        if not budget_exhausted:
            return scores

        scores = scores.clone()
        scores[..., self._config.answer_start_token_id] += self._config.transition_bias
        return scores


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class BudgetForcingDecoder:
    """Greedy autoregressive decoder with token budget forcing.

    Args:
        model_fn: Callable ``(input_ids: LongTensor(1, T)) -> logits (1, T, V)``.
        config:   :class:`BudgetConfig`.
    """

    def __init__(self, model_fn, config: BudgetConfig) -> None:
        self._model_fn = model_fn
        self._config = config

    def generate(
        self,
        prompt_ids: LongTensor,
        max_total_tokens: int = 200,
    ) -> LongTensor:
        """Greedy decode with budget forcing.

        Args:
            prompt_ids:       1-D or 2-D ``(1, T)`` LongTensor of prompt tokens.
            max_total_tokens: Maximum *new* tokens to generate.

        Returns:
            1-D LongTensor of newly generated tokens (prompt excluded).
        """
        # Normalise to (1, T)
        if prompt_ids.dim() == 1:
            input_ids: LongTensor = prompt_ids.unsqueeze(0)
        else:
            input_ids = prompt_ids

        tracker = BudgetTracker(self._config)
        processor = BudgetForcingLogitsProcessor(self._config, tracker)
        generated: list[int] = []

        for _ in range(max_total_tokens):
            # Forward pass — take logits for the last position
            logits: Tensor = self._model_fn(input_ids)  # (1, T, V)
            next_logits: Tensor = logits[0, -1, :]      # (V,)

            # Apply budget forcing
            next_logits = processor(input_ids, next_logits)

            # Greedy selection
            next_token_id: int = int(next_logits.argmax(dim=-1).item())

            # Append to running sequence
            next_token_tensor = torch.tensor(
                [[next_token_id]],
                dtype=torch.long,
                device=input_ids.device,
            )
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
            generated.append(next_token_id)

            # Inform the tracker *after* selecting the token
            tracker.step(next_token_id)

        return torch.tensor(generated, dtype=torch.long, device=prompt_ids.device)
