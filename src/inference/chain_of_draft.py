"""Chain of Draft: Thinking Faster by Writing Less (arXiv:2502.18600).

Chain of Draft (CoD) generates minimal "draft" reasoning steps bounded by a
per-step token budget, then produces the final answer.  It trades reasoning
verbosity for decoding speed while maintaining task accuracy.

Key paper terminology preserved:
  - draft step   : one concise reasoning step (≤ draft_budget tokens)
  - draft budget : hard token limit per draft step
  - budget forcing : stopping draft generation once the budget is exhausted
  - answer extraction : generating the final answer after all draft steps
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ChainOfDraftConfig:
    """Configuration for Chain of Draft decoding.

    Attributes
    ----------
    max_draft_steps:
        Maximum number of draft reasoning steps to generate.
    draft_budget:
        Maximum tokens per draft step (budget forcing limit).
    step_separator_id:
        Token id used as a step separator / stop signal within drafts.
        Default 198 corresponds to the newline character ``'\\n'`` in many
        byte-level tokenisers.
    answer_prefix_ids:
        Optional sequence of token ids (e.g. for ``"Answer:"`` ) that is
        appended to the context before generating the final answer.
    """

    max_draft_steps: int = 5
    draft_budget: int = 10
    step_separator_id: int = 198  # '\n' in GPT-2 / byte-level vocab
    answer_prefix_ids: list[int] = field(default_factory=list)


class ChainOfDraftDecoder:
    """Implements Chain of Draft decoding (arXiv:2502.18600).

    The decoder alternates between:
    1. Generating short *draft steps* — each capped at ``draft_budget`` tokens
       (budget forcing) or stopped early at ``step_separator_id``.
    2. After ``max_draft_steps`` steps (or exhaustion), generating the final
       answer from the accumulated context.

    Parameters
    ----------
    model:
        An ``nn.Module`` whose forward signature is
        ``(input_ids, past_key_values=None) -> (loss, logits, past_key_values)``
        where ``logits`` has shape ``(B, T, vocab_size)``.
    config:
        ``ChainOfDraftConfig`` instance.  Defaults to
        ``ChainOfDraftConfig()`` when ``None``.
    """

    def __init__(self, model: nn.Module, config: ChainOfDraftConfig | None = None) -> None:
        self.model = model
        self.config = config if config is not None else ChainOfDraftConfig()

    # ------------------------------------------------------------------
    # Core decoding primitives
    # ------------------------------------------------------------------

    def _greedy_next_token(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Run one forward pass and return the greedy next token and logits.

        Parameters
        ----------
        input_ids:
            Shape ``(B, T)``.

        Returns
        -------
        next_tokens:
            Shape ``(B, 1)`` — greedy argmax over the last position.
        last_logits:
            Shape ``(B, vocab_size)`` — raw logits at the last position
            (used for NaN / Inf checks in tests).
        """
        with torch.no_grad():
            _loss, logits, _pkv = self.model(input_ids)

        # logits: (B, T, vocab_size) — take last time-step
        last_logits = logits[:, -1, :]  # (B, vocab_size)
        next_tokens = torch.argmax(last_logits, dim=-1, keepdim=True)  # (B, 1)
        return next_tokens, last_logits

    def generate_draft_step(
        self,
        input_ids: Tensor,
        max_tokens: int,
        stop_ids: list[int],
    ) -> tuple[Tensor, bool]:
        """Generate one draft step with budget forcing.

        Decodes greedily up to *max_tokens* new tokens.  Generation stops
        early when any token in *stop_ids* is produced (budget forcing on
        separator).

        Parameters
        ----------
        input_ids:
            Shape ``(B, T)`` — current context.
        max_tokens:
            Hard token budget for this draft step.
        stop_ids:
            Stop generation immediately upon producing any of these token ids.

        Returns
        -------
        new_tokens:
            Shape ``(B, k)`` where ``k ≤ max_tokens`` — the generated tokens
            for this draft step (including the stop token if one was hit).
        hit_stop:
            ``True`` if generation halted because a stop token was produced.
        """
        stop_set = set(stop_ids)
        generated: list[Tensor] = []
        current = input_ids
        hit_stop = False

        for _ in range(max_tokens):
            next_tok, _ = self._greedy_next_token(current)  # (B, 1)
            generated.append(next_tok)
            current = torch.cat([current, next_tok], dim=1)

            # Budget forcing: stop if any item in the batch produced a stop id
            if any(int(next_tok[b, 0]) in stop_set for b in range(next_tok.size(0))):
                hit_stop = True
                break

        if not generated:
            # Return empty tensor with correct shape
            new_tokens = input_ids.new_empty((input_ids.size(0), 0))
        else:
            new_tokens = torch.cat(generated, dim=1)  # (B, k)

        return new_tokens, hit_stop

    # ------------------------------------------------------------------
    # Full CoD decode
    # ------------------------------------------------------------------

    def decode(
        self,
        input_ids: Tensor,
        max_answer_tokens: int = 50,
    ) -> tuple[Tensor, dict]:
        """Full Chain of Draft decode: draft steps followed by the final answer.

        The method:
        1. Runs up to ``config.max_draft_steps`` draft steps, each bounded by
           ``config.draft_budget`` tokens (budget forcing).
        2. Optionally appends ``config.answer_prefix_ids`` to signal the model
           to produce the final answer.
        3. Decodes the final answer greedily up to ``max_answer_tokens`` tokens.

        Parameters
        ----------
        input_ids:
            Shape ``(B, T)`` — the initial prompt.
        max_answer_tokens:
            Maximum tokens to generate for the final answer.

        Returns
        -------
        output_ids:
            Shape ``(B, T + draft_tokens + answer_tokens)`` — the full
            generated sequence (prompt + draft reasoning + answer).
        metadata:
            Dict with keys:
              - ``'n_draft_steps'`` (int): number of draft steps executed.
              - ``'draft_token_count'`` (int): total draft tokens generated.
              - ``'answer_token_count'`` (int): total answer tokens generated.
        """
        cfg = self.config
        stop_ids = [cfg.step_separator_id]

        current = input_ids.clone()
        draft_token_count = 0
        n_draft_steps = 0

        # --- draft phase ---------------------------------------------------
        for _step in range(cfg.max_draft_steps):
            new_tokens, _hit_stop = self.generate_draft_step(
                current,
                max_tokens=cfg.draft_budget,
                stop_ids=stop_ids,
            )
            if new_tokens.size(1) > 0:
                current = torch.cat([current, new_tokens], dim=1)
                draft_token_count += int(new_tokens.size(1))
                n_draft_steps += 1

        # --- optional answer prefix ----------------------------------------
        if cfg.answer_prefix_ids:
            prefix = (
                torch.tensor(
                    cfg.answer_prefix_ids,
                    dtype=torch.long,
                    device=current.device,
                )
                .unsqueeze(0)
                .expand(current.size(0), -1)
            )  # (B, P)
            current = torch.cat([current, prefix], dim=1)

        # --- answer phase --------------------------------------------------
        answer_tokens_list: list[Tensor] = []
        for _ in range(max_answer_tokens):
            next_tok, _ = self._greedy_next_token(current)  # (B, 1)
            answer_tokens_list.append(next_tok)
            current = torch.cat([current, next_tok], dim=1)

        answer_token_count = len(answer_tokens_list)

        metadata: dict = {
            "n_draft_steps": n_draft_steps,
            "draft_token_count": draft_token_count,
            "answer_token_count": answer_token_count,
        }
        return current, metadata

    # ------------------------------------------------------------------
    # Efficiency metric
    # ------------------------------------------------------------------

    @staticmethod
    def compute_draft_efficiency(
        draft_token_count: int,
        cot_token_count: int,
    ) -> float:
        """Compression ratio: ``draft_tokens / cot_tokens``.

        A value < 1.0 means CoD used fewer tokens than full chain-of-thought,
        i.e. it was more token-efficient.

        Parameters
        ----------
        draft_token_count:
            Total reasoning tokens produced by Chain of Draft.
        cot_token_count:
            Total reasoning tokens produced by full chain-of-thought.

        Returns
        -------
        float
            The ratio ``draft_token_count / cot_token_count``.  Returns
            ``1.0`` when both counts are equal, and ``0.0`` when
            ``cot_token_count`` is 0 and ``draft_token_count`` is also 0.

        Raises
        ------
        ZeroDivisionError
            If ``cot_token_count`` is 0 and ``draft_token_count`` > 0.
        """
        if cot_token_count == 0 and draft_token_count == 0:
            return 1.0
        return draft_token_count / cot_token_count
