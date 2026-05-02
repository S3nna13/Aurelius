"""Speculative decoding v3: clean implementation with vectorized acceptance sampling.

Implements the speculative decoding algorithm from "Fast Inference from Transformers
via Speculative Decoding" (Leviathan et al., 2023).  A small draft model proposes
n_draft_tokens in a single autoregressive roll-out; the target model verifies all of
them in one forward pass; accepted tokens are kept and the first rejection triggers a
corrected resample so the final distribution exactly matches the target model.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpeculativeConfig:
    """Hyperparameters for speculative decoding."""

    n_draft_tokens: int = 4
    """Number of tokens the draft model proposes per verification round."""

    temperature: float = 1.0
    """Softmax temperature applied to logits before sampling (1.0 = unscaled)."""

    top_p: float = 1.0
    """Nucleus-sampling threshold in (0, 1].  1.0 disables nucleus filtering."""

    max_new_tokens: int = 100
    """Maximum tokens to generate beyond the prompt."""

    eos_token_id: int = 2
    """Token id that signals end-of-sequence."""


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------


def sample_from_logits(
    logits: Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> int:
    """Sample a single token id from a logit vector.

    Args:
        logits: 1-D tensor of shape ``(vocab_size,)`` containing raw logits.
        temperature: Scaling factor applied before softmax.  Values < 1 make
            the distribution sharper; values > 1 make it flatter.  Must be > 0.
        top_p: Nucleus-sampling probability mass to retain.  Tokens outside the
            smallest set whose cumulative probability meets ``top_p`` are masked
            to ``-inf`` before sampling.  ``1.0`` keeps all tokens.

    Returns:
        Sampled token id as a plain Python ``int``.
    """
    if logits.dim() != 1:
        raise ValueError(f"logits must be 1-D, got shape {tuple(logits.shape)}")

    # Temperature scaling
    scaled = logits / max(temperature, 1e-8)

    # Convert to probabilities
    probs = F.softmax(scaled, dim=-1)

    # Nucleus (top-p) filtering
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=0)
        # Keep tokens up to (and including) the one that pushes cumsum >= top_p
        remove_mask = (cumulative - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)
        # Renormalise
        sorted_probs = sorted_probs / sorted_probs.sum().clamp(min=1e-12)
        # Sample in the sorted space, then map back
        sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1).item()
        token_id = sorted_indices[sampled_sorted_idx].item()
    else:
        token_id = torch.multinomial(probs, num_samples=1).item()

    return int(token_id)


# ---------------------------------------------------------------------------
# Acceptance-probability helpers
# ---------------------------------------------------------------------------


def compute_acceptance_prob(draft_prob: float, target_prob: float) -> float:
    """Scalar speculative-sampling acceptance probability.

    Returns ``min(1, target_prob / draft_prob)``.  Handles ``draft_prob == 0``
    gracefully by returning ``1.0`` (the token is accepted when the draft model
    assigns it zero probability – a degenerate but safe edge case).

    Args:
        draft_prob: Probability the draft model assigned to the token in [0, 1].
        target_prob: Probability the target model assigns to the token in [0, 1].

    Returns:
        Acceptance probability in ``[0, 1]``.
    """
    if draft_prob <= 0.0:
        return 1.0
    ratio = target_prob / draft_prob
    return float(min(1.0, ratio))


def speculative_sample_step(
    draft_probs: Tensor,
    target_probs: Tensor,
) -> tuple[Tensor, Tensor]:
    """Vectorised speculative-sampling acceptance for a sequence of draft tokens.

    For each draft position ``i``, accepts token ``t_i`` with probability
    ``min(1, target_probs[i, t_i] / draft_probs[i, t_i])``.  After the first
    rejection the remaining positions are not meaningful; the caller should stop
    consuming tokens at that boundary.  A corrected distribution is also returned
    so that the caller can resample a replacement token at the rejection site,
    maintaining the exact target distribution.

    Args:
        draft_probs: ``(n_draft, vocab_size)`` probability tensor from the draft
            model (rows must sum to 1).
        target_probs: ``(n_draft, vocab_size)`` probability tensor from the target
            model (rows must sum to 1).

    Returns:
        A 2-tuple ``(accepted_mask, corrected_probs)``:

        - ``accepted_mask``: ``BoolTensor`` of shape ``(n_draft,)`` – ``True`` for
          accepted positions.
        - ``corrected_probs``: ``Tensor`` of shape ``(vocab_size,)`` – the
          corrected resampling distribution to use at the first rejected position.
          Defined as ``max(0, target - draft)`` normalised to a valid probability
          distribution.  If all tokens are accepted this is the target distribution
          at the last position (for the bonus token).
    """
    n_draft, vocab_size = draft_probs.shape
    if target_probs.shape != (n_draft, vocab_size):
        raise ValueError(
            f"draft_probs {draft_probs.shape} and target_probs {target_probs.shape} must match"
        )

    # Identify which token was "chosen" by the draft model at each step
    # We treat the argmax as the drafted token for the purpose of computing
    # per-position acceptance (the standard single-token-per-step variant).
    draft_token_ids = draft_probs.argmax(dim=-1)  # (n_draft,)

    # Acceptance probabilities for each position
    draft_chosen = draft_probs[torch.arange(n_draft), draft_token_ids]  # (n_draft,)
    target_chosen = target_probs[torch.arange(n_draft), draft_token_ids]  # (n_draft,)

    # min(1, q/p), safe against p=0
    accept_probs = torch.where(
        draft_chosen > 0,
        torch.clamp(target_chosen / draft_chosen.clamp(min=1e-12), max=1.0),
        torch.ones_like(draft_chosen),
    )  # (n_draft,)

    # Stochastic accept/reject
    u = torch.rand_like(accept_probs)
    accepted_mask = u < accept_probs  # BoolTensor (n_draft,)

    # Corrected distribution: max(0, target - draft) at the first rejection
    # Find first rejection index
    rejected_indices = (~accepted_mask).nonzero(as_tuple=False)
    if rejected_indices.numel() > 0:
        first_rejection = rejected_indices[0].item()
        raw = (target_probs[first_rejection] - draft_probs[first_rejection]).clamp(min=0.0)
    else:
        # All accepted — corrected dist is the target bonus distribution at last position
        raw = target_probs[-1]

    total = raw.sum()
    corrected_probs = raw / total.clamp(min=1e-12) if total > 0 else target_probs[-1]

    return accepted_mask, corrected_probs


# ---------------------------------------------------------------------------
# SpeculativeDecoder
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """Speculative decoding wrapper around arbitrary draft/target model callables.

    Both ``draft_model_fn`` and ``target_model_fn`` must accept a 1-D ``LongTensor``
    of token ids (the full context so far) and return a 2-D ``FloatTensor`` of shape
    ``(seq_len, vocab_size)`` containing **logits** (not probabilities) for each
    position.

    Args:
        draft_model_fn: Callable implementing the *draft* (small/fast) model.
        target_model_fn: Callable implementing the *target* (large/accurate) model.
        config: Hyperparameters controlling the decoding loop.
    """

    def __init__(
        self,
        draft_model_fn: Callable[[Tensor], Tensor],
        target_model_fn: Callable[[Tensor], Tensor],
        config: SpeculativeConfig,
    ) -> None:
        self.draft_model_fn = draft_model_fn
        self.target_model_fn = target_model_fn
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _logits_to_probs(self, logits: Tensor) -> Tensor:
        """Convert a logit tensor to probabilities using config temperature/top-p."""
        cfg = self.config
        scaled = logits / max(cfg.temperature, 1e-8)
        probs = F.softmax(scaled, dim=-1)

        if cfg.top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = (cumulative - sorted_probs) >= cfg.top_p
            sorted_probs = sorted_probs.masked_fill(remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)
            # Scatter back to original order
            probs = torch.zeros_like(sorted_probs).scatter_(-1, sorted_idx, sorted_probs)

        return probs

    def _sample_token(self, probs: Tensor) -> int:
        """Sample a single token from a 1-D probability distribution."""
        return int(torch.multinomial(probs, num_samples=1).item())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draft_tokens(self, input_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Autoregressively generate ``n_draft_tokens`` using the draft model.

        Args:
            input_ids: 1-D ``LongTensor`` – the current context.

        Returns:
            A 2-tuple ``(draft_token_ids, draft_probs)``:

            - ``draft_token_ids``: ``LongTensor`` of shape ``(n_draft,)``
            - ``draft_probs``: ``FloatTensor`` of shape ``(n_draft, vocab_size)``
        """
        cfg = self.config
        context = input_ids.clone()
        token_ids_list = []
        probs_list = []

        for _ in range(cfg.n_draft_tokens):
            logits = self.draft_model_fn(context)  # (seq_len, vocab_size)
            next_logits = logits[-1]  # (vocab_size,)
            probs = self._logits_to_probs(next_logits)
            token_id = self._sample_token(probs)
            token_ids_list.append(token_id)
            probs_list.append(probs)
            context = torch.cat([context, torch.tensor([token_id], dtype=torch.long)])

        draft_token_ids = torch.tensor(token_ids_list, dtype=torch.long)
        draft_probs = torch.stack(probs_list, dim=0)  # (n_draft, vocab_size)
        return draft_token_ids, draft_probs

    def verify_and_accept(
        self,
        input_ids: Tensor,
        draft_token_ids: Tensor,
        draft_probs: Tensor,
    ) -> tuple[Tensor, int]:
        """Verify draft tokens with the target model and accept greedily.

        Runs the target model over ``input_ids + draft_token_ids`` in one forward
        pass, computes per-token acceptance, and returns the accepted token ids.

        Args:
            input_ids: 1-D ``LongTensor`` – context *before* draft tokens.
            draft_token_ids: 1-D ``LongTensor`` of shape ``(n_draft,)``.
            draft_probs: ``FloatTensor`` of shape ``(n_draft, vocab_size)`` from the
                draft model.

        Returns:
            A 2-tuple ``(accepted_token_ids, n_accepted)``:

            - ``accepted_token_ids``: ``LongTensor`` containing the accepted tokens
              (length 1 – n_draft+1 inclusive of a possible bonus/corrected token).
            - ``n_accepted``: ``int`` number of tokens accepted (``len(accepted_token_ids)``).
        """
        n_draft = draft_token_ids.shape[0]
        combined = torch.cat([input_ids, draft_token_ids])
        target_logits = self.target_model_fn(combined)  # (seq_len, vocab_size)

        # We need target probs at positions corresponding to the draft tokens and
        # the one bonus position.  The draft tokens start at index len(input_ids)-1
        # in the target output (the prediction *for* the next token after each prefix).
        start = len(input_ids) - 1  # position that predicts first draft token
        # target_probs[i] -> distribution for predicting draft_token_ids[i]
        target_probs_draft = self._logits_to_probs(
            target_logits[start : start + n_draft]
        )  # (n_draft, vocab_size)
        # Bonus distribution: target prediction after all draft tokens
        bonus_probs = self._logits_to_probs(target_logits[start + n_draft])  # (vocab_size,)

        accepted_tokens = []
        for i in range(n_draft):
            t = draft_token_ids[i].item()
            p_draft = draft_probs[i, t].item()
            p_target = target_probs_draft[i, t].item()
            alpha = compute_acceptance_prob(p_draft, p_target)
            u = torch.rand(1).item()
            if u < alpha:
                accepted_tokens.append(t)
            else:
                # Resample from corrected distribution and stop
                raw = (target_probs_draft[i] - draft_probs[i]).clamp(min=0.0)
                total = raw.sum()
                corrected = raw / total.clamp(min=1e-12) if total > 0 else target_probs_draft[i]
                bonus = self._sample_token(corrected)
                accepted_tokens.append(bonus)
                result = torch.tensor(accepted_tokens, dtype=torch.long)
                return result, len(accepted_tokens)

        # All draft tokens accepted — append a bonus token from the target
        bonus = self._sample_token(bonus_probs)
        accepted_tokens.append(bonus)
        result = torch.tensor(accepted_tokens, dtype=torch.long)
        return result, len(accepted_tokens)

    def decode(self, prompt_ids: Tensor) -> Tensor:
        """Full speculative decoding loop.

        Iterates speculative draft + verify rounds until ``max_new_tokens`` tokens
        have been generated or ``eos_token_id`` is produced.

        Args:
            prompt_ids: 1-D ``LongTensor`` containing the prompt token ids.

        Returns:
            1-D ``LongTensor`` of shape ``(prompt_len + n_generated,)`` containing
            the prompt followed by all generated tokens.
        """
        cfg = self.config
        context = prompt_ids.clone()
        n_generated = 0

        while n_generated < cfg.max_new_tokens:
            remaining = cfg.max_new_tokens - n_generated
            n_draft = min(cfg.n_draft_tokens, remaining)

            # Temporarily reduce config draft length if near the budget
            orig_n_draft = cfg.n_draft_tokens
            cfg.n_draft_tokens = n_draft

            draft_token_ids, draft_probs = self.draft_tokens(context)

            cfg.n_draft_tokens = orig_n_draft  # restore

            # Verify
            accepted, n_acc = self.verify_and_accept(context, draft_token_ids, draft_probs)

            # Append accepted tokens, stopping at EOS
            eos_hit = False
            for tok in accepted.tolist():
                if n_generated >= cfg.max_new_tokens:
                    break
                context = torch.cat([context, torch.tensor([tok], dtype=torch.long)])
                n_generated += 1
                if tok == cfg.eos_token_id:
                    eos_hit = True
                    break

            if eos_hit:
                break

        return context


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def compute_speedup_ratio(n_tokens_generated: int, n_target_calls: int) -> float:
    """Theoretical speedup from speculative decoding.

    Defined as the ratio of tokens generated to target-model forward passes.
    Vanilla autoregressive decoding requires one target call per token, so a
    ratio > 1 indicates a speedup.

    Args:
        n_tokens_generated: Total number of tokens produced.
        n_target_calls: Number of times the target model was called.

    Returns:
        ``n_tokens_generated / n_target_calls``, or ``0.0`` if
        ``n_target_calls == 0``.
    """
    if n_target_calls == 0:
        return 0.0
    return n_tokens_generated / n_target_calls
