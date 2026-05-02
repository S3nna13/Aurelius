"""Cascade Speculative Decoding (Spector & Re, 2023).

Uses a hierarchy of models for drafting instead of a single draft model.
Chain: [tiny drafter -> medium verifier/drafter -> large final verifier].
Each level verifies and extends the draft from the level below, achieving
higher acceptance rates than single-level speculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CascadeConfig:
    """Configuration for cascade speculative decoding."""

    draft_lengths: list[int] = field(default_factory=lambda: [4, 2])
    acceptance_thresholds: list[float] = field(default_factory=lambda: [0.0, 0.0])
    max_new_tokens: int = 512


class CascadeLevel:
    """A single level in the cascade: wraps a model for drafting and verification."""

    def __init__(
        self,
        model: nn.Module,
        draft_len: int,
        acceptance_threshold: float = 0.0,
    ) -> None:
        self.model = model
        self.draft_len = draft_len
        self.acceptance_threshold = acceptance_threshold

    @torch.no_grad()
    def draft(
        self,
        input_ids: torch.Tensor,  # (batch, T)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate draft_len tokens autoregressively.

        Returns:
            draft_ids:     (batch, draft_len)
            draft_logprobs: (batch, draft_len, vocab_size)
        """
        input_ids.shape[0]
        context = input_ids.clone()

        draft_ids_list: list[torch.Tensor] = []
        draft_logprobs_list: list[torch.Tensor] = []

        for _ in range(self.draft_len):
            _, logits, _ = self.model(context)
            # logits: (batch, seq_len, vocab_size) — take last position
            last_logits = logits[:, -1, :]  # (batch, vocab_size)
            log_probs = F.log_softmax(last_logits, dim=-1)  # (batch, vocab_size)
            probs = log_probs.exp()

            # Greedy: pick argmax (batch,)
            next_token = probs.argmax(dim=-1)  # (batch,)

            draft_ids_list.append(next_token.unsqueeze(1))  # (batch, 1)
            draft_logprobs_list.append(log_probs.unsqueeze(1))  # (batch, 1, vocab_size)

            context = torch.cat([context, next_token.unsqueeze(1)], dim=1)

        draft_ids = torch.cat(draft_ids_list, dim=1)  # (batch, draft_len)
        draft_logprobs = torch.cat(draft_logprobs_list, dim=1)  # (batch, draft_len, vocab_size)

        return draft_ids, draft_logprobs

    @torch.no_grad()
    def verify(
        self,
        input_ids: torch.Tensor,  # (batch, T)
        draft_ids: torch.Tensor,  # (batch, K)
    ) -> tuple[torch.Tensor, int]:
        """Verify draft tokens in a single forward pass using greedy matching.

        Accepts tokens while argmax of model logits matches draft token.
        Stops at the first mismatch.

        Returns:
            accepted_tokens: (batch, n_accepted)
            n_accepted:      int number of accepted tokens
        """
        K = draft_ids.shape[1]
        T = input_ids.shape[1]
        combined = torch.cat([input_ids, draft_ids], dim=1)  # (batch, T+K)

        _, logits, _ = self.model(combined)
        # logits: (batch, T+K, vocab_size)

        n_accepted = 0
        for k in range(K):
            # Position T-1+k in logits predicts draft token k
            pos_logits = logits[:, T - 1 + k, :]  # (batch, vocab_size)
            predicted = pos_logits.argmax(dim=-1)  # (batch,)
            draft_token = draft_ids[:, k]  # (batch,)

            # Check threshold: max probability must be >= acceptance_threshold
            max_prob = F.softmax(pos_logits, dim=-1).max(dim=-1).values  # (batch,)
            if (
                self.acceptance_threshold > 0.0
                and max_prob.min().item() < self.acceptance_threshold
            ):
                break

            if (predicted == draft_token).all():
                n_accepted += 1
            else:
                break

        if n_accepted > 0:
            accepted_tokens = draft_ids[:, :n_accepted]  # (batch, n_accepted)
        else:
            accepted_tokens = draft_ids[:, :0]  # (batch, 0) empty

        return accepted_tokens, n_accepted


class CascadeSpeculativeDecoder:
    """Cascade speculative decoder using a hierarchy of models.

    Level 0 (smallest) drafts tokens. Level 1 verifies level-0 draft and
    extends it. The final level verifies and accepts the full cascade draft.
    """

    def __init__(
        self,
        levels: list[CascadeLevel],
        max_new_tokens: int = 512,
    ) -> None:
        if len(levels) < 2:
            raise ValueError("CascadeSpeculativeDecoder requires at least 2 levels.")
        self.levels = levels
        self.max_new_tokens = max_new_tokens

        # Stats accumulators
        self._total_tokens: int = 0
        self._level_accepted: list[list[int]] = [[] for _ in levels]
        self._level_proposed: list[list[int]] = [[] for _ in levels]

    def decode_step(
        self,
        input_ids: torch.Tensor,  # (batch, T)
    ) -> tuple[torch.Tensor, dict]:
        """Run one cascade decode step.

        Algorithm:
          1. Level 0 drafts draft_len[0] tokens.
          2. Each intermediate level verifies the previous level's draft,
             accepts some prefix, then drafts additional tokens to reach
             its own draft_len.
          3. Final level verifies and returns accepted tokens.

        Returns:
            new_tokens: (batch, n_new) accepted tokens
            stats:      dict with per-level acceptance info
        """
        n_levels = len(self.levels)

        # Level 0: pure draft
        draft_ids, _ = self.levels[0].draft(input_ids)
        level_drafts = [draft_ids]  # draft_ids[i] = what level i+1 will verify

        # Intermediate and final levels: verify then optionally draft more
        for i in range(1, n_levels):
            prev_draft = level_drafts[i - 1]  # (batch, K_prev)
            accepted_tokens, n_acc = self.levels[i].verify(input_ids, prev_draft)

            proposed = prev_draft.shape[1]
            self._level_accepted[i - 1].append(n_acc)
            self._level_proposed[i - 1].append(proposed)

            if i < n_levels - 1:
                # Intermediate level: extend accepted tokens with more drafts
                if n_acc > 0:
                    extended_context = torch.cat([input_ids, accepted_tokens], dim=1)
                else:
                    extended_context = input_ids

                additional_needed = self.levels[i].draft_len
                if additional_needed > 0:
                    extra_ids, _ = self.levels[i].draft(extended_context)
                    if n_acc > 0:
                        combined_draft = torch.cat([accepted_tokens, extra_ids], dim=1)
                    else:
                        combined_draft = extra_ids
                else:
                    combined_draft = accepted_tokens

                level_drafts.append(combined_draft)
            else:
                # Final level: just record and return accepted tokens
                self._level_accepted[i].append(n_acc)
                self._level_proposed[i].append(proposed)

                # If nothing accepted at final level but something was drafted, return first token
                if n_acc == 0 and prev_draft.shape[1] > 0:
                    new_tokens = prev_draft[:, :1]
                    self._total_tokens += 1
                else:
                    new_tokens = accepted_tokens
                    self._total_tokens += n_acc

        stats = {
            "n_accepted": n_acc,
            "n_proposed": proposed,
        }
        return new_tokens, stats

    def generate(
        self,
        input_ids: torch.Tensor,  # (batch, T)
    ) -> torch.Tensor:
        """Full generation loop using cascade speculative decoding.

        Returns:
            output_ids: (batch, T + n_generated) complete sequence
        """
        context = input_ids.clone()
        n_generated = 0

        while n_generated < self.max_new_tokens:
            remaining = self.max_new_tokens - n_generated
            new_tokens, _ = self.decode_step(context)

            if new_tokens.shape[1] == 0:
                # Safety: generate one token greedily from the final model
                _, logits, _ = self.levels[-1].model(context)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                new_tokens = next_token

            # Trim to not overshoot max_new_tokens
            if new_tokens.shape[1] > remaining:
                new_tokens = new_tokens[:, :remaining]

            context = torch.cat([context, new_tokens], dim=1)
            n_generated += new_tokens.shape[1]

        return context

    def get_stats(self) -> dict:
        """Return stats dict with acceptance_rates per level, total_tokens, and speedup estimate.

        Returns:
            dict with keys:
              acceptance_rates: List[float] per level
              total_tokens: int total tokens generated
              cascade_speedup_estimate: float estimated speedup vs no speculation
        """
        acceptance_rates = []
        for i in range(len(self.levels)):
            accepted = self._level_accepted[i]
            proposed = self._level_proposed[i]
            if proposed and sum(proposed) > 0:
                rate = sum(accepted) / sum(proposed)
            else:
                rate = 0.0
            acceptance_rates.append(rate)

        # Speedup estimate: average tokens per forward pass of final model
        # vs 1 token per call without speculation
        final_proposed = self._level_proposed[-1]
        if final_proposed:
            sum(final_proposed) / len(final_proposed)
            final_accepted = self._level_accepted[-1]
            avg_accepted = sum(final_accepted) / len(final_accepted) if final_accepted else 0.0
            cascade_speedup_estimate = max(1.0, avg_accepted)
        else:
            cascade_speedup_estimate = 1.0

        return {
            "acceptance_rates": acceptance_rates,
            "total_tokens": self._total_tokens,
            "cascade_speedup_estimate": cascade_speedup_estimate,
        }


def compute_cascade_acceptance_rate(
    level_acceptances: list[list[int]],
) -> list[float]:
    """Compute per-level acceptance rate from acceptance counts per step.

    Args:
        level_acceptances: List of lists; level_acceptances[i] = list of
                           n_accepted per decode step for level i.

    Returns:
        List[float]: acceptance rate per level (accepted / steps).
    """
    rates = []
    for level_counts in level_acceptances:
        if not level_counts:
            rates.append(0.0)
        else:
            rates.append(sum(level_counts) / len(level_counts))
    return rates


def build_cascade(
    models: list[nn.Module],
    draft_lengths: list[int],
) -> CascadeSpeculativeDecoder:
    """Factory: build a CascadeSpeculativeDecoder from models and draft lengths.

    Args:
        models:        List of nn.Module, ordered from smallest to largest.
        draft_lengths: Number of tokens each level drafts per step.
                       Length must match len(models).

    Returns:
        CascadeSpeculativeDecoder
    """
    if len(models) != len(draft_lengths):
        raise ValueError(
            f"len(models)={len(models)} must equal len(draft_lengths)={len(draft_lengths)}"
        )
    if len(models) < 2:
        raise ValueError("build_cascade requires at least 2 models.")

    levels = [CascadeLevel(model=m, draft_len=d) for m, d in zip(models, draft_lengths)]
    return CascadeSpeculativeDecoder(levels=levels)
