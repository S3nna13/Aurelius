"""Speculative decoding v2: full rejection sampling with draft model and theoretical acceptance rate tracking."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding v2."""
    n_draft_tokens: int = 4       # tokens to draft per step
    max_new_tokens: int = 64
    temperature: float = 1.0      # target model temperature
    draft_temperature: float = 1.0
    track_acceptance: bool = True  # track acceptance statistics


def sample_from_logits(logits: torch.Tensor, temperature: float = 1.0) -> tuple[int, torch.Tensor]:
    """Sample a token from logits with temperature scaling.

    Args:
        logits: (V,) raw logits for a single position.
        temperature: Sampling temperature.

    Returns:
        Tuple of (token_id: int, probabilities: Tensor of shape (V,)).
    """
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    token_id = torch.multinomial(probs, 1).item()
    return int(token_id), probs


def rejection_sample(
    target_prob: torch.Tensor,
    draft_prob: torch.Tensor,
    draft_token: int,
) -> tuple[int, bool]:
    """Single token rejection sampling step from Leviathan et al.

    Args:
        target_prob: (V,) target model probability distribution.
        draft_prob: (V,) draft model probability distribution.
        draft_token: The token proposed by the draft model.

    Returns:
        Tuple of (accepted_token: int, was_accepted: bool).
    """
    t = target_prob[draft_token].clamp(min=1e-10)
    d = draft_prob[draft_token].clamp(min=1e-10)
    accept_prob = torch.clamp(t / d, max=1.0).item()

    if torch.rand(1).item() <= accept_prob:
        return draft_token, True
    else:
        # Sample from adjusted distribution: max(0, target - draft) / norm
        adjusted = (target_prob - draft_prob).clamp(min=0.0)
        norm = adjusted.sum()
        if norm < 1e-10:
            adjusted = target_prob.clone()
            norm = adjusted.sum()
        adjusted = adjusted / norm
        token_id = torch.multinomial(adjusted, 1).item()
        return int(token_id), False


def _get_logits(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run model and return logits. Returns (seq_len, vocab_size)."""
    _, logits, _ = model(input_ids)
    return logits[0]  # (seq_len, vocab_size)


@torch.no_grad()
def speculative_decode_step(
    target_model: nn.Module,
    draft_model: nn.Module,
    prompt_ids: torch.Tensor,
    n_draft: int,
    temperature: float = 1.0,
) -> tuple[list[int], int, int]:
    """One step of speculative decoding.

    Draft model autoregressively generates n_draft tokens; target model does
    a single forward pass to verify all of them in parallel. Rejection sampling
    is applied per token.

    Args:
        target_model: Large verification model.
        draft_model: Small draft model.
        prompt_ids: (1, seq_len) current context.
        n_draft: Number of draft tokens to propose.
        temperature: Temperature for both models.

    Returns:
        Tuple of (accepted_tokens: list[int], n_accepted: int, n_proposed: int).
        accepted_tokens may include a bonus token from the target model.
    """
    draft_tokens: list[int] = []
    draft_probs: list[torch.Tensor] = []
    draft_input = prompt_ids.clone()

    # Step 1: Draft model generates n_draft tokens autoregressively
    for _ in range(n_draft):
        logits = _get_logits(draft_model, draft_input)  # (seq, vocab)
        token_id, probs = sample_from_logits(logits[-1], temperature=temperature)
        draft_tokens.append(token_id)
        draft_probs.append(probs)
        draft_input = torch.cat(
            [draft_input, torch.tensor([[token_id]], dtype=prompt_ids.dtype, device=prompt_ids.device)],
            dim=1,
        )

    # Step 2: Target model verifies all n_draft tokens in one forward pass
    verify_input = torch.cat(
        [prompt_ids, torch.tensor([draft_tokens], dtype=prompt_ids.dtype, device=prompt_ids.device)],
        dim=1,
    )
    target_logits = _get_logits(target_model, verify_input)  # (prompt_len + n_draft, vocab)

    # Step 3: Rejection sampling for each draft token
    accepted_tokens: list[int] = []
    n_accepted = 0
    prompt_len = prompt_ids.shape[1]

    for i, (draft_tok, d_prob) in enumerate(zip(draft_tokens, draft_probs)):
        # target_logits[prompt_len - 1 + i] predicts the token at position prompt_len + i
        target_pos = prompt_len - 1 + i
        t_logits = target_logits[target_pos]
        if temperature != 1.0 and temperature > 0:
            t_logits = t_logits / temperature
        t_prob = F.softmax(t_logits, dim=-1)

        token_id, was_accepted = rejection_sample(t_prob, d_prob, draft_tok)
        accepted_tokens.append(token_id)

        if was_accepted:
            n_accepted += 1
        else:
            # Rejected: stop here (corrected token already appended)
            return accepted_tokens, n_accepted, n_draft

    # All n_draft accepted: sample bonus token from target
    bonus_pos = prompt_len + n_draft - 1
    bonus_logits = target_logits[bonus_pos]
    bonus_token_id, _ = sample_from_logits(bonus_logits, temperature=temperature)
    accepted_tokens.append(bonus_token_id)

    return accepted_tokens, n_accepted, n_draft


class AcceptanceTracker:
    """Tracks acceptance rate statistics across speculative decoding steps."""

    def __init__(self) -> None:
        self.n_accepted: int = 0
        self.n_proposed: int = 0
        self._tokens_per_step: list[int] = []

    def record(self, n_accepted: int, n_proposed: int) -> None:
        """Record the results of one speculative decode step.

        Args:
            n_accepted: Number of draft tokens accepted in this step.
            n_proposed: Number of draft tokens proposed in this step.
        """
        self.n_accepted += n_accepted
        self.n_proposed += n_proposed
        self._tokens_per_step.append(n_accepted)

    def acceptance_rate(self) -> float:
        """Compute acceptance rate: n_accepted / max(n_proposed, 1)."""
        return self.n_accepted / max(self.n_proposed, 1)

    def speedup_estimate(self) -> float:
        """Estimate speedup as mean tokens per step / n_draft_tokens (simplified).

        Uses mean tokens accepted per step normalized by average draft budget.
        Falls back to acceptance_rate if no steps recorded.
        """
        if not self._tokens_per_step:
            return self.acceptance_rate()
        mean_tokens = sum(self._tokens_per_step) / len(self._tokens_per_step)
        n_draft = self.n_proposed / max(len(self._tokens_per_step), 1)
        return mean_tokens / max(n_draft, 1)


class SpeculativeDecoderV2:
    """Full speculative decoding engine using rejection sampling (Leviathan et al.)."""

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        config: SpeculativeConfig,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config
        self.tracker = AcceptanceTracker()

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor) -> tuple[list[int], dict]:
        """Generate tokens using speculative decoding.

        Args:
            prompt_ids: (1, prompt_len) input token ids.

        Returns:
            Tuple of (generated_token_ids: list[int], stats: dict).
            stats contains: acceptance_rate, n_steps, total_tokens.
        """
        cfg = self.config
        current_ids = prompt_ids.clone()
        all_generated: list[int] = []
        n_steps = 0

        while len(all_generated) < cfg.max_new_tokens:
            n_draft = cfg.n_draft_tokens

            accepted_tokens, n_accepted, n_proposed = speculative_decode_step(
                target_model=self.target_model,
                draft_model=self.draft_model,
                prompt_ids=current_ids,
                n_draft=n_draft,
                temperature=cfg.temperature,
            )

            if cfg.track_acceptance:
                self.tracker.record(n_accepted, n_proposed)

            # Trim to max_new_tokens budget
            remaining = cfg.max_new_tokens - len(all_generated)
            accepted_tokens = accepted_tokens[:remaining]

            all_generated.extend(accepted_tokens)
            n_steps += 1

            # Update context
            new_tok_tensor = torch.tensor(
                [accepted_tokens],
                dtype=current_ids.dtype,
                device=current_ids.device,
            )
            current_ids = torch.cat([current_ids, new_tok_tensor], dim=1)

            if len(all_generated) >= cfg.max_new_tokens:
                break

        stats = {
            "acceptance_rate": self.tracker.acceptance_rate(),
            "n_steps": n_steps,
            "total_tokens": len(all_generated),
        }
        return all_generated, stats

    def reset_stats(self) -> None:
        """Reset the acceptance tracker."""
        self.tracker = AcceptanceTracker()


@torch.no_grad()
def estimate_draft_quality(
    target_model: nn.Module,
    draft_model: nn.Module,
    prompt_ids: torch.Tensor,
    n_eval: int = 8,
) -> dict[str, float]:
    """Measure how often the draft model agrees with the target model.

    Greedily generates n_eval tokens from both models and computes
    agreement rate and mean probability ratio.

    Args:
        target_model: Target (large) model.
        draft_model: Draft (small) model.
        prompt_ids: (1, seq_len) input context.
        n_eval: Number of tokens to evaluate.

    Returns:
        Dict with keys "agreement_rate" and "mean_prob_ratio".
    """
    current_ids = prompt_ids.clone()
    agreements = 0
    prob_ratios: list[float] = []

    for _ in range(n_eval):
        target_logits = _get_logits(target_model, current_ids)[-1]  # (vocab,)
        draft_logits = _get_logits(draft_model, current_ids)[-1]    # (vocab,)

        target_probs = F.softmax(target_logits, dim=-1)
        draft_probs = F.softmax(draft_logits, dim=-1)

        target_token = int(target_probs.argmax().item())
        draft_token = int(draft_probs.argmax().item())

        if target_token == draft_token:
            agreements += 1

        t_p = target_probs[draft_token].clamp(min=1e-10).item()
        d_p = draft_probs[draft_token].clamp(min=1e-10).item()
        prob_ratios.append(t_p / d_p)

        # Advance context with target's greedy token
        current_ids = torch.cat(
            [current_ids, torch.tensor([[target_token]], dtype=current_ids.dtype, device=current_ids.device)],
            dim=1,
        )

    return {
        "agreement_rate": agreements / max(n_eval, 1),
        "mean_prob_ratio": sum(prob_ratios) / max(len(prob_ratios), 1),
    }
