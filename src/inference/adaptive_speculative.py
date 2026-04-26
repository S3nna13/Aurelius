"""Adaptive speculative decoding with dynamic draft length adjustment.

Dynamically adjusts draft length based on EMA-smoothed acceptance rate history,
targeting a configurable acceptance rate to maximize throughput.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AdaptiveSpecConfig:
    """Configuration for adaptive speculative decoding."""

    init_draft_len: int = 4
    min_draft_len: int = 1
    max_draft_len: int = 8
    alpha: float = 0.1  # EMA smoothing for acceptance rate
    target_acceptance: float = 0.7  # target acceptance rate
    adjustment_interval: int = 10  # steps between length adjustments
    temperature: float = 1.0


class AcceptanceRateTracker:
    """EMA-smoothed acceptance rate tracker."""

    def __init__(self, alpha: float = 0.1, init_rate: float = 0.7) -> None:
        self._alpha = alpha
        self._init_rate = init_rate
        self._rate = init_rate

    def update(self, n_accepted: int, n_proposed: int) -> None:
        """Update EMA: rate = alpha * (n_accepted/n_proposed) + (1-alpha) * rate."""
        if n_proposed <= 0:
            return
        batch_rate = n_accepted / n_proposed
        self._rate = self._alpha * batch_rate + (1.0 - self._alpha) * self._rate

    def get_rate(self) -> float:
        return self._rate

    def reset(self) -> None:
        self._rate = self._init_rate


@torch.no_grad()
def sample_draft_tokens(
    draft_model: nn.Module,
    input_ids: torch.Tensor,  # (1, T) current context
    n_tokens: int,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively sample n_tokens from draft_model.

    Returns (draft_ids, draft_log_probs): shapes (1, n_tokens), (1, n_tokens).
    """
    draft_ids_list: list[torch.Tensor] = []
    draft_log_probs_list: list[torch.Tensor] = []

    context = input_ids.clone()

    for _ in range(n_tokens):
        _, logits, _ = draft_model(context)
        # logits: (1, seq_len, vocab_size) — take last position
        last_logits = logits[0, -1, :]  # (vocab_size,)

        if temperature != 1.0 and temperature > 0:
            last_logits = last_logits / temperature

        log_probs = F.log_softmax(last_logits, dim=-1)  # (vocab_size,)
        probs = log_probs.exp()

        next_token = torch.multinomial(probs, 1)  # (1,) scalar index
        token_log_prob = log_probs[next_token]  # (1,)

        draft_ids_list.append(next_token)
        draft_log_probs_list.append(token_log_prob)

        context = torch.cat([context, next_token.view(1, 1)], dim=1)

    # Stack to (n_tokens,) then unsqueeze to (1, n_tokens)
    draft_ids = torch.stack(draft_ids_list, dim=0).view(1, n_tokens)  # (1, n_tokens)
    draft_log_probs = torch.stack(draft_log_probs_list, dim=0).view(1, n_tokens)  # (1, n_tokens)

    return draft_ids, draft_log_probs


@torch.no_grad()
def compute_target_log_probs(
    target_model: nn.Module,
    input_ids: torch.Tensor,  # (1, T) context
    draft_ids: torch.Tensor,  # (1, K) draft tokens
) -> torch.Tensor:
    """Run target model on context+draft, return log probs for draft positions.

    Returns (1, K) log probs of the draft tokens at their positions.
    """
    K = draft_ids.shape[1]
    combined = torch.cat([input_ids, draft_ids], dim=1)  # (1, T+K)

    _, logits, _ = target_model(combined)
    # logits: (1, T+K, vocab_size)
    # Position T-1 predicts draft token 0, position T predicts draft token 1, ...
    # draft token k is at position T+k in combined, predicted by logit at position T-1+k
    T = input_ids.shape[1]

    target_log_probs_list: list[torch.Tensor] = []
    for k in range(K):
        pos_logits = logits[0, T - 1 + k, :]  # (vocab_size,)
        log_probs = F.log_softmax(pos_logits, dim=-1)  # (vocab_size,)
        token_id = draft_ids[0, k]
        target_log_probs_list.append(log_probs[token_id].unsqueeze(0))  # (1,)

    target_log_probs = torch.stack(target_log_probs_list, dim=0).view(1, K)  # (1, K)
    return target_log_probs


def speculative_verify(
    draft_ids: torch.Tensor,  # (1, K)
    draft_log_probs: torch.Tensor,  # (1, K)
    target_log_probs: torch.Tensor,  # (1, K)
) -> tuple[torch.Tensor, int]:
    """Token-by-token acceptance sampling.

    For each token k:
      ratio = exp(target_log_prob[k] - draft_log_prob[k]).clamp(max=1.0)
      accept if uniform() < ratio

    Returns (accepted_ids, n_accepted) where accepted_ids is shape (1, n_accepted).
    """
    K = draft_ids.shape[1]
    n_accepted = 0

    for k in range(K):
        log_ratio = target_log_probs[0, k] - draft_log_probs[0, k]
        ratio = log_ratio.exp().clamp(max=1.0)
        u = torch.rand(1, device=draft_ids.device)
        if u.item() < ratio.item():
            n_accepted += 1
        else:
            break

    if n_accepted > 0:
        accepted_ids = draft_ids[:, :n_accepted]  # (1, n_accepted)
    else:
        accepted_ids = draft_ids[:, :0]  # (1, 0) empty

    return accepted_ids, n_accepted


def adjust_draft_length(
    current_len: int,
    acceptance_rate: float,
    target_rate: float,
    min_len: int,
    max_len: int,
) -> int:
    """Adjust draft length based on acceptance rate vs target.

    If acceptance_rate > target_rate + 0.1: increase by 1 (more aggressive).
    If acceptance_rate < target_rate - 0.1: decrease by 1 (more conservative).
    Clamp to [min_len, max_len].
    """
    if acceptance_rate > target_rate + 0.1:
        new_len = current_len + 1
    elif acceptance_rate < target_rate - 0.1:
        new_len = current_len - 1
    else:
        new_len = current_len

    return max(min_len, min(max_len, new_len))


class AdaptiveSpeculativeDecoder:
    """Adaptive speculative decoding with dynamic draft length."""

    def __init__(
        self,
        target_model: nn.Module,
        draft_model: nn.Module,
        cfg: AdaptiveSpecConfig,
    ) -> None:
        self.target_model = target_model
        self.draft_model = draft_model
        self.cfg = cfg

        self._tracker = AcceptanceRateTracker(alpha=cfg.alpha, init_rate=cfg.target_acceptance)
        self._draft_len = cfg.init_draft_len

        # Stats accumulators
        self._n_target_calls = 0
        self._n_tokens_generated = 0
        self._draft_len_history: list[int] = []
        self._step = 0

    def decode(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> tuple[torch.Tensor, dict]:
        """Generate max_new_tokens using adaptive speculative decoding.

        Returns (generated_ids, stats) where:
          generated_ids: (1, n_tokens_generated)
          stats: dict with mean_draft_len, mean_acceptance_rate, n_target_calls, n_tokens_generated
        """
        cfg = self.cfg
        context = input_ids.clone()
        generated_tokens: list[torch.Tensor] = []

        self._n_target_calls = 0
        self._n_tokens_generated = 0
        self._draft_len_history = []
        self._step = 0
        self._tracker.reset()
        self._draft_len = cfg.init_draft_len

        while self._n_tokens_generated < max_new_tokens:
            remaining = max_new_tokens - self._n_tokens_generated
            draft_len = min(self._draft_len, remaining)
            if draft_len <= 0:
                break

            self._draft_len_history.append(draft_len)

            # 1. Sample draft tokens
            draft_ids, draft_log_probs = sample_draft_tokens(
                self.draft_model, context, draft_len, cfg.temperature
            )

            # 2. Verify with target model
            target_log_probs = compute_target_log_probs(self.target_model, context, draft_ids)
            self._n_target_calls += 1

            # 3. Accept verified tokens
            accepted_ids, n_accepted = speculative_verify(
                draft_ids, draft_log_probs, target_log_probs
            )

            # Update tracker
            self._tracker.update(n_accepted, draft_len)

            # If nothing accepted, fall back to one greedy token from target
            if n_accepted == 0:
                # Use the first draft token as a fallback (or resample)
                # Simple fallback: accept first draft token
                fallback_token = draft_ids[:, :1]  # (1, 1)
                generated_tokens.append(fallback_token)
                context = torch.cat([context, fallback_token], dim=1)
                self._n_tokens_generated += 1
            else:
                generated_tokens.append(accepted_ids)
                context = torch.cat([context, accepted_ids], dim=1)
                self._n_tokens_generated += n_accepted

            self._step += 1

            # 4. Every adjustment_interval steps: adjust draft_len
            if self._step % cfg.adjustment_interval == 0:
                self._draft_len = adjust_draft_length(
                    self._draft_len,
                    self._tracker.get_rate(),
                    cfg.target_acceptance,
                    cfg.min_draft_len,
                    cfg.max_draft_len,
                )

        # Assemble output
        if generated_tokens:
            generated_ids = torch.cat(generated_tokens, dim=1)  # (1, n_tokens_generated)
            # Trim to max_new_tokens if we overshot
            if generated_ids.shape[1] > max_new_tokens:
                generated_ids = generated_ids[:, :max_new_tokens]
        else:
            generated_ids = torch.zeros((1, 0), dtype=input_ids.dtype, device=input_ids.device)

        stats = self.get_stats()
        return generated_ids, stats

    def get_stats(self) -> dict[str, float]:
        """Return current stats dict."""
        mean_draft_len = (
            sum(self._draft_len_history) / len(self._draft_len_history)
            if self._draft_len_history
            else float(self.cfg.init_draft_len)
        )
        return {
            "mean_draft_len": mean_draft_len,
            "mean_acceptance_rate": self._tracker.get_rate(),
            "n_target_calls": float(self._n_target_calls),
            "n_tokens_generated": float(self._n_tokens_generated),
        }
