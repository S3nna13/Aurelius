"""Speculative decoding: draft model proposes tokens, target model verifies in parallel.

Public API
----------
SpeculativeConfig   — configuration dataclass
draft_tokens        — autoregressively sample n tokens from a callable draft model
verify_draft        — run target model once; rejection-sample accepted tokens
speculative_decode_step — one draft + verify step
SpeculativeDecoder  — full decode loop with statistics
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    n_draft_tokens: int = 4
    temperature: float = 1.0
    max_new_tokens: int = 50
    vocab_size: int = 32000


# ---------------------------------------------------------------------------
# Draft phase
# ---------------------------------------------------------------------------

def draft_tokens(
    draft_model_fn: Callable[[Tensor], Tensor],
    token_ids: Tensor,
    n: int,
    temperature: float,
) -> Tuple[Tensor, Tensor]:
    """Autoregressively sample *n* tokens from *draft_model_fn*.

    Parameters
    ----------
    draft_model_fn:
        Callable that accepts ``(B, T)`` int64 tensor and returns
        ``(B, T, V)`` logits.
    token_ids:
        ``(B, T)`` int64 prompt tensor.
    n:
        Number of tokens to draft.
    temperature:
        Sampling temperature (values ≤ 1e-8 treated as greedy).

    Returns
    -------
    draft_ids : ``(B, n)`` int64 — sampled draft token ids.
    draft_probs : ``(B, n)`` float — probability of each selected token
        under the draft distribution.
    """
    batch_size = token_ids.shape[0]
    all_ids: list[Tensor] = []
    all_probs: list[Tensor] = []

    current = token_ids
    temp = max(float(temperature), 1e-8)

    with torch.no_grad():
        for _ in range(n):
            logits = draft_model_fn(current)          # (B, T, V)
            last_logits = logits[:, -1, :]            # (B, V)

            if temp < 1e-7:
                next_tok = last_logits.argmax(dim=-1)  # (B,)
            else:
                probs_full = F.softmax(last_logits / temp, dim=-1)  # (B, V)
                next_tok = torch.multinomial(probs_full, num_samples=1).squeeze(-1)  # (B,)

            probs_full = F.softmax(last_logits / temp, dim=-1)  # (B, V)
            chosen_prob = probs_full[torch.arange(batch_size, device=token_ids.device), next_tok]  # (B,)

            all_ids.append(next_tok)
            all_probs.append(chosen_prob)

            current = torch.cat([current, next_tok.unsqueeze(1)], dim=1)

    draft_ids = torch.stack(all_ids, dim=1)    # (B, n)
    draft_probs = torch.stack(all_probs, dim=1)  # (B, n)
    return draft_ids, draft_probs


# ---------------------------------------------------------------------------
# Verification phase
# ---------------------------------------------------------------------------

def verify_draft(
    target_model_fn: Callable[[Tensor], Tensor],
    token_ids: Tensor,
    draft_ids: Tensor,
    draft_probs: Tensor,
    temperature: float,
) -> Tuple[Tensor, int]:
    """Run target model on ``[token_ids | draft_ids]`` and apply rejection sampling.

    For each draft position *i* with draft token *t*:

    * Compute ``accept_prob = min(1, p_target[t] / p_draft[t])``.
    * Draw ``u ~ Uniform(0, 1)``.
    * If ``u < accept_prob``: accept *t*.
    * Else: sample from the corrected distribution
      ``max(0, p_target - p_draft)`` (normalised), then stop.

    If all draft tokens are accepted a bonus token is sampled from the
    target distribution at the next position.

    Parameters
    ----------
    target_model_fn:
        Callable ``(B, T) -> (B, T, V)``.
    token_ids:
        ``(B, prompt_len)`` int64.
    draft_ids:
        ``(B, n_draft)`` int64.
    draft_probs:
        ``(B, n_draft)`` float — draft probability for each selected token.
    temperature:
        Sampling temperature.

    Returns
    -------
    accepted_ids : ``(B, k)`` int64 — accepted new tokens (k ≥ 0).
    n_accepted : int — number of draft tokens that were accepted.
    """
    batch_size = token_ids.shape[0]
    prompt_len = token_ids.shape[1]
    n_draft = draft_ids.shape[1]

    # Single forward pass over [prompt | draft]
    full_ids = torch.cat([token_ids, draft_ids], dim=1)  # (B, prompt_len + n_draft)

    with torch.no_grad():
        target_logits = target_model_fn(full_ids)          # (B, prompt_len + n_draft, V)

    temp = max(float(temperature), 1e-8)
    accepted: list[Tensor] = []
    n_accepted = 0

    for i in range(n_draft):
        # Target logit at position (prompt_len - 1 + i) predicts token at
        # position (prompt_len + i), which is draft_ids[:, i].
        target_pos = prompt_len - 1 + i
        t_logits = target_logits[:, target_pos, :]           # (B, V)
        t_probs = F.softmax(t_logits / temp, dim=-1)         # (B, V)

        draft_tok = draft_ids[:, i]                          # (B,)
        d_prob = draft_probs[:, i]                           # (B,)

        p_target = t_probs[torch.arange(batch_size, device=token_ids.device), draft_tok]  # (B,)

        # Acceptance probability: min(1, p_target / p_draft)
        accept_prob = torch.clamp(p_target / (d_prob + 1e-10), max=1.0)  # (B,)

        u = torch.rand(batch_size, device=token_ids.device)

        # Scalar decision driven by batch index 0 (standard single-batch inference).
        if u[0].item() <= accept_prob[0].item():
            accepted.append(draft_tok)
            n_accepted += 1
        else:
            # Corrected distribution: max(0, p_target - p_draft)
            adj = (t_probs - d_prob.unsqueeze(-1)).clamp(min=0.0)  # (B, V)
            adj_sum = adj.sum(dim=-1, keepdim=True)
            # Fall back to target distribution if adjusted is degenerate
            adj = torch.where(adj_sum < 1e-10, t_probs, adj / (adj_sum + 1e-10))
            fallback_tok = torch.multinomial(adj, num_samples=1).squeeze(-1)  # (B,)
            accepted.append(fallback_tok)
            break

    if n_accepted == n_draft:
        # Bonus token from target at the position after all drafts
        bonus_pos = prompt_len + n_draft - 1
        bonus_logits = target_logits[:, bonus_pos, :]
        bonus_probs = F.softmax(bonus_logits / temp, dim=-1)
        bonus_tok = torch.multinomial(bonus_probs, num_samples=1).squeeze(-1)
        accepted.append(bonus_tok)

    if accepted:
        accepted_ids = torch.stack(accepted, dim=1)  # (B, k)
    else:
        accepted_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=token_ids.device)

    return accepted_ids, n_accepted


# ---------------------------------------------------------------------------
# One speculative decode step
# ---------------------------------------------------------------------------

def speculative_decode_step(
    draft_model_fn: Callable[[Tensor], Tensor],
    target_model_fn: Callable[[Tensor], Tensor],
    token_ids: Tensor,
    config: SpeculativeConfig,
) -> Tuple[Tensor, int]:
    """Draft *K* tokens then verify them; return the new accepted tokens.

    Parameters
    ----------
    draft_model_fn, target_model_fn:
        Callables ``(B, T) -> (B, T, V)``.
    token_ids:
        ``(B, T)`` current context.
    config:
        :class:`SpeculativeConfig` instance.

    Returns
    -------
    new_token_ids : ``(B, k)`` int64 — newly accepted tokens (k ≥ 0).
    n_accepted : int — number of draft tokens accepted (before bonus).
    """
    draft_ids, draft_probs = draft_tokens(
        draft_model_fn, token_ids, config.n_draft_tokens, config.temperature
    )
    new_token_ids, n_accepted = verify_draft(
        target_model_fn, token_ids, draft_ids, draft_probs, config.temperature
    )
    return new_token_ids, n_accepted


# ---------------------------------------------------------------------------
# Full decode loop
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """Full speculative decoding loop with statistics.

    Parameters
    ----------
    draft_model_fn, target_model_fn:
        Callables ``(B, T) -> (B, T, V)``.
    config:
        :class:`SpeculativeConfig` instance.
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

        self._total_draft_tokens: int = 0
        self._total_accepted: int = 0
        self._n_steps: int = 0

    def decode(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Generate up to *max_new_tokens* tokens using speculative decoding.

        Falls back to a single target-model sample when ``n_accepted == 0``.

        Parameters
        ----------
        prompt_ids : ``(B, T)`` int64.
        max_new_tokens : int — maximum tokens to generate.

        Returns
        -------
        ``(B, T + generated)`` int64.
        """
        cfg = self.config
        temp = max(float(cfg.temperature), 1e-8)
        generated = prompt_ids.clone()
        tokens_generated = 0
        batch_size = prompt_ids.shape[0]

        while tokens_generated < max_new_tokens:
            remaining = max_new_tokens - tokens_generated
            # Temporarily cap n_draft_tokens to avoid overshooting
            orig_n_draft = cfg.n_draft_tokens
            cfg.n_draft_tokens = min(orig_n_draft, remaining)

            new_toks, n_accepted = speculative_decode_step(
                self.draft_model_fn, self.target_model_fn, generated, cfg
            )

            cfg.n_draft_tokens = orig_n_draft  # restore

            self._total_draft_tokens += cfg.n_draft_tokens
            self._total_accepted += n_accepted
            self._n_steps += 1

            if new_toks.shape[1] == 0 or n_accepted == 0:
                # Fallback: sample one token from the target model directly
                with torch.no_grad():
                    t_logits = self.target_model_fn(generated)  # (B, T, V)
                    last_logits = t_logits[:, -1, :]             # (B, V)
                    t_probs = F.softmax(last_logits / temp, dim=-1)
                    fallback = torch.multinomial(t_probs, num_samples=1)  # (B, 1)
                generated = torch.cat([generated, fallback], dim=1)
                tokens_generated += 1
            else:
                # Accept up to `remaining` of the returned tokens
                n_to_add = min(new_toks.shape[1], remaining)
                generated = torch.cat([generated, new_toks[:, :n_to_add]], dim=1)
                tokens_generated += n_to_add

        return generated

    def get_stats(self) -> Dict[str, float]:
        """Return decoding statistics.

        Returns
        -------
        dict with keys:
            ``total_draft_tokens``, ``total_accepted``, ``acceptance_rate``,
            ``n_steps``.
        """
        total_draft = float(self._total_draft_tokens)
        total_acc = float(self._total_accepted)
        rate = total_acc / total_draft if total_draft > 0 else 0.0
        return {
            "total_draft_tokens": total_draft,
            "total_accepted": total_acc,
            "acceptance_rate": rate,
            "n_steps": float(self._n_steps),
        }
