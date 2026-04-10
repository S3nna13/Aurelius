"""Lookahead decoding: parallel n-gram branch generation and verification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class LookaheadConfig:
    window_size: int = 4        # W: steps to look ahead
    n_gram_size: int = 3        # N: n-gram candidates to verify
    n_candidates: int = 5       # number of n-gram candidates per step
    temperature: float = 1.0
    max_new_tokens: int = 32


class NGramPool:
    """Stores and retrieves n-gram candidates for lookahead verification."""

    def __init__(self, n_gram_size: int, max_pool_size: int = 500) -> None:
        self.n_gram_size = n_gram_size
        self.max_pool_size = max_pool_size
        self._pool: list[Tensor] = []

    def add(self, tokens: Tensor) -> None:
        """Add all n-grams from a token sequence (1D tensor)."""
        tokens = tokens.reshape(-1)
        n = self.n_gram_size
        length = tokens.shape[0]
        if length < n:
            return
        for i in range(length - n + 1):
            ngram = tokens[i : i + n].clone()
            self._pool.append(ngram)
            if len(self._pool) > self.max_pool_size:
                self._pool.pop(0)

    def get_candidates(self, prefix: Tensor, n_candidates: int) -> list[Tensor]:
        """Return up to n_candidates n-grams that start with the last token of prefix.

        Each candidate is a (n_gram_size,) tensor.
        Returns empty list if pool is empty or no matches.
        """
        if len(self._pool) == 0:
            return []
        prefix = prefix.reshape(-1)
        if prefix.shape[0] == 0:
            return []
        last_token = prefix[-1].item()
        matches: list[Tensor] = []
        for ngram in self._pool:
            if ngram[0].item() == last_token:
                matches.append(ngram)
                if len(matches) >= n_candidates:
                    break
        return matches

    def __len__(self) -> int:
        return len(self._pool)


def verify_ngram(
    model: nn.Module,
    context_ids: Tensor,
    candidate: Tensor,
    temperature: float = 1.0,
) -> tuple[int, Tensor]:
    """Verify how many tokens of candidate are accepted by model.

    Run model on context, check if argmax matches candidate[0].
    If yes, extend context and check candidate[1], etc.

    Returns (n_accepted, accepted_tokens (n_accepted,)).
    """
    device = context_ids.device
    candidate = candidate.to(device)
    n_gram_size = candidate.shape[0]

    accepted_tokens: list[Tensor] = []
    current_context = context_ids.clone()

    with torch.no_grad():
        for i in range(n_gram_size):
            _, logits, _ = model(current_context)
            next_logits = logits[0, -1, :]
            if temperature != 1.0:
                next_logits = next_logits / temperature
            predicted_token = next_logits.argmax(dim=-1)

            if predicted_token.item() == candidate[i].item():
                accepted_tokens.append(candidate[i].clone())
                current_context = torch.cat(
                    [current_context, candidate[i].view(1, 1)], dim=1
                )
            else:
                break

    n_accepted = len(accepted_tokens)
    if n_accepted == 0:
        accepted = torch.empty(0, dtype=torch.long, device=device)
    else:
        accepted = torch.stack(accepted_tokens)

    return n_accepted, accepted


def lookahead_step(
    model: nn.Module,
    context_ids: Tensor,
    pool: NGramPool,
    config: LookaheadConfig,
    temperature: float = 1.0,
) -> tuple[Tensor, int]:
    """One lookahead step.

    1. Get n-gram candidates from pool
    2. Verify each candidate against model
    3. Accept the longest matching prefix
    4. If no candidates match, fall back to greedy decode (1 token)

    Returns (accepted_ids (n_accepted,), n_accepted).
    """
    device = context_ids.device

    candidates = pool.get_candidates(context_ids[0], config.n_candidates)

    best_n = 0
    best_tokens: Optional[Tensor] = None

    for candidate in candidates:
        n_accepted, accepted_tokens = verify_ngram(
            model, context_ids, candidate, temperature
        )
        if n_accepted > best_n:
            best_n = n_accepted
            best_tokens = accepted_tokens

    if best_n > 0 and best_tokens is not None:
        return best_tokens, best_n

    with torch.no_grad():
        _, logits, _ = model(context_ids)
        next_logits = logits[0, -1, :]
        if temperature != 1.0:
            next_logits = next_logits / temperature
        next_token = next_logits.argmax(dim=-1)

    accepted = next_token.unsqueeze(0)
    return accepted, 1


class LookaheadDecoder:
    """Full lookahead decoder with n-gram pool."""

    def __init__(self, model: nn.Module, config: LookaheadConfig) -> None:
        self.model = model
        self.config = config
        self.pool = NGramPool(n_gram_size=config.n_gram_size)

    def generate(self, input_ids: Tensor) -> tuple[Tensor, dict]:
        """Generate up to max_new_tokens using lookahead decoding.

        Returns (output_ids, stats) where stats has:
            'n_steps': int - number of lookahead_step calls
            'n_tokens': int - tokens generated
            'mean_accept_len': float - avg tokens accepted per step
        """
        self.model.eval()
        context = input_ids.clone()
        n_steps = 0
        n_tokens = 0
        total_accept = 0

        with torch.no_grad():
            while n_tokens < self.config.max_new_tokens:
                accepted_ids, n_accepted = lookahead_step(
                    self.model,
                    context,
                    self.pool,
                    self.config,
                    self.config.temperature,
                )
                remaining = self.config.max_new_tokens - n_tokens
                if n_accepted > remaining:
                    accepted_ids = accepted_ids[:remaining]
                    n_accepted = remaining

                self.pool.add(accepted_ids)

                context = torch.cat(
                    [context, accepted_ids.to(context.device).unsqueeze(0)], dim=1
                )

                n_steps += 1
                n_tokens += n_accepted
                total_accept += n_accepted

        stats = {
            "n_steps": n_steps,
            "n_tokens": n_tokens,
            "mean_accept_len": total_accept / max(n_steps, 1),
        }
        return context, stats

    def update_pool(self, tokens: Tensor) -> None:
        """Add token sequence to the n-gram pool."""
        self.pool.add(tokens.reshape(-1))
