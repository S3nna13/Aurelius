"""Lookahead decoding: parallel n-gram branch generation and verification.

Based on Fu et al. 2024, "Break the Sequential Dependency of LLM Inference
Using Lookahead Decoding". Proposes multiple candidate n-gram continuations
in parallel ("windows"), verifies them against the model's distribution,
and accepts as many tokens as possible per step.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LookaheadConfig:
    window_size: int = 5     # W — tokens per lookahead branch
    n_gram_size: int = 3     # N — n-gram size for the n-gram pool
    guess_set_size: int = 5  # G — max candidate guesses per step
    max_new_tokens: int = 50


# ---------------------------------------------------------------------------
# NGramPool
# ---------------------------------------------------------------------------

class NGramPool:
    """Stores and retrieves n-grams built from accepted token sequences.

    Keys are (n-1)-length prefix tuples; values are lists of full n-gram tuples.
    """

    def __init__(self, n: int) -> None:
        self.n = n
        # Maps prefix tuple (length n-1) to list of full n-gram tuples (length n)
        self._pool: Dict[Tuple[int, ...], List[Tuple[int, ...]]] = {}
        self._total: int = 0

    def update(self, tokens: List[int]) -> None:
        """Extract all n-grams from tokens and add to pool."""
        n = self.n
        if len(tokens) < n:
            return
        for i in range(len(tokens) - n + 1):
            ngram: Tuple[int, ...] = tuple(tokens[i : i + n])
            prefix: Tuple[int, ...] = ngram[: n - 1]
            if prefix not in self._pool:
                self._pool[prefix] = []
            if ngram not in self._pool[prefix]:
                self._pool[prefix].append(ngram)
                self._total += 1

    def lookup(self, prefix: Tuple[int, ...]) -> List[Tuple[int, ...]]:
        """Find n-grams whose first (n-1) tokens match prefix.

        prefix length should be n-1. Returns [] if not found.
        """
        return list(self._pool.get(prefix, []))

    def __len__(self) -> int:
        return self._total


# ---------------------------------------------------------------------------
# verify_candidates
# ---------------------------------------------------------------------------

def verify_candidates(
    model: nn.Module,
    input_ids: Tensor,
    candidates: List[Tensor],
) -> Tuple[Tensor, int]:
    """Run model on extended context; verify each candidate greedily.

    Extends input_ids with the longest candidate for a single forward pass,
    then checks how many leading tokens of each candidate match the model's
    greedy predictions at each position.

    Returns:
        best_accepted_tokens: 1-D Tensor of accepted token ids
        n_tokens_accepted: int >= 1
    """
    device = input_ids.device

    if not candidates:
        with torch.no_grad():
            _, logits, _ = model(input_ids)
        next_token = logits[0, -1, :].argmax(dim=-1)
        return next_token.unsqueeze(0), 1

    # Find longest candidate to set the forward-pass length
    longest = max(candidates, key=lambda c: c.shape[0])
    extended = torch.cat(
        [input_ids, longest.to(device).unsqueeze(0)], dim=1
    )  # (1, L + longest_len)

    with torch.no_grad():
        _, logits, _ = model(extended)
    # logits[0, input_len-1+i, :] predicts the token at position input_len+i
    # i.e. candidate[i]
    input_len = input_ids.shape[1]

    best_n = 0
    best_tokens: Optional[Tensor] = None

    for candidate in candidates:
        candidate = candidate.to(device)
        cand_len = candidate.shape[0]
        accepted: List[Tensor] = []
        for i in range(cand_len):
            predicted = logits[0, input_len - 1 + i, :].argmax(dim=-1)
            if predicted.item() == candidate[i].item():
                accepted.append(candidate[i].clone())
            else:
                break
        n = len(accepted)
        if n > best_n:
            best_n = n
            best_tokens = torch.stack(accepted) if accepted else None

    if best_n > 0 and best_tokens is not None:
        return best_tokens, best_n

    # No candidate matched even the first token — return greedy token
    greedy_first = logits[0, input_len - 1, :].argmax(dim=-1)
    return greedy_first.unsqueeze(0), 1


# ---------------------------------------------------------------------------
# lookahead_decode_step
# ---------------------------------------------------------------------------

def lookahead_decode_step(
    model: nn.Module,
    input_ids: Tensor,
    pool: NGramPool,
    config: LookaheadConfig,
) -> Tuple[Tensor, int]:
    """One lookahead decoding step.

    1. Use the last (n-1) tokens as prefix to look up candidates from the pool.
    2. If pool is empty: greedy decode 1 token, update pool, return (token, 1).
    3. Otherwise: verify top-G candidates, accept the longest match, update pool.

    Returns:
        accepted_tokens: 1-D Tensor
        n_accepted: int >= 1
    """
    device = input_ids.device
    n = config.n_gram_size

    # Build prefix from the last (n-1) tokens of input_ids
    seq = input_ids[0]  # shape (L,)
    prefix_len = n - 1
    if seq.shape[0] >= prefix_len and prefix_len > 0:
        prefix: Tuple[int, ...] = tuple(seq[-prefix_len:].tolist())
    else:
        prefix = tuple(seq.tolist())

    candidates_tuples = pool.lookup(prefix)

    if not candidates_tuples:
        # Greedy fallback
        with torch.no_grad():
            _, logits, _ = model(input_ids)
        next_token = logits[0, -1, :].argmax(dim=-1)
        accepted = next_token.unsqueeze(0)
        # Update pool with the accepted sequence (context + new token)
        all_tokens: List[int] = seq.tolist() + [int(next_token.item())]
        pool.update(all_tokens)
        return accepted, 1

    # Convert top-G candidate tuples to tensors (the suffix/continuation part)
    top_g = candidates_tuples[: config.guess_set_size]
    candidate_tensors: List[Tensor] = []
    for ngram_tuple in top_g:
        continuation = list(ngram_tuple[prefix_len:])
        if continuation:
            candidate_tensors.append(
                torch.tensor(continuation, dtype=torch.long, device=device)
            )

    accepted_tokens, n_accepted = verify_candidates(model, input_ids, candidate_tensors)

    # Update pool with the full new sequence
    all_tokens = seq.tolist() + accepted_tokens.tolist()
    pool.update(all_tokens)

    return accepted_tokens, n_accepted


# ---------------------------------------------------------------------------
# LookaheadDecoder
# ---------------------------------------------------------------------------

class LookaheadDecoder:
    """Full lookahead decoder using an n-gram pool for speculation."""

    def __init__(self, model: nn.Module, config: LookaheadConfig) -> None:
        self.model = model
        self.config = config
        self._pool = NGramPool(n=config.n_gram_size)

    def decode(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Autoregressively call lookahead_decode_step, accumulate tokens.

        Returns a 1-D tensor of generated tokens (batch=1 only).
        """
        self.model.train(False)
        context = input_ids.clone()
        generated: List[Tensor] = []
        total = 0

        with torch.no_grad():
            while total < max_new_tokens:
                accepted, n_accepted = lookahead_decode_step(
                    self.model, context, self._pool, self.config
                )
                remaining = max_new_tokens - total
                if n_accepted > remaining:
                    accepted = accepted[:remaining]
                    n_accepted = remaining

                generated.append(accepted)
                context = torch.cat(
                    [context, accepted.unsqueeze(0)], dim=1
                )
                total += n_accepted

        if not generated:
            return torch.empty(0, dtype=torch.long, device=input_ids.device)
        return torch.cat(generated, dim=0)

    def decode_with_stats(self, input_ids: Tensor, max_new_tokens: int) -> Dict:
        """Decode and return statistics.

        Returns:
            {
                'output': 1-D Tensor of generated tokens,
                'total_steps': int,
                'total_tokens': int,
                'mean_tokens_per_step': float,
            }
        """
        self.model.train(False)
        context = input_ids.clone()
        generated: List[Tensor] = []
        total_tokens = 0
        total_steps = 0

        with torch.no_grad():
            while total_tokens < max_new_tokens:
                accepted, n_accepted = lookahead_decode_step(
                    self.model, context, self._pool, self.config
                )
                remaining = max_new_tokens - total_tokens
                if n_accepted > remaining:
                    accepted = accepted[:remaining]
                    n_accepted = remaining

                generated.append(accepted)
                context = torch.cat(
                    [context, accepted.unsqueeze(0)], dim=1
                )
                total_tokens += n_accepted
                total_steps += 1

        output = (
            torch.cat(generated, dim=0)
            if generated
            else torch.empty(0, dtype=torch.long, device=input_ids.device)
        )

        return {
            "output": output,
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "mean_tokens_per_step": float(total_tokens) / max(total_steps, 1),
        }
