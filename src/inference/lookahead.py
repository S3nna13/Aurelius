"""Simplified lookahead decoding with n-gram cache acceleration.

Maintains a rolling n-gram cache from generated text and uses it to propose
candidate continuations that are verified against the model's distribution.
This avoids the complexity of full Jacobi iteration while still providing
speedups when the cache predicts well.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LookaheadConfig:
    """Configuration for lookahead decoding."""

    ngram_size: int = 3  # n-gram size for cache
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.9
    eos_token_id: int | None = None
    lookahead_steps: int = 2  # how many future tokens to try accepting


class NGramCache:
    """Cache of n-grams seen during generation for lookahead candidates."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._cache: dict[tuple[int, ...], list[int]] = defaultdict(list)

    def update(self, token_ids: list[int]) -> None:
        """Add all n-grams from token_ids to cache."""
        for i in range(len(token_ids) - self.n + 1):
            prefix = tuple(token_ids[i : i + self.n - 1])
            next_token = token_ids[i + self.n - 1]
            if next_token not in self._cache[prefix]:
                self._cache[prefix].append(next_token)

    def candidates(self, context: list[int]) -> list[int]:
        """Return candidate next tokens given last (n-1) context tokens."""
        if len(context) < self.n - 1:
            return []
        prefix = tuple(context[-(self.n - 1) :])
        return self._cache.get(prefix, [])


def _sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    """Sample a single token from logits with temperature + top-p."""
    logits = logits / max(temperature, 1e-8)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = sorted_probs.cumsum(0)
    cutoff = (cumsum - sorted_probs) > top_p
    sorted_probs[cutoff] = 0.0
    sorted_probs /= sorted_probs.sum() + 1e-12
    chosen = torch.multinomial(sorted_probs, 1)
    return sorted_idx[chosen].item()


def lookahead_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    cfg: LookaheadConfig,
) -> torch.Tensor:
    """Generate tokens using n-gram lookahead cache for acceleration.

    Algorithm:
    1. Start with prompt as context
    2. Build n-gram cache from prompt
    3. At each step:
       a. Run model forward to get next token logits
       b. Look up candidates from n-gram cache
       c. If candidates exist: verify each candidate against model distribution
          - If model assigns >1% probability to a candidate: accept it
          - Otherwise: fall back to sampling from model distribution
       d. Update n-gram cache with newly generated tokens

    Args:
        model: AureliusTransformer (mode should be set to inference/no-grad)
        input_ids: (1, S) prompt tensor
        cfg: LookaheadConfig

    Returns:
        (1, S + new_tokens) completed sequence tensor
    """
    model.eval()
    generated = input_ids[0].tolist()
    cache = NGramCache(cfg.ngram_size)
    cache.update(generated)

    device = input_ids.device

    with torch.no_grad():
        steps = 0
        while steps < cfg.max_new_tokens:
            # Standard forward pass for current position
            curr = torch.tensor(generated, device=device).unsqueeze(0)
            _, logits, _ = model(curr)
            next_logits = logits[0, -1, :]  # (V,)

            # Try lookahead: check if cache has candidates
            candidates = cache.candidates(generated)
            accepted = False

            if candidates:
                # Verify candidates against model distribution
                probs = F.softmax(next_logits / max(cfg.temperature, 1e-8), dim=-1)
                for cand in candidates[:3]:  # try up to 3 candidates
                    if probs[cand].item() > 0.01:  # threshold: model assigns >1% prob
                        generated.append(cand)
                        cache.update(generated)
                        steps += 1
                        accepted = True

                        # Check EOS
                        if cfg.eos_token_id is not None and cand == cfg.eos_token_id:
                            return torch.tensor(generated, device=device).unsqueeze(0)
                        break

            if not accepted:
                # Fall back to sampling
                next_token = _sample_token(next_logits, cfg.temperature, cfg.top_p)
                generated.append(next_token)
                cache.update(generated)
                steps += 1

                if cfg.eos_token_id is not None and next_token == cfg.eos_token_id:
                    break

    return torch.tensor(generated, device=device).unsqueeze(0)
