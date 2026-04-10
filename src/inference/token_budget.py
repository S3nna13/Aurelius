"""Token Budget & Adaptive Generation.

Provides budget-aware token generation with entropy-based complexity estimation.
Adjusts generation length based on input complexity: simple inputs get shorter
budgets, complex inputs get longer ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BudgetConfig:
    """Configuration for token budget and adaptive generation."""
    max_tokens: int = 512
    target_efficiency: float = 0.8
    adaptive: bool = True
    eos_token_id: int = 0
    pad_token_id: int = 0


# ---------------------------------------------------------------------------
# Token Budget tracker
# ---------------------------------------------------------------------------

class TokenBudget:
    """Tracks token consumption against a fixed budget.

    Args:
        max_tokens: Maximum number of tokens allowed.
    """

    def __init__(self, max_tokens: int) -> None:
        if max_tokens < 0:
            raise ValueError("max_tokens must be non-negative")
        self._max_tokens = max_tokens
        self._consumed = 0

    @property
    def consumed(self) -> int:
        """Number of tokens consumed so far."""
        return self._consumed

    @property
    def remaining(self) -> int:
        """Number of tokens remaining in the budget."""
        return max(self._max_tokens - self._consumed, 0)

    @property
    def is_exhausted(self) -> bool:
        """Whether the budget has been fully consumed."""
        return self._consumed >= self._max_tokens

    def consume(self, n: int) -> None:
        """Consume *n* tokens from the budget.

        Args:
            n: Number of tokens to consume. Must be non-negative.

        Raises:
            ValueError: If *n* is negative.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        self._consumed += n

    def reset(self) -> None:
        """Reset consumed count to zero."""
        self._consumed = 0


# ---------------------------------------------------------------------------
# Complexity estimation
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_sequence_complexity(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
) -> float:
    """Estimate the complexity of *input_ids* using mean per-token entropy.

    Runs a forward pass, computes Shannon entropy at each position, then
    normalises the mean entropy to the [0, 1] range using
    ``min(mean_H / ln(vocab_size), 1.0)``.

    Args:
        model: AureliusTransformer (or compatible). Forward returns
            ``(loss, logits, past_key_values)``.
        input_ids: ``(1, seq_len)`` token ids.

    Returns:
        Complexity score in [0, 1]. 0 = trivially simple, 1 = maximally complex.
    """
    _, logits, _ = model(input_ids)  # (1, seq_len, vocab_size)
    # logits shape: (batch, seq_len, vocab_size)
    logits_2d = logits[0]  # (seq_len, vocab_size)
    probs = F.softmax(logits_2d, dim=-1)
    # Shannon entropy per position
    ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (seq_len,)
    mean_ent = ent.mean().item()
    vocab_size = logits_2d.size(-1)
    max_ent = math.log(vocab_size)
    complexity = min(mean_ent / max_ent, 1.0)
    return max(complexity, 0.0)


# ---------------------------------------------------------------------------
# Adaptive budget calculation
# ---------------------------------------------------------------------------

def adaptive_max_tokens(
    base_budget: int,
    complexity: float,
    target_efficiency: float = 0.8,
) -> int:
    """Compute an adaptive token budget based on sequence complexity.

    Simple sequences (complexity < 0.3) get *half* the base budget.
    Complex sequences (complexity > 0.7) get *double* the base budget.
    Everything in between is linearly interpolated between 0.5x and 2.0x.

    The result is further scaled by ``target_efficiency``.

    Args:
        base_budget: The default maximum number of tokens.
        complexity: Complexity score in [0, 1].
        target_efficiency: Scaling factor (0, 1]. Lower values reduce the
            budget proportionally.

    Returns:
        Adjusted token budget (at least 1).
    """
    if complexity < 0.0 or complexity > 1.0:
        raise ValueError("complexity must be in [0, 1]")
    if target_efficiency <= 0.0 or target_efficiency > 1.0:
        raise ValueError("target_efficiency must be in (0, 1]")

    # Map complexity to a multiplier in [0.5, 2.0]
    if complexity <= 0.3:
        multiplier = 0.5
    elif complexity >= 0.7:
        multiplier = 2.0
    else:
        # Linear interpolation between 0.3->0.5x and 0.7->2.0x
        t = (complexity - 0.3) / 0.4
        multiplier = 0.5 + t * 1.5

    adjusted = base_budget * multiplier * target_efficiency
    return max(int(adjusted), 1)


# ---------------------------------------------------------------------------
# Budget-aware generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def budget_aware_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    budget: TokenBudget,
) -> torch.Tensor:
    """Generate tokens one at a time until *budget* is exhausted or EOS.

    Uses greedy (argmax) decoding. Stops when:
    - The budget is exhausted, **or**
    - The model emits EOS token id 0.

    Args:
        model: AureliusTransformer (or compatible).
        input_ids: ``(1, prompt_len)`` prompt token ids.
        budget: A :class:`TokenBudget` instance tracking remaining capacity.

    Returns:
        ``(num_generated,)`` tensor of generated token ids (excluding the prompt).
    """
    generated: list[int] = []
    current_ids = input_ids.clone()

    while not budget.is_exhausted:
        _, logits, _ = model(current_ids)
        next_logits = logits[0, -1, :]  # (vocab_size,)
        next_token = int(torch.argmax(next_logits).item())
        generated.append(next_token)
        budget.consume(1)

        # EOS check (eos_token_id defaults to 0)
        if next_token == 0:
            break

        next_t = torch.tensor([[next_token]], dtype=current_ids.dtype, device=current_ids.device)
        current_ids = torch.cat([current_ids, next_t], dim=1)

    return torch.tensor(generated, dtype=torch.long)


# ---------------------------------------------------------------------------
# BudgetedGenerator
# ---------------------------------------------------------------------------

class BudgetedGenerator:
    """High-level generator that ties together budget config, complexity
    estimation, and budget-aware generation.

    Args:
        model: AureliusTransformer (or compatible).
        config: :class:`BudgetConfig` controlling generation behaviour.
    """

    def __init__(self, model: torch.nn.Module, config: BudgetConfig | None = None) -> None:
        self.model = model
        self.config = config or BudgetConfig()

    def generate(self, input_ids: torch.Tensor) -> dict[str, Any]:
        """Generate tokens with adaptive budgeting.

        Args:
            input_ids: ``(1, prompt_len)`` prompt token ids.

        Returns:
            Dictionary with keys:

            - ``tokens`` — ``(num_generated,)`` tensor of token ids.
            - ``tokens_consumed`` — int, how many tokens were generated.
            - ``efficiency`` — float, ``tokens_consumed / budget_used``.
        """
        cfg = self.config

        if cfg.adaptive:
            complexity = estimate_sequence_complexity(self.model, input_ids)
            max_tok = adaptive_max_tokens(cfg.max_tokens, complexity, cfg.target_efficiency)
        else:
            max_tok = cfg.max_tokens

        budget = TokenBudget(max_tok)
        tokens = budget_aware_generate(self.model, input_ids, budget)
        tokens_consumed = int(budget.consumed)
        efficiency = tokens_consumed / max_tok if max_tok > 0 else 0.0

        return {
            "tokens": tokens,
            "tokens_consumed": tokens_consumed,
            "efficiency": efficiency,
        }
