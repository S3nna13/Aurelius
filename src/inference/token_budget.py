"""Adaptive token budget: stop early on confident outputs, extend on uncertain ones.

Monitors per-token entropy H(p) = -sum(p * log(p)) during generation.
Stops early when the model is confident for `patience` consecutive tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import torch
import torch.nn.functional as F


@dataclass
class TokenBudgetConfig:
    max_new_tokens: int = 256
    min_new_tokens: int = 1             # always generate at least this many
    low_entropy_threshold: float = 0.5  # nats — stop if below this
    patience: int = 3                   # consecutive low-entropy tokens to trigger stop
    temperature: float = 1.0
    top_p: float = 0.9
    eos_token_id: int | None = None


@dataclass
class TokenBudgetResult:
    token_ids: torch.Tensor       # (S_new,) generated token ids
    entropies: list[float]        # per-token entropy during generation
    stopped_early: bool           # True if stopped due to low entropy
    tokens_saved: int             # max_new_tokens - actual_new_tokens


def token_entropy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    """Compute Shannon entropy of next-token distribution. Returns value in nats.

    Args:
        logits: (vocab_size,) raw logits for one position.
        temperature: Scaling factor applied before softmax.

    Returns:
        Entropy in nats (float).
    """
    if temperature != 1.0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    # H = -sum(p * log(p))
    entropy = -(probs * torch.log(probs + 1e-10)).sum().item()
    return entropy


def nucleus_sample(logits: torch.Tensor, top_p: float, temperature: float) -> int:
    """Top-p nucleus sampling. Returns single token id (int).

    Args:
        logits: (vocab_size,) raw logits for one position.
        top_p: Nucleus sampling probability threshold.
        temperature: Scaling factor applied before softmax.

    Returns:
        Sampled token id as a Python int.
    """
    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    # Sort descending to build nucleus
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)

    # Remove tokens that push cumulative probability above top_p
    # (keep at least the top token)
    remove_mask = cumulative - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(remove_mask, 0.0)

    # Re-normalize
    sorted_probs = sorted_probs / sorted_probs.sum()

    # Scatter back to original vocabulary ordering
    nucleus_probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)

    token_id = torch.multinomial(nucleus_probs, num_samples=1).item()
    return int(token_id)


@torch.no_grad()
def generate_with_budget(
    model: "torch.nn.Module",
    input_ids: torch.Tensor,
    cfg: TokenBudgetConfig,
) -> TokenBudgetResult:
    """Generate tokens with adaptive early stopping based on entropy.

    Generates tokens one at a time (no KV cache). Stops early when the model
    is confident (low entropy) for `cfg.patience` consecutive tokens.

    Args:
        model: AureliusTransformer (or compatible) — forward returns (loss, logits, pkv).
        input_ids: (1, prompt_len) prompt token ids.
        cfg: TokenBudgetConfig controlling generation behavior.

    Returns:
        TokenBudgetResult with generated tokens, per-token entropies, and stop reason.
    """
    generated_ids: list[int] = []
    entropies: list[float] = []
    stopped_early = False
    consecutive_low = 0

    # Build the running context starting from the prompt
    current_ids = input_ids.clone()

    for step in range(cfg.max_new_tokens):
        # Forward pass — model returns (loss, logits, present_key_values)
        _, logits, _ = model(current_ids)

        # Extract last-position logits: (vocab_size,)
        next_logits = logits[0, -1, :]

        # Compute entropy of the next-token distribution
        H = token_entropy(next_logits, temperature=cfg.temperature)
        entropies.append(H)

        # Sample next token
        next_token_id = nucleus_sample(next_logits, top_p=cfg.top_p, temperature=cfg.temperature)
        generated_ids.append(next_token_id)

        # Append to context
        next_token_tensor = torch.tensor([[next_token_id]], dtype=current_ids.dtype, device=current_ids.device)
        current_ids = torch.cat([current_ids, next_token_tensor], dim=1)

        # Track consecutive low-entropy count
        if H < cfg.low_entropy_threshold:
            consecutive_low += 1
        else:
            consecutive_low = 0

        # Check early-stop condition (only after min_new_tokens)
        if (
            len(generated_ids) >= cfg.min_new_tokens
            and consecutive_low >= cfg.patience
        ):
            stopped_early = True
            break

        # Stop on EOS token
        if cfg.eos_token_id is not None and next_token_id == cfg.eos_token_id:
            break

    token_ids = torch.tensor(generated_ids, dtype=torch.long)
    tokens_saved = cfg.max_new_tokens - len(generated_ids)

    return TokenBudgetResult(
        token_ids=token_ids,
        entropies=entropies,
        stopped_early=stopped_early,
        tokens_saved=tokens_saved,
    )
