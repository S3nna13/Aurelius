"""
src/inference/nucleus_plus.py

Nucleus++ sampling: top-p with minimum probability floor, repetition penalty,
and surprise upweighting for more diverse yet coherent generation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class NucleusConfig:
    """Configuration for Nucleus++ sampling."""
    temperature: float = 1.0
    top_p: float = 0.9
    min_p: float = 0.02
    repetition_penalty: float = 1.1
    surprise_alpha: float = 0.0  # 0.0 = disabled


def apply_temperature(logits: Tensor, temp: float) -> Tensor:
    """Divide logits by temperature.

    Args:
        logits: Shape ``(vocab_size,)`` raw logits.
        temp: Temperature value. Must be > 0.

    Returns:
        Scaled logits of the same shape.
    """
    if temp == 1.0:
        return logits
    return logits / temp


def apply_repetition_penalty(logits: Tensor, prev_ids: List[int], penalty: float) -> Tensor:
    """Reduce scores of tokens that already appear in prev_ids.

    For each token in prev_ids, its logit is divided by penalty (positive logit)
    or multiplied by penalty (negative logit) to push it away from re-selection.

    Args:
        logits: Shape ``(vocab_size,)`` raw logits.
        prev_ids: List of previously generated token ids.
        penalty: Repetition penalty factor (>= 1.0; 1.0 = no penalty).

    Returns:
        Modified logits of the same shape.
    """
    if penalty == 1.0 or not prev_ids:
        return logits
    logits = logits.clone()
    unique_ids = set(prev_ids)
    for tok in unique_ids:
        if 0 <= tok < logits.shape[0]:
            if logits[tok] > 0:
                logits[tok] = logits[tok] / penalty
            else:
                logits[tok] = logits[tok] * penalty
    return logits


def apply_min_p(probs: Tensor, min_p: float) -> Tensor:
    """Zero out tokens whose probability is below min_p * max_prob, then renormalize.

    Args:
        probs: Shape ``(vocab_size,)`` probability distribution (non-negative, sums to 1).
        min_p: Floor fraction of the maximum probability.

    Returns:
        Renormalized probability tensor of the same shape.
    """
    if min_p <= 0.0:
        return probs
    max_prob = probs.max()
    threshold = min_p * max_prob
    mask = probs < threshold
    probs = probs.clone()
    probs[mask] = 0.0
    total = probs.sum()
    if total > 0:
        probs = probs / total
    return probs


def apply_surprise_upweight(probs: Tensor, alpha: float) -> Tensor:
    """Upweight rare tokens using surprise weighting: probs * (1 - probs)^alpha.

    When alpha == 0.0, returns probs unchanged (identity).

    Args:
        probs: Shape ``(vocab_size,)`` probability distribution.
        alpha: Surprise exponent. 0.0 = disabled, larger = stronger upweighting.

    Returns:
        Renormalized probability tensor of the same shape.
    """
    if alpha == 0.0:
        return probs
    weights = probs * (1.0 - probs) ** alpha
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights


def nucleus_plus_sample(logits: Tensor, prev_ids: List[int], config: NucleusConfig) -> int:
    """Draw one sample using Nucleus++ sampling.

    Applies in order: repetition penalty, temperature, softmax, top-p nucleus
    filtering, min-p floor, surprise upweighting, then samples.

    Args:
        logits: Shape ``(vocab_size,)`` raw logits for the next token.
        prev_ids: Previously generated token ids for repetition penalty.
        config: NucleusConfig with sampling hyperparameters.

    Returns:
        Sampled token id as a Python int.
    """
    # 1. Repetition penalty
    logits = apply_repetition_penalty(logits, prev_ids, config.repetition_penalty)
    # 2. Temperature
    logits = apply_temperature(logits, config.temperature)
    # 3. Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    # 4. Top-p nucleus filtering
    if config.top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        # Remove tokens with cumulative prob above top_p (shift by one to keep at least one)
        remove_mask = (cumsum - sorted_probs) >= config.top_p
        sorted_probs = sorted_probs.clone()
        sorted_probs[remove_mask] = 0.0
        # Scatter back
        probs = torch.zeros_like(probs)
        probs.scatter_(0, sorted_indices, sorted_probs)
        total = probs.sum()
        if total > 0:
            probs = probs / total
    # 5. Min-p floor
    probs = apply_min_p(probs, config.min_p)
    # 6. Surprise upweighting
    probs = apply_surprise_upweight(probs, config.surprise_alpha)
    # 7. Sample
    token_id = int(torch.multinomial(probs, num_samples=1).item())
    return token_id


class NucleusDecoder:
    """Autoregressive decoder using Nucleus++ sampling.

    Args:
        config: NucleusConfig controlling sampling behaviour.
    """

    def __init__(self, config: Optional[NucleusConfig] = None) -> None:
        self.config = config or NucleusConfig()

    def generate(
        self,
        model_fn: Callable[[Tensor], Tensor],
        input_ids: Tensor,
        max_new_tokens: int = 128,
        eos_id: int = 2,
    ) -> List[int]:
        """Generate tokens autoregressively with Nucleus++ sampling.

        Args:
            model_fn: Callable ``(input_ids: Tensor) -> logits Tensor`` where
                      input_ids has shape ``(1, seq_len)`` and logits has shape
                      ``(1, seq_len, vocab_size)``.
            input_ids: Prompt token ids, shape ``(seq_len,)`` or ``(1, seq_len)``.
            max_new_tokens: Maximum number of tokens to generate.
            eos_id: Token id that stops generation.

        Returns:
            List of generated token ids (not including the prompt).
        """
        if input_ids.dim() == 1:
            current_ids = input_ids.unsqueeze(0)
        else:
            current_ids = input_ids.clone()

        generated: List[int] = []

        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits_3d = model_fn(current_ids)  # (1, T, V)
            step_logits = logits_3d[0, -1, :]  # (V,)

            token_id = nucleus_plus_sample(step_logits, generated, self.config)
            generated.append(token_id)

            next_tensor = torch.tensor([[token_id]], dtype=torch.long, device=current_ids.device)
            current_ids = torch.cat([current_ids, next_tensor], dim=1)

            if token_id == eos_id:
                break

        return generated
