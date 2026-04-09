"""Contrastive decoding: subtract amateur model logits from expert to reduce repetition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ContrastiveDecodingConfig:
    """Configuration for contrastive decoding (Li et al. 2023)."""

    alpha: float = 0.1         # adaptive plausibility threshold (fraction of max prob)
    temperature: float = 1.0   # sampling temperature applied to both models
    max_new_tokens: int = 64   # number of tokens to generate
    top_k: int = 50            # top-k filtering applied after plausibility masking


def compute_cd_score(
    expert_logits: Tensor,
    amateur_logits: Tensor,
    temperature: float,
) -> Tensor:
    """Compute contrastive decoding score: log_softmax(expert/T) - log_softmax(amateur/T).

    Args:
        expert_logits:  Raw logits from expert model,  shape (B, vocab_size).
        amateur_logits: Raw logits from amateur model, shape (B, vocab_size).
        temperature:    Temperature to apply before softmax (must be > 0).

    Returns:
        CD scores of shape (B, vocab_size).
    """
    temp = max(temperature, 1e-8)
    log_p_expert = F.log_softmax(expert_logits / temp, dim=-1)
    log_p_amateur = F.log_softmax(amateur_logits / temp, dim=-1)
    return log_p_expert - log_p_amateur


def apply_adaptive_plausibility(
    expert_logits: Tensor,
    cd_scores: Tensor,
    alpha: float,
) -> Tensor:
    """Mask CD scores for tokens that are implausible under the expert.

    Tokens where ``softmax(expert_logits) < alpha * max_prob`` are set to -inf
    so they cannot be sampled.

    Args:
        expert_logits: Raw expert logits, shape (B, vocab_size).
        cd_scores:     CD scores to mask,  shape (B, vocab_size).
        alpha:         Plausibility threshold fraction in [0, 1].

    Returns:
        Masked CD scores of shape (B, vocab_size).
    """
    expert_probs = F.softmax(expert_logits, dim=-1)                     # (B, V)
    max_prob = expert_probs.max(dim=-1, keepdim=True).values            # (B, 1)
    plausible = expert_probs >= alpha * max_prob                        # (B, V)
    return cd_scores.masked_fill(~plausible, float("-inf"))


def _top_k_filter(scores: Tensor, top_k: int) -> Tensor:
    """Set all but the top-k entries in the last dim to -inf."""
    if top_k <= 0:
        return scores
    k = min(top_k, scores.size(-1))
    threshold = torch.topk(scores, k, dim=-1).values[..., -1:]
    return scores.masked_fill(scores < threshold, float("-inf"))


def contrastive_sample(
    expert_logits: Tensor,
    amateur_logits: Tensor,
    config: ContrastiveDecodingConfig,
) -> Tensor:
    """Sample next tokens using contrastive decoding.

    Steps:
    1. Compute CD scores.
    2. Apply adaptive plausibility mask.
    3. Apply top-k filter.
    4. Sample via multinomial over softmax of masked scores.

    Args:
        expert_logits:  Raw expert logits,  shape (B, vocab_size).
        amateur_logits: Raw amateur logits, shape (B, vocab_size).
        config:         ContrastiveDecodingConfig.

    Returns:
        Sampled token ids of shape (B,).
    """
    cd_scores = compute_cd_score(expert_logits, amateur_logits, config.temperature)
    masked = apply_adaptive_plausibility(expert_logits, cd_scores, config.alpha)
    filtered = _top_k_filter(masked, config.top_k)
    probs = F.softmax(filtered, dim=-1)
    # Replace any NaN rows (all -inf case) with uniform — safety fallback
    bad = probs.isnan().any(dim=-1, keepdim=True)
    if bad.any():
        uniform = torch.ones_like(probs) / probs.size(-1)
        probs = torch.where(bad, uniform, probs)
    token_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)     # (B,)
    return token_ids


class ContrastiveDecoder:
    """Autoregressive generator using Li et al. 2023 contrastive decoding.

    Args:
        expert_model:  Stronger AureliusTransformer (returns ``(loss, logits, past)``).
        amateur_model: Weaker  AureliusTransformer (same interface).
        config:        ContrastiveDecodingConfig.
    """

    def __init__(
        self,
        expert_model: nn.Module,
        amateur_model: nn.Module,
        config: ContrastiveDecodingConfig,
    ) -> None:
        self.expert = expert_model
        self.amateur = amateur_model
        self.config = config

    @torch.no_grad()
    def generate(self, input_ids: Tensor) -> tuple[Tensor, dict]:
        """Autoregressively generate tokens using contrastive decoding.

        Args:
            input_ids: Prompt token ids, shape (B, T).

        Returns:
            A tuple ``(generated_ids, stats)`` where:
            - ``generated_ids``: shape ``(B, max_new_tokens)`` — only newly generated tokens.
            - ``stats``: dict with ``"n_tokens"`` (int) and ``"mean_cd_score"`` (float).
        """
        seq = input_ids.clone()
        generated_tokens: list[Tensor] = []
        cd_score_sum = 0.0

        for _ in range(self.config.max_new_tokens):
            _, expert_logits_full, _ = self.expert(seq)    # (B, T, V)
            _, amateur_logits_full, _ = self.amateur(seq)  # (B, T, V)

            expert_last = expert_logits_full[:, -1, :]     # (B, V)
            amateur_last = amateur_logits_full[:, -1, :]   # (B, V)

            # Compute CD scores before masking to record selected token's score
            cd_scores_raw = compute_cd_score(
                expert_last, amateur_last, self.config.temperature
            )  # (B, V)

            next_tokens = contrastive_sample(expert_last, amateur_last, self.config)  # (B,)

            # Accumulate the mean CD score across batch for selected tokens
            # next_tokens: (B,) — gather the score for each selected token
            selected_scores = cd_scores_raw[
                torch.arange(next_tokens.size(0)), next_tokens
            ]  # (B,)
            cd_score_sum += selected_scores.mean().item()

            generated_tokens.append(next_tokens.unsqueeze(1))          # (B, 1)
            seq = torch.cat([seq, next_tokens.unsqueeze(1)], dim=1)

        generated_ids = torch.cat(generated_tokens, dim=1)             # (B, max_new_tokens)
        n = self.config.max_new_tokens
        stats: dict = {
            "n_tokens": n,
            "mean_cd_score": cd_score_sum / n if n > 0 else 0.0,
        }
        return generated_ids, stats


def measure_repetition(token_ids: Tensor, window: int = 16) -> float:
    """Measure n-gram repetition in a 1-D sequence of token ids.

    For each position ``t``, checks whether ``token_ids[t]`` appears anywhere
    in the previous ``window`` tokens.  Returns the fraction of positions that
    are repetitions (0.0 means no repetition, 1.0 means all repeated).

    Args:
        token_ids: 1-D tensor of token ids, shape (T,).
        window:    Look-back window length.

    Returns:
        Repetition fraction as a Python float in [0.0, 1.0].
    """
    ids = token_ids.tolist()
    T = len(ids)
    if T == 0:
        return 0.0

    n_repeated = 0
    for t in range(T):
        context = ids[max(0, t - window): t]
        if ids[t] in context:
            n_repeated += 1

    return n_repeated / T
