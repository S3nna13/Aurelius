"""Entropix adaptive entropy-based sampler for AureliusTransformer.

Implements the Entropix algorithm (xjdr, 2024): pick a sampling strategy based
on the Shannon entropy of the next-token distribution (H) and the variance of
per-head attention entropy (VH).

Strategy map
------------
Low  H + Low  VH  → greedy (argmax)
High H + Low  VH  → temperature sampling  (temp > 1)
Low  H + High VH  → min-p sampling
High H + High VH  → creative (high temperature)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class EntropixConfig:
    """Thresholds and hyper-parameters for the Entropix sampler."""

    # Entropy thresholds for next-token distribution
    low_ent_thresh: float = 0.1
    high_ent_thresh: float = 3.0

    # Varentropy thresholds for per-head attention entropy variance
    low_vent_thresh: float = 0.1
    high_vent_thresh: float = 2.0

    # Temperatures used by each strategy
    sample_temp: float = 0.8    # used in High-H / Low-VH branch
    creative_temp: float = 1.3  # used in High-H / High-VH branch

    # min-p cutoff for Low-H / High-VH branch
    min_p: float = 0.05


class EntropixSampler:
    """Adaptive sampler that selects a decoding strategy per token.

    Parameters
    ----------
    config:
        EntropixConfig instance; defaults are used when *None*.
    """

    def __init__(self, config: Optional[EntropixConfig] = None) -> None:
        self.config = config if config is not None else EntropixConfig()

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def token_entropy(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute Shannon entropy of the next-token distribution.

        Parameters
        ----------
        logits:
            Shape ``(B, V)`` – raw (un-normalised) logit scores.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` – per-sample entropy in nats.
        """
        # Replace -inf with a large negative number so softmax gives 0 prob
        # but we avoid propagating NaN through log(0).
        safe_logits = logits.clone()
        safe_logits[safe_logits == float("-inf")] = -1e9

        log_probs = F.log_softmax(safe_logits, dim=-1)   # (B, V)
        probs = log_probs.exp()                           # (B, V)

        # H = -sum p * log(p);  where p==0, p*log(p) == 0
        entropy = -(probs * log_probs).sum(dim=-1)        # (B,)
        return entropy

    def attn_varentropy(self, attn: torch.Tensor) -> torch.Tensor:
        """Compute the variance of per-head attention entropy.

        Parameters
        ----------
        attn:
            Shape ``(B, H, T, T)`` – attention weight tensors (already
            softmax-normalised, so each row sums to 1).

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` – variance across heads of the mean per-query
            attention entropy.
        """
        # attn: (B, H, T, T)
        # Clamp to avoid log(0); weights should already be non-negative.
        eps = 1e-9
        attn_clamped = attn.clamp(min=eps)

        # Per-position entropy: -sum_j w_j * log(w_j)  → shape (B, H, T)
        head_query_ent = -(attn_clamped * attn_clamped.log()).sum(dim=-1)  # (B, H, T)

        # Summarise over query positions → mean entropy per head: (B, H)
        head_ent = head_query_ent.mean(dim=-1)  # (B, H)

        # Variance across heads: (B,)
        varentropy = head_ent.var(dim=-1, unbiased=False)
        return varentropy

    # ------------------------------------------------------------------
    # Sampling strategies (private helpers)
    # ------------------------------------------------------------------

    @staticmethod
    def _greedy(logits: torch.Tensor) -> torch.Tensor:
        """Argmax decoding. Shape: (B, V) → (B,)."""
        return logits.argmax(dim=-1)

    @staticmethod
    def _temperature_sample(
        logits: torch.Tensor,
        temperature: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Sample from logits / temperature. Shape: (B, V) → (B,)."""
        scaled = logits / temperature
        probs = F.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)

    @staticmethod
    def _min_p_sample(
        logits: torch.Tensor,
        min_p: float,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Min-p sampling: keep tokens with prob >= min_p * max_prob.

        Shape: (B, V) → (B,).
        """
        probs = F.softmax(logits, dim=-1)           # (B, V)
        max_prob = probs.max(dim=-1, keepdim=True).values  # (B, 1)
        threshold = min_p * max_prob                # (B, 1)

        # Mask out tokens below threshold; replace masked logits with -inf
        filtered_logits = logits.clone()
        filtered_logits[probs < threshold] = float("-inf")

        filtered_probs = F.softmax(filtered_logits, dim=-1)
        return torch.multinomial(filtered_probs, num_samples=1, generator=generator).squeeze(-1)

    # ------------------------------------------------------------------
    # Public sample method
    # ------------------------------------------------------------------

    def sample(
        self,
        logits: torch.Tensor,
        attn: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Select and apply a sampling strategy per element in the batch.

        Parameters
        ----------
        logits:
            Shape ``(B, V)`` – next-token logits.
        attn:
            Shape ``(B, H, T, T)`` – attention weights from the final layer.
        generator:
            Optional :class:`torch.Generator` for reproducibility.

        Returns
        -------
        torch.Tensor
            Shape ``(B,)`` – sampled token indices in ``[0, V)``.
        """
        cfg = self.config
        B, V = logits.shape

        H_tok = self.token_entropy(logits)         # (B,)
        VH_attn = self.attn_varentropy(attn)       # (B,)

        tokens = torch.empty(B, dtype=torch.long, device=logits.device)

        for i in range(B):
            h = H_tok[i].item()
            vh = VH_attn[i].item()
            l_i = logits[i].unsqueeze(0)           # (1, V)

            low_h = h < cfg.high_ent_thresh
            low_vh = vh < cfg.high_vent_thresh

            if low_h and low_vh:
                # Confident, consistent → greedy
                tokens[i] = self._greedy(l_i).squeeze(0)
            elif (not low_h) and low_vh:
                # Uncertain distribution, consistent heads → explore with temp
                tokens[i] = self._temperature_sample(
                    l_i, cfg.sample_temp, generator
                ).squeeze(0)
            elif low_h and (not low_vh):
                # Peaked distribution, inconsistent heads → min-p
                tokens[i] = self._min_p_sample(l_i, cfg.min_p, generator).squeeze(0)
            else:
                # High entropy, high varentropy → most creative
                tokens[i] = self._temperature_sample(
                    l_i, cfg.creative_temp, generator
                ).squeeze(0)

        return tokens
