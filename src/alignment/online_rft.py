"""Aurelius — Online Rejection Fine-Tuning (Online RFT / RAFT++, 2025).

Online RFT interleaves generation and training: generate N candidates per
prompt, keep only correct ones, train on them immediately.  This is more
sample-efficient than offline RFT and adapts to the current model's
distribution.

Reference: RAFT++ and related online rejection-sampling fine-tuning work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class OnlineRFTConfig:
    """Configuration for OnlineRFTTrainer."""

    n_candidates: int = 8
    """Number of candidate responses to generate per prompt."""

    min_keep_ratio: float = 0.125
    """Keep at least this fraction of candidates (1/n_candidates by default)."""

    max_keep_ratio: float = 0.5
    """Keep at most this fraction of candidates."""

    temperature: float = 0.8
    """Sampling temperature."""

    sft_loss_weight: float = 1.0
    """Weight applied to the SFT (cross-entropy) loss component."""

    kl_penalty_weight: float = 0.1
    """Weight applied to the KL divergence penalty from the reference model."""

    filter_strategy: str = "correct_only"
    """Filtering strategy: 'correct_only' or 'top_k'."""

    top_k_ratio: float = 0.5
    """Fraction of candidates to keep when filter_strategy='top_k'."""


# ---------------------------------------------------------------------------
# RFTSample
# ---------------------------------------------------------------------------


@dataclass
class RFTSample:
    """A single candidate response with metadata."""

    prompt_tokens: list[int]
    response_tokens: list[int]
    is_correct: bool
    reward: float = 0.0
    log_probs: torch.Tensor | None = None
    """Per-token log probabilities, shape [T]."""


# ---------------------------------------------------------------------------
# OnlineRFTTrainer
# ---------------------------------------------------------------------------


class OnlineRFTTrainer:
    """Online Rejection Fine-Tuning trainer.

    Generates N candidates per prompt, filters to keep only useful ones, and
    trains on them immediately — adapting to the current model's distribution.
    """

    def __init__(self, config: OnlineRFTConfig | None = None) -> None:
        self.config = config if config is not None else OnlineRFTConfig()

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_candidates(self, candidates: list[RFTSample]) -> list[RFTSample]:
        """Filter candidates according to the configured strategy.

        For ``correct_only``:
            Keep all candidates where ``is_correct=True``.
            If fewer than ``ceil(n_candidates * min_keep_ratio)`` are correct,
            top-up by reward until the minimum is reached.

        For ``top_k``:
            Keep the top ``ceil(n_candidates * top_k_ratio)`` candidates sorted
            by reward descending.

        Args:
            candidates: List of RFTSample objects to filter.

        Returns:
            Filtered list of RFTSample objects.
        """
        cfg = self.config
        n = cfg.n_candidates
        min_keep = math.ceil(n * cfg.min_keep_ratio)

        if cfg.filter_strategy == "top_k":
            k = math.ceil(n * cfg.top_k_ratio)
            sorted_cands = sorted(candidates, key=lambda s: s.reward, reverse=True)
            return sorted_cands[:k]

        # Default: correct_only
        correct = [s for s in candidates if s.is_correct]

        if len(correct) >= min_keep:
            return correct

        # Top-up: take the best by reward from the full pool
        sorted_all = sorted(candidates, key=lambda s: s.reward, reverse=True)
        return sorted_all[:min_keep]

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------

    def compute_sft_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute SFT cross-entropy loss, ignoring pad tokens (label=0).

        Args:
            logits: ``[B, T, V]`` — raw model logits.
            labels: ``[B, T]`` — target token ids; positions with value 0 are
                    treated as padding and excluded from the loss.

        Returns:
            Scalar loss tensor.
        """
        B, T, V = logits.shape
        # Flatten for cross-entropy
        logits_flat = logits.reshape(B * T, V)
        labels_flat = labels.reshape(B * T)

        # Replace pad tokens (0) with ignore_index=-100 so they do not
        # contribute to the loss.
        labels_masked = labels_flat.clone()
        labels_masked[labels_flat == 0] = -100

        return F.cross_entropy(logits_flat, labels_masked, ignore_index=-100)

    def compute_kl_penalty(
        self,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL(policy || ref) as mean(policy_log_probs - ref_log_probs).

        Args:
            policy_log_probs: ``[B, T]`` per-token log probs from the current
                              policy.
            ref_log_probs: ``[B, T]`` per-token log probs from the reference
                           model.

        Returns:
            Scalar KL penalty tensor.
        """
        return (policy_log_probs - ref_log_probs).mean()

    def total_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        policy_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute weighted total loss = sft_weight * sft + kl_weight * kl.

        Args:
            logits: ``[B, T, V]`` model logits.
            labels: ``[B, T]`` target token ids (0 = pad).
            policy_log_probs: ``[B, T]`` policy log probs.
            ref_log_probs: ``[B, T]`` reference model log probs.

        Returns:
            Tuple of (total_loss_tensor, metrics_dict) where metrics_dict
            contains float values for keys ``"sft"``, ``"kl"``, ``"total"``.
        """
        cfg = self.config
        sft_loss = self.compute_sft_loss(logits, labels)
        kl_loss = self.compute_kl_penalty(policy_log_probs, ref_log_probs)

        total = cfg.sft_loss_weight * sft_loss + cfg.kl_penalty_weight * kl_loss
        metrics = {
            "sft": float(sft_loss.item()),
            "kl": float(kl_loss.item()),
            "total": float(total.item()),
        }
        return total, metrics

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def statistics(self, candidates: list[RFTSample], kept: list[RFTSample]) -> dict:
        """Compute summary statistics for a filtering round.

        Args:
            candidates: Full list of candidate samples before filtering.
            kept: Filtered samples retained for training.

        Returns:
            Dict with keys:
            - ``n_candidates``: total number of candidates.
            - ``n_kept``: number kept after filtering.
            - ``keep_rate``: fraction kept (n_kept / n_candidates).
            - ``n_correct``: number of candidates marked correct.
            - ``mean_reward``: mean reward across all candidates.
        """
        n_candidates = len(candidates)
        n_kept = len(kept)
        keep_rate = n_kept / n_candidates if n_candidates > 0 else 0.0
        n_correct = sum(1 for s in candidates if s.is_correct)
        mean_reward = sum(s.reward for s in candidates) / n_candidates if n_candidates > 0 else 0.0
        return {
            "n_candidates": n_candidates,
            "n_kept": n_kept,
            "keep_rate": keep_rate,
            "n_correct": n_correct,
            "mean_reward": mean_reward,
        }
