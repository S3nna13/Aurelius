"""Language model-guided data filtering using perplexity as a quality signal.

Keeps samples in the "learning zone": not too easy (low ppl) and not too
noisy (high ppl).  Optionally applies a reward model as a second gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class LMFilterConfig:
    ppl_low: float = 5.0            # below this = too easy, filter out
    ppl_high: float = 100.0         # above this = too noisy, filter out
    reward_threshold: float | None = None  # if set, also require reward >= threshold
    batch_size: int = 8


# ---------------------------------------------------------------------------
# FilterResult
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    kept: list[Tensor]
    rejected: list[Tensor]
    ppl_scores: list[float]          # ppl for kept samples only
    rejection_rate: float
    stats: dict[str, Any]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def compute_perplexity_batch(
    model,
    input_ids_list: list[Tensor],
) -> list[float]:
    """Compute per-example perplexity using the model's own cross-entropy loss.

    Each entry in *input_ids_list* is a 1-D tensor of token ids (variable
    length OK — we process one example at a time to avoid padding).

    Returns a list of float ppl values, one per example.
    """
    ppls: list[float] = []
    model.eval()
    with torch.no_grad():
        for input_ids in input_ids_list:
            # input_ids: (seq_len,) -> unsqueeze to (1, seq_len)
            ids = input_ids.unsqueeze(0)
            loss, _logits, _pkv = model(ids, labels=ids)
            ppl = math.exp(loss.item())
            ppls.append(ppl)
    return ppls


def filter_by_perplexity(
    model,
    dataset: list[Tensor],
    cfg: LMFilterConfig,
) -> tuple[list[Tensor], list[float]]:
    """Filter *dataset* keeping samples where cfg.ppl_low <= ppl <= cfg.ppl_high.

    Returns (kept_samples, kept_ppl_scores).
    """
    if not dataset:
        return [], []

    ppls = compute_perplexity_batch(model, dataset)
    kept_samples: list[Tensor] = []
    kept_scores: list[float] = []
    for sample, ppl in zip(dataset, ppls):
        if cfg.ppl_low <= ppl <= cfg.ppl_high:
            kept_samples.append(sample)
            kept_scores.append(ppl)
    return kept_samples, kept_scores


def filter_by_reward(
    reward_model,
    dataset: list[Tensor],
    threshold: float,
    batch_size: int = 8,
) -> tuple[list[Tensor], list[float]]:
    """Filter *dataset* keeping samples where reward_model score >= threshold.

    The reward model is expected to accept a 2-D input_ids tensor (1, seq_len)
    and return a scalar reward (or a tuple whose first element is the reward).

    Returns (kept_samples, kept_reward_scores).
    """
    if not dataset:
        return [], []

    kept_samples: list[Tensor] = []
    kept_scores: list[float] = []

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            chunk = dataset[i : i + batch_size]
            for sample in chunk:
                ids = sample.unsqueeze(0)
                output = reward_model(ids)
                # Support bare scalar tensor or tuple/list
                if isinstance(output, (tuple, list)):
                    reward_val = output[0]
                else:
                    reward_val = output
                score = float(reward_val.item() if isinstance(reward_val, Tensor) else reward_val)
                if score >= threshold:
                    kept_samples.append(sample)
                    kept_scores.append(score)

    return kept_samples, kept_scores


# ---------------------------------------------------------------------------
# LMFilter class
# ---------------------------------------------------------------------------

class LMFilter:
    """Combines perplexity-based and optional reward-based filtering."""

    def __init__(self, model, cfg: LMFilterConfig, reward_model=None):
        self.model = model
        self.cfg = cfg
        self.reward_model = reward_model

    def filter(self, dataset: list[Tensor]) -> FilterResult:
        """Filter *dataset* and return a FilterResult."""
        n_total = len(dataset)

        if n_total == 0:
            return FilterResult(
                kept=[],
                rejected=[],
                ppl_scores=[],
                rejection_rate=0.0,
                stats={"n_kept": 0, "n_rejected": 0, "mean_ppl_kept": float("nan"),
                       "mean_ppl_rejected": float("nan")},
            )

        # --- Step 1: perplexity filter ---
        all_ppls = compute_perplexity_batch(self.model, dataset)
        ppl_kept: list[Tensor] = []
        ppl_kept_scores: list[float] = []
        ppl_rejected: list[Tensor] = []
        ppl_rejected_scores: list[float] = []

        for sample, ppl in zip(dataset, all_ppls):
            if self.cfg.ppl_low <= ppl <= self.cfg.ppl_high:
                ppl_kept.append(sample)
                ppl_kept_scores.append(ppl)
            else:
                ppl_rejected.append(sample)
                ppl_rejected_scores.append(ppl)

        # --- Step 2: optional reward filter ---
        if self.reward_model is not None and self.cfg.reward_threshold is not None:
            reward_kept, _reward_scores = filter_by_reward(
                self.reward_model,
                ppl_kept,
                threshold=self.cfg.reward_threshold,
                batch_size=self.cfg.batch_size,
            )
            # Samples that passed ppl but failed reward -> also rejected
            reward_kept_set = {id(s) for s in reward_kept}
            extra_rejected = [s for s in ppl_kept if id(s) not in reward_kept_set]
            final_kept = reward_kept
            final_kept_ppls = [
                ppl for s, ppl in zip(ppl_kept, ppl_kept_scores)
                if id(s) in reward_kept_set
            ]
            final_rejected = ppl_rejected + extra_rejected
        else:
            final_kept = ppl_kept
            final_kept_ppls = ppl_kept_scores
            final_rejected = ppl_rejected

        n_kept = len(final_kept)
        n_rejected = len(final_rejected)
        rejection_rate = n_rejected / n_total if n_total > 0 else 0.0

        mean_ppl_kept = (
            sum(final_kept_ppls) / len(final_kept_ppls) if final_kept_ppls else float("nan")
        )
        mean_ppl_rejected = (
            sum(ppl_rejected_scores) / len(ppl_rejected_scores)
            if ppl_rejected_scores
            else float("nan")
        )

        stats = {
            "n_kept": n_kept,
            "n_rejected": n_rejected,
            "mean_ppl_kept": mean_ppl_kept,
            "mean_ppl_rejected": mean_ppl_rejected,
        }

        return FilterResult(
            kept=final_kept,
            rejected=final_rejected,
            ppl_scores=final_kept_ppls,
            rejection_rate=rejection_rate,
            stats=stats,
        )


# ---------------------------------------------------------------------------
# Gaussian importance weighting
# ---------------------------------------------------------------------------

def learning_zone_weights(
    ppl_scores: list[float],
    target_ppl: float = 20.0,
) -> list[float]:
    """Gaussian weighting centred at *target_ppl*.

    w_i = exp(-(ppl_i - target_ppl)^2 / (2 * sigma^2))
    sigma = target_ppl / 2

    Weights are normalized to sum to 1.  Higher weight for samples near
    *target_ppl* (the "learning zone" sweet spot).
    """
    if not ppl_scores:
        return []

    sigma = target_ppl / 2.0
    two_sigma2 = 2.0 * sigma * sigma

    raw = [math.exp(-((p - target_ppl) ** 2) / two_sigma2) for p in ppl_scores]
    total = sum(raw)
    if total == 0.0:
        # Degenerate: return uniform
        n = len(raw)
        return [1.0 / n] * n
    return [w / total for w in raw]
