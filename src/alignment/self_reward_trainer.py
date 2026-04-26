"""Aurelius — Self-Reward Trainer: LLM-as-own-judge preference generation + DPO.

Yuan et al. (2024) "Self-Rewarding Language Models"
https://arxiv.org/abs/2401.10020

The model acts as its own judge, scoring candidate responses on a 0–5 scale
(LLM-as-Judge prompting), then the scored pairs become DPO training data.

Core loop:
    1. Generate M candidate responses for each prompt.
    2. Score each response with LLM-as-judge prompting (model judges itself).
    3. Create preference pairs: (highest-scored, lowest-scored) where
       score_gap >= min_score_gap.
    4. Fine-tune with DPO on the preference pairs.

This module implements the reward-scoring infrastructure and DPO training
objective — live LLM calls are handled outside this file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class SelfRewardConfig:
    """Configuration for SelfRewardTrainer.

    Attributes:
        n_candidates:  Number of candidate responses (M) generated per prompt.
        min_score_gap: Minimum judge-score gap between chosen and rejected
                       to form a valid preference pair.
        max_score:     Maximum judge score value (scale is 0 to max_score).
        beta:          DPO temperature; controls KL-regularisation strength.
        eps:           Small constant for numerical stability.
    """

    n_candidates: int = 4
    min_score_gap: float = 2.0
    max_score: float = 5.0
    beta: float = 0.1
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ScoredCandidate:
    """A single model-generated candidate response with a judge score.

    Attributes:
        token_ids: Decoded token id sequence.
        log_probs: ``[T]`` per-token log-probabilities from the policy.
        mask:      ``[T]`` binary attention mask (1 = valid token).
        score:     LLM-as-judge score in ``[0, max_score]``.
    """

    token_ids: list[int]
    log_probs: torch.Tensor  # [T]
    mask: torch.Tensor  # [T]
    score: float


@dataclass
class SelfRewardBatch:
    """A batch of prompts with their scored candidates plus reference log-probs.

    Attributes:
        candidates:    ``[B prompts][M candidates each]`` scored candidates.
        ref_log_probs: Reference (frozen) per-token log-probs for every
                       candidate in ``candidates``, laid out as a flat list
                       ordered by prompt then by candidate index (``[T]`` each).
    """

    candidates: list[list[ScoredCandidate]]
    ref_log_probs: list[torch.Tensor]  # flat list, len == sum(len(c) for c in candidates)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SelfRewardTrainer:
    """Self-Rewarding Language Model trainer (DPO variant).

    Usage::

        config  = SelfRewardConfig(n_candidates=4, min_score_gap=2.0)
        trainer = SelfRewardTrainer(config)

        # Build ScoredCandidate objects from actual model outputs / judge scores
        batch = SelfRewardBatch(candidates=..., ref_log_probs=...)
        out = trainer.compute_loss(batch)
        out["loss"].backward()
    """

    def __init__(self, config: SelfRewardConfig | None = None) -> None:
        self.config = config if config is not None else SelfRewardConfig()

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    def sequence_log_prob(
        self,
        log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Mean log-prob over masked (valid) tokens.

        Args:
            log_probs: ``[T]`` per-token log-probabilities.
            mask:      ``[T]`` binary mask; 1 = valid token, 0 = padding.

        Returns:
            Scalar float: mean log-prob over valid positions.
        """
        valid = mask.sum().clamp(min=1)
        return ((log_probs * mask).sum() / valid).item()

    # ------------------------------------------------------------------
    # Preference pair construction
    # ------------------------------------------------------------------

    def create_preference_pairs(
        self,
        candidates: list[ScoredCandidate],
    ) -> list[tuple[ScoredCandidate, ScoredCandidate]]:
        """Enumerate all valid (chosen, rejected) pairs from a candidate list.

        A pair is valid when:
            score(chosen) > score(rejected)  AND
            score(chosen) - score(rejected) >= min_score_gap

        All valid pairs are returned (not just top-1 vs bottom-1), which gives
        richer training signal when multiple score levels are present.

        Args:
            candidates: List of ``ScoredCandidate`` objects for one prompt.

        Returns:
            List of ``(chosen, rejected)`` pairs; empty if no valid pair exists.
        """
        gap = self.config.min_score_gap
        pairs: list[tuple[ScoredCandidate, ScoredCandidate]] = []

        n = len(candidates)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                c = candidates[i]
                r = candidates[j]
                if c.score > r.score and (c.score - r.score) >= gap:
                    pairs.append((c, r))

        return pairs

    # ------------------------------------------------------------------
    # DPO loss
    # ------------------------------------------------------------------

    def dpo_loss(
        self,
        chosen_lp: torch.Tensor,
        chosen_ref_lp: torch.Tensor,
        rejected_lp: torch.Tensor,
        rejected_ref_lp: torch.Tensor,
    ) -> torch.Tensor:
        """Standard DPO loss over a batch of preference pairs.

        L_DPO = -logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
        where log_ratio_x = log_p_policy(x) - log_p_ref(x).

        Args:
            chosen_lp:      ``[B]`` policy sequence log-probs for chosen.
            chosen_ref_lp:  ``[B]`` reference sequence log-probs for chosen.
            rejected_lp:    ``[B]`` policy sequence log-probs for rejected.
            rejected_ref_lp: ``[B]`` reference sequence log-probs for rejected.

        Returns:
            Scalar loss tensor.
        """
        beta = self.config.beta
        log_ratio_chosen = chosen_lp - chosen_ref_lp
        log_ratio_rejected = rejected_lp - rejected_ref_lp
        logits = beta * (log_ratio_chosen - log_ratio_rejected)
        return -F.logsigmoid(logits).mean()

    # ------------------------------------------------------------------
    # Full loss computation
    # ------------------------------------------------------------------

    def compute_loss(self, batch: SelfRewardBatch) -> dict[str, torch.Tensor]:
        """Compute the self-reward DPO loss over a batch of prompts.

        Steps:
            1. For each prompt's candidates call ``create_preference_pairs``.
            2. Collect all valid ``(chosen, rejected)`` pairs across prompts.
            3. Compute sequence log-probs (policy + reference) for each pair.
            4. Compute DPO loss over the collected pairs.

        Args:
            batch: ``SelfRewardBatch`` with candidates and reference log-probs.

        Returns:
            Dict with keys:

            - ``"loss"``            — DPO loss scalar (0 if no valid pairs).
            - ``"n_pairs"``         — number of valid preference pairs (scalar).
            - ``"mean_score_gap"``  — average score gap of valid pairs (scalar).
            - ``"reward_accuracy"`` — fraction of pairs where policy agrees with
                                     judge ordering (chosen_lp > rejected_lp).
        """
        # Build a flat index: ref_log_probs[prompt_idx * M + cand_idx]
        # Precompute offset per prompt
        prompt_offsets: list[int] = []
        offset = 0
        for cands in batch.candidates:
            prompt_offsets.append(offset)
            offset += len(cands)

        chosen_policy_lps: list[torch.Tensor] = []
        chosen_ref_lps: list[torch.Tensor] = []
        rejected_policy_lps: list[torch.Tensor] = []
        rejected_ref_lps: list[torch.Tensor] = []
        score_gaps: list[float] = []

        for p_idx, cands in enumerate(batch.candidates):
            pairs = self.create_preference_pairs(cands)
            base = prompt_offsets[p_idx]

            # Build identity map: id(candidate) -> position index
            id_to_idx = {id(c): i for i, c in enumerate(cands)}

            for chosen, rejected in pairs:
                c_idx = id_to_idx[id(chosen)]
                r_idx = id_to_idx[id(rejected)]

                # Policy sequence log-probs (differentiable)
                c_lp = (chosen.log_probs * chosen.mask).sum() / chosen.mask.sum().clamp(min=1)
                r_lp = (rejected.log_probs * rejected.mask).sum() / rejected.mask.sum().clamp(min=1)

                # Reference sequence log-probs (detached / frozen)
                ref_c = batch.ref_log_probs[base + c_idx]
                ref_r = batch.ref_log_probs[base + r_idx]
                c_ref_lp = (ref_c * chosen.mask).sum() / chosen.mask.sum().clamp(min=1)
                r_ref_lp = (ref_r * rejected.mask).sum() / rejected.mask.sum().clamp(min=1)

                chosen_policy_lps.append(c_lp)
                chosen_ref_lps.append(c_ref_lp)
                rejected_policy_lps.append(r_lp)
                rejected_ref_lps.append(r_ref_lp)
                score_gaps.append(chosen.score - rejected.score)

        n_pairs = len(chosen_policy_lps)

        if n_pairs == 0:
            zero = torch.tensor(0.0)
            return {
                "loss": zero,
                "n_pairs": torch.tensor(0.0),
                "mean_score_gap": torch.tensor(0.0),
                "reward_accuracy": torch.tensor(0.0),
            }

        # Stack into [n_pairs] tensors
        c_lp_t = torch.stack(chosen_policy_lps)  # [P]
        c_ref_t = torch.stack(chosen_ref_lps)  # [P]
        r_lp_t = torch.stack(rejected_policy_lps)  # [P]
        r_ref_t = torch.stack(rejected_ref_lps)  # [P]

        loss = self.dpo_loss(c_lp_t, c_ref_t, r_lp_t, r_ref_t)

        with torch.no_grad():
            reward_acc = (c_lp_t.detach() > r_lp_t.detach()).float().mean()
            mean_gap = torch.tensor(sum(score_gaps) / len(score_gaps))

        return {
            "loss": loss,
            "n_pairs": torch.tensor(float(n_pairs)),
            "mean_score_gap": mean_gap,
            "reward_accuracy": reward_acc,
        }

    # ------------------------------------------------------------------
    # Score statistics
    # ------------------------------------------------------------------

    def score_statistics(
        self,
        candidates_per_prompt: list[list[ScoredCandidate]],
    ) -> dict[str, float]:
        """Compute descriptive statistics over all judge scores.

        Args:
            candidates_per_prompt: Outer list is per prompt; inner list is
                per candidate (same shape as ``SelfRewardBatch.candidates``).

        Returns:
            Dict with keys:

            - ``"mean_score"``         — mean over all candidate scores.
            - ``"std_score"``          — standard deviation of all scores.
            - ``"max_score_observed"`` — highest score seen.
            - ``"min_score_observed"`` — lowest score seen.
            - ``"pairs_created"``      — total valid preference pairs across
                                        all prompts.
        """
        all_scores: list[float] = []
        for cands in candidates_per_prompt:
            for c in cands:
                all_scores.append(c.score)

        n = len(all_scores)
        if n == 0:
            return {
                "mean_score": 0.0,
                "std_score": 0.0,
                "max_score_observed": 0.0,
                "min_score_observed": 0.0,
                "pairs_created": 0.0,
            }

        scores_t = torch.tensor(all_scores, dtype=torch.float32)
        mean_s = scores_t.mean().item()
        std_s = scores_t.std(correction=0).item() if n > 1 else 0.0
        max_s = scores_t.max().item()
        min_s = scores_t.min().item()

        total_pairs = sum(
            len(self.create_preference_pairs(cands)) for cands in candidates_per_prompt
        )

        return {
            "mean_score": mean_s,
            "std_score": std_s,
            "max_score_observed": max_s,
            "min_score_observed": min_s,
            "pairs_created": float(total_pairs),
        }
