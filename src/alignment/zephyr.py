"""Aurelius -- Zephyr-style AIF / self-play preference data generation.

Implements the data generation and distillation components from:
    Zephyr: Direct Distillation of LM Alignment
    Tunstall et al., arXiv:2310.16944

Two-stage alignment pipeline:
    1. dSFT  (distilled SFT):  fine-tune on teacher-generated completions.
    2. dDPO  (distilled DPO):  apply DPO using AI-feedback-scored preference pairs.

Reference:
    https://arxiv.org/abs/2310.16944
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# AIFeedbackScorer
# ---------------------------------------------------------------------------


class AIFeedbackScorer:
    """Score completions using mean log-probability (lower perplexity = higher score).

    In the Zephyr pipeline a teacher LLM (or the model itself) scores each
    sampled completion by its per-token log-probability.  Higher mean log-prob
    indicates a more probable / higher-quality completion under the model.

    Args:
        temperature: Softmax temperature applied before scoring.  When 1.0
            (default) the raw log-probs are used unchanged.
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def score_batch(self, log_probs_list: list[torch.Tensor]) -> torch.Tensor:
        """Score a batch of completions by their mean token log-probability.

        Args:
            log_probs_list: List of N tensors, each of shape (T_i,) containing
                per-token log-probabilities for one completion.

        Returns:
            Tensor of shape (N,) with one score per completion.  Higher is
            better (less negative means lower perplexity).
        """
        scores = []
        for lp in log_probs_list:
            if lp.numel() == 0:
                # Edge-case: empty completion gets the worst possible score.
                scores.append(torch.tensor(float("-inf"), dtype=torch.float32))
            else:
                # Apply temperature scaling then take mean.
                scores.append((lp / self.temperature).mean())
        return torch.stack(scores)


# ---------------------------------------------------------------------------
# PreferenceDataBuilder
# ---------------------------------------------------------------------------


class PreferenceDataBuilder:
    """Build (chosen, rejected) preference pairs from multiple completions.

    Given N completions for a single prompt, the builder scores them with an
    :class:`AIFeedbackScorer` and selects the best as *chosen* and the worst
    as *rejected*.  Pairs where the score gap is below ``margin`` are
    discarded as uninformative.

    Args:
        scorer: An :class:`AIFeedbackScorer` instance.
        margin:  Minimum score gap required to form a usable pair.  Pairs
            with ``score_best - score_worst < margin`` return ``(None, None)``.
    """

    def __init__(self, scorer: AIFeedbackScorer, margin: float = 0.1) -> None:
        self.scorer = scorer
        self.margin = margin

    def build_pairs(
        self, completions_log_probs: list[torch.Tensor]
    ) -> tuple[int | None, int | None]:
        """Return (chosen_idx, rejected_idx) for a set of completions.

        Args:
            completions_log_probs: List of N tensors, each of shape (T_i,).

        Returns:
            ``(chosen_idx, rejected_idx)`` — integer indices into
            ``completions_log_probs``, or ``(None, None)`` if the pair is
            not usable (margin too small or fewer than 2 completions).
        """
        if len(completions_log_probs) < 2:
            return (None, None)

        scores = self.scorer.score_batch(completions_log_probs)  # (N,)
        chosen_idx = int(scores.argmax().item())
        rejected_idx = int(scores.argmin().item())

        best_score = scores[chosen_idx].item()
        worst_score = scores[rejected_idx].item()

        if (best_score - worst_score) < self.margin:
            return (None, None)

        return (chosen_idx, rejected_idx)

    def build_batch_pairs(
        self, batch: list[list[torch.Tensor]]
    ) -> list[tuple[int | None, int | None]]:
        """Process a list of per-prompt completion sets.

        Args:
            batch: List of M items, each being a list of N_i completion
                log-prob tensors (one per sampled completion for that prompt).

        Returns:
            List of M ``(chosen_idx, rejected_idx)`` tuples.
        """
        return [self.build_pairs(completions) for completions in batch]


# ---------------------------------------------------------------------------
# dDPO Loss
# ---------------------------------------------------------------------------


class dDPOLoss:
    """DPO loss for distilled training (dDPO).

    Applies the standard DPO objective (Rafailov et al., 2023) using
    AI-feedback-scored preference pairs rather than human-annotated ones.

    Loss formula::

        L = -logsigmoid(beta * ((pi_w - ref_w) - (pi_l - ref_l)))

    where ``w`` / ``l`` denote chosen / rejected completions.

    Args:
        beta: KL-regularisation coefficient.  Larger values keep the policy
            closer to the reference model.
    """

    def __init__(self, beta: float = 0.1) -> None:
        self.beta = beta

    def __call__(
        self,
        pi_lp_chosen: torch.Tensor,
        pi_lp_rejected: torch.Tensor,
        ref_lp_chosen: torch.Tensor,
        ref_lp_rejected: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dDPO loss and metrics.

        Args:
            pi_lp_chosen:   (B,) mean log-probs of chosen completions under policy.
            pi_lp_rejected: (B,) mean log-probs of rejected completions under policy.
            ref_lp_chosen:  (B,) mean log-probs of chosen completions under reference.
            ref_lp_rejected:(B,) mean log-probs of rejected completions under reference.

        Returns:
            ``(loss, metrics)`` where ``loss`` is a scalar tensor and
            ``metrics`` is a dict with keys:
                - ``'reward_chosen'``:   mean implicit reward for chosen responses.
                - ``'reward_rejected'``: mean implicit reward for rejected responses.
                - ``'accuracy'``:        fraction of pairs where chosen reward > rejected reward.
        """
        reward_chosen = self.beta * (pi_lp_chosen - ref_lp_chosen)
        reward_rejected = self.beta * (pi_lp_rejected - ref_lp_rejected)

        loss = -F.logsigmoid(reward_chosen - reward_rejected).mean()

        accuracy = (reward_chosen > reward_rejected).float().mean()

        metrics: dict[str, torch.Tensor] = {
            "reward_chosen": reward_chosen.mean(),
            "reward_rejected": reward_rejected.mean(),
            "accuracy": accuracy,
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# ZephyrTrainer
# ---------------------------------------------------------------------------


class ZephyrTrainer:
    """Two-stage Zephyr trainer (dSFT + dDPO).

    Stage 1 — dSFT:
        Fine-tune on teacher-generated completions using standard NLL loss.

    Stage 2 — dDPO:
        Fine-tune on AI-feedback preference pairs using :class:`dDPOLoss`.

    Args:
        model:     The policy model being trained.
        ref_model: The frozen reference model used by dDPO.
        optimizer: Optimizer for ``model`` parameters.
        loss_fn:   A :class:`dDPOLoss` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: dDPOLoss,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    # ------------------------------------------------------------------
    # Stage 1: dSFT
    # ------------------------------------------------------------------

    def dsft_step(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the dSFT (distilled SFT) NLL loss and update model.

        The standard teacher-forcing cross-entropy loss: predict token t+1
        from logits at position t.

        Args:
            logits:    (B, T, V) unnormalised logit tensor from the model.
            token_ids: (B, T)   ground-truth token ids (teacher completions).

        Returns:
            Scalar NLL loss tensor (after backward + optimizer step).
        """
        # Shift: predict token i+1 from position i.
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = token_ids[:, 1:].contiguous()  # (B, T-1)

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    # ------------------------------------------------------------------
    # Stage 2: dDPO
    # ------------------------------------------------------------------

    def ddpo_step(
        self,
        pi_lp_w: torch.Tensor,
        pi_lp_l: torch.Tensor,
        ref_lp_w: torch.Tensor,
        ref_lp_l: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute dDPO loss, update model, and return metrics.

        Args:
            pi_lp_w:  (B,) policy mean log-probs for chosen completions.
            pi_lp_l:  (B,) policy mean log-probs for rejected completions.
            ref_lp_w: (B,) reference mean log-probs for chosen completions.
            ref_lp_l: (B,) reference mean log-probs for rejected completions.

        Returns:
            ``(loss, metrics)`` — see :meth:`dDPOLoss.__call__` for metrics keys.
        """
        loss, metrics = self.loss_fn(pi_lp_w, pi_lp_l, ref_lp_w, ref_lp_l)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss, metrics

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def freeze_ref(self) -> None:
        """Freeze all parameters of the reference model."""
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
