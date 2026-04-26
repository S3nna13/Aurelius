"""ORPO and IPO: DPO variants without reference model issues.

ORPO (Hong et al. 2024) — Odds Ratio Preference Optimization:
    Combines SFT loss + odds-ratio preference loss in one forward pass.
    No reference model required.

IPO (Azar et al. 2024) — Identity Preference Optimization:
    Replaces DPO's log-sigmoid loss with a squared Bregman divergence,
    guaranteeing a unique minimizer and preventing overfitting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# ORPO Loss
# ---------------------------------------------------------------------------


class ORPOLoss(nn.Module):
    """ORPO: Odds Ratio Preference Optimization.

    No reference model needed. Jointly optimises:
    1. SFT loss on chosen responses (standard LM loss)
    2. Relative preference loss using odds ratio

    L_ORPO = L_SFT + lambda_ * L_OR

    Where:
        odds(y|x) = p(y|x) / (1 - p(y|x))
        L_OR = -log sigmoid(log_odds_chosen - log_odds_rejected)

    log_odds(y|x) = log_prob - log(1 - exp(log_prob))
                  = log_prob - softplus(-log_prob)  [numerically stable]

    Args:
        lambda_: weight for the odds-ratio loss component (default 0.1).
    """

    def __init__(self, lambda_: float = 0.1) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def log_odds(self, log_probs: torch.Tensor) -> torch.Tensor:
        """Compute log odds from per-sequence mean log probabilities.

        Args:
            log_probs: (B,) mean log prob per sequence. Values must be < 0.

        Returns:
            (B,) log-odds tensor.
        """
        # log_odds = log_prob - log(1 - exp(log_prob))
        #          = log_prob - softplus(-log_prob)   [stable when log_prob < 0]
        return log_probs - F.softplus(-log_probs)

    def forward(
        self,
        chosen_logps: torch.Tensor,  # (B,) mean log probs for chosen
        rejected_logps: torch.Tensor,  # (B,) mean log probs for rejected
        sft_loss: torch.Tensor,  # scalar SFT loss on chosen
    ) -> tuple[torch.Tensor, dict]:
        """Compute ORPO loss.

        Returns:
            (total_loss, metrics) where metrics has keys:
            'sft_loss', 'or_loss', 'log_odds_ratio', 'or_reward_margin'
        """
        log_odds_chosen = self.log_odds(chosen_logps)
        log_odds_rejected = self.log_odds(rejected_logps)

        log_odds_ratio = log_odds_chosen - log_odds_rejected
        or_loss = -F.logsigmoid(log_odds_ratio).mean()

        total_loss = sft_loss + self.lambda_ * or_loss

        metrics = {
            "sft_loss": sft_loss.detach(),
            "or_loss": or_loss.detach(),
            "log_odds_ratio": log_odds_ratio.detach().mean(),
            "or_reward_margin": log_odds_ratio.detach().mean(),
        }
        return total_loss, metrics


# ---------------------------------------------------------------------------
# IPO Loss
# ---------------------------------------------------------------------------


class IPOLoss(nn.Module):
    """IPO: Identity Preference Optimization.

    Replaces DPO's log-sigmoid loss with a squared Bregman divergence,
    guaranteeing a unique minimizer and preventing overfitting.

    L_IPO = E[(h_chosen - h_rejected - 1/(2*tau))^2]

    Where:
        h = log(pi/pi_ref) for a response
        tau controls regularisation strength

    When tau → ∞, IPO → DPO. When tau → 0, IPO → pure preference.

    Args:
        tau: regularisation strength (default 0.1).
        beta: temperature for implicit reward (default 0.1, same as DPO).
    """

    def __init__(self, tau: float = 0.1, beta: float = 0.1) -> None:
        super().__init__()
        self.tau = tau
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,  # (B,)
        policy_rejected_logps: torch.Tensor,  # (B,)
        ref_chosen_logps: torch.Tensor,  # (B,)
        ref_rejected_logps: torch.Tensor,  # (B,)
    ) -> tuple[torch.Tensor, dict]:
        """Compute IPO loss.

        Steps:
            1. h_chosen   = policy_chosen_logps   - ref_chosen_logps
            2. h_rejected = policy_rejected_logps - ref_rejected_logps
            3. ipo_target = h_chosen - h_rejected - 1/(2*tau)
            4. loss       = mean(ipo_target^2)

        Returns:
            (loss, metrics) where metrics has keys:
            'h_chosen', 'h_rejected', 'ipo_margin', 'loss'
        """
        h_chosen = policy_chosen_logps - ref_chosen_logps
        h_rejected = policy_rejected_logps - ref_rejected_logps

        target = 1.0 / (2.0 * self.tau)
        ipo_target = h_chosen - h_rejected - target
        loss = (ipo_target**2).mean()

        metrics = {
            "h_chosen": h_chosen.detach().mean(),
            "h_rejected": h_rejected.detach().mean(),
            "ipo_margin": (h_chosen - h_rejected).detach().mean(),
            "loss": loss.detach(),
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# ORPO Trainer
# ---------------------------------------------------------------------------


class ORPOTrainer:
    """Train a model with ORPO loss (no reference model needed)."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        lambda_: float = 0.1,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = ORPOLoss(lambda_=lambda_)

    def compute_logps(
        self,
        prompt_ids: torch.Tensor,
        response_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean log prob over response tokens.

        Args:
            prompt_ids:   (seq_len_p,) prompt token ids.
            response_ids: (seq_len_r,) response token ids.

        Returns:
            Scalar mean log probability over response tokens.
        """
        # Build full sequence: [prompt | response]
        full_ids = torch.cat([prompt_ids, response_ids], dim=0).unsqueeze(0)  # (1, T)
        prompt_len = prompt_ids.shape[0]

        _, logits, _ = self.model(full_ids)  # (1, T, vocab)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, vocab)

        # Gather log probs for actual next tokens
        targets = full_ids[:, 1:]  # (1, T-1)
        token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)

        # Only care about response tokens (positions prompt_len onward in next-token preds)
        # Position i in token_lp predicts full_ids[i+1]
        # Response starts at index prompt_len → token_lp starts at prompt_len-1 (predicts prompt_len)  # noqa: E501
        response_lp = token_lp[0, prompt_len - 1 :]  # cover all response token predictions

        return response_lp.mean()

    def compute_sft_loss(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Standard LM loss (NLL) on chosen response tokens.

        Args:
            prompt_ids: (seq_len_p,) prompt token ids.
            chosen_ids: (seq_len_r,) chosen response token ids.

        Returns:
            Scalar NLL loss.
        """
        full_ids = torch.cat([prompt_ids, chosen_ids], dim=0).unsqueeze(0)  # (1, T)
        prompt_len = prompt_ids.shape[0]

        _, logits, _ = self.model(full_ids)  # (1, T, vocab)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T-1, vocab)

        targets = full_ids[:, 1:]  # (1, T-1)
        token_lp = log_probs.gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (1, T-1)

        # SFT loss only on response tokens
        response_lp = token_lp[0, prompt_len - 1 :]
        return -response_lp.mean()

    def train_step(
        self,
        prompt_ids: torch.Tensor,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> dict:
        """Run one ORPO training step.

        Returns:
            Metrics dict including 'loss'.
        """
        self.model.train()
        self.optimizer.zero_grad()

        chosen_logps = self.compute_logps(prompt_ids, chosen_ids)
        rejected_logps = self.compute_logps(prompt_ids, rejected_ids)
        sft_loss = self.compute_sft_loss(prompt_ids, chosen_ids)

        loss, metrics = self.loss_fn(
            chosen_logps=chosen_logps.unsqueeze(0),
            rejected_logps=rejected_logps.unsqueeze(0),
            sft_loss=sft_loss,
        )

        loss.backward()
        self.optimizer.step()

        metrics["loss"] = loss.detach()
        return metrics
