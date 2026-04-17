"""ORPO: Odds Ratio Preference Optimization (Hong et al. 2024).

Combines SFT loss with a reference-free preference loss in a single
forward pass. No reference model is required.

Loss:
    L_ORPO = L_SFT + lambda_ * L_OR

Where:
    L_SFT  = -log p_theta(y_w | x)          (NLL on chosen)
    L_OR   = -log sigma(log(odds_w / odds_l))
    odds_x = p(y_x | x) / (1 - p(y_x | x))

Using mean log-prob as the proxy for p, the log-odds simplify to:
    log_odds(log_p) = log_p - log1p(-exp(log_p))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ORPOConfig:
    """Configuration for ORPO loss.

    Attributes:
        lambda_: Weight on the odds-ratio term (default 0.1).
        beta:    Label smoothing / temperature — reserved for future
                 extensions; not used in the base loss (default 0.1).
    """
    lambda_: float = 0.1
    beta: float = 0.1


# ---------------------------------------------------------------------------
# ORPOLoss
# ---------------------------------------------------------------------------

class ORPOLoss(nn.Module):
    """ORPO loss combining SFT NLL with a reference-free odds-ratio term.

    Args:
        config: :class:`ORPOConfig` holding ``lambda_`` and ``beta``.
    """

    def __init__(self, config: ORPOConfig) -> None:
        super().__init__()
        self.config = config

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def log_odds(self, log_p: Tensor) -> Tensor:
        """Convert a mean log-probability to log-odds.

        log_odds(p) = log(p / (1-p)) = log_p - log(1 - exp(log_p))
                    = log_p - log1p(-exp(log_p))

        Args:
            log_p: (B,) mean log-probabilities.  Values must be < 0.

        Returns:
            (B,) log-odds tensor (finite for valid inputs).
        """
        # Clamp to avoid log(0) at the boundaries.
        log_p = log_p.clamp(-30.0, -1e-7)
        return log_p - torch.log1p(-torch.exp(log_p))

    def sft_loss(self, logits: Tensor, labels: torch.LongTensor) -> Tensor:
        """Standard causal NLL loss ignoring positions labelled -100.

        Args:
            logits: (B, T, V) raw (un-normalised) logits.
            labels: (B, T) target token ids; -100 marks positions to ignore.

        Returns:
            Scalar mean NLL loss.  Returns ``torch.tensor(0.)`` when every
            label is -100 to avoid NaN.
        """
        B, T, V = logits.shape
        # Shift: predict token t+1 from position t.
        shift_logits = logits[:, :-1, :].contiguous().view(-1, V)  # ((B*(T-1)), V)
        shift_labels = labels[:, 1:].contiguous().view(-1)          # (B*(T-1),)

        mask = shift_labels != -100
        if mask.sum() == 0:
            return torch.tensor(0.0, dtype=logits.dtype, device=logits.device)

        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return loss

    def odds_ratio_loss(self, log_p_w: Tensor, log_p_l: Tensor) -> Tensor:
        """Odds-ratio preference loss.

        L_OR = -log_sigmoid(log_odds(w) - log_odds(l))

        Args:
            log_p_w: (B,) mean log-prob for chosen responses.
            log_p_l: (B,) mean log-prob for rejected responses.

        Returns:
            Scalar loss.
        """
        ratio = self.log_odds(log_p_w) - self.log_odds(log_p_l)
        return -F.logsigmoid(ratio).mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        chosen_logits: Tensor,
        rejected_logits: Tensor,
        chosen_labels: torch.LongTensor,
        rejected_labels: torch.LongTensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute ORPO loss.

        Args:
            chosen_logits:   (B, T, V) logits for chosen responses.
            rejected_logits: (B, T, V) logits for rejected responses.
            chosen_labels:   (B, T) labels for chosen; -100 = ignore.
            rejected_labels: (B, T) labels for rejected; -100 = ignore.

        Returns:
            (total_loss, metrics) where metrics contains:
            ``sft_loss``, ``odds_ratio_loss``, ``total_loss``, ``accuracy``.
        """
        # ---- SFT loss on chosen ----
        l_sft = self.sft_loss(chosen_logits, chosen_labels)

        # ---- Per-sequence mean log-probs ----
        log_p_w = self._mean_log_probs(chosen_logits, chosen_labels)
        log_p_l = self._mean_log_probs(rejected_logits, rejected_labels)

        # ---- Odds-ratio loss ----
        l_or = self.odds_ratio_loss(log_p_w, log_p_l)

        # ---- Total ----
        total = l_sft + self.config.lambda_ * l_or

        # ---- Accuracy: fraction where chosen log-odds > rejected log-odds ----
        with torch.no_grad():
            lo_w = self.log_odds(log_p_w.detach())
            lo_l = self.log_odds(log_p_l.detach())
            accuracy = (lo_w > lo_l).float().mean()

        metrics: Dict[str, Tensor] = {
            "sft_loss": l_sft.detach(),
            "odds_ratio_loss": l_or.detach(),
            "total_loss": total.detach(),
            "accuracy": accuracy,
        }
        return total, metrics

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_log_probs(logits: Tensor, labels: torch.LongTensor) -> Tensor:
        """Compute mean per-token log-prob over non-masked positions for each sequence.

        Args:
            logits: (B, T, V)
            labels: (B, T)  — -100 marks positions to ignore.

        Returns:
            (B,) mean log-prob per sequence.
        """
        B, T, V = logits.shape
        # Shift as in causal LM: logit[t] predicts label[t+1].
        shift_logits = logits[:, :-1, :]          # (B, T-1, V)
        shift_labels = labels[:, 1:].clone()       # (B, T-1)

        log_probs = F.log_softmax(shift_logits, dim=-1)  # (B, T-1, V)

        # Gather the log-prob of the actual target token.
        valid_labels = shift_labels.clone()
        valid_labels[valid_labels == -100] = 0          # safe index; masked out below
        per_token_lp = log_probs.gather(
            2, valid_labels.unsqueeze(-1)
        ).squeeze(-1)                                    # (B, T-1)

        mask = (shift_labels != -100).float()            # (B, T-1)
        # Avoid division by zero for fully-masked sequences.
        denom = mask.sum(dim=1).clamp(min=1.0)
        mean_lp = (per_token_lp * mask).sum(dim=1) / denom  # (B,)
        return mean_lp


# ---------------------------------------------------------------------------
# ORPOTrainer
# ---------------------------------------------------------------------------

class ORPOTrainer:
    """Thin training wrapper for ORPO.

    Args:
        model:     The policy model (``nn.Module``).
        optimizer: PyTorch optimiser.
        loss_fn:   An :class:`ORPOLoss` instance.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: ORPOLoss,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compute_log_probs(
        self,
        model: nn.Module,
        logits: Tensor,
        labels: torch.LongTensor,
    ) -> Tensor:
        """Compute mean per-token log-prob over non-masked positions.

        Args:
            model:  Policy model (accepted for API symmetry; not called here
                    since logits are already computed).
            logits: (B, T, V) raw logits.
            labels: (B, T) targets; -100 = ignore.

        Returns:
            (B,) mean log-prob per sequence.
        """
        return ORPOLoss._mean_log_probs(logits, labels)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        chosen_logits: Tensor,
        rejected_logits: Tensor,
        chosen_labels: torch.LongTensor,
        rejected_labels: torch.LongTensor,
    ) -> Dict[str, Tensor]:
        """Run one ORPO optimisation step.

        ``chosen_logits`` **must** be the output of a forward pass through
        ``self.model`` so that gradients can flow back to model parameters.

        Args:
            chosen_logits:   (B, T, V) — must be computed via model.
            rejected_logits: (B, T, V).
            chosen_labels:   (B, T).
            rejected_labels: (B, T).

        Returns:
            Metrics dict with keys: ``sft_loss``, ``odds_ratio_loss``,
            ``total_loss``, ``accuracy``.
        """
        self.model.train()
        self.optimizer.zero_grad()

        total_loss, metrics = self.loss_fn(
            chosen_logits,
            rejected_logits,
            chosen_labels,
            rejected_labels,
        )

        total_loss.backward()
        self.optimizer.step()

        return metrics
