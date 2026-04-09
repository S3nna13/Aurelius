"""Preference optimization: ORPO, KTO, and RRHF for alignment without reference models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreferenceOptConfig:
    """Configuration for ORPO, KTO, and RRHF preference optimization."""

    method: str = "orpo"               # "orpo" | "kto" | "rrhf"
    beta: float = 0.1
    lambda_: float = 1.0               # ORPO SFT loss weight
    desirable_weight: float = 1.0      # KTO weight for chosen responses
    undesirable_weight: float = 1.0    # KTO weight for rejected responses


# ---------------------------------------------------------------------------
# Shared utility: compute per-sequence log-probabilities
# ---------------------------------------------------------------------------

def compute_sequence_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Compute per-sequence sum of log-probabilities.

    Args:
        model: Policy model. Forward returns (_, logits, _).
        input_ids: Shape (B, seq_len).
        labels: Shape (B, seq_len). Positions with -100 are masked (padding).

    Returns:
        Shape (B,) — sum of log-probs over non-masked positions.
    """
    _, logits, _ = model(input_ids)  # (B, seq_len, vocab_size)

    # Shift: logits at position t predict token at position t+1
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)  # (B, seq_len-1, vocab_size)

    # Shift labels to align with shifted logits
    shifted_labels = labels[:, 1:]  # (B, seq_len-1)

    # Mask out padding positions (label == -100)
    pad_mask = (shifted_labels != -100).float()  # (B, seq_len-1)

    # Replace -100 with 0 so gather doesn't fail; masked positions are zeroed anyway
    gather_labels = shifted_labels.clone()
    gather_labels[shifted_labels == -100] = 0

    token_lp = log_probs.gather(2, gather_labels.unsqueeze(-1)).squeeze(-1)  # (B, seq_len-1)

    return (token_lp * pad_mask).sum(dim=-1)  # (B,)


# ---------------------------------------------------------------------------
# ORPO loss
# ---------------------------------------------------------------------------

def orpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    policy_chosen_logits: torch.Tensor,
    labels: torch.Tensor,
    lambda_: float,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """Odds Ratio Preference Optimization loss (Hong et al. 2024).

    ORPO combines SFT and preference learning without a reference model by
    penalizing the log odds ratio between chosen and rejected responses.

    Args:
        policy_chosen_logps: Shape (B,) — log-probs for chosen sequences.
        policy_rejected_logps: Shape (B,) — log-probs for rejected sequences.
        policy_chosen_logits: Shape (B, seq_len, vocab_size) — logits for chosen
            sequences (used to compute SFT cross-entropy loss).
        labels: Shape (B, seq_len) — target token ids; -100 = padding/ignored.
        lambda_: Weight for the SFT loss term.
        beta: Weight for the odds-ratio penalty term.

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict has keys
        "sft_loss" and "ratio".
    """
    # SFT loss: negative mean of chosen log-probs (average over the batch)
    sft_loss = -policy_chosen_logps.mean()

    # Odds ratio penalty (log odds ratio between chosen and rejected)
    # odds = p / (1 - p) = exp(logp) / (1 - exp(logp))
    # Using log: log_odds = logp - log(1 - exp(logp) + eps)
    odds_chosen = policy_chosen_logps - torch.log(
        1.0 - policy_chosen_logps.exp() + 1e-8
    )
    odds_rejected = policy_rejected_logps - torch.log(
        1.0 - policy_rejected_logps.exp() + 1e-8
    )

    # Log odds ratio: log(odds_chosen / odds_rejected)
    ratio = torch.log(torch.exp(odds_chosen) / (torch.exp(odds_rejected) + 1e-8) + 1e-8)

    loss = lambda_ * sft_loss - beta * ratio.mean()

    return loss, {
        "sft_loss": sft_loss.item(),
        "ratio": ratio.mean().item(),
    }


# ---------------------------------------------------------------------------
# KTO loss
# ---------------------------------------------------------------------------

def kto_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    desirable_weight: float,
    undesirable_weight: float,
    beta: float,
) -> tuple[torch.Tensor, dict]:
    """Kahneman-Tversky Optimization loss (Ethayarajh et al. 2024).

    KTO maximizes the utility of desirable (chosen) responses and minimizes
    the utility of undesirable (rejected) responses using prospect theory.

    Args:
        policy_chosen_logps: Shape (B,) — policy log-probs for chosen.
        policy_rejected_logps: Shape (B,) — policy log-probs for rejected.
        ref_chosen_logps: Shape (B,) — reference log-probs for chosen.
        ref_rejected_logps: Shape (B,) — reference log-probs for rejected.
        desirable_weight: Weight applied to chosen response utility loss.
        undesirable_weight: Weight applied to rejected response utility loss.
        beta: Scaling coefficient for KL divergence terms.

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict has keys
        "chosen_utility" and "rejected_utility".
    """
    # Reference partition: mean log-prob difference used as a baseline
    z_ref = (ref_chosen_logps + ref_rejected_logps).mean()

    # KL-like terms: clamped to be non-negative (only penalize if policy is worse than ref)
    chosen_KL = (policy_chosen_logps - ref_chosen_logps - z_ref).clamp(min=0)
    rejected_KL = (policy_rejected_logps - ref_rejected_logps - z_ref).clamp(min=0)

    # Utility: desirable should have high utility (close to 1), undesirable low
    chosen_loss = 1.0 - torch.sigmoid(beta * chosen_KL)
    rejected_loss = torch.sigmoid(beta * rejected_KL)

    loss = desirable_weight * chosen_loss.mean() + undesirable_weight * rejected_loss.mean()

    return loss, {
        "chosen_utility": chosen_loss.mean().item(),
        "rejected_utility": rejected_loss.mean().item(),
    }


# ---------------------------------------------------------------------------
# RRHF loss
# ---------------------------------------------------------------------------

def rrhf_loss(
    ranked_logps: list[torch.Tensor],
    margin: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """Rank Responses from Human Feedback loss (Yuan et al. 2023).

    For every pair (i, j) where response i has a higher rank (lower index)
    than response j, penalize if log-prob of i is less than log-prob of j.

    Args:
        ranked_logps: List of (B,) tensors ordered by reward, highest rank first.
            ranked_logps[0] is the best response, ranked_logps[-1] the worst.
        margin: Hinge margin added to each pairwise comparison.

    Returns:
        Tuple of (loss, metrics_dict) where metrics_dict has keys
        "n_pairs" and "mean_margin".
    """
    n = len(ranked_logps)
    total_loss = torch.zeros(1, device=ranked_logps[0].device, dtype=ranked_logps[0].dtype)
    n_pairs = 0
    margin_sum = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            # i has higher rank than j → we want logp_i > logp_j
            # Penalize if logp_i < logp_j (+ margin)
            pair_loss = torch.clamp(ranked_logps[j] - ranked_logps[i] + margin, min=0.0)
            total_loss = total_loss + pair_loss.mean()
            margin_sum += (ranked_logps[i] - ranked_logps[j]).mean().item()
            n_pairs += 1

    if n_pairs > 0:
        loss = total_loss / n_pairs
        mean_margin = margin_sum / n_pairs
    else:
        loss = total_loss
        mean_margin = 0.0

    return loss.squeeze(), {"n_pairs": n_pairs, "mean_margin": mean_margin}


# ---------------------------------------------------------------------------
# PreferenceOptTrainer
# ---------------------------------------------------------------------------

class PreferenceOptTrainer:
    """Trainer for ORPO, KTO, and RRHF preference optimization.

    Args:
        policy_model: The trainable policy model.
        ref_model: Frozen reference model (used by KTO; can be None for ORPO/RRHF).
        config: PreferenceOptConfig specifying method and hyperparameters.
        optimizer: PyTorch optimizer for policy_model parameters.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module | None,
        config: PreferenceOptConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.config = config
        self.optimizer = optimizer

    def _build_labels(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Build next-token labels from input_ids; last position is masked (-100)."""
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100
        return labels

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> dict:
        """Run a single training step.

        Args:
            chosen_ids: Shape (B, seq_len) — chosen token sequences.
            rejected_ids: Shape (B, seq_len) — rejected token sequences.

        Returns:
            Dict with at minimum keys "loss" and "method".
        """
        self.policy_model.train()
        cfg = self.config

        chosen_labels = self._build_labels(chosen_ids)
        rejected_labels = self._build_labels(rejected_ids)

        if cfg.method == "orpo":
            # ORPO: reference-free, uses policy logits directly
            _, chosen_logits, _ = self.policy_model(chosen_ids)
            policy_chosen_logps = compute_sequence_log_probs(
                self.policy_model, chosen_ids, chosen_labels
            )
            policy_rejected_logps = compute_sequence_log_probs(
                self.policy_model, rejected_ids, rejected_labels
            )

            loss, metrics = orpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                chosen_logits,
                chosen_labels,
                lambda_=cfg.lambda_,
                beta=cfg.beta,
            )

        elif cfg.method == "kto":
            policy_chosen_logps = compute_sequence_log_probs(
                self.policy_model, chosen_ids, chosen_labels
            )
            policy_rejected_logps = compute_sequence_log_probs(
                self.policy_model, rejected_ids, rejected_labels
            )

            with torch.no_grad():
                ref_chosen_logps = compute_sequence_log_probs(
                    self.ref_model, chosen_ids, chosen_labels
                )
                ref_rejected_logps = compute_sequence_log_probs(
                    self.ref_model, rejected_ids, rejected_labels
                )

            loss, metrics = kto_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
                desirable_weight=cfg.desirable_weight,
                undesirable_weight=cfg.undesirable_weight,
                beta=cfg.beta,
            )

        elif cfg.method == "rrhf":
            # RRHF: rank chosen above rejected (two-level ranking)
            policy_chosen_logps = compute_sequence_log_probs(
                self.policy_model, chosen_ids, chosen_labels
            )
            policy_rejected_logps = compute_sequence_log_probs(
                self.policy_model, rejected_ids, rejected_labels
            )

            loss, metrics = rrhf_loss([policy_chosen_logps, policy_rejected_logps])

        else:
            raise ValueError(
                f"Unknown method: {cfg.method!r}. Choose 'orpo', 'kto', or 'rrhf'."
            )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item(), "method": cfg.method, **metrics}
