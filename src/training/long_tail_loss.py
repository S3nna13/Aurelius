"""Long-tail loss functions for handling class imbalance in token-frequency distributions.

Token vocabularies in LLMs follow a heavy-tailed frequency distribution — a small
fraction of tokens account for most occurrences.  Standard cross-entropy treats all
tokens equally and therefore tends to under-fit rare tokens.  The functions below
provide several strategies for re-balancing gradient signal:

  * Inverse-frequency class weights (``compute_class_weights``)
  * Class-balanced weights via effective number (``compute_effective_num_weights``)
  * Weighted cross-entropy (``class_balanced_loss``)
  * Logit adjustment (``logit_adjusted_loss``)
  * Balanced softmax (``balanced_softmax_loss``)
  * Seesaw loss (``SeesawLoss``)

All implementations are pure PyTorch — no external ML libraries required.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class LongTailConfig:
    """Hyperparameters shared across long-tail loss variants.

    Attributes:
        n_classes:           Vocabulary / class count.  Default: 50257 (GPT-2 vocab).
        smoothing:           Label-smoothing factor in [0, 1).  0 = no smoothing.
        focal_gamma:         Focal-loss focusing parameter.  0 = no focal weighting.
        class_freq_power:    Exponent used in inverse-frequency weighting.
        effective_num_beta:  Beta for effective-number weighting (close to 1).
        margin:              Additive margin applied to target logit (LDAM-style).
    """

    n_classes: int = 50257
    smoothing: float = 0.0
    focal_gamma: float = 0.0
    class_freq_power: float = 0.5
    effective_num_beta: float = 0.9999
    margin: float = 0.0


# ---------------------------------------------------------------------------
# Weight computation helpers
# ---------------------------------------------------------------------------


def compute_class_weights(class_counts: torch.Tensor, power: float = 0.5) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    For each class *c*:

        w_c = 1 / count_c^power

    Weights are then normalised so that they sum to ``n_classes`` (the number of
    elements in *class_counts*).

    Args:
        class_counts: (n_classes,) tensor of per-class occurrence counts.
                      Values must be positive.
        power:        Exponent applied to the counts before inversion.
                      0.5 = square-root frequency weighting (default).

    Returns:
        (n_classes,) float tensor of normalised class weights.
    """
    counts = class_counts.float().clamp(min=1e-8)
    weights = 1.0 / counts.pow(power)
    n_classes = counts.numel()
    weights = weights * (n_classes / weights.sum())
    return weights


def compute_effective_num_weights(class_counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    """Compute class-balanced weights using the effective number of samples.

    Following Cui et al. (2019), "Class-Balanced Loss Based on Effective Number
    of Samples":

        E_c = (1 - beta^n_c) / (1 - beta)
        w_c = 1 / E_c

    Weights are normalised to sum to ``n_classes``.

    Args:
        class_counts: (n_classes,) tensor of per-class occurrence counts.
        beta:         Hyperparameter in [0, 1).  Larger values give stronger
                      re-weighting.  Typical: 0.9, 0.99, 0.999, 0.9999.

    Returns:
        (n_classes,) float tensor of normalised class-balanced weights.
    """
    counts = class_counts.float().clamp(min=1.0)
    effective_num = (1.0 - beta**counts) / (1.0 - beta)
    weights = 1.0 / effective_num.clamp(min=1e-8)
    n_classes = counts.numel()
    weights = weights * (n_classes / weights.sum())
    return weights


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def class_balanced_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Weighted cross-entropy using per-class weights.

    Each token's cross-entropy is multiplied by the weight corresponding to its
    target class.  The loss is averaged over valid (non-ignored) positions.

    Args:
        logits:       (B, T, n_classes) or (N, n_classes) raw model logits.
        labels:       (B, T) or (N,) long tensor of target class indices.
        weights:      (n_classes,) per-class weight tensor.
        ignore_index: Target value marking positions to exclude.

    Returns:
        Scalar mean loss.
    """
    n_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, n_classes)
    labels_flat = labels.reshape(-1)

    # Per-token CE (unreduced)
    ce = F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=ignore_index)

    valid_mask = labels_flat != ignore_index

    if not valid_mask.any():
        return logits_flat.new_zeros(())

    # Gather per-token class weight
    w = weights.to(logits_flat.device, dtype=logits_flat.dtype)
    # For ignored positions use dummy index 0; weight will be zeroed out anyway
    safe_labels = labels_flat.clone()
    safe_labels[~valid_mask] = 0
    token_weights = w[safe_labels]

    loss = (token_weights * ce)[valid_mask]
    return loss.sum() / valid_mask.sum().clamp(min=1)


def logit_adjusted_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    log_prior: torch.Tensor,
    tau: float = 1.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Logit-adjusted cross-entropy loss.

    Adjusts logits by adding ``tau * log_prior`` before computing cross-entropy.
    This encourages the model to learn the Bayes-optimal classifier for long-tail
    distributions (Menon et al., 2021).

        adjusted_logit_c = logit_c + tau * log(pi_c)

    where *pi_c* is the class prior probability.

    Args:
        logits:       (B, T, n_classes) or (N, n_classes) raw model logits.
        labels:       (B, T) or (N,) long tensor of target class indices.
        log_prior:    (n_classes,) log of class prior probabilities
                      (i.e. log of normalised class frequencies).
        tau:          Temperature scaling for the prior adjustment (default 1.0).
        ignore_index: Target value marking positions to exclude.

    Returns:
        Scalar mean loss.
    """
    n_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, n_classes)
    labels_flat = labels.reshape(-1)

    lp = log_prior.to(logits_flat.device, dtype=logits_flat.dtype)
    adjusted = logits_flat + tau * lp.unsqueeze(0)

    return F.cross_entropy(adjusted, labels_flat, reduction="mean", ignore_index=ignore_index)


def balanced_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_counts: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Balanced softmax cross-entropy loss.

    Subtracts ``log(n_c / n_total)`` from each class logit before cross-entropy:

        adjusted_logit_c = logit_c - log(n_c / n_total)
                         = logit_c + log(n_total) - log(n_c)

    This encourages the model to represent each class proportionally to its
    frequency in the training set (Ren et al., 2020).

    Args:
        logits:       (B, T, n_classes) or (N, n_classes) raw model logits.
        labels:       (B, T) or (N,) long tensor of target class indices.
        class_counts: (n_classes,) per-class occurrence counts.
        ignore_index: Target value marking positions to exclude.

    Returns:
        Scalar mean loss.
    """
    n_classes = logits.shape[-1]
    logits_flat = logits.reshape(-1, n_classes)
    labels_flat = labels.reshape(-1)

    counts = class_counts.float().clamp(min=1e-8).to(logits_flat.device)
    n_total = counts.sum()
    # log(n_c / n_total) = log(n_c) - log(n_total)
    log_freq = counts.log() - n_total.log()
    adjusted = logits_flat - log_freq.unsqueeze(0)

    return F.cross_entropy(adjusted, labels_flat, reduction="mean", ignore_index=ignore_index)


# ---------------------------------------------------------------------------
# Seesaw Loss
# ---------------------------------------------------------------------------


class SeesawLoss(nn.Module):
    """Seesaw Loss for long-tailed recognition.

    Combines two complementary mechanisms (Wang et al., 2021):

    * **Mitigation factor** — downweights predictions for more-frequent classes
      when they compete with a rare class:

          mitigation_{ij} = (n_j / n_i)^p   if n_j < n_i  else 1

    * **Compensation factor** — re-upweights classes that are incorrectly scored
      higher than the ground-truth class:

          compensation_{ij} = sigmoid(q * (s_j - s_i))

    The combined seesaw weight for class *j* given ground-truth *i* is:

        sw_{ij} = mitigation_{ij} * compensation_{ij}

    The per-sample loss replaces the softmax denominator with a weighted sum.

    Args:
        n_classes: Number of output classes.
        p:         Mitigation factor exponent (default 0.8).
        q:         Compensation factor temperature (default 2.0).
    """

    def __init__(self, n_classes: int, p: float = 0.8, q: float = 2.0) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.p = p
        self.q = q

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute seesaw loss.

        Args:
            logits:       (B, T, n_classes) or (N, n_classes) raw model logits.
            labels:       (B, T) or (N,) long tensor of target class indices.
                          Positions with value -100 are ignored.
            class_counts: (n_classes,) per-class occurrence counts used for the
                          mitigation factor.

        Returns:
            Scalar mean seesaw loss over valid positions.
        """
        n_classes = self.n_classes
        logits_flat = logits.reshape(-1, n_classes)  # (N, C)
        labels_flat = labels.reshape(-1)  # (N,)

        valid_mask = labels_flat != -100
        if not valid_mask.any():
            return logits_flat.new_zeros(())

        lf = logits_flat[valid_mask]  # (M, C)
        lt = labels_flat[valid_mask]  # (M,)

        counts = class_counts.float().clamp(min=1.0).to(lf.device)  # (C,)

        # ---- Mitigation factor ------------------------------------------------
        # n_i: count of the ground-truth class for each sample  (M, 1)
        n_i = counts[lt].unsqueeze(1)  # (M, 1)
        # n_j: count of every class                             (1, C)
        n_j = counts.unsqueeze(0)  # (1, C)

        ratio = (n_j / n_i.clamp(min=1e-8)).clamp(max=1.0)  # (M, C) in [0, 1]
        mitigation = ratio.pow(self.p)  # (M, C)

        # ---- Compensation factor ----------------------------------------------
        # s_i: score of ground-truth class for each sample     (M, 1)
        s_i = lf.gather(1, lt.unsqueeze(1))  # (M, 1)
        # s_j: score of every class                            (M, C)
        s_j = lf  # (M, C)

        compensation = torch.sigmoid(self.q * (s_j - s_i))  # (M, C)

        # ---- Combined seesaw weight ------------------------------------------
        seesaw_w = mitigation * compensation  # (M, C)

        # ---- Weighted cross-entropy ------------------------------------------
        # Replace standard softmax denominator with seesaw-weighted sum
        # log_p_i = s_i - log( sum_j( sw_{ij} * exp(s_j) ) )
        weighted_logits = lf + seesaw_w.log().clamp(min=-1e9)  # (M, C)
        loss = F.cross_entropy(weighted_logits, lt, reduction="mean")

        return loss
