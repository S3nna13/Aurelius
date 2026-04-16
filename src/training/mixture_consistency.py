"""Mixture Consistency Training (MixCon-style cross-consistency regularization).

Trains models to produce consistent outputs across semantically equivalent
(paraphrased/augmented) views of the same input. The key regularization
objective: if two inputs are semantically equivalent, the model output
distributions should be similar.

Typical usage::

    config = MixtureConsistencyConfig()
    loss_fn = MixtureConsistencyLoss(
        consistency_weight=config.consistency_weight,
        temperature=config.temperature,
        distance_fn=config.distance_fn,
    )
    trainer = MixtureConsistencyTrainer(model, base_loss_fn=ce_loss)
    result = trainer.train_step(input_ids, labels)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MixtureConsistencyConfig:
    """Configuration for Mixture Consistency Training.

    Attributes:
        consistency_weight: Weight applied to the consistency regularization
            term when added to the base task loss. Larger values enforce
            stronger consistency at the cost of task performance.
        temperature: Softmax temperature for converting logits to
            distributions. Values < 1 sharpen; values > 1 smooth.
        distance_fn: Distribution distance function used to measure
            inconsistency between two augmented views. One of:
            ``'kl'`` (symmetric KL), ``'js'`` (Jensen-Shannon),
            ``'mse'`` (mean-squared error on logits), ``'cosine'``.
        dropout_rate: Fraction of non-PAD tokens randomly replaced with
            UNK in the token-dropout augmentation strategy.
        n_augmented_views: Number of augmented views to create per example.
            Currently only 2 views are used for pairwise consistency.
    """

    consistency_weight: float = 0.1
    temperature: float = 1.0
    distance_fn: str = "kl"
    dropout_rate: float = 0.15
    n_augmented_views: int = 2


# ---------------------------------------------------------------------------
# Standalone augmentation function
# ---------------------------------------------------------------------------

def token_dropout_augment(
    input_ids: Tensor,
    dropout_rate: float = 0.15,
    unk_id: int = 0,
) -> Tensor:
    """Randomly replace tokens with *unk_id* at the given *dropout_rate*.

    Args:
        input_ids: Integer token tensor of shape ``(batch, seq_len)`` or
            ``(seq_len,)``.
        dropout_rate: Fraction of tokens to replace. ``0.0`` leaves the
            tensor unchanged.
        unk_id: Replacement token id (typically the ``[UNK]`` id).

    Returns:
        New tensor with the same shape and dtype as *input_ids*, with a
        random subset of positions replaced by *unk_id*.
    """
    if dropout_rate <= 0.0:
        return input_ids.clone()

    mask = torch.rand_like(input_ids.float()) < dropout_rate
    result = input_ids.clone()
    result[mask] = unk_id
    return result


# ---------------------------------------------------------------------------
# Loss class
# ---------------------------------------------------------------------------

class MixtureConsistencyLoss(nn.Module):
    """Computes consistency loss between two augmented views of an input.

    Supports four distance functions between output distributions:

    * ``'kl'``     -- symmetric KL divergence (average of KL(p||q) and KL(q||p))
    * ``'js'``     -- Jensen-Shannon divergence (bounded in [0, log 2])
    * ``'mse'``    -- mean-squared error between raw logits
    * ``'cosine'`` -- cosine distance (1 - mean cosine similarity)

    Args:
        consistency_weight: Scalar multiplier for the consistency term.
        temperature: Softmax temperature applied before distance computation.
        distance_fn: Which distance function to use.
    """

    _VALID_DISTANCE_FNS = frozenset({"kl", "js", "mse", "cosine"})

    def __init__(
        self,
        consistency_weight: float = 0.1,
        temperature: float = 1.0,
        distance_fn: str = "kl",
    ) -> None:
        super().__init__()
        if distance_fn not in self._VALID_DISTANCE_FNS:
            raise ValueError(
                f"distance_fn must be one of {sorted(self._VALID_DISTANCE_FNS)}, "
                f"got '{distance_fn}'"
            )
        self.consistency_weight = consistency_weight
        self.temperature = temperature
        self.distance_fn = distance_fn

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def compute_consistency_loss(
        self,
        logits1: Tensor,
        logits2: Tensor,
    ) -> Tensor:
        """Compute scalar consistency loss between two sets of logits.

        Args:
            logits1: Raw model logits, shape ``(batch, seq, vocab)``.
            logits2: Raw model logits for the second view, same shape.

        Returns:
            Scalar consistency loss (mean over batch and sequence).
        """
        if self.distance_fn == "mse":
            return F.mse_loss(logits1, logits2)

        p = F.softmax(logits1 / self.temperature, dim=-1)
        q = F.softmax(logits2 / self.temperature, dim=-1)

        if self.distance_fn == "kl":
            return self.kl_consistency(p, q)
        if self.distance_fn == "js":
            return self.js_divergence(p, q)
        # cosine -- operate on distribution vectors
        return self.cosine_consistency(p, q)

    # ------------------------------------------------------------------
    # Distance primitives
    # ------------------------------------------------------------------

    def kl_consistency(self, p: Tensor, q: Tensor) -> Tensor:
        """Symmetric KL divergence: ``(KL(p||q) + KL(q||p)) / 2``.

        Args:
            p: Distribution tensor, shape ``(batch, seq, vocab)``.
            q: Distribution tensor, same shape.

        Returns:
            Scalar non-negative symmetric KL value.
        """
        eps = 1e-10
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)

        kl_pq = (p * (p.log() - q.log())).sum(dim=-1)  # (batch, seq)
        kl_qp = (q * (q.log() - p.log())).sum(dim=-1)  # (batch, seq)
        return ((kl_pq + kl_qp) / 2.0).mean()

    def js_divergence(self, p: Tensor, q: Tensor) -> Tensor:
        """Jensen-Shannon divergence: ``(KL(p||m) + KL(q||m)) / 2``, m=(p+q)/2.

        The JS divergence lies in ``[0, log(2)]``.

        Args:
            p: Distribution tensor, shape ``(batch, seq, vocab)``.
            q: Distribution tensor, same shape.

        Returns:
            Scalar JS divergence value in ``[0, log(2)]``.
        """
        eps = 1e-10
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        m = ((p + q) / 2.0).clamp(min=eps)

        kl_pm = (p * (p.log() - m.log())).sum(dim=-1)
        kl_qm = (q * (q.log() - m.log())).sum(dim=-1)
        return ((kl_pm + kl_qm) / 2.0).mean()

    def cosine_consistency(self, h1: Tensor, h2: Tensor) -> Tensor:
        """Cosine distance: ``1 - mean(cosine_similarity(h1, h2))``.

        Args:
            h1: Hidden-state or distribution tensor,
                shape ``(batch, seq, d)``.
            h2: Second tensor, same shape.

        Returns:
            Scalar cosine distance in ``[0, 2]`` (0 = identical directions).
        """
        sim = F.cosine_similarity(h1, h2, dim=-1)  # (batch, seq)
        return (1.0 - sim).mean()


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------

class MixtureConsistencyTrainer:
    """Training wrapper that combines task loss with consistency regularization.

    For each training step the trainer:

    1. Computes the base task loss on the *original* input.
    2. Creates two augmented views of the input via *augment_fn*.
    3. Runs forward passes on both views to get output logits.
    4. Measures the consistency loss between the two output distributions.
    5. Returns ``total_loss = base_loss + consistency_weight * consistency_loss``.

    Args:
        model: A ``torch.nn.Module`` whose forward signature is
            ``(input_ids) -> logits`` where logits has shape
            ``(batch, seq, vocab)``.
        base_loss_fn: Callable ``(logits, labels) -> scalar_loss``.
        consistency_weight: Weight for the consistency regularization term.
        augment_fn: Optional custom augmentation function
            ``(input_ids) -> input_ids``. Defaults to
            :func:`token_dropout_augment` with the default dropout rate.
    """

    def __init__(
        self,
        model: nn.Module,
        base_loss_fn: Callable[[Tensor, Tensor], Tensor],
        consistency_weight: float = 0.1,
        augment_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> None:
        self.model = model
        self.base_loss_fn = base_loss_fn
        self.consistency_weight = consistency_weight
        self.augment_fn = augment_fn if augment_fn is not None else token_dropout_augment
        self._consistency_loss_fn = MixtureConsistencyLoss(
            consistency_weight=consistency_weight
        )

    def create_augmented_pair(
        self, input_ids: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Create two independently augmented views of *input_ids*.

        Args:
            input_ids: Integer token tensor, shape ``(batch, seq_len)``.

        Returns:
            ``(view1, view2)`` -- a tuple of two tensors each with the same
            shape as *input_ids*.
        """
        view1 = self.augment_fn(input_ids)
        view2 = self.augment_fn(input_ids)
        return view1, view2

    def train_step(
        self,
        input_ids: Tensor,
        labels: Tensor,
    ) -> dict:
        """Perform one training step combining task and consistency losses.

        Args:
            input_ids: Integer token tensor, shape ``(batch, seq_len)``.
            labels: Target token ids, shape ``(batch, seq_len)``.

        Returns:
            Dictionary with keys:

            * ``'loss'``             -- total combined loss (scalar tensor)
            * ``'base_loss'``        -- task loss on original inputs
            * ``'consistency_loss'`` -- consistency regularization loss
            * ``'ratio'``            -- ``consistency_loss / base_loss``
              (``0.0`` if base_loss is zero)
        """
        # Base task loss on original input
        logits_orig = self.model(input_ids)
        base_loss = self.base_loss_fn(logits_orig, labels)

        # Consistency loss between two augmented views
        view1, view2 = self.create_augmented_pair(input_ids)
        logits1 = self.model(view1)
        logits2 = self.model(view2)
        consistency_loss = self._consistency_loss_fn.compute_consistency_loss(
            logits1, logits2
        )

        total_loss = base_loss + self.consistency_weight * consistency_loss

        base_val = base_loss.detach()
        if base_val.abs() > 1e-12:
            ratio = consistency_loss.detach() / base_val
        else:
            ratio = torch.zeros_like(base_val)

        return {
            "loss": total_loss,
            "base_loss": base_loss,
            "consistency_loss": consistency_loss,
            "ratio": ratio,
        }


# ---------------------------------------------------------------------------
# Evaluation utility
# ---------------------------------------------------------------------------

def compute_pairwise_consistency(
    model: nn.Module,
    dataset: List[Tensor],
    n_pairs: int = 100,
) -> float:
    """Measure a model's output consistency on augmented pairs from *dataset*.

    For each sampled example two augmented views are created; the consistency
    score is ``1 - cosine_distance`` between their softmax output distributions
    (averaged over sequence positions and then over sampled pairs).

    Args:
        model: A ``torch.nn.Module`` mapping ``input_ids -> logits``
            ``(batch, seq, vocab)``.
        dataset: List of 1-D or 2-D integer token tensors.
        n_pairs: Number of example pairs to sample and score.

    Returns:
        Mean consistency score in ``[0, 1]``:  ``1`` means perfectly
        consistent, ``0`` means maximally inconsistent.
    """
    model.train(False)
    scores: List[float] = []
    n_samples = min(n_pairs, len(dataset))

    indices = torch.randperm(len(dataset))[:n_samples].tolist()

    with torch.no_grad():
        for idx in indices:
            example = dataset[idx]
            if example.dim() == 1:
                example = example.unsqueeze(0)  # (1, seq_len)

            view1 = token_dropout_augment(example)
            view2 = token_dropout_augment(example)

            logits1 = model(view1)
            logits2 = model(view2)

            p = F.softmax(logits1, dim=-1)
            q = F.softmax(logits2, dim=-1)

            # cosine similarity over vocab dimension -> (batch, seq)
            sim = F.cosine_similarity(p, q, dim=-1).mean().item()
            scores.append(max(0.0, min(1.0, sim)))

    return float(sum(scores) / len(scores)) if scores else 0.0
