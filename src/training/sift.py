"""SIFT: Selective Fine-Tuning via Influence-Based Data Filtering.

Implements selective SFT where each training sample is weighted by its
estimated influence on a target validation metric.  Influence is approximated
via cosine similarity of gradients (no Hessian inversion required):

    I(i) = -∇L_val · ∇L_i / (||∇L_val|| * ||∇L_i||)

A positive score means sample i's gradient aligns with the validation
gradient → likely helpful.  Negative → likely harmful.

Ref: Yi et al., "SIFT: Grounding LLM Reasoning in Contexts via Stochastic
     Instructions Fine-Tuning", arXiv:2403.11399.

Usage::

    scorer = InfluenceScorer(model, probe_params=["lm_head.weight"])
    sift_filter = SIFTFilter(threshold=0.0)
    loss_fn = SIFTLoss()
    trainer = SIFTTrainer(model, optimizer, sift_filter, loss_fn)

    stats = trainer.train_step(
        batch_logits, batch_targets, val_logits, val_targets
    )
    # stats: {'loss': float, 'n_kept': int, 'fraction_kept': float}
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# InfluenceScorer
# ---------------------------------------------------------------------------


class InfluenceScorer:
    """Estimates gradient-based influence via cosine similarity.

    The influence of a training sample on validation loss is approximated as
    the cosine similarity between the validation gradient and the training
    sample gradient with respect to a (possibly small) subset of parameters.

    Args:
        model: The model whose parameters are used for gradient probing.
        probe_params: Optional list of parameter *names* to restrict gradient
            computation to.  If ``None``, all parameters that require gradients
            are used.

    Example::

        scorer = InfluenceScorer(model, probe_params=["transformer.ln_f.weight"])
        score = scorer.score(train_loss, val_loss)  # float in [-1, 1]
    """

    def __init__(
        self,
        model: nn.Module,
        probe_params: list[str] | None = None,
    ) -> None:
        self.model = model

        # Resolve probe parameters: list of nn.Parameter objects.
        named = dict(model.named_parameters())
        if probe_params is not None:
            missing = [n for n in probe_params if n not in named]
            if missing:
                raise ValueError(f"InfluenceScorer: probe_params not found in model: {missing}")
            self._params: list[nn.Parameter] = [named[n] for n in probe_params]
        else:
            self._params = [p for p in model.parameters() if p.requires_grad]

        if not self._params:
            raise ValueError("InfluenceScorer: no parameters available for gradient probing.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, train_loss: Tensor, val_loss: Tensor) -> float:
        """Return cosine similarity between val and train gradients.

        Args:
            train_loss: Scalar loss tensor computed on a training sample
                (must have a grad_fn so autograd can differentiate).
            val_loss: Scalar loss tensor computed on a validation batch.

        Returns:
            Cosine similarity in [-1, 1].  A value > 0 indicates the training
            sample gradient aligns with the validation gradient (helpful); < 0
            indicates misalignment (harmful).
        """
        if train_loss.dim() != 0 or val_loss.dim() != 0:
            raise ValueError("train_loss and val_loss must be scalar tensors.")

        # Compute gradients — create_graph=False is sufficient here.
        grads_train = torch.autograd.grad(
            train_loss,
            self._params,
            create_graph=False,
            allow_unused=True,
        )
        grads_val = torch.autograd.grad(
            val_loss,
            self._params,
            create_graph=False,
            allow_unused=True,
        )

        # Flatten and concatenate, treating unused gradients as zero.
        def _flatten(grads):
            parts = []
            for g, p in zip(grads, self._params):
                if g is None:
                    parts.append(torch.zeros(p.numel(), device=p.device, dtype=p.dtype))
                else:
                    parts.append(g.detach().reshape(-1))
            return torch.cat(parts)

        g_train = _flatten(grads_train)
        g_val = _flatten(grads_val)

        # Cosine similarity: dot / (||a|| * ||b||).  Guard against zero norms.
        norm_train = g_train.norm()
        norm_val = g_val.norm()

        if norm_train == 0.0 or norm_val == 0.0:
            return 0.0

        cos_sim = (g_val @ g_train) / (norm_val * norm_train)
        return float(cos_sim.item())


# ---------------------------------------------------------------------------
# SIFTFilter
# ---------------------------------------------------------------------------


class SIFTFilter:
    """Filters a batch of samples by influence score.

    Two filtering modes are supported:

    * **Threshold mode** (default): keep samples with ``score >= threshold``.
    * **Top-k mode**: keep the ``ceil(top_k * N)`` samples with the highest
      scores.  ``top_k`` should be in ``(0, 1]``.

    Args:
        threshold: Minimum influence score to retain a sample (used when
            ``top_k`` is ``None``).  Default: ``0.0``.
        top_k: If set, fraction of samples to keep (e.g. ``0.5`` keeps the top
            50%).  Overrides ``threshold`` when provided.

    Example::

        f = SIFTFilter(threshold=0.0)
        mask = f.filter(scores)   # BoolTensor of shape (N,)
    """

    def __init__(
        self,
        threshold: float = 0.0,
        top_k: float | None = None,
    ) -> None:
        if top_k is not None and not (0.0 < top_k <= 1.0):
            raise ValueError(f"top_k must be in (0, 1], got {top_k}.")
        self.threshold = threshold
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, scores: Tensor) -> Tensor:
        """Return a boolean mask of samples to keep.

        Args:
            scores: 1-D float tensor of shape ``(N,)`` with influence scores.

        Returns:
            Boolean tensor of shape ``(N,)`` where ``True`` means keep.
        """
        if scores.dim() != 1:
            raise ValueError(f"scores must be 1-D, got shape {tuple(scores.shape)}.")

        n = scores.shape[0]

        if self.top_k is not None:
            k = math.ceil(self.top_k * n)
            k = max(1, min(k, n))
            # argsort descending; take top-k indices.
            topk_indices = torch.topk(scores, k=k, largest=True).indices
            mask = torch.zeros(n, dtype=torch.bool, device=scores.device)
            mask[topk_indices] = True
            return mask

        # Threshold mode.
        return scores >= self.threshold


# ---------------------------------------------------------------------------
# SIFTLoss
# ---------------------------------------------------------------------------


class SIFTLoss:
    """Weighted SFT loss with optional per-sample influence weighting.

    When ``weights`` are provided the loss is the weighted average of
    per-sample mean token losses.  Without weights this reduces to standard
    cross-entropy over all tokens.

    Args:
        base_loss_fn: Callable ``(logits, targets) → Tensor``.  Must return a
            **per-sample** 1-D tensor of shape ``(B,)``.  If ``None``, uses
            cross-entropy reduced over the sequence dimension.

    Example::

        loss_fn = SIFTLoss()
        loss = loss_fn(logits, targets)               # standard CE
        loss = loss_fn(logits, targets, weights)      # weighted CE
    """

    def __init__(self, base_loss_fn=None) -> None:
        self.base_loss_fn = base_loss_fn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _per_sample_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Return per-sample mean CE loss; shape (B,)."""
        if self.base_loss_fn is not None:
            return self.base_loss_fn(logits, targets)

        # logits: (B, T, V) — reshape for F.cross_entropy.
        B, T, V = logits.shape
        # (B*T, V) vs (B*T,) → unreduced, then reshape.
        per_token = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            reduction="none",
        )  # (B*T,)
        return per_token.reshape(B, T).mean(dim=1)  # (B,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        logits: Tensor,
        targets: Tensor,
        weights: Tensor | None = None,
    ) -> Tensor:
        """Compute (optionally weighted) SFT loss.

        Args:
            logits: Shape ``(B, T, V)``.
            targets: Shape ``(B, T)``, integer token ids.
            weights: Optional shape ``(B,)`` float.  Samples with weight 0
                contribute nothing; weights are *not* required to sum to 1
                (they are normalised internally).

        Returns:
            Scalar loss tensor with gradient.
        """
        per_sample = self._per_sample_loss(logits, targets)  # (B,)

        if weights is None:
            return per_sample.mean()

        # Normalise weights so the result is a proper weighted mean.
        w = weights.to(per_sample.device, dtype=per_sample.dtype)
        w_sum = w.sum()
        if w_sum == 0.0:
            # All weights zero — return zero loss (avoids div-by-zero).
            return (per_sample * w).sum()
        return (per_sample * w).sum() / w_sum


# ---------------------------------------------------------------------------
# SIFTTrainer
# ---------------------------------------------------------------------------


class SIFTTrainer:
    """Orchestrates one selective fine-tuning step.

    Per-sample losses on the *training* batch are used as a cheap proxy for
    influence scores (high training loss → the model hasn't learned the sample
    → positive influence direction).  A sign-flip is applied so that
    *lower* proxy loss → *higher* influence, matching the intuition that
    well-fitting examples are most helpful for small learning-rate updates.

    The actual influence scoring via ``InfluenceScorer`` is optional and can
    be layered on top by the caller; ``SIFTTrainer`` operates directly on
    user-supplied proxy scores.

    Args:
        model: The model being fine-tuned.
        optimizer: A ``torch.optim.Optimizer`` instance.
        filter: A ``SIFTFilter`` that selects which samples to train on.
        loss_fn: A ``SIFTLoss`` for computing the training objective.

    Example::

        trainer = SIFTTrainer(model, optimizer, SIFTFilter(), SIFTLoss())
        stats = trainer.train_step(
            batch_logits, batch_targets, val_logits, val_targets
        )
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        filter: SIFTFilter,
        loss_fn: SIFTLoss,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.filter = filter
        self.loss_fn = loss_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_step(
        self,
        batch_logits: Tensor,
        batch_targets: Tensor,
        val_logits: Tensor,
        val_targets: Tensor,
    ) -> dict[str, float]:
        """Perform one selective SFT update step.

        Influence proxy: per-sample validation loss minus per-sample training
        loss.  Samples whose training loss is lower than validation loss have
        positive proxy scores (the model generalises well from them).

        Args:
            batch_logits: Training logits, shape ``(B, T, V)``.
            batch_targets: Training targets, shape ``(B, T)``.
            val_logits: Validation logits, shape ``(B_val, T, V)``.
            val_targets: Validation targets, shape ``(B_val, T)``.

        Returns:
            Dict with keys ``'loss'`` (float), ``'n_kept'`` (int),
            ``'fraction_kept'`` (float).
        """
        B = batch_logits.shape[0]

        # ------------------------------------------------------------------
        # 1. Compute per-sample proxy influence scores (no grad needed).
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Per-sample training losses: (B,)
            train_per_sample = self.loss_fn._per_sample_loss(batch_logits, batch_targets)

            # Scalar validation loss used as reference baseline.
            val_per_sample = self.loss_fn._per_sample_loss(val_logits, val_targets)
            val_mean = val_per_sample.mean()

            # Proxy: samples whose loss > val_mean are "harder" → more influence.
            # Negate so higher training loss → positive score.
            proxy_scores = train_per_sample - val_mean  # (B,)

        # ------------------------------------------------------------------
        # 2. Filter samples.
        # ------------------------------------------------------------------
        mask = self.filter.filter(proxy_scores)  # (B,) bool
        n_kept = int(mask.sum().item())
        fraction_kept = n_kept / B if B > 0 else 0.0

        # ------------------------------------------------------------------
        # 3. Train on kept samples.
        # ------------------------------------------------------------------
        if n_kept == 0:
            return {
                "loss": 0.0,
                "n_kept": n_kept,
                "fraction_kept": fraction_kept,
            }

        kept_logits = batch_logits[mask]  # (n_kept, T, V)
        kept_targets = batch_targets[mask]  # (n_kept, T)

        # Optionally weight by proxy score magnitude for kept samples.
        kept_scores = proxy_scores[mask]
        # Clamp to [eps, inf] — kept samples already have score >= threshold.
        weights = kept_scores.clamp(min=1e-6)

        self.optimizer.zero_grad()
        loss = self.loss_fn(kept_logits, kept_targets, weights=weights)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "n_kept": n_kept,
            "fraction_kept": fraction_kept,
        }
