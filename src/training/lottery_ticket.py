"""Lottery Ticket Hypothesis: Iterative Magnitude Pruning (IMP).

Implementation of the Lottery Ticket Hypothesis from:
    Frankle & Carlin, "The Lottery Ticket Hypothesis: Finding Sparse,
    Trainable Neural Networks", arXiv:1803.03635.

Key insight: a randomly initialized dense network (θ_0) contains a sparse
subnetwork — the "winning ticket" — whose initialization alone enables it
to train to full accuracy.  The winning ticket is identified via Iterative
Magnitude Pruning (IMP):

    1. Randomly initialize network  →  θ_0
    2. Train for k iterations       →  θ_k
    3. Prune p% of weights with smallest |θ_k|  →  binary mask m
    4. RESET remaining weights to θ_0  (critical: rewind, not retrain)
    5. Repeat from step 2 with the masked network

This module provides:
    - ``LotteryConfig``       — hyperparameter dataclass
    - ``LotteryTicketPruner`` — stateful pruner that tracks θ_0 and current masks

Paper notation used throughout:
    θ_0  — initial weights (stored as ``initial_weights``)
    θ_k  — weights after k training iterations (live model parameters)
    m    — binary mask (1 = keep, 0 = prune)
    p    — pruning rate per round (``LotteryConfig.pruning_rate``)
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class LotteryConfig:
    """Hyperparameters for Iterative Magnitude Pruning (IMP).

    Attributes:
        pruning_rate:   Fraction of *currently unpruned* weights to remove each
                        round.  Paper uses 20 % globally (p = 0.20).
        n_rounds:       Number of IMP rounds to run when calling
                        ``run_iterative_pruning``.
        global_pruning: If True, threshold is computed over *all* non-bias
                        parameters jointly (global magnitude pruning).
                        If False, each parameter tensor is pruned independently.
    """
    pruning_rate: float = 0.20
    n_rounds: int = 5
    global_pruning: bool = True


# ── Core Pruner ───────────────────────────────────────────────────────────────

class LotteryTicketPruner:
    """Stateful implementation of the Lottery Ticket IMP algorithm.

    Usage::

        pruner = LotteryTicketPruner(model)
        pruner.save_initial_weights()        # store θ_0

        # ... train model for k steps ...

        masks = pruner.run_round()           # prune + rewind to θ_0
        print(f"sparsity: {pruner.sparsity():.2%}")

        # ... train again, repeat ...

    Only *weight* parameters (non-bias) are pruned.  Bias parameters are
    always left untouched, matching the paper's convention.
    """

    def __init__(self, model: nn.Module, config: LotteryConfig | None = None) -> None:
        self.model = model
        self.config = config if config is not None else LotteryConfig()
        # θ_0: deep copies of the initial parameter tensors, keyed by name.
        self.initial_weights: dict[str, Tensor] = {}
        # Accumulated masks from all rounds so far.
        self._masks: dict[str, Tensor] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def save_initial_weights(self) -> None:
        """Store θ_0 — deep copies of the current parameter values.

        Must be called immediately after model initialization, *before* any
        training, so that the rewind target represents the true random init.
        """
        self.initial_weights = {
            name: param.data.clone()
            for name, param in self.model.named_parameters()
            if self._is_weight(name)
        }
        logger.debug("Saved initial weights for %d parameter tensors.", len(self.initial_weights))

    def compute_masks(self, pruning_rate: float | None = None) -> dict[str, Tensor]:
        """Compute binary masks by global or per-layer magnitude threshold.

        Masks are computed relative to the *current* (trained) weights θ_k so
        that low-magnitude weights are identified for pruning.  Weights already
        masked to zero in a previous round remain masked (masks are cumulative).

        Args:
            pruning_rate: Fraction of *remaining* (unmasked) weights to prune.
                          Defaults to ``self.config.pruning_rate``.

        Returns:
            A dict mapping parameter name → binary mask Tensor (1 = keep,
            0 = prune), with the same shape as the parameter.
        """
        if pruning_rate is None:
            pruning_rate = self.config.pruning_rate

        weight_params = {
            name: param
            for name, param in self.model.named_parameters()
            if self._is_weight(name)
        }

        if self.config.global_pruning:
            masks = self._compute_global_masks(weight_params, pruning_rate)
        else:
            masks = self._compute_layerwise_masks(weight_params, pruning_rate)

        return masks

    def apply_masks(self, masks: dict[str, Tensor]) -> None:
        """Zero out pruned weights in-place (m ⊙ θ_k).

        Stores the provided masks as the current accumulated mask state.

        Args:
            masks: Dict of {param_name: binary_mask} as returned by
                   ``compute_masks``.
        """
        self._masks = masks
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in masks:
                    param.data.mul_(masks[name])

    def rewind_to_initial(self, masks: dict[str, Tensor]) -> None:
        """Reset unpruned weights to θ_0; pruned positions stay zero.

        This is the critical step that distinguishes LTH from standard
        pruning: rather than fine-tuning pruned weights from θ_k, we
        rewind the *remaining* weights all the way back to their original
        random initialization θ_0.

        Args:
            masks: Dict of {param_name: binary_mask}.  Unmasked positions
                   (mask == 1) are restored; masked positions (mask == 0)
                   are kept at zero.
        """
        if not self.initial_weights:
            raise RuntimeError(
                "Initial weights not saved.  Call save_initial_weights() first."
            )

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in masks and name in self.initial_weights:
                    # Restore unpruned weights to θ_0, zero out pruned ones.
                    param.data.copy_(self.initial_weights[name] * masks[name])

    def sparsity(self) -> float:
        """Fraction of zero weights across all non-bias parameters.

        Returns:
            Float in [0, 1].  0.0 means no pruning; 1.0 means fully pruned.
        """
        total = 0
        zeros = 0
        for name, param in self.model.named_parameters():
            if self._is_weight(name):
                total += param.numel()
                zeros += (param.data == 0).sum().item()
        if total == 0:
            return 0.0
        return zeros / total

    def run_round(self) -> dict[str, Tensor]:
        """Execute one IMP round: compute masks → apply → rewind to θ_0.

        This combines ``compute_masks``, ``apply_masks``, and
        ``rewind_to_initial`` into a single convenience call.

        Returns:
            The binary masks computed this round.
        """
        masks = self.compute_masks()
        apply_masks = masks  # reuse variable for clarity
        self.apply_masks(apply_masks)
        self.rewind_to_initial(apply_masks)
        return apply_masks

    def run_iterative_pruning(self) -> list[dict[str, Tensor]]:
        """Run ``config.n_rounds`` IMP rounds consecutively.

        Note: In the full LTH workflow each round is interleaved with
        training.  This method performs all pruning rounds *without*
        training, which is useful for offline analysis.  For proper LTH
        training, call ``run_round()`` after each training phase.

        Returns:
            List of mask dicts, one per round.
        """
        all_masks = []
        for round_idx in range(self.config.n_rounds):
            masks = self.run_round()
            all_masks.append(masks)
            logger.info(
                "IMP round %d/%d — sparsity: %.2f%%",
                round_idx + 1,
                self.config.n_rounds,
                self.sparsity() * 100,
            )
        return all_masks

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_weight(name: str) -> bool:
        """Return True for non-bias weight parameters.

        Excludes parameters whose name ends with ``.bias`` or equals
        ``bias``, matching the paper's convention of only pruning weight
        matrices.
        """
        parts = name.split(".")
        return parts[-1] != "bias"

    def _compute_global_masks(
        self,
        weight_params: dict[str, nn.Parameter],
        pruning_rate: float,
    ) -> dict[str, Tensor]:
        """Global magnitude pruning: one threshold across all weight tensors.

        Weights already zeroed by previous rounds do not contribute to the
        threshold computation — we prune ``pruning_rate`` of the *remaining*
        live (non-zero) weights.
        """
        # Collect all live (non-zero) magnitudes from all weight params.
        all_mags: list[Tensor] = []
        for name, param in weight_params.items():
            mags = param.data.abs()
            # Consider only currently unmasked weights for threshold.
            if name in self._masks:
                live = mags[self._masks[name].bool()]
            else:
                live = mags.flatten()
            all_mags.append(live)

        if not all_mags:
            return {}

        live_all = torch.cat(all_mags)

        if live_all.numel() == 0:
            # Already fully pruned.
            return {
                name: torch.zeros_like(param.data)
                for name, param in weight_params.items()
            }

        # Compute the magnitude threshold corresponding to pruning_rate.
        k = max(1, int(pruning_rate * live_all.numel()))
        threshold = torch.kthvalue(live_all, k).values.item()

        masks: dict[str, Tensor] = {}
        for name, param in weight_params.items():
            mags = param.data.abs()
            # Build new mask: keep weights above threshold.
            new_mask = (mags > threshold).to(dtype=param.data.dtype)
            # Intersect with existing mask (never un-prune a weight).
            if name in self._masks:
                new_mask = new_mask * self._masks[name]
            masks[name] = new_mask

        return masks

    def _compute_layerwise_masks(
        self,
        weight_params: dict[str, nn.Parameter],
        pruning_rate: float,
    ) -> dict[str, Tensor]:
        """Per-layer magnitude pruning: independent threshold per tensor."""
        masks: dict[str, Tensor] = {}
        for name, param in weight_params.items():
            mags = param.data.abs()
            if name in self._masks:
                live = mags[self._masks[name].bool()]
            else:
                live = mags.flatten()

            if live.numel() == 0:
                masks[name] = torch.zeros_like(mags)
                continue

            k = max(1, int(pruning_rate * live.numel()))
            threshold = torch.kthvalue(live, k).values.item()
            new_mask = (mags > threshold).to(dtype=mags.dtype)
            if name in self._masks:
                new_mask = new_mask * self._masks[name]
            masks[name] = new_mask

        return masks
