"""DELLA (Drop and rEscaLe via mAgnitude) model merging — Bhatnagar et al. 2024.

arXiv: https://arxiv.org/abs/2406.11617
GitHub: https://github.com/declare-lab/della

Unlike DARE (which drops parameters *randomly*), DELLA uses *magnitude-based*
sparsification (MAGPRUNE):
    1. Compute delta = finetuned_weights - pretrained_weights.
    2. Rank delta values by absolute magnitude.
    3. Keep only the top `density` fraction (by magnitude); zero the rest.
    4. Rescale surviving deltas by 1 / density to preserve expected magnitude.
    5. Merge multiple deltas via weighted mean (or TIES-style sign election).
    6. Add merged delta back to the pretrained base.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DELLAConfig:
    """Configuration for the DELLA model merging algorithm.

    Attributes:
        density: Fraction of delta values to *keep* (top by |magnitude|).
            0.0 keeps nothing; 1.0 keeps everything.
        rescale: If True, surviving delta values are scaled by 1 / density
            so that the expected magnitude is preserved.
        merge_method: How to combine multiple pruned deltas before adding
            back to the base.  One of ``"mean"``, ``"weighted"``, ``"ties"``.
        weights: Per-model scalar weights used when ``merge_method="weighted"``.
            Must have the same length as ``finetuned_models``.  Values are
            normalised internally so they do not need to sum to 1.
    """

    density: float = 0.5
    rescale: bool = True
    merge_method: str = "mean"
    weights: list[float] | None = None


# ---------------------------------------------------------------------------
# Core helper: magnitude-based pruning of a single tensor
# ---------------------------------------------------------------------------


def magnitude_prune_delta(
    delta: torch.Tensor,
    density: float,
    rescale: bool = True,
) -> torch.Tensor:
    """Keep the top ``density`` fraction of ``delta`` by absolute magnitude.

    Elements in the bottom ``(1 - density)`` fraction are set to zero.
    Optionally rescales surviving values by ``1 / density``.

    Args:
        delta: Arbitrary-shape float tensor representing a weight delta.
        density: Fraction of values to keep (0.0 – 1.0 inclusive).
        rescale: Whether to scale surviving values by ``1 / density``.

    Returns:
        Pruned (and optionally rescaled) tensor with the same shape/dtype as
        ``delta``.
    """
    if density <= 0.0:
        return torch.zeros_like(delta)
    if density >= 1.0:
        return delta.clone()

    flat_abs = delta.abs().flatten()
    # Threshold: the (1 - density) quantile of |delta|.
    # Values *strictly* below this are zeroed.
    threshold = torch.quantile(flat_abs.float(), 1.0 - density)
    mask = (delta.abs() >= threshold).to(delta.dtype)

    result = delta * mask
    if rescale:
        result = result / density
    return result


# ---------------------------------------------------------------------------
# Functional API
# ---------------------------------------------------------------------------


def della_merge(
    base_model: dict[str, torch.Tensor],
    finetuned_models: list[dict[str, torch.Tensor]],
    config: DELLAConfig,
) -> dict[str, torch.Tensor]:
    """Merge multiple fine-tuned models into a single merged state dict.

    Implements the DELLA pipeline:
        delta_i   = finetuned_i - base
        pruned_i  = magnitude_prune_delta(delta_i, density, rescale)
        merged    = base + combine(pruned_1, ..., pruned_n)

    The ``combine`` step depends on ``config.merge_method``:
        - ``"mean"``     — element-wise mean of pruned deltas.
        - ``"weighted"`` — weighted mean using ``config.weights``.
        - ``"ties"``     — sign-elected mean (majority-vote per position).

    Args:
        base_model: State dict of the pretrained base model.
        finetuned_models: List of fine-tuned model state dicts.
        config: DELLA hyper-parameters.

    Returns:
        Merged state dict with the same keys as ``base_model``.
    """
    merger = DELLAMerger(config)
    if config.merge_method == "ties":
        return merger.merge_with_ties(base_model, finetuned_models)
    return merger.merge(base_model, finetuned_models)


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------


class DELLAMerger:
    """High-level object-oriented interface to the DELLA merge algorithm.

    Args:
        config: A :class:`DELLAConfig` instance controlling merge behaviour.
    """

    def __init__(self, config: DELLAConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------

    def compute_delta(
        self,
        base: dict[str, torch.Tensor],
        finetuned: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return ``{key: finetuned[key] - base[key]}`` for weight tensors.

        Only keys that appear in *both* dicts with matching shapes are included.

        Args:
            base: Base model state dict.
            finetuned: Fine-tuned model state dict.

        Returns:
            Dict of delta tensors (float).
        """
        delta: dict[str, torch.Tensor] = {}
        for key in base:
            if key in finetuned and finetuned[key].shape == base[key].shape:
                delta[key] = finetuned[key].float() - base[key].float()
        return delta

    def prune_delta(self, delta: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply magnitude pruning to every tensor in ``delta``.

        Uses ``config.density`` and ``config.rescale``.

        Args:
            delta: Dict of delta tensors produced by :meth:`compute_delta`.

        Returns:
            New dict with pruned tensors.
        """
        return {
            key: magnitude_prune_delta(tensor, self.config.density, self.config.rescale)
            for key, tensor in delta.items()
        }

    # ------------------------------------------------------------------
    # Merge methods
    # ------------------------------------------------------------------

    def merge(
        self,
        base: dict[str, torch.Tensor],
        finetuned_list: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Merge fine-tuned models into ``base`` using mean or weighted merge.

        For ``config.merge_method == "weighted"`` the caller must provide
        ``config.weights``.

        Args:
            base: Pretrained base model state dict.
            finetuned_list: List of fine-tuned model state dicts.

        Returns:
            Merged state dict.
        """
        if not finetuned_list:
            return {k: v.clone() for k, v in base.items()}

        # Compute and prune per-model deltas
        pruned_deltas = [self.prune_delta(self.compute_delta(base, ft)) for ft in finetuned_list]

        # Determine per-model weights
        n = len(pruned_deltas)
        if self.config.merge_method == "weighted" and self.config.weights is not None:
            raw_weights = list(self.config.weights)
            w_sum = sum(raw_weights)
            if w_sum == 0.0:
                norm_weights = [0.0] * n
            else:
                norm_weights = [w / w_sum for w in raw_weights]
        else:
            # "mean" or fallback: uniform weights
            norm_weights = [1.0 / n] * n

        # Combine deltas key-by-key
        merged_state: dict[str, torch.Tensor] = {}
        for key in base:
            base_param = base[key]
            combined = torch.zeros_like(base_param, dtype=torch.float)
            total_w = 0.0
            for pd, w in zip(pruned_deltas, norm_weights):
                if key in pd:
                    combined = combined + w * pd[key].float()
                    total_w += w
            if total_w > 0.0:
                # Already normalised — just add
                merged_state[key] = (base_param.float() + combined).to(base_param.dtype)
            else:
                merged_state[key] = base_param.clone()

        return merged_state

    def merge_with_ties(
        self,
        base: dict[str, torch.Tensor],
        finetuned_list: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """TIES-style merge: elect sign per parameter before averaging.

        After magnitude pruning, for each scalar position the "elected sign"
        is determined by majority vote (sign of the element-wise sum of all
        pruned deltas).  Only deltas whose sign agrees with the elected sign
        contribute to the final average.

        Args:
            base: Pretrained base model state dict.
            finetuned_list: List of fine-tuned model state dicts.

        Returns:
            Merged state dict.
        """
        if not finetuned_list:
            return {k: v.clone() for k, v in base.items()}

        pruned_deltas = [self.prune_delta(self.compute_delta(base, ft)) for ft in finetuned_list]

        merged_state: dict[str, torch.Tensor] = {}
        for key in base:
            base_param = base[key]

            tensors = [pd[key].float() for pd in pruned_deltas if key in pd]
            if not tensors:
                merged_state[key] = base_param.clone()
                continue

            # Elect sign: sign of sum across all pruned deltas
            stacked = torch.stack(tensors, dim=0)  # (n, *shape)
            elected_sign = torch.sign(stacked.sum(dim=0))  # (*shape)

            # Average only agreeing contributions
            merged_delta = torch.zeros_like(base_param, dtype=torch.float)
            count = torch.zeros_like(base_param, dtype=torch.float)
            for t in tensors:
                agrees = (t.sign() == elected_sign) & (t != 0)
                merged_delta = merged_delta + t * agrees.float()
                count = count + agrees.float()

            count = count.clamp(min=1.0)
            merged_delta = merged_delta / count

            merged_state[key] = (base_param.float() + merged_delta).to(base_param.dtype)

        return merged_state
