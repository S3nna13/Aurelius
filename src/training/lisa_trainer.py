"""LISA: Layerwise Importance Sampling for Memory-Efficient LLM Fine-Tuning.

Pan et al. 2024 — https://arxiv.org/abs/2403.17919

Key idea: at each training step, randomly sample K inner layers to unfreeze.
All other inner layers are frozen. Embedding + LM-head are always active.
This reduces peak activation memory to roughly K/N of full fine-tuning.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class LayerActivationSchedule:
    """Determines which layers are active at each training step.

    At each step, ``activate_layers`` indices are sampled from ``[0, n_layers)``
    without replacement, using a deterministic per-step RNG so that experiments
    are reproducible given the same ``seed``.
    """

    def __init__(self, n_layers: int, activate_layers: int, seed: int = 0) -> None:
        if activate_layers < 1:
            raise ValueError(f"activate_layers must be >= 1, got {activate_layers}")
        if activate_layers > n_layers:
            raise ValueError(
                f"activate_layers ({activate_layers}) must be <= n_layers ({n_layers})"
            )
        self.n_layers = n_layers
        self.activate_layers = activate_layers
        self.seed = seed

    def sample(self, step: int) -> list[int]:
        """Return a sorted list of ``activate_layers`` layer indices for *step*.

        Uses ``torch.randperm`` with a generator seeded on ``(seed + step)`` so
        results are independent across steps yet reproducible.
        """
        gen = torch.Generator()
        gen.manual_seed(self.seed + step)
        perm = torch.randperm(self.n_layers, generator=gen)
        indices = perm[: self.activate_layers].tolist()
        return sorted(indices)

    def activation_fraction(self) -> float:
        """Fraction of layers active at any given step."""
        return self.activate_layers / self.n_layers


class LISALayerManager:
    """Manages per-step freeze/unfreeze of transformer inner layers.

    Args:
        named_layer_params: Mapping from layer index (integer key) to the list
            of ``nn.Parameter`` objects belonging to that layer.  Keys may be
            integers or strings of integers.
        schedule: A :class:`LayerActivationSchedule` that decides which layers
            are active at each step.
    """

    def __init__(
        self,
        named_layer_params: dict[int, list[nn.Parameter]],
        schedule: LayerActivationSchedule,
    ) -> None:
        # Normalise keys to int so callers can pass either int or str keys.
        self._layer_params: dict[int, list[nn.Parameter]] = {
            int(k): v for k, v in named_layer_params.items()
        }
        self.schedule = schedule

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_all_requires_grad(self, value: bool) -> None:
        for params in self._layer_params.values():
            for p in params:
                p.requires_grad_(value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def activate_step(self, step: int) -> list[int]:
        """Freeze all layers then unfreeze the sampled subset for *step*.

        Returns the sorted list of active layer indices.
        """
        active_indices = self.schedule.sample(step)
        active_set = set(active_indices)

        # Freeze everything first.
        self._set_all_requires_grad(False)

        # Unfreeze only the sampled layers.
        for idx in active_set:
            if idx in self._layer_params:
                for p in self._layer_params[idx]:
                    p.requires_grad_(True)

        return active_indices

    def unfreeze_all(self) -> None:
        """Restore ``requires_grad=True`` on every managed parameter."""
        self._set_all_requires_grad(True)


class LISATrainer:
    """Thin training loop wrapper that applies LISA layer sampling each step.

    Args:
        model: The model being fine-tuned.
        optimizer: An already-constructed ``torch.optim.Optimizer`` wrapping the
            model's parameters.
        layer_manager: A :class:`LISALayerManager` configured for the model.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        layer_manager: LISALayerManager,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.layer_manager = layer_manager

    def train_step(self, loss_fn: Callable[[], torch.Tensor], step: int) -> dict[str, float]:
        """Execute one LISA training step.

        1. Activate the sampled layers for *step*.
        2. Zero gradients.
        3. Compute loss via ``loss_fn()``.
        4. Backward pass.
        5. Optimizer step.

        Returns a dict with:
            ``loss``          – scalar loss value (Python float).
            ``active_layers`` – number of layers unfrozen this step.
            ``n_active_params``– total count of trainable parameters this step.
        """
        active_indices = self.layer_manager.activate_step(step)

        self.optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        self.optimizer.step()

        n_active_params: int = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        return {
            "loss": loss.item(),
            "active_layers": len(active_indices),
            "n_active_params": n_active_params,
        }
