"""Forward-hook-based profiler that records per-layer activation statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


@dataclass
class LayerStats:
    """Activation statistics for a single layer."""

    name: str
    mean: float
    std: float
    abs_max: float
    sparsity: float  # fraction of elements with |x| < 1e-3
    num_elements: int


class ActivationProfiler:
    """Context manager that attaches forward hooks to record activation stats.

    Usage:
        with ActivationProfiler(model) as profiler:
            model(input_ids)
        stats = profiler.stats  # dict[str, LayerStats]
        report = profiler.summary()  # sorted text report
    """

    def __init__(self, model: nn.Module, module_types: tuple = (nn.Linear,)) -> None:
        """
        Args:
            model: The model to profile.
            module_types: Tuple of module types to hook. Default: nn.Linear only.
        """
        self._model = model
        self._module_types = module_types
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._stats: dict[str, LayerStats] = {}

    def __enter__(self) -> ActivationProfiler:
        for name, module in self._model.named_modules():
            if isinstance(module, self._module_types):
                hook = module.register_forward_hook(self._make_hook(name))
                self._hooks.append(hook)
        return self

    def __exit__(self, *args: Any) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _make_hook(self, name: str):
        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            if isinstance(output, tuple):
                output = output[0]
            t = output.detach().float()
            self._stats[name] = LayerStats(
                name=name,
                mean=t.mean().item(),
                std=t.std().item(),
                abs_max=t.abs().max().item(),
                sparsity=(t.abs() < 1e-3).float().mean().item(),
                num_elements=t.numel(),
            )

        return hook_fn

    @property
    def stats(self) -> dict[str, LayerStats]:
        """Return recorded stats keyed by module name."""
        return self._stats

    def summary(self, top_n: int = 10) -> str:
        """Return a human-readable summary sorted by sparsity descending.

        Shows top_n most sparse layers.
        """
        sorted_stats = sorted(self._stats.values(), key=lambda s: s.sparsity, reverse=True)[:top_n]

        lines = [
            f"{'Layer':<60} {'Mean':>10} {'Std':>10} {'AbsMax':>10} {'Sparsity':>10} {'Elements':>12}",  # noqa: E501
            "-" * 112,
        ]
        for s in sorted_stats:
            lines.append(
                f"{s.name:<60} {s.mean:>10.4f} {s.std:>10.4f} {s.abs_max:>10.4f} {s.sparsity:>10.4f} {s.num_elements:>12d}"  # noqa: E501
            )
        return "\n".join(lines)
