"""Advanced weight decay strategies: layer-wise scaling, parameter grouping, and scheduled decay."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class WeightDecayConfig:
    """Configuration for advanced weight decay strategies."""

    base_wd: float = 0.1
    wd_schedule: str = "constant"  # "constant" | "cosine" | "linear_warmup"
    layer_wise_scaling: bool = False  # scale WD per layer depth
    layer_scale_factor: float = 0.9  # multiply WD by this per layer
    exclude_patterns: list[str] = field(default_factory=lambda: ["bias", "norm", "embedding"])
    warmup_steps: int = 100
    total_steps: int = 10000


def should_decay(name: str, exclude_patterns: list[str]) -> bool:
    """Return False if any pattern appears as substring in name, True otherwise."""
    for pattern in exclude_patterns:
        if pattern in name:
            return False
    return True


def compute_layer_index(name: str) -> int:
    """Extract layer index from parameter name (e.g., 'layers.5.attn.w_q.weight' -> 5).

    Returns -1 if no layer index found.
    """
    match = re.search(r"layers\.(\d+)\.", name)
    if match:
        return int(match.group(1))
    return -1


def layer_wise_wd(base_wd: float, layer_idx: int, n_layers: int, scale_factor: float) -> float:
    """Compute layer-wise weight decay.

    Deeper layers get lower WD: base_wd * scale_factor^(n_layers - 1 - layer_idx).
    Layer 0 (first) gets highest WD, last layer gets lowest.
    """
    return base_wd * (scale_factor**layer_idx)


def scheduled_wd(base_wd: float, step: int, config: WeightDecayConfig) -> float:
    """Compute scheduled weight decay at a given step.

    Schedules:
        - "constant": return base_wd
        - "cosine": base_wd * 0.5 * (1 + cos(pi * step / total_steps))
        - "linear_warmup": linearly ramp from 0 to base_wd over warmup_steps, then constant
    """
    if config.wd_schedule == "constant":
        return base_wd
    elif config.wd_schedule == "cosine":
        return base_wd * 0.5 * (1.0 + math.cos(math.pi * step / config.total_steps))
    elif config.wd_schedule == "linear_warmup":
        if step < config.warmup_steps:
            return base_wd * (step / config.warmup_steps)
        return base_wd
    else:
        raise ValueError(f"Unknown WD schedule: {config.wd_schedule}")


def build_param_groups(model: nn.Module, config: WeightDecayConfig) -> list[dict]:
    """Group model parameters for optimizer with appropriate weight decay.

    Returns list of optimizer param groups:
        [{"params": [...], "weight_decay": float, "name": str}, ...]
    """
    if config.layer_wise_scaling:
        # Determine number of layers
        n_layers = 0
        for name, _ in model.named_parameters():
            idx = compute_layer_index(name)
            if idx >= 0:
                n_layers = max(n_layers, idx + 1)

        # Group by layer index for decaying params
        layer_groups: dict[int, list] = {}
        no_decay_params: list = []
        non_layer_decay_params: list = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if not should_decay(name, config.exclude_patterns):
                no_decay_params.append(param)
            else:
                idx = compute_layer_index(name)
                if idx >= 0:
                    layer_groups.setdefault(idx, []).append(param)
                else:
                    non_layer_decay_params.append(param)

        groups = []

        # Non-layer decaying params (e.g., lm_head) get base WD
        if non_layer_decay_params:
            groups.append(
                {
                    "params": non_layer_decay_params,
                    "weight_decay": config.base_wd,
                    "name": "decay_non_layer",
                }
            )

        # Per-layer groups
        for idx in sorted(layer_groups.keys()):
            wd = layer_wise_wd(config.base_wd, idx, n_layers, config.layer_scale_factor)
            groups.append(
                {
                    "params": layer_groups[idx],
                    "weight_decay": wd,
                    "name": f"decay_layer_{idx}",
                }
            )

        # No-decay group
        if no_decay_params:
            groups.append(
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "name": "no_decay",
                }
            )

        return groups
    else:
        # Simple two-group split
        decay_params = []
        no_decay_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if should_decay(name, config.exclude_patterns):
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        groups = []
        if decay_params:
            groups.append(
                {
                    "params": decay_params,
                    "weight_decay": config.base_wd,
                    "name": "decay",
                }
            )
        if no_decay_params:
            groups.append(
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                    "name": "no_decay",
                }
            )

        return groups


class WeightDecayScheduler:
    """Schedules weight decay updates for optimizer param groups."""

    def __init__(self, optimizer, config: WeightDecayConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        # Store initial WD for each group
        self._initial_wds: dict[str, float] = {}
        for group in optimizer.param_groups:
            name = group.get("name", "unnamed")
            self._initial_wds[name] = group["weight_decay"]

    def step(self, current_step: int) -> None:
        """Update weight decay in optimizer param groups based on schedule."""
        for group in self.optimizer.param_groups:
            name = group.get("name", "unnamed")
            base = self._initial_wds.get(name, group["weight_decay"])
            if base > 0.0:
                group["weight_decay"] = scheduled_wd(base, current_step, self.config)
            # Keep no-decay groups at 0

    def get_current_wd(self) -> dict[str, float]:
        """Return current WD per param group name."""
        return {
            group.get("name", "unnamed"): group["weight_decay"]
            for group in self.optimizer.param_groups
        }


def count_decayed_params(param_groups: list[dict]) -> dict[str, int]:
    """Count parameters in decayed vs non-decayed groups.

    Returns {"decayed": int, "non_decayed": int, "total": int}.
    """
    decayed = 0
    non_decayed = 0
    for group in param_groups:
        count = sum(p.numel() for p in group["params"])
        if group["weight_decay"] > 0.0:
            decayed += count
        else:
            non_decayed += count
    return {
        "decayed": decayed,
        "non_decayed": non_decayed,
        "total": decayed + non_decayed,
    }
