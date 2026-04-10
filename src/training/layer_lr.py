"""Layer-wise learning rate decay and warmup scheduling."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class LayerLRConfig:
    base_lr: float = 1e-4
    decay_factor: float = 0.9     # lr_layer_i = base_lr * decay_factor^(n_layers - i - 1)
    min_lr: float = 1e-6
    warmup_steps: int = 100
    decay_steps: int = 1000
    decay_type: str = "cosine"    # "cosine" | "linear" | "constant"


def layer_learning_rates(
    n_layers: int,
    base_lr: float,
    decay_factor: float,
    min_lr: float = 1e-6,
) -> list[float]:
    """Compute per-layer learning rates.

    Layer n_layers-1 (top): base_lr
    Layer i: base_lr * decay_factor^(n_layers - 1 - i)
    All capped at min_lr from below.

    Returns list of n_layers floats (index 0 = bottom layer).
    """
    lrs = []
    for i in range(n_layers):
        exponent = n_layers - 1 - i
        lr = base_lr * (decay_factor ** exponent)
        lr = max(lr, min_lr)
        lrs.append(lr)
    return lrs


def warmup_lr_schedule(
    step: int,
    warmup_steps: int,
    base_lr: float,
) -> float:
    """Linear warmup from 0 to base_lr over warmup_steps.
    Returns current lr as float."""
    if warmup_steps <= 0:
        return base_lr
    if step >= warmup_steps:
        return base_lr
    return base_lr * (step / warmup_steps)


def cosine_decay_schedule(
    step: int,
    warmup_steps: int,
    decay_steps: int,
    base_lr: float,
    min_lr: float,
) -> float:
    """Cosine decay from base_lr to min_lr between warmup_steps and decay_steps.
    Returns current lr.
    Before warmup_steps: linear warmup.
    After decay_steps: min_lr."""
    if step < warmup_steps:
        return warmup_lr_schedule(step, warmup_steps, base_lr)
    if step >= decay_steps:
        return min_lr
    # Cosine decay phase
    progress = (step - warmup_steps) / max(decay_steps - warmup_steps, 1)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine_factor


def build_layer_param_groups(
    model: nn.Module,
    config: LayerLRConfig,
) -> list[dict]:
    """Build optimizer param groups with per-layer learning rates.

    Groups:
    - embed: base_lr * decay_factor^n_layers (lowest)
    - layers[i]: layer_learning_rates[i]
    - head (lm_head, norm): base_lr (highest)

    Returns list of dicts: [{"params": [...], "lr": float}, ...]
    """
    n_layers = len(model.layers)
    per_layer_lrs = layer_learning_rates(
        n_layers, config.base_lr, config.decay_factor, config.min_lr
    )

    # Embed group — one step below the bottom layer
    embed_lr = max(config.base_lr * (config.decay_factor ** n_layers), config.min_lr)

    # Head group — full base_lr
    head_lr = config.base_lr

    param_groups: list[dict] = []

    # 1. Embedding
    embed_params = list(model.embed.parameters())
    param_groups.append({"params": embed_params, "lr": embed_lr})

    # 2. One group per transformer layer
    for i, layer in enumerate(model.layers):
        layer_params = list(layer.parameters())
        param_groups.append({"params": layer_params, "lr": per_layer_lrs[i]})

    # 3. Head: norm + lm_head
    # Collect all params already assigned to avoid duplicates (e.g. tied embeddings).
    already_assigned: set[int] = set()
    for g in param_groups:
        for p in g["params"]:
            already_assigned.add(id(p))

    raw_head_params = list(model.norm.parameters()) + list(model.lm_head.parameters())
    head_params = [p for p in raw_head_params if id(p) not in already_assigned]
    param_groups.append({"params": head_params, "lr": head_lr})

    return param_groups


class LayerLROptimizer:
    """Wraps optimizer with layer-wise LR and scheduling."""

    def __init__(
        self,
        model: nn.Module,
        config: LayerLRConfig,
        optimizer_cls: type = torch.optim.AdamW,
    ) -> None:
        self.config = config
        self._n_layers = len(model.layers)
        param_groups = build_layer_param_groups(model, config)
        self.optimizer = optimizer_cls(param_groups)

    def _compute_scale(self, step: int) -> float:
        """Compute the LR scale factor for the current step based on decay_type."""
        if self.config.decay_type == "cosine":
            scheduled_lr = cosine_decay_schedule(
                step,
                self.config.warmup_steps,
                self.config.decay_steps,
                self.config.base_lr,
                self.config.min_lr,
            )
        elif self.config.decay_type == "linear":
            if step < self.config.warmup_steps:
                scheduled_lr = warmup_lr_schedule(step, self.config.warmup_steps, self.config.base_lr)
            elif step >= self.config.decay_steps:
                scheduled_lr = self.config.min_lr
            else:
                progress = (step - self.config.warmup_steps) / max(
                    self.config.decay_steps - self.config.warmup_steps, 1
                )
                scheduled_lr = self.config.base_lr + (self.config.min_lr - self.config.base_lr) * progress
        else:  # "constant"
            scheduled_lr = warmup_lr_schedule(step, self.config.warmup_steps, self.config.base_lr)

        # Scale relative to base_lr
        if self.config.base_lr > 0:
            return scheduled_lr / self.config.base_lr
        return 1.0

    def step(self, step: int) -> None:
        """Update learning rates for current step, then optimizer.step()."""
        scale = self._compute_scale(step)

        # Retrieve the base LR that each group was initialized with
        for group in self.optimizer.param_groups:
            # group["lr"] holds the current (already-scaled) lr;
            # group["_base_lr"] holds the original per-group lr set at init.
            if "_base_lr" not in group:
                group["_base_lr"] = group["lr"]
            group["lr"] = max(group["_base_lr"] * scale, self.config.min_lr)

        self.optimizer.step()

    def zero_grad(self) -> None:
        self.optimizer.zero_grad()

    def get_lrs(self) -> list[float]:
        """Return current learning rate per param group."""
        return [group["lr"] for group in self.optimizer.param_groups]


class LayerLRTrainer:
    """Trains model with layer-wise learning rate decay."""

    def __init__(
        self,
        model: nn.Module,
        config: LayerLRConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = LayerLROptimizer(model, config)

    def train_step(self, input_ids: Tensor, step: int) -> dict:
        """Forward + backward + layer-lr optimizer step.
        Returns dict with: 'loss', 'min_lr', 'max_lr'"""
        self.model.train()
        self.optimizer.zero_grad()

        # Shift labels: predict next token
        labels = input_ids[:, 1:].contiguous()
        model_input = input_ids[:, :-1].contiguous()

        loss, logits, past_key_values = self.model(model_input, labels=labels)

        loss.backward()
        self.optimizer.step(step)

        lrs = self.optimizer.get_lrs()
        return {
            "loss": loss.item(),
            "min_lr": min(lrs),
            "max_lr": max(lrs),
        }
