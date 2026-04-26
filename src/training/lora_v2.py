"""LoRA v2: Low-Rank Adaptation with LoRALinear, LoRAModel, and helpers.

Implements low-rank decomposition adapters B @ A added to frozen pretrained
weights. Only lora_A and lora_B are trained; base weights are frozen.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
https://arxiv.org/abs/2106.09685
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class LoRAConfig:
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    merge_weights: bool = False


# ---------------------------------------------------------------------------
# LoRALinear
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a LoRA adapter."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int,
        lora_alpha: float,
        lora_dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = lora_alpha / r

        # Frozen base weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False) if bias else None

        # Trainable low-rank adapters
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # Dropout on LoRA path
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        # Stored delta for unmerge
        self._merged_delta: Tensor | None = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B already zeros

    def forward(self, x: Tensor) -> Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora_delta = (
            F.linear(F.linear(self.lora_dropout(x), self.lora_A), self.lora_B) * self.scaling
        )
        return base + lora_delta

    def merge(self) -> None:
        """Fold lora_B @ lora_A * scaling into weight."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self._merged_delta = delta.detach().clone()
        self.weight.data += delta.detach()
        self.lora_A.data.zero_()
        self.lora_B.data.zero_()

    def unmerge(self) -> None:
        """Subtract the previously merged delta from weight."""
        if self._merged_delta is not None:
            self.weight.data -= self._merged_delta
            self._merged_delta = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def freeze_non_lora_params(model: nn.Module) -> None:
    """Freeze all parameters except lora_A and lora_B in LoRALinear layers."""
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def apply_lora(
    linear: nn.Linear,
    r: int,
    lora_alpha: float,
    lora_dropout: float = 0.0,
) -> LoRALinear:
    """Wrap an existing nn.Linear with LoRA, copying its weight and bias."""
    has_bias = linear.bias is not None
    lora_linear = LoRALinear(
        in_features=linear.in_features,
        out_features=linear.out_features,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias=has_bias,
    )
    lora_linear.weight.data.copy_(linear.weight.data)
    if has_bias and linear.bias is not None:
        lora_linear.bias.data.copy_(linear.bias.data)
    return lora_linear


# ---------------------------------------------------------------------------
# LoRAModel
# ---------------------------------------------------------------------------


class LoRAModel(nn.Module):
    """Wraps a base model and replaces target Linear layers with LoRALinear."""

    def __init__(
        self,
        base_model: nn.Module,
        config: LoRAConfig,
        module_names: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.config = config
        self._replace_linears(module_names)

    def _replace_linears(self, module_names: list[str] | None) -> None:
        for name, module in list(self.base_model.named_modules()):
            if not isinstance(module, nn.Linear):
                continue
            if module_names is not None and name not in module_names:
                continue
            lora_layer = apply_lora(
                module, self.config.r, self.config.lora_alpha, self.config.lora_dropout
            )
            # Navigate to the parent and swap the child
            parts = name.split(".")
            parent = self.base_model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], lora_layer)

    def forward(self, *args, **kwargs):
        return self.base_model(*args, **kwargs)

    def get_lora_params(self) -> list[Tensor]:
        """Return all lora_A and lora_B parameter tensors."""
        params = []
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                params.append(module.lora_A)
                params.append(module.lora_B)
        return params

    def merge_all(self) -> None:
        """Merge all LoRALinear adapters into their base weights."""
        for module in self.base_model.modules():
            if isinstance(module, LoRALinear):
                module.merge()

    def save_adapter(self) -> dict[str, Tensor]:
        """Return a state dict containing only LoRA adapter parameters."""
        adapter_state = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, LoRALinear):
                adapter_state[f"{name}.lora_A"] = module.lora_A.data.clone()
                adapter_state[f"{name}.lora_B"] = module.lora_B.data.clone()
        return adapter_state
