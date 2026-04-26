"""LoRA++ and RSLoRA: Improved LoRA training dynamics.

LoRA++: Different learning rates for A (small) and B (large) matrices.
RSLoRA: Scale by 1/sqrt(r) instead of alpha/r for stable training at high ranks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer


@dataclass
class LoRAPlusConfig:
    rank: int = 8
    alpha: float = 16.0  # scaling factor
    lr_ratio: float = 16.0  # B lr = lr_A * lr_ratio
    use_rslora: bool = False  # if True, use RSLoRA scaling (1/sqrt(r))
    dropout: float = 0.0


class LoRAPlusLinear(nn.Module):
    """LoRA++ linear layer with asymmetric A/B learning rates.

    If use_rslora=True: scaling = 1/sqrt(rank) instead of alpha/rank
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cfg: LoRAPlusConfig,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cfg = cfg
        r = cfg.rank

        # Base weight (frozen)
        self.weight = nn.Parameter(torch.zeros(out_features, in_features), requires_grad=False)

        # LoRA matrices
        self.A = nn.Parameter(torch.empty(r, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, r))

        # Kaiming init for A, zero init for B (standard LoRA)
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # LoRA dropout
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        # RSLoRA scaling: 1/sqrt(r) vs alpha/r
        if cfg.use_rslora:
            self.scaling = 1.0 / math.sqrt(r)
        else:
            self.scaling = cfg.alpha / r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = F.linear(x, self.weight)
        lora_out = self.scaling * F.linear(F.linear(self.dropout(x), self.A), self.B)
        return base_out + lora_out

    def get_param_groups(self, lr_a: float) -> list[dict]:
        """Get separate parameter groups for A and B with asymmetric lr.

        Returns list of param_group dicts ready for optimizer:
        [{"params": [A], "lr": lr_a}, {"params": [B], "lr": lr_a * lr_ratio}]
        """
        return [
            {"params": [self.A], "lr": lr_a},
            {"params": [self.B], "lr": lr_a * self.cfg.lr_ratio},
        ]


def create_loraplus_optimizer(
    loraplus_layers: list[LoRAPlusLinear],
    base_lr: float = 1e-4,
    optimizer_cls=None,  # defaults to torch.optim.AdamW
    **optimizer_kwargs,
) -> Optimizer:
    """Create an optimizer with asymmetric A/B learning rates for LoRA++ layers.

    Args:
        loraplus_layers: List of LoRAPlusLinear modules
        base_lr: Learning rate for A matrices
        optimizer_cls: Optimizer class to use (default: AdamW)

    Returns:
        Optimizer with separate param groups for A (base_lr) and B (base_lr * lr_ratio)
    """
    if optimizer_cls is None:
        from torch.optim import AdamW

        optimizer_cls = AdamW

    param_groups = []
    for layer in loraplus_layers:
        param_groups.extend(layer.get_param_groups(base_lr))

    return optimizer_cls(param_groups, **optimizer_kwargs)
