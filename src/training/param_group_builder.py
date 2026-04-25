from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Callable


@dataclass
class ParamGroupConfig:
    weight_decay: float = 0.1
    no_decay_patterns: tuple[str, ...] = ("bias", "norm", "embedding", "ln_")


class ParamGroupBuilder:
    """Splits model parameters into decay and no-decay groups for AdamW."""

    def build(self, model: nn.Module, config: ParamGroupConfig) -> list[dict]:
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            # 1D params (bias, norm weight) or matching no_decay_patterns → no decay
            if param.ndim == 1 or any(pat in name for pat in config.no_decay_patterns):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def count_params(self, model: nn.Module) -> dict:
        """Returns total, trainable, and non-trainable param counts."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def get_param_stats(self, model: nn.Module) -> dict:
        """Per-layer parameter counts for diagnostics."""
        return {name: p.numel() for name, p in model.named_parameters()}


PARAM_GROUP_BUILDER = ParamGroupBuilder()
