"""Adapter tuning for parameter-efficient fine-tuning of frozen transformer models.

Inserts small trainable bottleneck modules into frozen layers so only adapter
parameters are updated during fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AdapterConfig:
    """Configuration for adapter tuning."""

    d_model: int = 512
    bottleneck_dim: int = 64
    adapter_dropout: float = 0.0
    init_scale: float = 1e-3
    adapter_type: str = "bottleneck"  # "bottleneck", "parallel", "prefix"


class BottleneckAdapter(nn.Module):
    """Sequential bottleneck adapter with residual connection.

    Applies LayerNorm -> down_proj -> GELU -> up_proj and adds the result
    to the input (residual).  Near-zero initialisation ensures the adapter
    starts as an approximate identity transformation.

    forward(x) = x + up_proj(gelu(down_proj(LN(x))))
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        dropout: float = 0.0,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.down_proj = nn.Linear(d_model, bottleneck_dim, bias=True)
        self.up_proj = nn.Linear(bottleneck_dim, d_model, bias=True)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        # Near-zero init: kaiming then scale down so adapter ≈ identity at start
        nn.init.kaiming_uniform_(self.down_proj.weight)
        nn.init.kaiming_uniform_(self.up_proj.weight)
        with torch.no_grad():
            self.down_proj.weight.mul_(init_scale)
            self.up_proj.weight.mul_(init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """x + adapter(x); shape preserved."""
        normed = self.layer_norm(x)
        hidden = F.gelu(self.down_proj(normed))
        hidden = self.dropout(hidden)
        out = self.up_proj(hidden)
        return x + out


class ParallelAdapter(nn.Module):
    """Parallel adapter without LayerNorm or residual.

    Computes down -> GELU -> up and returns the result.  The caller is
    responsible for adding this to the main branch output.

    forward(x) = up_proj(gelu(down_proj(x)))
    """

    def __init__(
        self,
        d_model: int,
        bottleneck_dim: int,
        init_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.down_proj = nn.Linear(d_model, bottleneck_dim, bias=True)
        self.up_proj = nn.Linear(bottleneck_dim, d_model, bias=True)

        nn.init.kaiming_uniform_(self.down_proj.weight)
        nn.init.kaiming_uniform_(self.up_proj.weight)
        with torch.no_grad():
            self.down_proj.weight.mul_(init_scale)
            self.up_proj.weight.mul_(init_scale)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Returns adapter(x) only; caller adds to main branch."""
        return self.up_proj(F.gelu(self.down_proj(x)))


class AdaptedLayer(nn.Module):
    """Wraps a base layer and adds a BottleneckAdapter in parallel.

    forward(x) = base_layer(x) + adapter(x)
    """

    def __init__(self, base_layer: nn.Module, adapter: BottleneckAdapter) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.adapter = adapter

    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x) + self.adapter(x)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_adapter(module: nn.Module) -> bool:
    """Return True if *module* is one of the adapter types."""
    return isinstance(module, (BottleneckAdapter, ParallelAdapter))


def count_adapter_parameters(model: nn.Module) -> int:
    """Count the total number of parameters inside adapter submodules."""
    counted: set[int] = set()
    total = 0
    for module in model.modules():
        if _is_adapter(module):
            for p in module.parameters():
                if id(p) not in counted:
                    counted.add(id(p))
                    total += p.numel()
    return total


def freeze_base_parameters(model: nn.Module) -> None:
    """Freeze all parameters that are NOT inside an adapter submodule."""
    # Collect parameter ids belonging to adapters
    adapter_param_ids: set[int] = set()
    for module in model.modules():
        if _is_adapter(module):
            for p in module.parameters():
                adapter_param_ids.add(id(p))

    for p in model.parameters():
        if id(p) not in adapter_param_ids:
            p.requires_grad = False


# ---------------------------------------------------------------------------
# AdapterModel
# ---------------------------------------------------------------------------

class AdapterModel(nn.Module):
    """Wraps a base model, inserting BottleneckAdapters after every nn.Linear.

    Only adapter parameters are trainable; all base parameters are frozen.
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: AdapterConfig,
        n_layers: int,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_layers = n_layers

        # Replace every nn.Linear in base_model with an AdaptedLayer.
        # We iterate over named children recursively using _replace_linears.
        self._replace_linears(base_model, config)
        self.base_model = base_model

        # Freeze everything except adapters
        freeze_base_parameters(self)

    @staticmethod
    def _replace_linears(module: nn.Module, config: AdapterConfig) -> None:
        """Recursively replace nn.Linear children with AdaptedLayer."""
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                adapter = BottleneckAdapter(
                    d_model=child.out_features,
                    bottleneck_dim=config.bottleneck_dim,
                    dropout=config.adapter_dropout,
                    init_scale=config.init_scale,
                )
                setattr(module, name, AdaptedLayer(child, adapter))
            else:
                AdapterModel._replace_linears(child, config)

    def forward(self, x: Tensor) -> Tensor:
        return self.base_model(x)

    def get_adapter_state_dict(self) -> Dict[str, Tensor]:
        """Return a state-dict containing only adapter parameters."""
        # Collect parameter ids that belong to any adapter submodule
        adapter_param_ids: set[int] = set()
        for module in self.modules():
            if _is_adapter(module):
                for p in module.parameters():
                    adapter_param_ids.add(id(p))

        adapter_sd: Dict[str, Tensor] = {}
        for name, param in self.named_parameters():
            if id(param) in adapter_param_ids:
                adapter_sd[name] = param
        return adapter_sd


# ---------------------------------------------------------------------------
# Efficiency metric
# ---------------------------------------------------------------------------

def compute_adapter_efficiency(base_params: int, adapter_params: int) -> float:
    """Fraction of total parameters that are in the adapter.

    Returns adapter_params / (base_params + adapter_params).
    """
    total = base_params + adapter_params
    if total == 0:
        return 0.0
    return adapter_params / total
