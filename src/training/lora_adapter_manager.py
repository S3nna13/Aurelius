from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    merge_weights: bool = False


class LoRALayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout_layer = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scaling * self.lora_B(self.lora_A(self.dropout_layer(x)))

    def merge(self) -> torch.Tensor:
        return self.lora_B.weight @ self.lora_A.weight * self.scaling


class LoRAAdapterManager:
    """Manage LoRA adapters: add, save, load, merge, switch."""

    def __init__(self, config: LoRAConfig | None = None) -> None:
        self.config = config or LoRAConfig()
        self._adapters: dict[str, dict[str, LoRALayer]] = {}

    def create_adapter(self, name: str, module: nn.Module) -> dict[str, LoRALayer]:
        layers: dict[str, LoRALayer] = {}
        for mod_name, mod in module.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            if not any(t in mod_name for t in self.config.target_modules):
                continue
            layers[mod_name] = LoRALayer(
                in_features=mod.in_features,
                out_features=mod.out_features,
                rank=self.config.rank,
                alpha=self.config.alpha,
                dropout=self.config.dropout,
            )
        self._adapters[name] = layers
        return layers

    def register_adapter(self, name: str, layers: dict[str, LoRALayer]) -> None:
        self._adapters[name] = layers

    def get_adapter(self, name: str) -> dict[str, LoRALayer] | None:
        return self._adapters.get(name)

    def list_adapters(self) -> list[str]:
        return list(self._adapters.keys())

    def remove_adapter(self, name: str) -> bool:
        if name in self._adapters:
            del self._adapters[name]
            return True
        return False

    def adapter_params(self, name: str) -> int:
        adapter = self._adapters.get(name)
        if adapter is None:
            return 0
        return sum(p.numel() for layer in adapter.values() for p in layer.parameters())
