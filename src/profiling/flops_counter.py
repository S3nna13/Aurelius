from dataclasses import dataclass, field
from typing import Any
import torch.nn as nn


@dataclass
class FLOPsConfig:
    include_bias: bool = True
    count_activations: bool = False


@dataclass(frozen=True)
class ModuleFLOPs:
    module_name: str
    module_type: str
    flops: int
    params: int


class FLOPsCounter:
    def __init__(self, config: FLOPsConfig | None = None):
        self.config = config if config is not None else FLOPsConfig()

    def count_linear(
        self,
        in_features: int,
        out_features: int,
        batch_size: int = 1,
        seq_len: int = 1,
    ) -> int:
        flops = 2 * batch_size * seq_len * in_features * out_features
        if self.config.include_bias:
            flops += out_features
        return flops

    def count_attention(
        self,
        seq_len: int,
        d_model: int,
        n_heads: int,
        batch_size: int = 1,
    ) -> int:
        return 4 * 2 * batch_size * seq_len * d_model * d_model

    def count_module(self, module: nn.Module, input_shape: tuple) -> list[ModuleFLOPs]:
        results = []
        batch_size = input_shape[0] if len(input_shape) > 0 else 1
        seq_len = input_shape[1] if len(input_shape) > 1 else 1

        for name, mod in module.named_modules():
            if isinstance(mod, nn.Linear):
                flops = self.count_linear(
                    mod.in_features,
                    mod.out_features,
                    batch_size=batch_size,
                    seq_len=seq_len,
                )
                params = mod.weight.numel()
                if mod.bias is not None:
                    params += mod.bias.numel()
                results.append(
                    ModuleFLOPs(
                        module_name=name or "linear",
                        module_type="Linear",
                        flops=flops,
                        params=params,
                    )
                )
            elif isinstance(mod, nn.MultiheadAttention):
                d_model = mod.embed_dim
                n_heads = mod.num_heads
                flops = self.count_attention(
                    seq_len=seq_len,
                    d_model=d_model,
                    n_heads=n_heads,
                    batch_size=batch_size,
                )
                params = sum(p.numel() for p in mod.parameters())
                results.append(
                    ModuleFLOPs(
                        module_name=name or "attention",
                        module_type="MultiheadAttention",
                        flops=flops,
                        params=params,
                    )
                )

        return results

    def total_flops(self, module_flops: list[ModuleFLOPs]) -> int:
        return sum(mf.flops for mf in module_flops)

    def summary(self, module: nn.Module, input_shape: tuple) -> dict:
        module_flops = self.count_module(module, input_shape)
        return {
            "total_flops": self.total_flops(module_flops),
            "total_params": sum(mf.params for mf in module_flops),
            "by_layer": [
                {
                    "module_name": mf.module_name,
                    "module_type": mf.module_type,
                    "flops": mf.flops,
                    "params": mf.params,
                }
                for mf in module_flops
            ],
        }


FLOPS_REGISTRY: dict[str, type[FLOPsCounter]] = {"default": FLOPsCounter}
