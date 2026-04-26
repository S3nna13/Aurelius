from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FSDPConfig:
    sharding_strategy: str = "full_shard"
    cpu_offload: bool = False
    mixed_precision_dtype: str = "bfloat16"
    min_num_params: int = 100_000


class FSDPWrapper:
    def __init__(self, config: FSDPConfig | None = None) -> None:
        self.config = config or FSDPConfig()

    def wrap(self, model: nn.Module) -> nn.Module:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            from torch.distributed.fsdp import FullyShardedDataParallel

            return FullyShardedDataParallel(model)
        return model

    def get_mixed_precision_policy(self):
        try:
            from torch.distributed.fsdp import MixedPrecision

            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            dtype = dtype_map.get(self.config.mixed_precision_dtype, torch.bfloat16)
            return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
        except Exception:
            return None

    def count_wrapped_modules(self, model: nn.Module) -> int:
        count = 0
        for module in model.modules():
            if module is model:
                continue
            n_params = sum(p.numel() for p in module.parameters(recurse=False))
            if n_params > self.config.min_num_params:
                count += 1
        return count


FSDP_REGISTRY: dict[str, type[FSDPWrapper]] = {"default": FSDPWrapper}
