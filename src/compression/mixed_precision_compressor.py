from __future__ import annotations
import torch
import torch.nn as nn

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}

def compress_tensor(t: torch.Tensor, dtype: str = "fp16") -> torch.Tensor:
    return t.to(DTYPE_MAP.get(dtype, torch.float32))

class MixedPrecisionCompressor:
    def __init__(self, skip_embedding: bool = False) -> None:
        self.skip_embedding = skip_embedding
    def compress(self, model: nn.Module, dtype: str = "fp16") -> None:
        target = DTYPE_MAP.get(dtype, torch.float16)
        skip_types = (nn.Embedding, nn.EmbeddingBag) if self.skip_embedding else ()
        for module in model.modules():
            if isinstance(module, skip_types):
                continue
            for param in module.parameters(recurse=False):
                if param.ndim >= 2:
                    param.data = param.data.to(target)
