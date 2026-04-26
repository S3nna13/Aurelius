from __future__ import annotations

import torch.nn as nn


def flops_for_linear(layer: nn.Linear) -> int:
    in_f, out_f = layer.in_features, layer.out_features
    if in_f == 0 or out_f == 0:
        return 0
    flops = 2 * in_f * out_f
    if layer.bias is not None:
        flops += out_f
    return flops


def flops_for_attention(seq_len: int, d_model: int, n_heads: int) -> int:
    head_dim = d_model // n_heads
    qkv = 3 * 2 * d_model * d_model
    if seq_len <= 1:
        return qkv
    attn = 2 * seq_len * seq_len * head_dim * n_heads
    out_proj = 2 * d_model * d_model
    return qkv + attn + out_proj


class FlopsProfiler:
    def profile(self, model: nn.Module, input_shape: tuple[int, ...]) -> int:
        total = 0
        for module in model.modules():
            if isinstance(module, nn.Linear):
                total += flops_for_linear(module)
        return total

    def profile_by_module(self, model: nn.Module, input_shape: tuple[int, ...]) -> dict[str, int]:
        results: dict[str, int] = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                results[name or module.__class__.__name__] = flops_for_linear(module)
        return results


FLOPS_PROFILER = FlopsProfiler()
