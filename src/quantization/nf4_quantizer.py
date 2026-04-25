from __future__ import annotations

import math

import torch


_NF4_LEVELS = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
])


def nf4_quantize(x: torch.Tensor) -> tuple[torch.Tensor, float]:
    absmax = x.abs().max().item()
    if absmax == 0:
        return torch.zeros(x.numel() // 2, dtype=torch.uint8), 0.0
    scaled = x.float() / absmax
    flat = scaled.view(-1)
    n = flat.numel()
    n_doubles = (n + 1) // 2
    # Map to nearest NF4 level
    expanded = flat.unsqueeze(1).expand(n, 16)
    dists = (expanded - _NF4_LEVELS.unsqueeze(0)).abs()
    indices = dists.argmin(dim=1)
    packed = torch.zeros(n_doubles, dtype=torch.uint8)
    for i in range(n_doubles):
        lo = indices[2 * i].item() if 2 * i < n else 0
        hi = indices[2 * i + 1].item() if 2 * i + 1 < n else 0
        packed[i] = (hi << 4) | lo
    return packed, absmax


def nf4_dequantize(q: torch.Tensor, scale: float, shape: torch.Size) -> torch.Tensor:
    n = 1
    for s in shape:
        n *= s
    if q.numel() * 2 < n:
        raise ValueError(f"shape {shape} has {n} elements but only {q.numel() * 2} available")
    indices = torch.zeros(n, dtype=torch.uint8)
    for i in range(q.numel()):
        byte_val = q[i].item()
        indices[2 * i] = byte_val & 0xF
        if 2 * i + 1 < n:
            indices[2 * i + 1] = (byte_val >> 4) & 0xF
    levels = _NF4_LEVELS.to(q.device)
    values = levels[indices.long()] * scale
    return values.reshape(shape)


class NF4Quantizer:
    def __init__(self) -> None:
        self._nf4_levels = _NF4_LEVELS

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        return nf4_quantize(x)

    def dequantize(self, q: torch.Tensor, scale: float, shape: torch.Size) -> torch.Tensor:
        return nf4_dequantize(q, scale, shape)


NF4_QUANTIZER = NF4Quantizer()
