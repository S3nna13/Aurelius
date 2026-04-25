from __future__ import annotations

import torch


class Int8DynamicQuantizer:
    def __init__(self, symmetric: bool = False, per_channel: bool = False) -> None:
        self.symmetric = symmetric
        self.per_channel = per_channel

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qmin, qmax = -128, 127
        if self.per_channel and x.dim() >= 2:
            row_norms = x.abs().max(dim=1, keepdim=True).values if self.symmetric else x.max(dim=1, keepdim=True).values
            min_vals = x.min(dim=1, keepdim=True).values
            if self.symmetric:
                scales = row_norms.squeeze(1) / qmax
                zero_points = torch.zeros(x.shape[0], dtype=torch.int8)
            else:
                ranges = (row_norms.squeeze(1) - min_vals.squeeze(1)).clamp(min=1e-8)
                scales = ranges / (qmax - qmin)
                zero_points = (qmin - min_vals.squeeze(1) / scales).round().clamp(qmin, qmax).to(torch.int8)
            x_reshaped = x.reshape(x.shape[0], -1)
            q = (x_reshaped / scales.unsqueeze(1).clamp(min=1e-8) + zero_points.unsqueeze(1).to(x.dtype)).round().clamp(qmin, qmax).to(torch.int8)
            return q.reshape(x.shape), scales, zero_points
        else:
            xmax = x.abs().max() if self.symmetric else x.max()
            xmin = x.min()
            if self.symmetric:
                if xmax == 0:
                    return torch.zeros_like(x, dtype=torch.int8), torch.tensor(1.0), torch.tensor(0, dtype=torch.int8)
                scale = (xmax / qmax).item()
                zp = torch.tensor(0, dtype=torch.int8)
            else:
                if xmax == xmin:
                    val = xmax.item()
                    if val == 0:
                        return torch.zeros_like(x, dtype=torch.int8), torch.tensor(1.0), torch.tensor(0, dtype=torch.int8)
                    scale_val = abs(val) / qmax
                    q_val = qmax if val >= 0 else qmin
                    return torch.full_like(x, q_val, dtype=torch.int8), torch.tensor(scale_val), torch.tensor(0, dtype=torch.int8)
                scale = ((xmax - xmin) / (qmax - qmin)).item()
                zp = round(qmin - xmin.item() / scale)
                zp = max(qmin, min(qmax, zp))
                zp = torch.tensor(zp, dtype=torch.int8)
            scale_t = torch.tensor(scale)
            q = (x / max(scale, 1e-8) + zp.float()).round().clamp(qmin, qmax).to(torch.int8)
            return q, scale_t.view(1), zp.view(1) if zp.dim() == 0 else zp

    def dequantize(self, q: torch.Tensor, scale: torch.Tensor, zp: torch.Tensor, shape: tuple[int, ...] | torch.Size) -> torch.Tensor:
        q_reshaped = q.reshape(-1, q.shape[-1]) if q.dim() >= 2 and self.per_channel else q
        s = scale.unsqueeze(1) if self.per_channel and scale.dim() == 1 else scale
        z = zp.unsqueeze(1) if self.per_channel and zp.dim() == 1 else zp
        x_hat = (q_reshaped.float() - z.float()) * s.float()
        return x_hat.reshape(shape)


INT8_QUANTIZER = Int8DynamicQuantizer()
