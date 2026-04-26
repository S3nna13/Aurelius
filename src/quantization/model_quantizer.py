"""Model quantization utilities for Aurelius."""
from __future__ import annotations

import torch

MODEL_QUANTIZER_REGISTRY: dict[str, type] = {}


class ModelQuantizer:
    """Per-tensor uniform quantizer supporting symmetric and asymmetric schemes."""

    def __init__(self, bits: int = 8, scheme: str = "symmetric") -> None:
        if bits not in {4, 8, 16}:
            raise ValueError(f"bits must be in {{4, 8, 16}}, got {bits}")
        if scheme not in {"symmetric", "asymmetric"}:
            raise ValueError(f"scheme must be 'symmetric' or 'asymmetric', got {scheme}")
        self.bits = bits
        self.scheme = scheme
        self._qmax = 2**bits - 1

    def _get_qmin_qmax(self) -> tuple[int, int]:
        if self.scheme == "symmetric":
            half = 2 ** (self.bits - 1)
            return -half, half - 1
        return 0, self._qmax

    def _get_dtype(self) -> torch.dtype:
        if self.bits == 4:
            return torch.int8
        if self.bits == 8:
            return torch.uint8 if self.scheme == "asymmetric" else torch.int8
        # bits == 16
        return torch.int32 if self.scheme == "asymmetric" else torch.int16

    def quantize_tensor(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize a float tensor to integer values.

        Returns:
            A tuple of ``(quantized_int_tensor, scale, zero_point)``.
        """
        if self.scheme == "symmetric":
            max_abs = t.abs().max()
            min_val = -max_abs
            max_val = max_abs
        else:
            min_val = t.min()
            max_val = t.max()

        dtype = self._get_dtype()
        qmin, qmax = self._get_qmin_qmax()

        if max_val.item() == min_val.item():
            if max_val.item() == 0:
                scale = torch.tensor(1.0, dtype=torch.float32)
                zero_point = torch.tensor(0, dtype=dtype)
                q = torch.zeros_like(t, dtype=dtype)
            else:
                # Asymmetric constant non-zero tensor
                val = max_val.item()
                scale = torch.tensor(abs(val) / qmax, dtype=torch.float32)
                zero_point = torch.tensor(0 if val > 0 else qmax, dtype=dtype)
                q_val = qmax if val > 0 else qmin
                q = torch.full_like(t, q_val, dtype=dtype)
            return q, scale, zero_point

        scale = (max_val - min_val) / self._qmax

        if self.scheme == "symmetric":
            zero_point = torch.tensor(0, dtype=dtype)
            q = (t / scale).round().clamp(qmin, qmax).to(dtype)
        else:
            zero_point = (-min_val / scale).round().clamp(qmin, qmax).to(dtype)
            q = (t / scale + zero_point.float()).round().clamp(qmin, qmax).to(dtype)

        return q, scale, zero_point

    def dequantize_tensor(
        self, q: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct a float tensor from quantized integer values."""
        return (q.float() - zero_point.float()) * scale.float()

    def quantize_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, tuple]:
        """Quantize all tensors in a state dict."""
        return {k: self.quantize_tensor(v) for k, v in state_dict.items()}

    def dequantize_state_dict(self, q_state_dict: dict[str, tuple]) -> dict[str, torch.Tensor]:
        """Dequantize a state dict produced by :meth:`quantize_state_dict`."""
        return {k: self.dequantize_tensor(*v) for k, v in q_state_dict.items()}


MODEL_QUANTIZER_REGISTRY["default"] = ModelQuantizer
