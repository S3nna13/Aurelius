from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class MemorySnapshot:
    allocated_mb: float
    reserved_mb: float
    peak_mb: float
    device: str


class MemoryProfiler:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def _is_cuda(self) -> bool:
        return self.device.startswith("cuda")

    def snapshot(self) -> MemorySnapshot:
        if self._is_cuda():
            allocated_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2)
            reserved_mb = torch.cuda.memory_reserved(self.device) / (1024 ** 2)
            peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
        else:
            allocated_mb = 0.0
            reserved_mb = 0.0
            peak_mb = 0.0
        return MemorySnapshot(
            allocated_mb=allocated_mb,
            reserved_mb=reserved_mb,
            peak_mb=peak_mb,
            device=self.device,
        )

    def reset_peak(self) -> None:
        if self._is_cuda():
            torch.cuda.reset_peak_memory_stats(self.device)

    def estimate_activation_mb(
        self, batch: int, seq_len: int, d_model: int, n_layers: int
    ) -> float:
        return float((batch * seq_len * d_model * n_layers * 4) / (1024 ** 2))

    def oom_guard(
        self,
        model: nn.Module,
        sample_input: torch.Tensor,
        safety_factor: float = 1.5,
    ) -> bool:
        if not self._is_cuda():
            return True
        batch = sample_input.shape[0]
        seq_len = sample_input.shape[1] if sample_input.dim() > 1 else 1
        d_model = sample_input.shape[-1] if sample_input.dim() > 1 else sample_input.shape[-1]
        n_layers = sum(1 for _ in model.parameters())
        estimated = self.estimate_activation_mb(batch, seq_len, d_model, n_layers)
        free_bytes = torch.cuda.get_device_properties(self.device).total_memory - torch.cuda.memory_allocated(self.device)
        free_mb = free_bytes / (1024 ** 2)
        return estimated * safety_factor <= free_mb


MEMORY_PROFILER_REGISTRY: dict[str, type[MemoryProfiler]] = {"default": MemoryProfiler}
