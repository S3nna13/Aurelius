"""
Training Throughput Profiler
Measures and analyzes training efficiency: tokens/sec, flops/sec, memory usage,
step time breakdown, and bottleneck identification.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StepProfile:
    step: int
    forward_time_ms: float
    backward_time_ms: float
    optimizer_time_ms: float
    total_time_ms: float
    tokens_per_sec: float
    loss: float
    batch_size: int
    seq_len: int
    peak_memory_mb: float  # GPU peak memory, 0.0 on CPU


@dataclass
class ProfileSummary:
    n_steps: int
    mean_tokens_per_sec: float
    mean_step_time_ms: float
    mean_forward_ms: float
    mean_backward_ms: float
    mean_optimizer_ms: float
    peak_memory_mb: float
    forward_fraction: float    # forward_ms / total_ms
    backward_fraction: float   # backward_ms / total_ms
    bottleneck: str            # "forward" | "backward" | "optimizer" | "balanced"


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, sync_cuda: bool = False):
        self.sync_cuda = sync_cuda
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1000.0


def estimate_model_flops(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
) -> int:
    """
    Estimate FLOPs for one forward pass.
    For Linear layers: 2 * in_features * out_features per token.
    Returns total FLOPs (int).
    """
    total_flops = 0
    n_tokens = batch_size * seq_len
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # 2 * in_features * out_features per token (multiply-accumulate)
            total_flops += 2 * module.in_features * module.out_features * n_tokens
    return int(total_flops)


def estimate_model_params(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_model_memory_mb(model: nn.Module, dtype: torch.dtype = torch.float32) -> float:
    """Estimate model memory in MB based on parameter count and dtype."""
    dtype_to_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.float64: 8,
        torch.int8: 1,
        torch.int32: 4,
        torch.int64: 8,
    }
    bytes_per_param = dtype_to_bytes.get(dtype, 4)
    n_params = sum(p.numel() for p in model.parameters())
    return (n_params * bytes_per_param) / (1024 ** 2)


class MemoryTracker:
    """Track memory usage during training."""

    def __init__(self):
        self._peak_mb: float = 0.0

    def reset(self) -> None:
        self._peak_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self) -> Dict[str, float]:
        """Return {'allocated_mb': ..., 'reserved_mb': ..., 'peak_mb': ...}"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            allocated = 0.0
            reserved = 0.0
            peak = 0.0
        self._peak_mb = max(self._peak_mb, peak)
        return {
            "allocated_mb": allocated,
            "reserved_mb": reserved,
            "peak_mb": peak,
        }

    def get_peak_mb(self) -> float:
        self.snapshot()  # refresh peak
        return self._peak_mb


class ThroughputProfiler:
    """Profile training throughput step by step."""

    def __init__(
        self,
        model: nn.Module,
        warmup_steps: int = 3,
        sync_cuda: bool = False,
    ):
        self.model = model
        self.warmup_steps = warmup_steps
        self.sync_cuda = sync_cuda
        self._history: List[StepProfile] = []

    def profile_step(
        self,
        input_ids: torch.Tensor,     # (B, T)
        optimizer: torch.optim.Optimizer,
        step: int = 0,
    ) -> StepProfile:
        """
        Profile one training step:
        1. Time forward pass: compute loss from model(input_ids)
        2. Time backward pass: loss.backward()
        3. Time optimizer step: optimizer.step()
        Reset gradients before and after.
        """
        batch_size, seq_len = input_ids.shape
        n_tokens = batch_size * seq_len

        # Reset gradients before the step
        optimizer.zero_grad()

        # Reset CUDA memory stats for accurate peak measurement
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Time forward pass
        with Timer(sync_cuda=self.sync_cuda) as fwd_timer:
            output = self.model(input_ids)
            # Support models returning (loss, ...) tuple or just a loss tensor
            if isinstance(output, (tuple, list)):
                loss = output[0]
            else:
                loss = output

        forward_time_ms = fwd_timer.elapsed_ms

        # Time backward pass
        with Timer(sync_cuda=self.sync_cuda) as bwd_timer:
            loss.backward()

        backward_time_ms = bwd_timer.elapsed_ms

        # Time optimizer step
        with Timer(sync_cuda=self.sync_cuda) as opt_timer:
            optimizer.step()

        optimizer_time_ms = opt_timer.elapsed_ms

        # Reset gradients after step
        optimizer.zero_grad()

        total_time_ms = forward_time_ms + backward_time_ms + optimizer_time_ms
        tokens_per_sec = n_tokens / (total_time_ms / 1000.0) if total_time_ms > 0 else 0.0

        # Get peak memory
        if torch.cuda.is_available():
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_memory_mb = 0.0

        profile = StepProfile(
            step=step,
            forward_time_ms=forward_time_ms,
            backward_time_ms=backward_time_ms,
            optimizer_time_ms=optimizer_time_ms,
            total_time_ms=total_time_ms,
            tokens_per_sec=tokens_per_sec,
            loss=loss.item(),
            batch_size=batch_size,
            seq_len=seq_len,
            peak_memory_mb=peak_memory_mb,
        )
        self._history.append(profile)
        return profile

    def run(
        self,
        input_ids: torch.Tensor,    # (B, T) — same batch reused for profiling
        optimizer: torch.optim.Optimizer,
        n_steps: int = 10,
    ) -> ProfileSummary:
        """Profile n_steps and return summary (excluding warmup_steps)."""
        for i in range(n_steps):
            self.profile_step(input_ids, optimizer, step=i)

        # Exclude warmup steps from summary
        measured = self._history[-n_steps:]  # steps from this run
        measured = measured[self.warmup_steps:]

        if not measured:
            # Edge case: all steps are warmup
            measured = self._history[-n_steps:]

        mean_tokens_per_sec = sum(s.tokens_per_sec for s in measured) / len(measured)
        mean_step_time_ms = sum(s.total_time_ms for s in measured) / len(measured)
        mean_forward_ms = sum(s.forward_time_ms for s in measured) / len(measured)
        mean_backward_ms = sum(s.backward_time_ms for s in measured) / len(measured)
        mean_optimizer_ms = sum(s.optimizer_time_ms for s in measured) / len(measured)
        peak_memory_mb = max(s.peak_memory_mb for s in measured)

        if mean_step_time_ms > 0:
            forward_fraction = mean_forward_ms / mean_step_time_ms
            backward_fraction = mean_backward_ms / mean_step_time_ms
        else:
            forward_fraction = 0.0
            backward_fraction = 0.0

        summary = ProfileSummary(
            n_steps=len(measured),
            mean_tokens_per_sec=mean_tokens_per_sec,
            mean_step_time_ms=mean_step_time_ms,
            mean_forward_ms=mean_forward_ms,
            mean_backward_ms=mean_backward_ms,
            mean_optimizer_ms=mean_optimizer_ms,
            peak_memory_mb=peak_memory_mb,
            forward_fraction=forward_fraction,
            backward_fraction=backward_fraction,
            bottleneck=self.identify_bottleneck(
                ProfileSummary(
                    n_steps=len(measured),
                    mean_tokens_per_sec=mean_tokens_per_sec,
                    mean_step_time_ms=mean_step_time_ms,
                    mean_forward_ms=mean_forward_ms,
                    mean_backward_ms=mean_backward_ms,
                    mean_optimizer_ms=mean_optimizer_ms,
                    peak_memory_mb=peak_memory_mb,
                    forward_fraction=forward_fraction,
                    backward_fraction=backward_fraction,
                    bottleneck="",  # placeholder
                )
            ),
        )
        return summary

    def get_history(self) -> List[StepProfile]:
        """Return all profiled step records."""
        return list(self._history)

    def identify_bottleneck(self, summary: ProfileSummary) -> str:
        """Return 'forward', 'backward', 'optimizer', or 'balanced'."""
        fwd = summary.mean_forward_ms
        bwd = summary.mean_backward_ms
        opt = summary.mean_optimizer_ms
        total = fwd + bwd + opt

        if total == 0:
            return "balanced"

        fwd_frac = fwd / total
        bwd_frac = bwd / total
        opt_frac = opt / total

        # Consider "balanced" if no single phase dominates by > 20% margin
        THRESHOLD = 0.5  # dominates if > 50% of total
        if fwd_frac > THRESHOLD:
            return "forward"
        elif bwd_frac > THRESHOLD:
            return "backward"
        elif opt_frac > THRESHOLD:
            return "optimizer"
        else:
            return "balanced"


def compute_mfu(
    model: nn.Module,
    tokens_per_sec: float,
    batch_size: int,
    seq_len: int,
    theoretical_flops_per_sec: float = 312e12,  # A100 BF16 theoretical
) -> float:
    """
    Model FLOPs Utilization (MFU).
    mfu = actual_flops_per_sec / theoretical_flops_per_sec
    actual_flops_per_sec = model_flops * tokens_per_sec / seq_len
    Returns float (typically 0.0 to 1.0, but can exceed 1.0 for estimates).
    """
    model_flops = estimate_model_flops(model, batch_size=1, seq_len=seq_len)
    # actual_flops_per_sec = flops_per_token * tokens_per_sec
    flops_per_token = model_flops / seq_len if seq_len > 0 else 0
    actual_flops_per_sec = flops_per_token * tokens_per_sec
    if theoretical_flops_per_sec <= 0:
        return 0.0
    return actual_flops_per_sec / theoretical_flops_per_sec
