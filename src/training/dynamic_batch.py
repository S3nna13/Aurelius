"""Dynamic batch size finder with gradient accumulation scaling.

Automatically finds the max batch size that fits in memory by running
forward+backward passes with increasing sizes. Computes gradient
accumulation steps to match target global batch tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DynamicBatchConfig:
    target_global_batch_tokens: int = 524_288  # 512K tokens (standard LLM pretraining)
    seq_len: int = 2048
    min_batch_size: int = 1
    max_batch_size: int = 128
    safety_factor: float = 0.9  # use only 90% of max to avoid OOM during training
    device: str = "cpu"


@dataclass
class BatchScaleResult:
    """Result of dynamic batch size search."""

    max_batch_size: int  # largest batch that fit
    safe_batch_size: int  # max * safety_factor (rounded down)
    grad_accum_steps: int  # steps to reach target_global_batch_tokens
    actual_global_batch_tokens: int  # safe_batch_size * seq_len * grad_accum_steps
    utilization: float  # actual / target (1.0 = perfect)


def try_batch_size(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    device: str,
) -> bool:
    """Try a single forward+backward pass with given batch_size.

    Returns True if successful, False if OOM or other error.
    Clears gradients and cache after each attempt.
    """
    try:
        # Create random integer input tokens
        x = torch.randint(0, 256, (batch_size, seq_len), device=device)
        # Forward pass — model returns logits or (logits, ...) depending on architecture
        out = model(x)
        # Handle tuple output (e.g. (logits, kv_cache))
        if isinstance(out, (tuple, list)):
            out = out[0]
        # Reduce to scalar and backward
        loss = out.float().sum()
        loss.backward()
        return True
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        return False
    finally:
        # Always clear gradients and CUDA cache
        model.zero_grad(set_to_none=True)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def find_max_batch_size(
    model: nn.Module,
    cfg: DynamicBatchConfig,
) -> int:
    """Binary search for maximum batch size that fits in memory.

    Uses try_batch_size() as the probe. Returns the largest successful batch_size.
    """
    lo = cfg.min_batch_size
    hi = cfg.max_batch_size

    # If even the minimum doesn't fit, raise immediately
    if not try_batch_size(model, lo, cfg.seq_len, cfg.device):
        raise RuntimeError(f"Minimum batch size {lo} does not fit in memory.")

    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_batch_size(model, mid, cfg.seq_len, cfg.device):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def compute_grad_accum_steps(
    safe_batch_size: int,
    seq_len: int,
    target_global_batch_tokens: int,
) -> int:
    """Compute gradient accumulation steps.

    grad_accum = ceil(target / (safe_batch_size * seq_len))
    Minimum 1.
    """
    tokens_per_step = safe_batch_size * seq_len
    return max(1, math.ceil(target_global_batch_tokens / tokens_per_step))


def scale_batch(
    model: nn.Module,
    cfg: DynamicBatchConfig | None = None,
) -> BatchScaleResult:
    """Find max batch size and compute grad accum steps.

    Returns BatchScaleResult.
    """
    if cfg is None:
        cfg = DynamicBatchConfig()

    max_batch_size = find_max_batch_size(model, cfg)
    safe_batch_size = max(1, int(max_batch_size * cfg.safety_factor))

    grad_accum_steps = compute_grad_accum_steps(
        safe_batch_size, cfg.seq_len, cfg.target_global_batch_tokens
    )

    actual_global_batch_tokens = safe_batch_size * cfg.seq_len * grad_accum_steps
    utilization = actual_global_batch_tokens / cfg.target_global_batch_tokens

    return BatchScaleResult(
        max_batch_size=max_batch_size,
        safe_batch_size=safe_batch_size,
        grad_accum_steps=grad_accum_steps,
        actual_global_batch_tokens=actual_global_batch_tokens,
        utilization=utilization,
    )
