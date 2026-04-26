"""Attention backend detection and selection utilities."""

import contextlib
import time
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class AttentionBackend(Enum):
    FLASH = "flash"  # Flash Attention 2 (CUDA only)
    EFFICIENT = "efficient"  # xFormers memory-efficient (CUDA)
    MATH = "math"  # Standard math fallback (CPU/CUDA)
    AUTO = "auto"  # Let PyTorch choose


@dataclass
class BackendInfo:
    name: str
    available: bool
    requires_cuda: bool
    notes: str


def detect_available_backends() -> dict[AttentionBackend, BackendInfo]:
    """Detect which SDPA backends are available on current hardware."""
    cuda_available = torch.cuda.is_available()
    backends = {
        AttentionBackend.MATH: BackendInfo(
            name="math",
            available=True,
            requires_cuda=False,
            notes="Standard scaled dot-product attention (always available)",
        ),
        AttentionBackend.FLASH: BackendInfo(
            name="flash_attention",
            available=cuda_available,
            requires_cuda=True,
            notes="Flash Attention 2 (CUDA only, not available on CPU)",
        ),
        AttentionBackend.EFFICIENT: BackendInfo(
            name="efficient_attention",
            available=cuda_available,
            requires_cuda=True,
            notes="Memory-efficient attention (CUDA only)",
        ),
    }
    return backends


def get_sdpa_backend_context(backend: AttentionBackend):
    """Return a context manager that forces a specific SDPA backend.

    Uses torch.backends.cuda.sdp_kernel() if available, else returns a no-op context.
    """
    if not torch.cuda.is_available():
        return contextlib.nullcontext()

    # torch.backends.cuda.sdp_kernel available in PyTorch 2.0+
    try:
        if backend == AttentionBackend.FLASH:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            )
        elif backend == AttentionBackend.EFFICIENT:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=False, enable_mem_efficient=True
            )
        elif backend == AttentionBackend.MATH:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            )
        else:
            return contextlib.nullcontext()
    except AttributeError:
        return contextlib.nullcontext()


def run_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Run scaled dot-product attention with automatic backend selection.

    Args:
        q: (B, H, S, D)
        k: (B, H, S_kv, D)
        v: (B, H, S_kv, D)
        mask: optional additive attention mask
        is_causal: apply causal mask
        dropout_p: dropout probability

    Returns:
        (B, H, S, D) attention output
    """
    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=is_causal
    )


@dataclass
class AttentionBenchmarkResult:
    backend: str
    time_ms: float
    output_shape: tuple
    max_diff_from_math: float  # max absolute difference vs math backend


def benchmark_attention(
    batch_size: int = 2,
    n_heads: int = 4,
    seq_len: int = 128,
    head_dim: int = 64,
    n_warmup: int = 3,
    n_trials: int = 10,
    device: str = "cpu",
) -> list[AttentionBenchmarkResult]:
    """Benchmark available attention backends and compare outputs.

    Returns list of benchmark results for each available backend.
    """
    q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)

    # Get reference (math) output
    with torch.no_grad():
        ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    results = []

    # Only benchmark MATH backend on CPU (others need CUDA)
    backends_to_test = ["math"]

    for backend_name in backends_to_test:
        for _ in range(n_warmup):
            F.scaled_dot_product_attention(q, k, v, is_causal=True)

        times = []
        for _ in range(n_trials):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        max_diff = (out - ref).abs().max().item()
        results.append(
            AttentionBenchmarkResult(
                backend=backend_name,
                time_ms=sum(times) / len(times),
                output_shape=tuple(out.shape),
                max_diff_from_math=max_diff,
            )
        )

    return results
