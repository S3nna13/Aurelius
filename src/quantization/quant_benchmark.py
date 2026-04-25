"""Quantization method benchmarking harness for Aurelius.

Runs the GPTQ and AWQ quantizers on a common weight tensor, records MSE,
compression ratio and wall-clock quantization time, and produces a compact
ASCII report suitable for inclusion in experiment logs.

Torch is imported lazily inside the methods that actually need it so the
module can be imported in torch-free environments.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .awq_quantizer import AWQConfig, AWQQuantizer
from .gptq_quantizer import GPTQConfig, GPTQQuantizer

if TYPE_CHECKING:  # pragma: no cover
    import torch


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkResult:
    method: str
    bits: int
    group_size: int
    mse: float
    compression_ratio: float
    quantize_time_ms: float


# ---------------------------------------------------------------------------
# Benchmark driver
# ---------------------------------------------------------------------------

class QuantBenchmark:
    """Run and compare quantization methods on identical weight tensors."""

    @staticmethod
    def _compression_ratio(bits: int, original_bits: int = 32) -> float:
        return original_bits / bits

    # ------------------------------------------------------------------
    # Individual runners
    # ------------------------------------------------------------------

    def run_gptq(
        self,
        weight: "torch.Tensor",
        config: GPTQConfig | None = None,
    ) -> BenchmarkResult:
        cfg = config if config is not None else GPTQConfig()
        quantizer = GPTQQuantizer(cfg)

        t0 = time.perf_counter()
        layer = quantizer.quantize_weight(weight)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        mse = quantizer.quantize_error(weight, layer)
        return BenchmarkResult(
            method="gptq",
            bits=cfg.bits,
            group_size=cfg.group_size,
            mse=mse,
            compression_ratio=self._compression_ratio(cfg.bits),
            quantize_time_ms=elapsed_ms,
        )

    def run_awq(
        self,
        weight: "torch.Tensor",
        activation_sample: "torch.Tensor",
        config: AWQConfig | None = None,
    ) -> BenchmarkResult:
        cfg = config if config is not None else AWQConfig()
        quantizer = AWQQuantizer(cfg)

        t0 = time.perf_counter()
        stats = quantizer.collect_activation_stats(activation_sample)
        scale_factor = quantizer.compute_scale_factor(weight, stats)
        w_int, scale, zero = quantizer.quantize_with_scales(weight, scale_factor)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        mse = quantizer.reconstruction_error(weight, w_int, scale, zero)
        return BenchmarkResult(
            method="awq",
            bits=cfg.bits,
            group_size=cfg.group_size,
            mse=mse,
            compression_ratio=self._compression_ratio(cfg.bits),
            quantize_time_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def compare(self, results: list[BenchmarkResult]) -> dict:
        if not results:
            return {"best_mse": None, "best_compression": None, "summary": []}
        best_mse = min(results, key=lambda r: r.mse)
        best_comp = max(results, key=lambda r: r.compression_ratio)
        summary = sorted(results, key=lambda r: r.mse)
        return {
            "best_mse": best_mse,
            "best_compression": best_comp,
            "summary": summary,
        }

    def report(self, results: list[BenchmarkResult]) -> str:
        header = f"{'method':<8} {'bits':>4} {'group':>6} {'mse':>12} {'compression':>12} {'time_ms':>10}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in results:
            lines.append(
                f"{r.method:<8} {r.bits:>4d} {r.group_size:>6d} "
                f"{r.mse:>12.6f} {r.compression_ratio:>12.3f} "
                f"{r.quantize_time_ms:>10.3f}"
            )
        if not results:
            lines.append("(no results)")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
QUANT_BENCHMARK_REGISTRY: dict[str, type] = {"default": QuantBenchmark}
