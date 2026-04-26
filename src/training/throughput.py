import time
from dataclasses import dataclass

import torch


@dataclass
class ProfileResult:
    step_time_ms: float  # wall-clock time for the step in milliseconds
    tokens_per_sec: float  # tokens processed per second
    peak_memory_mb: float  # peak GPU/CPU memory in MB (0 if no CUDA)
    n_tokens: int  # total tokens in this step
    flops_estimate: float  # estimated FLOPs for this step
    mfu: float  # model FLOP utilization (0-1, requires hardware_flops_per_sec)

    def summary(self) -> str:
        lines = [
            f"Step time:    {self.step_time_ms:.1f} ms",
            f"Throughput:   {self.tokens_per_sec:.0f} tokens/sec",
            f"Peak memory:  {self.peak_memory_mb:.1f} MB",
            f"FLOPs (est):  {self.flops_estimate:.2e}",
            f"MFU:          {self.mfu:.1%}"
            if self.mfu > 0
            else "MFU:          N/A (no hardware spec)",
        ]
        return "\n".join(lines)


class ThroughputProfiler:
    """Profiles a training step's throughput.

    Usage:
        profiler = ThroughputProfiler(
            model_params=1_395_000_000,
            hardware_flops_per_sec=312e12,  # A100 BF16 peak
        )

        with profiler.profile(n_tokens=batch_size * seq_len) as ctx:
            loss = model(input_ids, labels=labels)[0]
            loss.backward()
            optimizer.step()

        result = ctx.result
        print(result.summary())

    Or step-by-step:
        profiler.start(n_tokens=1024)
        # ... training step ...
        result = profiler.stop()
    """

    def __init__(
        self,
        model_params: int = 0,
        hardware_flops_per_sec: float = 0.0,
    ):
        self.model_params = model_params
        self.hardware_flops_per_sec = hardware_flops_per_sec
        self._start_time: float | None = None
        self._n_tokens: int = 0
        self._results: list[ProfileResult] = []

    def start(self, n_tokens: int) -> None:
        """Begin profiling a step."""
        self._n_tokens = n_tokens
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._start_time = time.perf_counter()

    def stop(self) -> ProfileResult:
        """End profiling and return result."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - self._start_time
        step_ms = elapsed * 1000
        tps = self._n_tokens / elapsed

        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated()
            peak_mb = peak_bytes / (1024**2)
        else:
            peak_mb = 0.0

        # FLOPs estimate: 6 * params * n_tokens (standard transformer estimate)
        # 6 = 2 (fwd matmul) * 3 (bwd ≈ 2x fwd, total ~6x)
        flops = 6.0 * self.model_params * self._n_tokens

        mfu = 0.0
        if self.hardware_flops_per_sec > 0:
            achieved_flops = flops / elapsed
            mfu = achieved_flops / self.hardware_flops_per_sec

        result = ProfileResult(
            step_time_ms=step_ms,
            tokens_per_sec=tps,
            peak_memory_mb=peak_mb,
            n_tokens=self._n_tokens,
            flops_estimate=flops,
            mfu=mfu,
        )
        self._results.append(result)
        return result

    def profile(self, n_tokens: int) -> "_ProfileContext":
        """Context manager for profiling."""
        return _ProfileContext(self, n_tokens)

    @property
    def results(self) -> list[ProfileResult]:
        return self._results

    def average_result(self) -> ProfileResult | None:
        """Return average over all recorded results."""
        if not self._results:
            return None
        return ProfileResult(
            step_time_ms=sum(r.step_time_ms for r in self._results) / len(self._results),
            tokens_per_sec=sum(r.tokens_per_sec for r in self._results) / len(self._results),
            peak_memory_mb=max(r.peak_memory_mb for r in self._results),
            n_tokens=self._results[-1].n_tokens,
            flops_estimate=sum(r.flops_estimate for r in self._results) / len(self._results),
            mfu=sum(r.mfu for r in self._results) / len(self._results),
        )


class _ProfileContext:
    def __init__(self, profiler: ThroughputProfiler, n_tokens: int):
        self.profiler = profiler
        self.n_tokens = n_tokens
        self.result: ProfileResult | None = None

    def __enter__(self):
        self.profiler.start(self.n_tokens)
        return self

    def __exit__(self, *args):
        self.result = self.profiler.stop()
