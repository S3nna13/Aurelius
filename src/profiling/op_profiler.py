from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpRecord:
    name: str
    elapsed_ms: float
    call_count: int = 1


class OpProfiler:
    def __init__(self):
        self._data: dict[str, dict] = {}

    @contextmanager
    def profile(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if name in self._data:
                self._data[name]["elapsed_ms"] += elapsed_ms
                self._data[name]["call_count"] += 1
            else:
                self._data[name] = {"elapsed_ms": elapsed_ms, "call_count": 1}

    def records(self) -> list[OpRecord]:
        return sorted(
            [
                OpRecord(
                    name=name,
                    elapsed_ms=v["elapsed_ms"],
                    call_count=v["call_count"],
                )
                for name, v in self._data.items()
            ],
            key=lambda r: r.elapsed_ms,
            reverse=True,
        )

    def total_ms(self) -> float:
        return sum(v["elapsed_ms"] for v in self._data.values())

    def top_k(self, k: int) -> list[OpRecord]:
        return self.records()[:k]

    def reset(self):
        self._data.clear()

    def report(self) -> str:
        recs = self.records()
        if not recs:
            return "No records."
        header = f"{'name':<30} {'calls':>6} {'total_ms':>10} {'avg_ms':>10}"
        sep = "-" * len(header)
        lines = [header, sep]
        for r in recs:
            avg = r.elapsed_ms / r.call_count
            lines.append(f"{r.name:<30} {r.call_count:>6} {r.elapsed_ms:>10.3f} {avg:>10.3f}")
        return "\n".join(lines)


OP_PROFILER_REGISTRY: dict[str, type[OpProfiler]] = {"default": OpProfiler}
