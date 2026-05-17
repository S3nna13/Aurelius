"""
from __future__ import annotations
Pareto Frontier Router — v4 Enhancement
Dynamic multi-backend selection optimizing Quality/Cost/Latency tradeoff.

Based on: "Pareto-Lenient Consensus" (2025) & LLM routing frontier analysis.
Integrates with existing router.py to add Pareto-optimal backend selection.

Key idea:
  Frontiers are non-dominated backends. No single backend is best on all 3 metrics.
  Build empirical frontier from historical request data and select knee-point
  or constraint-based (budget/latency) point.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BackendConfig:
    """Static configuration for a backend."""

    name: str  # e.g. "local", "step", "claude"
    label: str  # display name
    cost_per_1k_tokens: float  # USD per 1K output tokens
    latency_per_token_ms: float  # estimated per-token latency
    quality: float  # estimated quality score [0-1]
    priority: int = 1  # lower = higher priority
    max_context: int = 32768  # max context tokens supported
    # Runtime estimates updated by ParetoRouter based on history
    empirical_multiplier_cost: float = 1.0
    empirical_multiplier_latency: float = 1.0
    empirical_quality_delta: float = 0.0  # quality adjustment from baseline


@dataclass
class Request:
    """Incoming request properties for routing decision."""

    prompt: str
    max_tokens: int = 512
    context_len: int = 0  # actual prompt length
    task_type: str = "chat"  # chat | code | reasoning | creative
    quality_requirement: float = 0.80  # min quality [0-1]
    latency_requirement_ms: int | None = None  # max allowed latency
    budget_per_1k_tokens: float | None = None  # max cost willing to pay


@dataclass
class RoutingDecision:
    backend: BackendConfig
    estimated_cost: float
    estimated_latency_ms: float
    estimated_quality: float
    reason: str


class ParetoFrontierRouter:
    """
    Enhanced router: selects backend from empirical Pareto frontier.

    Modes:
      simple        — round-robin / priority only (baseline)
      cost_aware    — cheapest that meets quality & latency
      pareto        — full frontier optimization (default)
    """

    def __init__(
        self,
        backends: list[BackendConfig],
        mode: str = "pareto",
        history_window: int = 1000,
        quality_threshold: float = 0.85,
        latency_target_p99: int = 5000,
    ):
        self.backends = {b.name: b for b in backends}
        self.mode = mode
        self.history_window = history_window
        self.quality_threshold = quality_threshold
        self.latency_target = latency_target_p99

        # Empirical measurements: (backend, cost, latency, quality, timestamp)
        self.history: deque = deque(maxlen=history_window)

        # Cached frontier (recomputed periodically)
        self._frontier: list[BackendConfig] = []
        self._frontier_last_update = 0
        self._frontier_update_interval = 100  # requests

    # ─── PUBLIC API ────────────────────────────────────────────────────────────

    def select(self, req: Request) -> RoutingDecision:
        """
        Select best backend for request.

        Args:
            req: Request with quality/latency/budget constraints

        Returns:
            RoutingDecision with selected backend and estimates
        """
        # Recompute frontier periodically
        self._maybe_update_frontier()

        # Score all backends
        candidates = []
        for backend in self.backends.values():
            estimate = self._estimate(backend, req)
            if estimate is None:
                continue

            # Apply constraints
            if estimate.estimated_quality < req.quality_requirement:
                continue
            if (
                req.latency_requirement_ms
                and estimate.estimated_latency_ms > req.latency_requirement_ms
            ):
                continue
            if req.budget_per_1k_tokens and estimate.estimated_cost > req.budget_per_1k_tokens:
                continue

            candidates.append(estimate)

        if not candidates:
            # No backend meets constraints — pick least-worst quality
            # but warn
            print(
                f"[WARN] No backend meets constraints for {req.task_type} task; "
                f"falling back to highest-quality available"
            )
            # Re-evaluate without quality constraint
            candidates = [
                self._estimate(b, req)
                for b in self.backends.values()
                if self._estimate(b, req) is not None
            ]
            if not candidates:
                raise RuntimeError("No backends available!")

        if self.mode == "simple":
            # Choose highest priority
            best = min(candidates, key=lambda c: self.backends[c.backend.name].priority)
        elif self.mode == "cost_aware":
            # Cheapest among those meeting thresholds
            best = min(candidates, key=lambda c: c.estimated_cost)
        else:  # pareto
            # Pick frontier point balancing all 3
            best = self._select_pareto_optimal(candidates, req)

        return best

    def record(self, backend_name: str, cost: float, latency_ms: float, quality: float):
        """Record actual outcome for empirical tuning."""
        self.history.append(
            {
                "backend": backend_name,
                "cost": cost,
                "latency_ms": latency_ms,
                "quality": quality,
                "ts": time.time(),
            }
        )

        # Trigger frontier recompute if enough new data
        if len(self.history) % self._frontier_update_interval == 0:
            self._compute_frontier()

    # ─── INTERNAL ───────────────────────────────────────────────────────────────

    def _estimate(self, backend: BackendConfig, req: Request) -> RoutingDecision | None:
        """
        Estimate cost/latency/quality for (backend, request).

        Uses baseline config + empirical multipliers from history.
        """
        # Context length impact
        ctx_factor = 1.0 + 0.1 * (req.context_len / 1000)  # +10% per 1k context

        # Cost = tokens × base_cost × empirical_multiplier
        cost = (
            req.max_tokens
            * backend.cost_per_1k_tokens
            / 1000
            * backend.empirical_multiplier_cost
            * ctx_factor
        )

        # Latency = tokens × per-token latency × empirical × length_scale
        # Longer contexts incur KV cache lookup overhead
        length_factor = 1.0 + 0.2 * (req.context_len / 4096)  # +20% per 4k context
        latency_ms = (
            req.max_tokens
            * backend.latency_per_token_ms
            * backend.empirical_multiplier_latency
            * length_factor
        )

        # Quality = base - degradation from long context + empirical delta
        quality = backend.quality + backend.empirical_quality_delta
        # Quality drop for very long contexts (model-dependent)
        if req.context_len > 32000:
            quality -= 0.05
        elif req.context_len > 128000:
            quality -= 0.15

        quality = max(0.0, min(1.0, quality))

        return RoutingDecision(
            backend=backend,
            estimated_cost=cost,
            estimated_latency_ms=latency_ms,
            estimated_quality=quality,
            reason="baseline",
        )

    def _select_pareto_optimal(
        self, candidates: list[RoutingDecision], req: Request
    ) -> RoutingDecision:
        """
        Choose point on Pareto frontier.

        Strategy: Find knee point (max quality improvement per unit cost).
        If latency constraint tight → prioritize latency.
        If budget constraint tight → prioritize cost.
        """
        # Sort by quality descending
        candidates.sort(key=lambda c: c.estimated_quality, reverse=True)

        # Compute cost/quality tradeoff ratios
        if len(candidates) == 1:
            return candidates[0]

        best = candidates[0]  # default: highest quality
        best_ratio = -1.0

        for i in range(1, len(candidates)):
            prev, curr = candidates[i - 1], candidates[i]
            quality_gain = prev.estimated_quality - curr.estimated_quality
            cost_saved = prev.estimated_cost - curr.estimated_cost
            if cost_saved > 0:
                ratio = quality_gain / cost_saved
                if ratio > best_ratio:
                    best_ratio = ratio
                    best = curr

        # Latency constraint enforcement
        if req.latency_requirement_ms:
            # Filter to those meeting latency
            latency_ok = [
                c for c in candidates if c.estimated_latency_ms <= req.latency_requirement_ms
            ]
            if latency_ok:
                # Choose highest quality among latency-ok
                best = max(latency_ok, key=lambda c: c.estimated_quality)

        return best

    def _maybe_update_frontier(self):
        """Recompute frontier every N new records."""
        if len(self.history) - self._frontier_last_update >= self._frontier_update_interval:
            self._compute_frontier()
            self._frontier_last_update = len(self.history)

    def _compute_frontier(self):
        """
        Compute empirical frontier from recent history.
        Backend B is dominated if ∃ A:
        cost_A ≤ cost_B AND latency_A ≤ latency_B AND quality_A ≥ quality_B,
        with at least one strict.
        """
        # Aggregate by backend: use median of recent samples
        stats = {}
        for record in self.history:
            b = record["backend"]
            if b not in stats:
                stats[b] = {"cost": [], "latency": [], "quality": []}
            stats[b]["cost"].append(record["cost"])
            stats[b]["latency"].append(record["latency_ms"])
            stats[b]["quality"].append(record["quality"])

        # Compute medians
        empirical = {}
        for b, vals in stats.items():
            if b not in self.backends:
                continue  # removed backend
            base = self.backends[b]
            empirical[b] = {
                "cost_mult": np.median(vals["cost"]) / max(1e-6, base.cost_per_1k_tokens / 1000),
                "latency_mult": np.median(vals["latency"]) / max(1e-6, base.latency_per_token_ms),
                "quality_delta": np.median(vals["quality"]) - base.quality,
            }

        # Update backend configs
        for bname, mults in empirical.items():
            self.backends[bname].empirical_multiplier_cost = mults["cost_mult"]
            self.backends[bname].empirical_multiplier_latency = mults["latency_mult"]
            self.backends[bname].empirical_quality_delta = mults["quality_delta"]

        # Build frontier (non-dominated set)
        self._frontier = self._calculate_frontier()

    def _calculate_frontier(self) -> list[BackendConfig]:
        """
        Return list of non-dominated backends sorted by cost ascending.

        A dominates B if:
          cost_A < cost_B AND latency_A ≤ latency_B AND quality_A ≥ quality_B
          OR
          cost_A ≤ cost_B AND latency_A < latency_B AND quality_A ≥ quality_B
          OR
          cost_A ≤ cost_B AND latency_A ≤ latency_B AND quality_A > quality_B
        """
        backends = list(self.backends.values())
        frontier = []

        for i, b in enumerate(backends):
            dominated = False
            for j, other in enumerate(backends):
                if i == j:
                    continue
                # Estimate metrics using current empirical configs
                # (simplified: use dummy request with average context)
                dummy_req = Request(prompt="", max_tokens=512, context_len=4096)
                b_est = self._estimate(b, dummy_req)
                o_est = self._estimate(other, dummy_req)
                if b_est is None or o_est is None:
                    continue

                # Does other dominate b?
                if (
                    o_est.estimated_cost <= b_est.estimated_cost
                    and o_est.estimated_latency_ms <= b_est.estimated_latency_ms
                    and o_est.estimated_quality >= b_est.estimated_quality
                ):
                    if (
                        o_est.estimated_cost < b_est.estimated_cost
                        or o_est.estimated_latency_ms < b_est.estimated_latency_ms
                        or o_est.estimated_quality > b_est.estimated_quality
                    ):
                        dominated = True
                        break
            if not dominated:
                frontier.append(b)

        # Sort by cost ascending
        frontier.sort(
            key=lambda b: (
                self._estimate(b, dummy_req).estimated_cost
                if self._estimate(b, dummy_req)
                else float("inf")
            )
        )
        return frontier

    def get_frontier_report(self) -> str:
        """Human-readable frontier status."""
        lines = ["=== Pareto Frontier (empirical) ==="]
        dummy_req = Request(prompt="", max_tokens=512, context_len=4096)
        for i, backend in enumerate(self._frontier):
            est = self._estimate(backend, dummy_req)
            if est:
                lines.append(
                    f"{i + 1}. {backend.name:12s} "
                    f"cost=${est.estimated_cost:.4f} "
                    f"lat={est.estimated_latency_ms:.0f}ms "
                    f"qual={est.estimated_quality:.3f}"
                )
        return "\n".join(lines)


# ─── FACTORY: DEFAULT BACKENDS ─────────────────────────────────────────────────


def default_backends() -> list[BackendConfig]:
    """
    Default backend configurations.
    Cost/latency baselines are estimates — calibrate with actual measurements.
    """
    return [
        BackendConfig(
            name="local",
            label="Aurelius Local",
            cost_per_1k_tokens=0.0,
            latency_per_token_ms=5.0,  # 7B @ 4-bit on RTX 4090
            quality=0.92,
            priority=1,
        ),
        BackendConfig(
            name="step",
            label="StepFun Step",
            cost_per_1k_tokens=0.002,
            latency_per_token_ms=50.0,
            quality=0.97,
            priority=2,
        ),
        BackendConfig(
            name="claude",
            label="Anthropic Claude",
            cost_per_1k_tokens=0.008,
            latency_per_token_ms=100.0,
            quality=0.98,
            priority=3,
        ),
        BackendConfig(
            name="deepseek",
            label="DeepSeek V3",
            cost_per_1k_tokens=0.0015,
            latency_per_token_ms=60.0,
            quality=0.96,
            priority=2,
        ),
    ]


# ─── USAGE EXAMPLE ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    router = ParetoFrontierRouter(backends=default_backends(), mode="pareto")

    req = Request(
        prompt="Explain quantum entanglement",
        max_tokens=256,
        context_len=100,
        quality_requirement=0.90,
        budget_per_1k_tokens=0.005,
    )

    decision = router.select(req)
    logger.info(f"Selected: {decision.backend.name}")
    logger.info(f"  Est. cost: ${decision.estimated_cost:.4f}")
    logger.info(f"  Est. latency: {decision.estimated_latency_ms:.0f} ms")
    logger.info(f"  Est. quality: {decision.estimated_quality:.3f}")
    logger.info(f"  Reason: {decision.reason}")

    # Record outcome after serving
    router.record(
        backend_name=decision.backend.name,
        cost=decision.estimated_cost,  # replace with actual billing
        latency_ms=150.0,  # replace with actual measured
        quality=0.93,  # replace with human/eval score
    )

    print("\n" + router.get_frontier_report())
