"""
speculative_acceptance_eval.py — Speculative Decoding Acceptance Evaluator
(Cycle 137-F)

Measures how well a draft model's tokens survive the target model's acceptance
filter in speculative decoding, and the resulting generation throughput gain.

Metrics
-------
- acceptance_rate        : total accepted draft tokens / total draft tokens (α)
- mean_accepted_per_step : avg tokens yielded per target-model call
                           (accepted + bonus token when present)
- theoretical_speedup    : E[tokens per step] = (1 - α^(K+1)) / (1 - α)
                           relative to the 1-token baseline of standard decoding
- draft_efficiency       : same as acceptance_rate (accepted / total_draft)
- per_position_acceptance: per-draft-position acceptance probability (decay profile)

References
----------
- Leviathan et al. "Fast Inference from Transformers via Speculative Decoding"
  (ICML 2023) — §3 acceptance / rejection sampling derivation.
- Chen et al. "Accelerating Large Language Model Decoding with Speculative
  Sampling" (arXiv 2302.01318) — empirical speedup analysis.

Registry
--------
  BENCHMARK_REGISTRY["speculative_acceptance"] = SpeculativeAcceptanceEval
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch  # imported for project consistency; tensor ops used in per-position

from src.eval import BENCHMARK_REGISTRY

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpecAcceptConfig:
    """Hyper-parameters for the speculative acceptance evaluator."""

    max_draft_len: int = 8   # K: maximum draft tokens per speculative step
    n_trials: int = 1000     # Monte Carlo trial count (used by callers if needed)


# ---------------------------------------------------------------------------
# Per-step data container
# ---------------------------------------------------------------------------


@dataclass
class SpecStep:
    """
    Records a single speculative decoding step.

    Attributes
    ----------
    draft_tokens    : token ids proposed by the draft model (length ≤ K)
    accepted_tokens : subset of draft_tokens that the target model accepted,
                      in proposal order (always a prefix of draft_tokens)
    bonus_token     : extra token sampled from the target distribution after
                      partial or full acceptance (None if the step was aborted
                      before bonus sampling)
    target_calls    : number of target-model forward passes used this step
                      (normally 1 in standard speculative decoding)
    """

    draft_tokens: List[int]
    accepted_tokens: List[int]
    bonus_token: Optional[int] = None
    target_calls: int = 1


# ---------------------------------------------------------------------------
# Evaluation result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SpecEvalResult:
    """
    Aggregate metrics from evaluating a list of :class:`SpecStep` objects.

    Attributes
    ----------
    n_steps                 : number of speculative steps evaluated
    total_draft_tokens      : sum of len(step.draft_tokens) over all steps
    total_accepted_tokens   : sum of len(step.accepted_tokens) over all steps
    total_generated_tokens  : total_accepted_tokens + number of bonus tokens
    acceptance_rate         : total_accepted / total_draft  (α)
    mean_accepted_per_step  : (accepted + bonus) averaged across steps
    theoretical_speedup     : E[tokens/step] using the closed-form formula
    draft_efficiency        : alias of acceptance_rate
    """

    n_steps: int
    total_draft_tokens: int
    total_accepted_tokens: int
    total_generated_tokens: int
    acceptance_rate: float
    mean_accepted_per_step: float
    theoretical_speedup: float
    draft_efficiency: float


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


class SpeculativeAcceptanceEval:
    """
    Evaluates speculative decoding acceptance statistics from observed steps.

    Parameters
    ----------
    config : SpecAcceptConfig
        Controls ``max_draft_len`` (K) and ``n_trials``.
    """

    def __init__(self, config: Optional[SpecAcceptConfig] = None) -> None:
        self.config = config or SpecAcceptConfig()

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def compute_acceptance_rate(self, steps: List[SpecStep]) -> float:
        """
        Empirical acceptance rate α = total_accepted / total_draft across
        all steps.

        Returns 0.0 when there are no draft tokens at all.
        """
        total_draft = sum(len(s.draft_tokens) for s in steps)
        if total_draft == 0:
            return 0.0
        total_accepted = sum(len(s.accepted_tokens) for s in steps)
        return total_accepted / total_draft

    def mean_accepted_per_step(self, steps: List[SpecStep]) -> float:
        """
        Mean number of tokens generated per target-model call.

        Each step contributes ``len(accepted_tokens) + (1 if bonus_token else 0)``
        tokens.  Returns 0.0 on an empty step list.
        """
        if not steps:
            return 0.0
        total = sum(
            len(s.accepted_tokens) + (1 if s.bonus_token is not None else 0)
            for s in steps
        )
        return total / len(steps)

    def theoretical_speedup(self, alpha: float, k: int) -> float:
        """
        Expected tokens generated per speculative step under the geometric
        acceptance model.

        For alpha < 1:
            E[tokens] = (1 - alpha^(k+1)) / (1 - alpha)

        For alpha == 1.0 (perfect acceptance):
            E[tokens] = k + 1  (K draft tokens + 1 bonus)

        Parameters
        ----------
        alpha : float
            Per-token acceptance probability in [0, 1].
        k : int
            Number of draft tokens per step (max_draft_len).

        Returns
        -------
        float
            Expected tokens per speculative step.  At baseline (k=0 or α=0)
            this equals 1.0 (one bonus token).
        """
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if alpha >= 1.0:
            return float(k + 1)
        # Numerically stable closed form
        return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)

    def per_position_acceptance(self, steps: List[SpecStep]) -> List[float]:
        """
        For each draft position ``i`` in ``0..K-1``, estimate the probability
        that the token at position ``i`` is accepted.

        Uses PyTorch tensors for accumulation then converts back to plain
        Python floats.

        Returns a list whose length equals the maximum observed draft length
        (capped at ``config.max_draft_len``).  Positions never presented
        (because some steps had shorter drafts) still accumulate only from
        steps where that position existed.
        """
        K = self.config.max_draft_len
        counts = torch.zeros(K, dtype=torch.float32)    # steps with token at pos i
        accepted = torch.zeros(K, dtype=torch.float32)  # steps where pos i accepted

        for step in steps:
            n_draft = len(step.draft_tokens)
            n_accepted = len(step.accepted_tokens)
            for i in range(min(n_draft, K)):
                counts[i] += 1.0
                if i < n_accepted:
                    accepted[i] += 1.0

        # Only return positions that were actually observed
        max_pos = int((counts > 0).float().sum().item())
        if max_pos == 0:
            return []

        rates = (accepted[:max_pos] / counts[:max_pos].clamp(min=1.0)).tolist()
        return rates

    # ------------------------------------------------------------------
    # Aggregate evaluation
    # ------------------------------------------------------------------

    def evaluate(self, steps: List[SpecStep]) -> SpecEvalResult:
        """
        Compute all acceptance metrics from a list of :class:`SpecStep` objects.

        Parameters
        ----------
        steps : list of SpecStep
            Observed speculative decoding steps.

        Returns
        -------
        SpecEvalResult
            Populated with all metrics.
        """
        n_steps = len(steps)
        total_draft = sum(len(s.draft_tokens) for s in steps)
        total_accepted = sum(len(s.accepted_tokens) for s in steps)
        bonus_count = sum(1 for s in steps if s.bonus_token is not None)
        total_generated = total_accepted + bonus_count

        alpha = self.compute_acceptance_rate(steps)
        mean_per_step = self.mean_accepted_per_step(steps)
        speedup = self.theoretical_speedup(alpha, self.config.max_draft_len)

        return SpecEvalResult(
            n_steps=n_steps,
            total_draft_tokens=total_draft,
            total_accepted_tokens=total_accepted,
            total_generated_tokens=total_generated,
            acceptance_rate=alpha,
            mean_accepted_per_step=mean_per_step,
            theoretical_speedup=speedup,
            draft_efficiency=alpha,  # same quantity by definition
        )

    def aggregate(self, results: List[SpecEvalResult]) -> Dict[str, float]:
        """
        Compute the mean of each scalar metric across *results*.

        Returns
        -------
        dict with keys:
            acceptance_rate_mean, speedup_mean, efficiency_mean,
            mean_accepted_per_step_mean, n_steps_mean
        """
        if not results:
            return {
                "acceptance_rate_mean": 0.0,
                "speedup_mean": 0.0,
                "efficiency_mean": 0.0,
                "mean_accepted_per_step_mean": 0.0,
                "n_steps_mean": 0.0,
            }
        n = len(results)
        return {
            "acceptance_rate_mean": sum(r.acceptance_rate for r in results) / n,
            "speedup_mean": sum(r.theoretical_speedup for r in results) / n,
            "efficiency_mean": sum(r.draft_efficiency for r in results) / n,
            "mean_accepted_per_step_mean": sum(
                r.mean_accepted_per_step for r in results
            ) / n,
            "n_steps_mean": sum(r.n_steps for r in results) / n,
        }

    def acceptance_curve(
        self, alpha_range: List[float], k: int
    ) -> Dict[float, float]:
        """
        Compute the theoretical speedup for each alpha value in *alpha_range*
        with draft length *k*.

        Parameters
        ----------
        alpha_range : list of float
            Acceptance-rate values to evaluate, each in [0, 1].
        k : int
            Draft length (max speculative tokens per step).

        Returns
        -------
        dict mapping each alpha to its theoretical speedup.
        """
        return {alpha: self.theoretical_speedup(alpha, k) for alpha in alpha_range}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY["speculative_acceptance"] = SpeculativeAcceptanceEval
