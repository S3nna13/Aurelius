"""Arena-Hard LLM-as-judge scoring harness (Li et al., 2024; LMSYS).

Arena-Hard uses a curated set of *challenging* prompts and asks a judge LLM to
compare two candidate model responses head-to-head. Results are aggregated
into a Bradley-Terry (BT) skill rating with bootstrapped confidence intervals.

Distinct from sibling harnesses:

  * MT-Bench (``mtbench_judge``): single-answer 1-10 rating OR pairwise, with
    multi-turn scaffolding. Arena-Hard is always pairwise and single-turn on a
    single hard prompt.
  * AlpacaEval (``alpacaeval_scorer``): candidate vs. a fixed baseline
    reference answer; win-rate only. Arena-Hard is a round-robin among N
    candidate models with NO privileged reference.

Bradley-Terry primer (informal):
    For two players a, b with log-ratings r_a, r_b:
        P(a beats b) = sigmoid(r_a - r_b) = 1/(1 + exp(-(r_a - r_b)))
    Given a set of pairwise observations we fit log-ratings by gradient ascent
    on the log-likelihood:
        dL/dr_a = sum over games involving a of (win_a - P(a beats opp))
    Ties count as 0.5 for each side.

The concrete update rule implemented here is:
    r_m += lr * sum over games of (observed_m - expected_m)
followed by zero-mean normalization so a lone model rates 0.0 by convention.

Pure stdlib (``math``, ``random``, ``statistics``). No foreign imports.
"""

from __future__ import annotations

import math
import random
import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ArenaProblem:
    """One curated hard prompt."""

    prompt_id: str
    prompt: str
    category: str = "general"


@dataclass
class ArenaComparison:
    """One pairwise comparison outcome.

    ``winner`` is one of ``{"A", "B", "tie", "invalid"}`` referring to
    ``model_a`` vs. ``model_b`` after order-swap reconciliation.
    """

    prompt_id: str
    model_a: str
    model_b: str
    winner: str


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------


PAIRWISE_SYSTEM_PROMPT = (
    "Please act as an impartial judge on a curated, challenging prompt. "
    "Compare the two model responses and pick the better answer overall "
    "(accuracy, helpfulness, depth, and instruction-following). Avoid "
    "position bias. After a brief explanation, output your final verdict by "
    'strictly following this format: "[[A]]" if assistant A is better, '
    '"[[B]]" if assistant B is better, and "[[C]]" for a tie.'
)


_BRACKET_VERDICT_RE = re.compile(r"\[\[\s*([A-Za-z]+)\s*\]\]")


def _format_pairwise_prompt(prompt: str, answer_a: str, answer_b: str) -> str:
    return "\n".join(
        [
            PAIRWISE_SYSTEM_PROMPT,
            "",
            "[User Prompt]",
            prompt,
            "",
            "[The Start of Assistant A's Answer]",
            answer_a,
            "[The End of Assistant A's Answer]",
            "",
            "[The Start of Assistant B's Answer]",
            answer_b,
            "[The End of Assistant B's Answer]",
        ]
    )


def _parse_verdict(output: str) -> str:
    """Return one of {'A', 'B', 'tie', 'invalid'}."""
    if not isinstance(output, str):
        return "invalid"
    m = _BRACKET_VERDICT_RE.search(output)
    if m is None:
        return "invalid"
    tok = m.group(1).strip().lower()
    if tok == "a":
        return "A"
    if tok == "b":
        return "B"
    if tok in ("c", "tie"):
        return "tie"
    return "invalid"


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------


class ArenaHardScorer:
    """Arena-Hard round-robin judge + Bradley-Terry rating fit."""

    def __init__(
        self,
        judge_fn: Callable[[str], str],
        swap_order: bool = True,
    ) -> None:
        if not callable(judge_fn):
            raise TypeError("judge_fn must be callable")
        self._judge_fn = judge_fn
        self.swap_order = bool(swap_order)

    # ----------------------------------------------------------- judge helper
    def _call_judge(self, prompt: str) -> str:
        try:
            out = self._judge_fn(prompt)
        except Exception as exc:  # noqa: BLE001 -- convert to invalid marker
            return f"__JUDGE_RAISED__ {type(exc).__name__}: {exc}"
        if not isinstance(out, str):
            return ""
        return out

    # ------------------------------------------------------------------ compare
    def compare(
        self,
        problem: ArenaProblem,
        response_a: str,
        response_b: str,
        model_a_name: str,
        model_b_name: str,
    ) -> ArenaComparison:
        """Compare two responses; returns winner in {A, B, tie, invalid}."""
        if not isinstance(response_a, str) or not isinstance(response_b, str):
            raise TypeError("response_a and response_b must be strings")
        if not model_a_name or not model_b_name:
            raise ValueError("model_a_name and model_b_name must be non-empty")
        if model_a_name == model_b_name:
            raise ValueError("model_a_name and model_b_name must differ")

        prompt_ab = _format_pairwise_prompt(problem.prompt, response_a, response_b)
        v_ab = _parse_verdict(self._call_judge(prompt_ab))

        if not self.swap_order:
            return ArenaComparison(
                prompt_id=problem.prompt_id,
                model_a=model_a_name,
                model_b=model_b_name,
                winner=v_ab,
            )

        # Swap: judge sees responses in reversed order. Map back.
        prompt_ba = _format_pairwise_prompt(problem.prompt, response_b, response_a)
        v_ba_raw = _parse_verdict(self._call_judge(prompt_ba))
        if v_ba_raw == "A":
            v_ba = "B"  # "A" in swapped call == model B in original frame
        elif v_ba_raw == "B":
            v_ba = "A"
        else:
            v_ba = v_ba_raw  # "tie" or "invalid"

        # Reconciliation:
        #   both invalid -> invalid
        #   one invalid  -> tie (safe fallback; position bias uncorroborated)
        #   agree        -> that verdict
        #   disagree     -> tie (position-bias disagreement)
        if v_ab == "invalid" and v_ba == "invalid":
            final = "invalid"
        elif v_ab == "invalid" or v_ba == "invalid":
            final = "tie"
        elif v_ab == v_ba:
            final = v_ab
        else:
            final = "tie"

        return ArenaComparison(
            prompt_id=problem.prompt_id,
            model_a=model_a_name,
            model_b=model_b_name,
            winner=final,
        )

    # -------------------------------------------------------- run_round_robin
    def run_round_robin(
        self,
        problems: Sequence[ArenaProblem],
        responses: dict[str, list[str]],
    ) -> list[ArenaComparison]:
        """All unordered model pairs on all problems.

        ``responses[model_name][i]`` is the model's answer for ``problems[i]``.
        """
        if not isinstance(responses, dict):
            raise TypeError("responses must be a dict[model_name, list[str]]")
        model_names = list(responses.keys())
        n = len(problems)
        for m in model_names:
            answers = responses[m]
            if not isinstance(answers, list):
                raise TypeError(f"responses[{m!r}] must be a list")
            if len(answers) != n:
                raise ValueError(
                    f"responses[{m!r}] has {len(answers)} answers; expected {n} to match problems"
                )

        comparisons: list[ArenaComparison] = []
        for i, problem in enumerate(problems):
            for a_idx in range(len(model_names)):
                for b_idx in range(a_idx + 1, len(model_names)):
                    ma = model_names[a_idx]
                    mb = model_names[b_idx]
                    comparisons.append(
                        self.compare(
                            problem,
                            responses[ma][i],
                            responses[mb][i],
                            ma,
                            mb,
                        )
                    )
        return comparisons

    # ---------------------------------------------------- fit_bradley_terry
    @staticmethod
    def fit_bradley_terry(
        comparisons: Sequence[ArenaComparison],
        model_names: Sequence[str],
        n_iters: int = 100,
        lr: float = 0.1,
    ) -> dict[str, float]:
        """Fit BT log-ratings via iterative (observed - expected) updates.

        Ties count as 0.5 for each side. Invalid comparisons are skipped.
        Returns a dict ``{model_name: rating}``; ratings are zero-mean
        normalized so a single model has rating ``0.0``.
        """
        names = list(model_names)
        if not isinstance(names, list):
            raise TypeError("model_names must be a sequence")
        if len(names) != len(set(names)):
            raise ValueError("model_names must be unique")
        for n in names:
            if not isinstance(n, str) or not n:
                raise ValueError("model_names must be non-empty strings")

        ratings: dict[str, float] = {m: 0.0 for m in names}
        if not names:
            return ratings

        # Filter relevant comparisons up front.
        valid: list[ArenaComparison] = []
        name_set = set(names)
        for c in comparisons:
            if c.winner == "invalid":
                continue
            if c.model_a not in name_set or c.model_b not in name_set:
                continue
            valid.append(c)

        if not valid:
            return ratings

        for _ in range(max(0, int(n_iters))):
            grad: dict[str, float] = {m: 0.0 for m in names}
            for c in valid:
                ra = ratings[c.model_a]
                rb = ratings[c.model_b]
                # P(A beats B) = sigmoid(ra - rb)
                diff = ra - rb
                # Numerically-stable sigmoid.
                if diff >= 0:
                    z = math.exp(-diff)
                    p_a = 1.0 / (1.0 + z)
                else:
                    z = math.exp(diff)
                    p_a = z / (1.0 + z)
                if c.winner == "A":
                    obs_a = 1.0
                elif c.winner == "B":
                    obs_a = 0.0
                else:  # tie
                    obs_a = 0.5
                residual = obs_a - p_a
                grad[c.model_a] += residual
                grad[c.model_b] -= residual
            for m in names:
                ratings[m] += lr * grad[m]

        # Zero-mean normalize so a single model ends at 0.0 and scale is
        # interpretable across fits.
        mean_r = statistics.fmean(ratings.values())
        for m in names:
            ratings[m] -= mean_r
        return ratings

    # ---------------------------------------------- bootstrap_confidence_intervals
    @staticmethod
    def bootstrap_confidence_intervals(
        comparisons: Sequence[ArenaComparison],
        model_names: Sequence[str],
        n_bootstrap: int = 200,
        ci: float = 0.95,
        n_iters: int = 100,
        lr: float = 0.1,
        seed: int | None = None,
    ) -> dict[str, tuple[float, float, float]]:
        """Non-parametric bootstrap over the *comparisons* list.

        Returns ``{model_name: (mean, lo, hi)}`` at the requested two-sided
        confidence level. With no valid comparisons, every model maps to
        ``(0.0, 0.0, 0.0)``.
        """
        if not 0.0 < ci < 1.0:
            raise ValueError("ci must be in (0, 1)")
        names = list(model_names)
        n_bootstrap = int(n_bootstrap)
        if n_bootstrap < 1:
            raise ValueError("n_bootstrap must be >= 1")

        rng = random.Random(seed)
        comps = list(comparisons)

        # Point estimate on the full set, for anchor "mean".
        point_ratings = ArenaHardScorer.fit_bradley_terry(comps, names, n_iters=n_iters, lr=lr)

        if not comps:
            return {m: (point_ratings.get(m, 0.0), 0.0, 0.0) for m in names}

        samples: dict[str, list[float]] = {m: [] for m in names}
        n = len(comps)
        for _ in range(n_bootstrap):
            resample = [comps[rng.randrange(n)] for _ in range(n)]
            r = ArenaHardScorer.fit_bradley_terry(resample, names, n_iters=n_iters, lr=lr)
            for m in names:
                samples[m].append(r[m])

        alpha = (1.0 - ci) / 2.0
        out: dict[str, tuple[float, float, float]] = {}
        for m in names:
            vals = sorted(samples[m])
            lo_idx = max(0, min(len(vals) - 1, int(math.floor(alpha * len(vals)))))
            hi_idx = max(0, min(len(vals) - 1, int(math.ceil((1.0 - alpha) * len(vals))) - 1))
            lo = vals[lo_idx]
            hi = vals[hi_idx]
            mean_val = statistics.fmean(vals)
            # Guarantee lo <= mean <= hi (floating point defensive).
            lo = min(lo, mean_val)
            hi = max(hi, mean_val)
            out[m] = (mean_val, lo, hi)
        return out


__all__ = [
    "ArenaProblem",
    "ArenaComparison",
    "ArenaHardScorer",
    "PAIRWISE_SYSTEM_PROMPT",
]
