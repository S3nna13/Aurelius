"""AlpacaEval LLM-as-judge scoring harness (Li et al., 2023; arXiv:2305.14387).

Pairwise-only: the candidate model is compared against a fixed baseline
reference answer. The judge LLM is caller-provided as ``judge_fn``. Aggregates
into win rate and a *simplified* length-controlled win rate.

Convention (documented in the pairwise prompt this file emits):
  * First call: Assistant A = candidate, Assistant B = reference.
    - Judge answering ``[[A]]`` means candidate wins.
    - Judge answering ``[[B]]`` means reference wins.
  * With ``swap_order=True`` we make a second call with A/B swapped; if the
    two verdicts disagree (position bias), we record a tie.

Simplified length-controlled win rate (LC):
  LC = raw_win_rate - 0.1 * mean(|length_ratio - 1.0|)
  where length_ratio = candidate_length / max(1, reference_length).
  This is a penalty-style proxy for the GLM-based AlpacaEval 2.0 LC; the full
  GLM fit is intentionally out of scope (spec: "simplified LC").

Pure stdlib: ``re`` + ``statistics``. No foreign imports.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AlpacaProblem:
    """A single AlpacaEval problem: an instruction + baseline reference output."""

    instruction: str
    reference_output: str
    reference_model: str = "baseline"


@dataclass
class AlpacaComparison:
    """Outcome of one pairwise comparison.

    ``winner`` is one of ``{"candidate", "reference", "tie", "invalid"}``.
    """

    instruction: str
    winner: str
    candidate_length: int
    reference_length: int


# ---------------------------------------------------------------------------
# Prompt template (pairwise, AlpacaEval-style)
# ---------------------------------------------------------------------------


PAIRWISE_SYSTEM_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the two "
    "responses provided by AI assistants to the user instruction displayed "
    "below. Choose the assistant whose answer is more helpful, accurate, and "
    "relevant. Avoid position bias. After a brief explanation, output your "
    'final verdict by strictly following this format: "[[A]]" if assistant '
    'A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.'
)


_BRACKET_VERDICT_RE = re.compile(r"\[\[\s*([A-Za-z]+)\s*\]\]")


def _format_pairwise_prompt(instruction: str, answer_a: str, answer_b: str) -> str:
    return "\n".join(
        [
            PAIRWISE_SYSTEM_PROMPT,
            "",
            "[Instruction]",
            instruction,
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
    """Return one of {'A','B','tie','invalid'}."""
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


class AlpacaEvalScorer:
    """AlpacaEval pairwise LLM-as-judge harness with optional order swap.

    Convention on the first call: A = candidate, B = reference. Therefore a
    raw judge verdict of ``[[A]]`` maps to ``winner="candidate"`` and
    ``[[B]]`` maps to ``winner="reference"``.
    """

    def __init__(
        self,
        judge_fn: Callable[[str], str],
        swap_order: bool = True,
    ) -> None:
        if not callable(judge_fn):
            raise TypeError("judge_fn must be callable")
        self._judge_fn = judge_fn
        self.swap_order = bool(swap_order)

    # -------------------------------------------------------------- internals
    def _call_judge(self, prompt: str) -> str:
        try:
            out = self._judge_fn(prompt)
        except Exception as exc:  # noqa: BLE001 -- explicit invalid marker
            return f"__JUDGE_RAISED__ {type(exc).__name__}: {exc}"
        if not isinstance(out, str):
            return ""
        return out

    # ---------------------------------------------------------------- compare
    def compare(self, problem: AlpacaProblem, candidate: str) -> AlpacaComparison:
        cand_len = len(candidate) if isinstance(candidate, str) else 0
        ref_len = len(problem.reference_output)

        # First call: A = candidate, B = reference.
        prompt_ab = _format_pairwise_prompt(
            problem.instruction, candidate, problem.reference_output
        )
        out_ab = self._call_judge(prompt_ab)
        v_ab = _parse_verdict(out_ab)

        def _ab_to_winner(v: str) -> str:
            if v == "A":
                return "candidate"
            if v == "B":
                return "reference"
            if v == "tie":
                return "tie"
            return "invalid"

        winner_first = _ab_to_winner(v_ab)

        if not self.swap_order:
            return AlpacaComparison(
                instruction=problem.instruction,
                winner=winner_first,
                candidate_length=cand_len,
                reference_length=ref_len,
            )

        # Second call with swap: A = reference, B = candidate.
        prompt_ba = _format_pairwise_prompt(
            problem.instruction, problem.reference_output, candidate
        )
        out_ba = self._call_judge(prompt_ba)
        v_ba = _parse_verdict(out_ba)

        # Map swapped verdict back to candidate/reference space.
        if v_ba == "A":
            winner_second = "reference"
        elif v_ba == "B":
            winner_second = "candidate"
        elif v_ba == "tie":
            winner_second = "tie"
        else:
            winner_second = "invalid"

        # Both invalid -> invalid. One invalid -> tie (disagreement is safe
        # default since we cannot corroborate). Otherwise agree or tie.
        if winner_first == "invalid" and winner_second == "invalid":
            final = "invalid"
        elif winner_first == "invalid" or winner_second == "invalid":
            final = "tie"
        elif winner_first == winner_second:
            final = winner_first
        else:
            # Position bias disagreement -> tie.
            final = "tie"

        return AlpacaComparison(
            instruction=problem.instruction,
            winner=final,
            candidate_length=cand_len,
            reference_length=ref_len,
        )

    # ------------------------------------------------------------------ score
    def score(
        self,
        problems: Sequence[AlpacaProblem],
        candidates: list[str],
    ) -> dict:
        """Aggregate win/tie/reference rates and simplified LC winrate.

        Returned keys:
          * ``win_rate``: fraction of valid comparisons where candidate won
          * ``tie_rate``: fraction of valid comparisons tied
          * ``reference_rate``: fraction where reference won
          * ``length_controlled_winrate``: win_rate penalized by mean
            absolute deviation of length ratio from 1.0, scaled by 0.1
          * ``n_valid``: number of non-invalid comparisons
          * ``n_total``: total comparisons attempted

        An empty ``problems`` list returns zeros rather than NaN.
        """
        if len(problems) != len(candidates):
            raise ValueError(
                f"problems ({len(problems)}) and candidates ({len(candidates)}) length mismatch"
            )
        comparisons = [self.compare(p, c) for p, c in zip(problems, candidates)]
        return self.aggregate(comparisons)

    # -------------------------------------------------------------- aggregate
    @staticmethod
    def aggregate(comparisons: Sequence[AlpacaComparison]) -> dict:
        n_total = len(comparisons)
        wins = sum(1 for c in comparisons if c.winner == "candidate")
        ties = sum(1 for c in comparisons if c.winner == "tie")
        refs = sum(1 for c in comparisons if c.winner == "reference")
        n_valid = wins + ties + refs

        if n_valid == 0:
            return {
                "win_rate": 0.0,
                "tie_rate": 0.0,
                "reference_rate": 0.0,
                "length_controlled_winrate": 0.0,
                "n_valid": 0,
                "n_total": n_total,
            }

        win_rate = wins / n_valid
        tie_rate = ties / n_valid
        ref_rate = refs / n_valid

        # Simplified length-controlled winrate.
        ratios_dev: list[float] = []
        for c in comparisons:
            if c.winner == "invalid":
                continue
            denom = c.reference_length if c.reference_length > 0 else 1
            ratio = c.candidate_length / denom
            ratios_dev.append(abs(ratio - 1.0))
        mean_dev = statistics.fmean(ratios_dev) if ratios_dev else 0.0
        lc = win_rate - 0.1 * mean_dev
        # Clamp to [0, 1] so downstream consumers see a rate, not a signed
        # residual. The penalty is intentionally mild (factor 0.1).
        if lc < 0.0:
            lc = 0.0
        elif lc > 1.0:
            lc = 1.0

        return {
            "win_rate": win_rate,
            "tie_rate": tie_rate,
            "reference_rate": ref_rate,
            "length_controlled_winrate": lc,
            "n_valid": n_valid,
            "n_total": n_total,
        }


__all__ = [
    "AlpacaProblem",
    "AlpacaComparison",
    "AlpacaEvalScorer",
    "PAIRWISE_SYSTEM_PROMPT",
]
