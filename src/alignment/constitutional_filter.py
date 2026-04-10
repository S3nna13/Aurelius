"""Constitutional AI data filtering pipeline.

Automated dataset filtering using constitutional principles. This module
provides heuristic and model-based scoring to accept or reject (prompt, response)
pairs based on safety, honesty, helpfulness, and coherence criteria.

Distinct from constitutional.py (self-critique / revision loop) and
constitutional_scoring.py (rule-based scoring utilities).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConstitutionalPrinciple:
    """A single constitutional principle used for dataset filtering."""

    name: str
    description: str
    critique_prompt: str
    revision_prompt: str
    weight: float = 1.0


@dataclass
class FilterScore:
    """Scoring result for one principle applied to one sample."""

    principle_name: str
    score: float          # in [0, 1]
    passed: bool          # score >= threshold (default 0.5)
    critique: str = ""


# ---------------------------------------------------------------------------
# Built-in principles
# ---------------------------------------------------------------------------

SAFETY_PRINCIPLES: list[ConstitutionalPrinciple] = [
    ConstitutionalPrinciple(
        name="harmlessness",
        description="The response must not contain harmful, dangerous, or illegal content.",
        critique_prompt=(
            "Does the following response contain harmful, dangerous, or illegal content? "
            "Identify any problematic phrases."
        ),
        revision_prompt=(
            "Rewrite the following response to remove any harmful, dangerous, or illegal "
            "content while preserving the helpful parts:"
        ),
        weight=1.0,
    ),
    ConstitutionalPrinciple(
        name="honesty",
        description="The response must not be deceptive or present falsehoods with unwarranted certainty.",
        critique_prompt=(
            "Is the following response honest? Does it avoid deception and acknowledge "
            "uncertainty where appropriate?"
        ),
        revision_prompt=(
            "Rewrite the following response to be more honest, avoiding deception and "
            "appropriately expressing uncertainty:"
        ),
        weight=1.0,
    ),
    ConstitutionalPrinciple(
        name="helpfulness",
        description="The response must actually address the question or request.",
        critique_prompt=(
            "Does the following response genuinely address the user's question or request? "
            "Is it substantive and useful?"
        ),
        revision_prompt=(
            "Rewrite the following response to be more helpful and directly address "
            "the user's question:"
        ),
        weight=1.0,
    ),
    ConstitutionalPrinciple(
        name="coherence",
        description="The response must be internally consistent and well-structured.",
        critique_prompt=(
            "Is the following response internally consistent and coherent? "
            "Does it avoid contradictions and repetition?"
        ),
        revision_prompt=(
            "Rewrite the following response to be more coherent, removing contradictions "
            "and unnecessary repetition:"
        ),
        weight=1.0,
    ),
]


# ---------------------------------------------------------------------------
# Heuristic scoring (no model required)
# ---------------------------------------------------------------------------

# Keywords that suggest harmful content
_HARM_KEYWORDS: frozenset[str] = frozenset([
    "harm", "danger", "dangerous", "illegal", "kill", "weapon", "murder",
    "suicide", "bomb", "explosive", "poison", "attack", "terrorist",
])

# Epistemic uncertainty markers (honest hedging)
_UNCERTAINTY_MARKERS: frozenset[str] = frozenset([
    "i think", "probably", "maybe", "i'm not sure", "i am not sure",
    "i believe", "it seems", "might be", "could be", "not certain",
])


def score_text_heuristic(
    text: str,
    principle: ConstitutionalPrinciple,
) -> FilterScore:
    """Score *text* against *principle* using lightweight heuristics (no model).

    Returns a FilterScore with score in [0, 1] and passed = score >= 0.5.
    """
    name = principle.name
    lower = text.lower()

    if name == "harmlessness":
        # Count harmful keyword hits
        hits = sum(1 for kw in _HARM_KEYWORDS if kw in lower)
        # Each hit reduces score by 0.2, floor at 0.0
        score = max(0.0, 1.0 - hits * 0.2)
        critique = (
            f"Found {hits} harmful keyword(s) in text." if hits else ""
        )

    elif name == "honesty":
        # Penalise strong, unqualified assertions on uncertain claims.
        # Reward epistemic markers (show author acknowledges limits).
        marker_hits = sum(1 for m in _UNCERTAINTY_MARKERS if m in lower)
        # A few markers are good; too many (>3) looks evasive — mild penalty
        if marker_hits == 0:
            # No epistemic markers at all — could be over-confident
            score = 0.6
        elif marker_hits <= 3:
            score = 0.85
        else:
            # Excessively hedged
            score = max(0.5, 0.85 - (marker_hits - 3) * 0.05)
        critique = f"{marker_hits} epistemic marker(s) found."

    elif name == "helpfulness":
        # Minimum length check
        if len(text.strip()) <= 20:
            score = 0.2
            critique = "Response is too short to be helpful."
        else:
            # Count words with more than 5 alphabetic characters (proxy for content words)
            content_words = [w for w in re.findall(r"[a-zA-Z]+", text) if len(w) > 5]
            if len(content_words) >= 3:
                score = 0.8
                critique = ""
            elif len(content_words) >= 1:
                score = 0.6
                critique = "Limited substantive vocabulary detected."
            else:
                score = 0.3
                critique = "Response lacks substantive content words."

    elif name == "coherence":
        # Check for repeated consecutive sentences
        sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
        repeated = any(
            sentences[i] == sentences[i + 1]
            for i in range(len(sentences) - 1)
        )
        # Check reasonable punctuation (ends with . ! ? or is very short)
        has_punctuation = bool(re.search(r"[.!?]", text))
        if repeated:
            score = 0.2
            critique = "Repeated consecutive sentences detected."
        elif not has_punctuation and len(text) > 30:
            score = 0.5
            critique = "Response lacks terminal punctuation."
        else:
            score = 0.9
            critique = ""

    else:
        # Unknown principle — neutral score
        score = 0.5
        critique = f"No heuristic available for principle '{name}'."

    return FilterScore(
        principle_name=name,
        score=score,
        passed=score >= 0.5,
        critique=critique,
    )


# ---------------------------------------------------------------------------
# Model-based scorer
# ---------------------------------------------------------------------------

class ModelBasedScorer:
    """Use an Aurelius model to evaluate and revise responses.

    Parameters
    ----------
    model:
        Aurelius model callable — ``loss, logits, pkv = model(input_ids)``.
    tokenizer_encode:
        Callable ``str -> list[int]``.
    tokenizer_decode:
        Callable ``list[int] -> str``.
    principles:
        Principles this scorer knows about.
    """

    # Token IDs used for yes/no classification
    _YES_TOKEN_ID: int = 121
    _NO_TOKEN_ID: int = 110

    def __init__(
        self,
        model,
        tokenizer_encode: Callable[[str], list[int]],
        tokenizer_decode: Callable[[list[int]], str],
        principles: list[ConstitutionalPrinciple],
    ) -> None:
        self.model = model
        self.encode = tokenizer_encode
        self.decode = tokenizer_decode
        self.principles = {p.name: p for p in principles}

    def score_response(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
    ) -> FilterScore:
        """Score *response* against *principle* using the model.

        Builds ``"{critique_prompt}\\nResponse: {response}\\nIs this good? Answer yes or no:"``
        and interprets p("yes") as the score.
        """
        import torch

        eval_text = (
            f"{principle.critique_prompt}\n"
            f"Response: {response}\n"
            f"Is this good? Answer yes or no:"
        )
        token_ids = self.encode(eval_text)
        input_ids = torch.tensor([token_ids], dtype=torch.long)

        with torch.no_grad():
            _loss, logits, _pkv = self.model(input_ids)

        # logits shape: (batch, seq_len, vocab_size)
        # Take the last token position for next-token prediction
        last_logits = logits[0, -1, :]  # (vocab_size,)
        yes_logit = last_logits[self._YES_TOKEN_ID]
        no_logit = last_logits[self._NO_TOKEN_ID]

        # Softmax over just the two tokens
        import math
        yes_exp = math.exp(float(yes_logit))
        no_exp = math.exp(float(no_logit))
        p_yes = yes_exp / (yes_exp + no_exp)
        p_yes = float(max(0.0, min(1.0, p_yes)))

        return FilterScore(
            principle_name=principle.name,
            score=p_yes,
            passed=p_yes >= 0.5,
            critique=f"Model p(yes)={p_yes:.3f}",
        )

    def revise_response(
        self,
        prompt: str,
        response: str,
        principle: ConstitutionalPrinciple,
        max_new_tokens: int = 64,
    ) -> str:
        """Generate a revised response using the model.

        Prepends ``revision_prompt + "\\n" + response`` and autoregressively
        generates up to *max_new_tokens* tokens.
        """
        import torch

        revision_text = f"{principle.revision_prompt}\n{response}"
        token_ids = self.encode(revision_text)
        generated_ids: list[int] = list(token_ids)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                input_ids = torch.tensor([generated_ids], dtype=torch.long)
                _loss, logits, _pkv = self.model(input_ids)
                next_token_logits = logits[0, -1, :]
                next_token_id = int(torch.argmax(next_token_logits).item())
                generated_ids.append(next_token_id)
                # Simple EOS: token 0 or 1 often used as EOS in small models
                if next_token_id in (0, 1, 2):
                    break

        new_token_ids = generated_ids[len(token_ids):]
        revised = self.decode(new_token_ids)
        # Fallback: if decoding yields empty string, return the prompt echo
        return revised if revised.strip() else self.decode(generated_ids)


# ---------------------------------------------------------------------------
# Constitutional filter
# ---------------------------------------------------------------------------

class ConstitutionalFilter:
    """Filter a dataset of (prompt, response) pairs using constitutional principles.

    Parameters
    ----------
    principles:
        List of principles to enforce. Defaults to ``SAFETY_PRINCIPLES``.
    threshold:
        Weighted mean score threshold; samples below this are rejected.
    use_model:
        If True, use a model-based scorer (requires ``ModelBasedScorer`` to be
        set up separately). Currently, only heuristic scoring is used inside
        this class; set ``use_model=True`` as a flag for downstream integration.
    """

    def __init__(
        self,
        principles: list[ConstitutionalPrinciple] | None = None,
        threshold: float = 0.5,
        use_model: bool = False,
    ) -> None:
        self.principles = principles if principles is not None else list(SAFETY_PRINCIPLES)
        self.threshold = threshold
        self.use_model = use_model

    def filter_sample(
        self,
        prompt: str,
        response: str,
    ) -> tuple[bool, list[FilterScore]]:
        """Score a single (prompt, response) pair against all principles.

        Returns
        -------
        passed_all : bool
            True if weighted mean score >= threshold.
        scores : list[FilterScore]
            One FilterScore per principle.
        """
        scores: list[FilterScore] = [
            score_text_heuristic(response, principle)
            for principle in self.principles
        ]

        total_weight = sum(p.weight for p in self.principles)
        if total_weight == 0:
            weighted_mean = 0.0
        else:
            weighted_sum = sum(
                s.score * p.weight
                for s, p in zip(scores, self.principles)
            )
            weighted_mean = weighted_sum / total_weight

        passed_all = weighted_mean >= self.threshold
        return passed_all, scores

    def filter_dataset(
        self,
        samples: list[tuple[str, str]],
    ) -> tuple[list[tuple[str, str]], dict]:
        """Filter a list of (prompt, response) pairs.

        Returns
        -------
        accepted_samples : list[tuple[str, str]]
            Samples that passed all principles.
        stats : dict
            ``{"total": int, "accepted": int, "rejection_rate": float,
               "per_principle_pass_rate": dict[str, float]}``
        """
        accepted: list[tuple[str, str]] = []
        # Per-principle pass counts
        pass_counts: dict[str, int] = {p.name: 0 for p in self.principles}

        for prompt, response in samples:
            passed_all, scores = self.filter_sample(prompt, response)
            if passed_all:
                accepted.append((prompt, response))
            for score in scores:
                if score.passed:
                    pass_counts[score.principle_name] = (
                        pass_counts.get(score.principle_name, 0) + 1
                    )

        total = len(samples)
        accepted_count = len(accepted)
        rejection_rate = 1.0 - (accepted_count / total) if total > 0 else 0.0

        per_principle_pass_rate: dict[str, float] = {
            name: (count / total if total > 0 else 0.0)
            for name, count in pass_counts.items()
        }

        stats = {
            "total": total,
            "accepted": accepted_count,
            "rejection_rate": rejection_rate,
            "per_principle_pass_rate": per_principle_pass_rate,
        }
        return accepted, stats

    def get_rejection_reasons(self, scores: list[FilterScore]) -> list[str]:
        """Return the names of principles whose scores did not pass."""
        return [s.principle_name for s in scores if not s.passed]
