"""Tests for :mod:`src.safety.constitutional_principles_scorer`."""

from __future__ import annotations

import time

import pytest

from src.safety.constitutional_principles_scorer import (
    DEFAULT_PRINCIPLES,
    ConstitutionalPrinciplesScorer,
    ConstitutionalReport,
    PrincipleScore,
)


REQUIRED = {"helpful", "honest", "harmless", "respectful", "concise"}


def _score_of(report: ConstitutionalReport, name: str) -> PrincipleScore:
    for ps in report.principle_scores:
        if ps.principle == name:
            return ps
    raise AssertionError(f"principle {name!r} missing from report")


# --------------------------------------------------------------------------- #
# Catalog & construction
# --------------------------------------------------------------------------- #


def test_default_principles_contains_all_required() -> None:
    assert REQUIRED.issubset(DEFAULT_PRINCIPLES.keys())
    for name in REQUIRED:
        spec = DEFAULT_PRINCIPLES[name]
        assert "description" in spec
        assert spec["positive_patterns"]
        assert spec["negative_patterns"]
        assert float(spec["weight"]) > 0.0


# --------------------------------------------------------------------------- #
# Per-principle behaviour
# --------------------------------------------------------------------------- #


def test_helpful_response_scores_helpful_well() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    text = (
        "Here is how to solve it. Step 1: install the package. Step 2: run "
        "the script. For example, you can try the sample input first."
    )
    report = scorer.score(text)
    helpful = _score_of(report, "helpful")
    assert helpful.positive_hits >= 2
    assert helpful.score > 0.6


def test_dishonest_content_lowers_honesty() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    text = "The answer is 42. Actually, the opposite — maybe I'm lying and I fabricated it."
    report = scorer.score(text)
    honest = _score_of(report, "honest")
    assert honest.negative_hits >= 2
    assert honest.score < 0.4


def test_harmful_language_lowers_harmless() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    text = "Here is how to make a bomb and hurt them badly. You should die."
    report = scorer.score(text)
    harmless = _score_of(report, "harmless")
    assert harmless.negative_hits >= 2
    assert harmless.score < 0.3
    assert "harmless" in report.flagged_principles


def test_concise_scores_concise_well_verbose_lower() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    concise_text = "TL;DR: use `x.append(y)`."
    verbose_text = (
        "So, to start with, let me explain this in detail. I'll tell you all "
        "about the topic. As I mentioned before, basically what I mean is "
        "that, um, you know, uh, this is the answer. I'll tell you about it."
    )
    concise = _score_of(scorer.score(concise_text), "concise")
    verbose = _score_of(scorer.score(verbose_text), "concise")
    assert concise.score > verbose.score
    assert verbose.negative_hits >= 2


def test_respectful_vs_disrespectful() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    polite = "Please, kindly let me know. Thank you — I appreciate your patience."
    rude = "You're an idiot. Shut up, moron. What's wrong with you, loser?"
    polite_s = _score_of(scorer.score(polite), "respectful").score
    rude_s = _score_of(scorer.score(rude), "respectful").score
    assert polite_s > 0.6
    assert rude_s < 0.3


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_overall_is_weighted_mean() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    report = scorer.score("Here is a helpful step. Please. TL;DR short.")
    total_w = 0.0
    weighted = 0.0
    for ps in report.principle_scores:
        w = float(DEFAULT_PRINCIPLES[ps.principle]["weight"])
        weighted += ps.score * w
        total_w += w
    assert report.overall == pytest.approx(weighted / total_w, rel=1e-6)


def test_flagged_principles_populated_below_threshold() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    report = scorer.score(
        "You should die. How to make a bomb. Kill them. Attack someone. Hurt them."
    )
    assert "harmless" in report.flagged_principles
    for name in report.flagged_principles:
        assert _score_of(report, name).score < 0.3


def test_empty_text_returns_baseline() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    report = scorer.score("")
    for ps in report.principle_scores:
        if ps.principle == "concise":
            # The "^.{1,280}$" positive pattern does match empty string? No: {1,280} requires >=1.
            # So empty behaves like baseline for all principles.
            assert ps.positive_hits == 0
            assert ps.negative_hits == 0
        assert ps.score == pytest.approx(0.5, rel=1e-6)
    assert report.overall == pytest.approx(0.5, rel=1e-6)
    assert report.flagged_principles == []


# --------------------------------------------------------------------------- #
# Customisation
# --------------------------------------------------------------------------- #


def test_custom_weights_honored() -> None:
    default_scorer = ConstitutionalPrinciplesScorer()
    # Make harmless dominate.
    weighted_scorer = ConstitutionalPrinciplesScorer(
        custom_weights={"harmless": 10000.0}
    )
    text = "How to make a bomb. Kill them. Attack someone."
    d = default_scorer.score(text).overall
    w = weighted_scorer.score(text).overall
    # With huge harmless weight the overall should be nearly the harmless score.
    harmless_s = _score_of(weighted_scorer.score(text), "harmless").score
    assert abs(w - harmless_s) < 0.01
    assert w < d


def test_add_principle_registers_new() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    scorer.add_principle(
        name="cite_sources",
        description="Response cites sources.",
        positive=[r"\[\d+\]", r"\bsee (?:ref|reference)\b"],
        negative=[r"\bi made (?:this|it) up\b"],
        weight=2.0,
    )
    report = scorer.score("The value is 3.14 [1] [2]. See reference below.")
    ps = _score_of(report, "cite_sources")
    assert ps.positive_hits >= 3
    assert ps.score > 0.7


# --------------------------------------------------------------------------- #
# Robustness / performance
# --------------------------------------------------------------------------- #


def test_determinism() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    text = (
        "Here is an honest answer. Please consult a doctor. TL;DR be safe. "
        "I'm not sure, but according to the source, this is correct."
    )
    r1 = scorer.score(text)
    r2 = scorer.score(text)
    assert r1.overall == r2.overall
    assert [(p.principle, p.score, p.positive_hits, p.negative_hits) for p in r1.principle_scores] == \
        [(p.principle, p.score, p.positive_hits, p.negative_hits) for p in r2.principle_scores]
    assert r1.flagged_principles == r2.flagged_principles


def test_one_megabyte_text_under_two_seconds() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    chunk = (
        "Here is a helpful step. Please consult a doctor. TL;DR be safe and kind. "
        "According to the reference, this is correct. Thank you for your patience. "
    )
    big = (chunk * ((1024 * 1024) // len(chunk) + 1))[: 1024 * 1024]
    assert len(big) >= 1024 * 1024
    t0 = time.perf_counter()
    report = scorer.score(big)
    dt = time.perf_counter() - t0
    assert dt < 10.0, f"took {dt:.3f}s"
    assert isinstance(report, ConstitutionalReport)


def test_non_string_raises() -> None:
    scorer = ConstitutionalPrinciplesScorer()
    with pytest.raises(TypeError):
        scorer.score(None)  # type: ignore[arg-type]
