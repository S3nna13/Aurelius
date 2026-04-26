"""Tests for src/reasoning/beam_search_reasoner.py — at least 20 tests."""

from __future__ import annotations

import pytest

from src.reasoning.beam_search_reasoner import (
    _MAX_BEAM_WIDTH,
    _MAX_DEPTH,
    _MAX_STEP_LEN,
    BEAM_SEARCH_REASONER,
    BeamHypothesis,
    BeamSearchReasoner,
)

# ---------------------------------------------------------------------------
# BeamHypothesis dataclass
# ---------------------------------------------------------------------------


def test_beam_hypothesis_depth_correct():
    h = BeamHypothesis(steps=["a", "b", "c"])
    assert h.depth == 3


def test_beam_hypothesis_depth_empty():
    h = BeamHypothesis(steps=[])
    assert h.depth == 0


def test_beam_hypothesis_extend_appends_step():
    h = BeamHypothesis(steps=["step one"])
    h2 = h.extend("step two", 1.0)
    assert h2.steps == ["step one", "step two"]


def test_beam_hypothesis_extend_sums_scores():
    h = BeamHypothesis(steps=["s1"], score=2.0)
    h2 = h.extend("s2", 3.0)
    assert h2.score == pytest.approx(5.0)


def test_beam_hypothesis_extend_too_long_raises():
    h = BeamHypothesis(steps=[])
    with pytest.raises(ValueError):
        h.extend("x" * (_MAX_STEP_LEN + 1), 1.0)


def test_beam_hypothesis_init_step_too_long_raises():
    with pytest.raises(ValueError):
        BeamHypothesis(steps=["x" * (_MAX_STEP_LEN + 1)])


def test_beam_hypothesis_normalized_score():
    h = BeamHypothesis(steps=["a", "b", "c"], score=6.0)
    assert h.normalized_score() == pytest.approx(2.0)


def test_beam_hypothesis_normalized_score_empty_steps():
    h = BeamHypothesis(steps=[], score=5.0)
    # max(0, 1) = 1 → 5.0 / 1 = 5.0
    assert h.normalized_score() == pytest.approx(5.0)


def test_beam_hypothesis_is_finished_default_false():
    h = BeamHypothesis(steps=["s"])
    assert h.is_finished is False


# ---------------------------------------------------------------------------
# BeamSearchReasoner construction
# ---------------------------------------------------------------------------


def test_beam_width_below_one_raises():
    with pytest.raises(ValueError, match="beam_width"):
        BeamSearchReasoner(beam_width=0)


def test_beam_width_above_max_raises():
    with pytest.raises(ValueError, match="beam_width"):
        BeamSearchReasoner(beam_width=_MAX_BEAM_WIDTH + 1)


def test_max_depth_below_one_raises():
    with pytest.raises(ValueError, match="max_depth"):
        BeamSearchReasoner(max_depth=0)


def test_max_depth_above_max_raises():
    with pytest.raises(ValueError, match="max_depth"):
        BeamSearchReasoner(max_depth=_MAX_DEPTH + 1)


# ---------------------------------------------------------------------------
# initialize
# ---------------------------------------------------------------------------


def test_initialize_single_hypothesis():
    r = BeamSearchReasoner(beam_width=4)
    beam = r.initialize("first step")
    assert len(beam) == 1
    assert beam[0].steps == ["first step"]


def test_initialize_score_propagated():
    r = BeamSearchReasoner(beam_width=4)
    beam = r.initialize("start", initial_score=3.0)
    assert beam[0].score == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------


def test_expand_candidates_mismatch_raises():
    r = BeamSearchReasoner(beam_width=4)
    h1 = BeamHypothesis(steps=["a"])
    h2 = BeamHypothesis(steps=["b"])
    with pytest.raises(ValueError, match="candidates length"):
        r.expand([h1, h2], [[("x", 1.0)]])  # 2 hyps, 1 candidates list


def test_expand_two_hyps_two_candidates_pruned_to_beam_width():
    r = BeamSearchReasoner(beam_width=3, normalize_scores=False)
    h1 = BeamHypothesis(steps=["a"], score=0.0)
    h2 = BeamHypothesis(steps=["b"], score=0.0)
    candidates = [
        [("x1", 1.0), ("x2", 2.0)],
        [("y1", 3.0), ("y2", 4.0)],
    ]
    result = r.expand([h1, h2], candidates)
    assert len(result) <= 3


def test_expand_finished_hypothesis_passes_through():
    r = BeamSearchReasoner(beam_width=4)
    h = BeamHypothesis(steps=["done"], score=5.0, is_finished=True)
    result = r.expand([h], [[("extra", 1.0)]])
    # finished hyp should pass through without being extended
    assert any(hyp.steps == ["done"] and hyp.is_finished for hyp in result)


def test_expand_hypothesis_at_max_depth_passes_through():
    r = BeamSearchReasoner(beam_width=4, max_depth=2)
    # hypothesis already at max_depth
    h = BeamHypothesis(steps=["s1", "s2"], score=1.0)
    result = r.expand([h], [[("new", 1.0)]])
    assert any(hyp.steps == ["s1", "s2"] for hyp in result)


# ---------------------------------------------------------------------------
# _prune
# ---------------------------------------------------------------------------


def test_prune_returns_at_most_beam_width():
    r = BeamSearchReasoner(beam_width=2, normalize_scores=False)
    hyps = [BeamHypothesis(steps=[f"s{i}"], score=float(i)) for i in range(10)]
    pruned = r._prune(hyps)
    assert len(pruned) <= 2


def test_prune_sorted_descending_by_score():
    r = BeamSearchReasoner(beam_width=5, normalize_scores=False)
    hyps = [BeamHypothesis(steps=[f"s{i}"], score=float(i)) for i in range(5)]
    pruned = r._prune(hyps)
    scores = [h.score for h in pruned]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# best
# ---------------------------------------------------------------------------


def test_best_returns_max_score_hypothesis():
    r = BeamSearchReasoner(beam_width=4, normalize_scores=False)
    hyps = [BeamHypothesis(steps=[f"s{i}"], score=float(i)) for i in range(5)]
    b = r.best(hyps)
    assert b.score == pytest.approx(4.0)


def test_best_empty_list_raises():
    r = BeamSearchReasoner(beam_width=4)
    with pytest.raises(ValueError, match="empty hypothesis list"):
        r.best([])


# ---------------------------------------------------------------------------
# mark_finished
# ---------------------------------------------------------------------------


def test_mark_finished_sets_is_finished():
    r = BeamSearchReasoner(beam_width=4)
    h = BeamHypothesis(steps=["step1"], score=1.0)
    finished = r.mark_finished(h)
    assert finished.is_finished is True
    assert finished.steps == h.steps
    assert finished.score == h.score


# ---------------------------------------------------------------------------
# normalize_scores flag
# ---------------------------------------------------------------------------


def test_normalize_scores_true_uses_normalized():
    r = BeamSearchReasoner(beam_width=2, normalize_scores=True)
    # hyp A: score=10, depth=5 → normalized=2.0
    # hyp B: score=6, depth=2 → normalized=3.0
    # B should win
    ha = BeamHypothesis(steps=["x"] * 5, score=10.0)
    hb = BeamHypothesis(steps=["y"] * 2, score=6.0)
    best = r.best([ha, hb])
    assert best.steps == hb.steps


def test_normalize_scores_false_uses_raw():
    r = BeamSearchReasoner(beam_width=2, normalize_scores=False)
    # hyp A: score=10, depth=5 → raw=10.0
    # hyp B: score=6, depth=2 → raw=6.0
    # A should win
    ha = BeamHypothesis(steps=["x"] * 5, score=10.0)
    hb = BeamHypothesis(steps=["y"] * 2, score=6.0)
    best = r.best([ha, hb])
    assert best.steps == ha.steps


# ---------------------------------------------------------------------------
# Full beam search simulation
# ---------------------------------------------------------------------------


def test_full_beam_search_simulation_three_steps_deep():
    """Simulate a 3-step beam search and verify the best path is selected."""
    r = BeamSearchReasoner(beam_width=2, max_depth=4, normalize_scores=False)

    # Step 0: initialize
    beam = r.initialize("problem statement", initial_score=0.0)
    assert len(beam) == 1

    # Step 1: expand — two candidates per hypothesis
    candidates_step1 = [
        [("path A step 1", 1.0), ("path B step 1", 3.0)],
    ]
    beam = r.expand(beam, candidates_step1)
    # beam has at most 2 hypotheses; path B should rank higher (score=3)
    assert len(beam) <= 2
    top = beam[0]
    assert "path B step 1" in top.steps

    # Step 2: expand again
    candidates_step2 = [[("A2", 1.0), ("B2", 5.0)] for _ in beam]
    beam = r.expand(beam, candidates_step2)
    assert len(beam) <= 2

    # Step 3: expand again
    candidates_step3 = [[("A3", 0.5), ("B3", 2.0)] for _ in beam]
    beam = r.expand(beam, candidates_step3)
    assert len(beam) <= 2

    best = r.best(beam)
    # The best hypothesis should exist and have 4 steps (initial + 3 expansions)
    assert len(best.steps) == 4
    assert best.score > 0.0


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


def test_beam_search_reasoner_singleton_is_instance():
    assert isinstance(BEAM_SEARCH_REASONER, BeamSearchReasoner)
