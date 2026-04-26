"""Tests for LeastToMost."""

from __future__ import annotations

from src.reasoning.least_to_most import (
    L2M_REGISTRY,
    L2MConfig,
    L2MResult,
    LeastToMost,
    SubProblem,
)


def test_decompose_splits_numbered_lines():
    l2m = LeastToMost()
    response = "1. Find the speed\n2. Find the distance\n3. Compute total"
    parts = l2m.decompose("some question", lambda q: response)
    assert parts == ["Find the speed", "Find the distance", "Compute total"]


def test_decompose_splits_bullet_lines():
    l2m = LeastToMost()
    response = "- Step one\n- Step two"
    parts = l2m.decompose("q", lambda q: response)
    assert parts == ["Step one", "Step two"]


def test_decompose_respects_max_subproblems():
    l2m = LeastToMost(L2MConfig(max_subproblems=2))
    response = "1. a\n2. b\n3. c\n4. d"
    parts = l2m.decompose("q", lambda q: response)
    assert len(parts) == 2


def test_decompose_ignores_blank_lines():
    l2m = LeastToMost()
    response = "\n1. Only one\n\n"
    parts = l2m.decompose("q", lambda q: response)
    assert parts == ["Only one"]


def test_solve_sequential_accumulates_answers():
    l2m = LeastToMost()
    calls = []

    def solve(q, ctx):
        calls.append((q, list(ctx)))
        return f"ans_{q}"

    resolved = l2m.solve_sequential(["a", "b", "c"], solve)
    assert [r.answer for r in resolved] == ["ans_a", "ans_b", "ans_c"]
    assert calls[1][1] == ["ans_a"]
    assert calls[2][1] == ["ans_a", "ans_b"]


def test_solve_sequential_returns_subproblems():
    l2m = LeastToMost()
    resolved = l2m.solve_sequential(["q1"], lambda q, ctx: "42")
    assert len(resolved) == 1
    assert isinstance(resolved[0], SubProblem)
    assert resolved[0].is_resolved


def test_run_returns_l2m_result():
    l2m = LeastToMost()
    decomp_response = "1. sub1\n2. sub2"
    result = l2m.run(
        "big question",
        lambda q: decomp_response,
        solve_fn=lambda q, ctx: "solved",
    )
    assert isinstance(result, L2MResult)


def test_run_n_steps_matches_subproblems():
    l2m = LeastToMost()
    decomp_response = "1. a\n2. b\n3. c"
    result = l2m.run(
        "q",
        lambda q: decomp_response,
        solve_fn=lambda q, ctx: "ok",
    )
    assert result.n_steps == 3


def test_run_final_answer_is_last_subproblem_answer():
    l2m = LeastToMost()
    decomp_response = "1. first\n2. second"
    answers = iter(["ans1", "ans2"])
    result = l2m.run(
        "q",
        lambda q: decomp_response,
        solve_fn=lambda q, ctx: next(answers),
    )
    assert result.final_answer == "ans2"


def test_run_uses_generate_fn_as_solve_when_no_solve_fn():
    l2m = LeastToMost()
    decomp_done = False

    def gen(q):
        nonlocal decomp_done
        if not decomp_done:
            decomp_done = True
            return "1. sub"
        return "fallback_answer"

    result = l2m.run("q", gen)
    assert result.final_answer == "fallback_answer"


def test_registry_key():
    assert "default" in L2M_REGISTRY
    assert L2M_REGISTRY["default"] is LeastToMost
