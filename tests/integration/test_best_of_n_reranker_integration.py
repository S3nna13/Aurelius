"""Integration tests for Best-of-N reranker.

Verifies end-to-end usage via the public ``src.alignment`` package and that
existing alignment imports are not broken by the new module.
"""

from __future__ import annotations


def test_exposed_via_src_alignment():
    # Importable via the package path (even though __init__.py is minimal).
    from src.alignment.best_of_n_reranker import BestOfNReranker, BoNCandidate

    assert BestOfNReranker is not None
    assert BoNCandidate is not None


def test_end_to_end_best_of_4():
    from src.alignment.best_of_n_reranker import BestOfNReranker

    # Simulate an agentic-coding reward: reward longer responses containing
    # the word "def".
    candidates = [
        "no code here",
        "def f():\n    return 1",
        "def g(x):\n    return x * 2  # longer answer",
        "short",
    ]
    it = iter(candidates)

    def gen(prompt: str) -> str:
        return next(it)

    def reward(prompt: str, response: str) -> float:
        return float(len(response)) + (10.0 if "def " in response else 0.0)

    rr = BestOfNReranker(gen, reward, n=4, aggregation="max")
    top = rr.best("write a python function")
    assert "def " in top.response
    assert top.rank == 0
    # The longest "def" candidate should win.
    assert top.response == candidates[2]


def test_existing_alignment_imports_still_work():
    # bond is the sibling module called out in the spec.
    import src.alignment  # noqa: F401
    import src.alignment.bond  # noqa: F401

    # And the new module sits alongside it.
    from src.alignment import best_of_n_reranker  # noqa: F401


def test_weighted_vote_end_to_end_math_style():
    from src.alignment.best_of_n_reranker import BestOfNReranker

    # Simulate 5 chain-of-thought samples for a math problem; majority
    # arrives at "42".
    samples = [
        "steps... final answer: 42",
        "steps... final answer: 41",
        "steps... final answer: 42",
        "steps... final answer: 42",
        "steps... final answer: 43",
    ]
    it = iter(samples)

    def gen(prompt: str) -> str:
        return next(it)

    def reward(prompt: str, response: str) -> float:
        # Flat reward; majority should dominate via sum.
        return 1.0

    def extract(resp: str) -> str:
        return resp.rsplit(":", 1)[-1].strip()

    rr = BestOfNReranker(gen, reward, n=5, aggregation="weighted_vote")
    winner = rr.weighted_vote("solve 6*7", extract)
    assert winner == "42"
