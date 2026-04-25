import pytest

from src.inference.difficulty_router import (
    DifficultyRouter,
    RouterStrategy,
    RoutingDecision,
)


@pytest.fixture()
def router() -> DifficultyRouter:
    return DifficultyRouter(threshold=0.5, strategy=RouterStrategy.HYBRID)


def test_default_threshold_and_strategy():
    r = DifficultyRouter()
    assert r.threshold == 0.5
    assert r.strategy == RouterStrategy.HYBRID


def test_score_heuristic_empty_prompt(router):
    score = router.score_heuristic("")
    assert 0.0 <= score <= 1.0


def test_score_heuristic_short_simple_prompt(router):
    score = router.score_heuristic("hi")
    assert score < 0.5


def test_score_heuristic_math_keywords_boost(router):
    base = router.score_heuristic("explain something")
    boosted = router.score_heuristic("prove the theorem about integral calculus")
    assert boosted > base


def test_score_heuristic_code_keywords_boost(router):
    base = router.score_heuristic("explain something")
    boosted = router.score_heuristic("implement a recursive algorithm in python")
    assert boosted > base


def test_score_heuristic_greet_penalty(router):
    base = router.score_heuristic("explain something quickly")
    penalised = router.score_heuristic("hi thanks")
    assert penalised <= base


def test_score_heuristic_clamps_to_zero_one(router):
    score = router.score_heuristic("thanks" * 100)
    assert 0.0 <= score <= 1.0
    long_math = ("prove theorem " * 50)
    score2 = router.score_heuristic(long_math)
    assert 0.0 <= score2 <= 1.0


def test_score_mf_deterministic(router):
    prompt = "What is the capital of France?"
    assert router.score_mf(prompt) == router.score_mf(prompt)


def test_score_mf_range(router):
    for prompt in ["hello", "complex math proof", "algorithm", ""]:
        score = router.score_mf(prompt)
        assert 0.0 <= score <= 1.0


def test_score_mf_different_prompts_differ(router):
    s1 = router.score_mf("hello world")
    s2 = router.score_mf("prove the Riemann hypothesis")
    assert s1 != s2


def test_route_returns_routing_decision(router):
    decision = router.route("What is 2+2?")
    assert isinstance(decision, RoutingDecision)


def test_route_routed_to_strong_above_threshold():
    r = DifficultyRouter(threshold=0.0, strategy=RouterStrategy.HEURISTIC)
    decision = r.route("prove the theorem about matrix eigenvalue decomposition calculus integral gradient")
    assert decision.routed_to == "strong"


def test_route_routed_to_weak_below_threshold():
    r = DifficultyRouter(threshold=1.0, strategy=RouterStrategy.HEURISTIC)
    decision = r.route("hi")
    assert decision.routed_to == "weak"


def test_route_heuristic_strategy():
    r = DifficultyRouter(strategy=RouterStrategy.HEURISTIC)
    decision = r.route("hello world")
    assert decision.strategy == RouterStrategy.HEURISTIC
    assert "heuristic" in decision.reason


def test_route_mf_strategy():
    r = DifficultyRouter(strategy=RouterStrategy.MATRIX_FACTOR)
    decision = r.route("hello world")
    assert decision.strategy == RouterStrategy.MATRIX_FACTOR
    assert "matrix-factor" in decision.reason


def test_route_hybrid_strategy(router):
    decision = router.route("hello world")
    assert decision.strategy == RouterStrategy.HYBRID
    assert "hybrid" in decision.reason


def test_route_threshold_stored_in_decision(router):
    decision = router.route("something")
    assert decision.threshold == router.threshold


def test_batch_route_length(router):
    prompts = ["hello", "prove theorem", "implement algorithm"]
    decisions = router.batch_route(prompts)
    assert len(decisions) == len(prompts)


def test_batch_route_all_decisions(router):
    prompts = ["a", "b", "c"]
    decisions = router.batch_route(prompts)
    assert all(isinstance(d, RoutingDecision) for d in decisions)


def test_batch_route_empty(router):
    assert router.batch_route([]) == []


def test_complexity_score_in_range(router):
    for prompt in ["hello", "prove theorem about eigenvalues", "implement async algorithm"]:
        d = router.route(prompt)
        assert 0.0 <= d.complexity_score <= 1.0
