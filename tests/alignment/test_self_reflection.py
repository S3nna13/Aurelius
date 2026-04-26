"""Tests for self_reflection module."""

from __future__ import annotations

import pytest
import torch

from src.alignment.self_reflection import (
    ReflectionConfig,
    ReflectionDataCollector,
    ReflectionStep,
    ResponseScorer,
    SelfCritiqueGenerator,
    SelfReflectionLoop,
    compute_self_consistency_score,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(42)
    cfg = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )
    model = AureliusTransformer(cfg)
    model.eval()
    return model


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8")[:64])


def decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def scorer(small_model):
    return ResponseScorer(small_model, encode, decode)


@pytest.fixture(scope="module")
def default_config():
    return ReflectionConfig()


@pytest.fixture(scope="module")
def critic(small_model, default_config):
    return SelfCritiqueGenerator(small_model, encode, decode, default_config)


@pytest.fixture(scope="module")
def reflection_loop(small_model, scorer, critic, default_config):
    return SelfReflectionLoop(small_model, scorer, critic, default_config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_reflection_config_defaults():
    """ReflectionConfig should have the specified default values."""
    cfg = ReflectionConfig()
    assert cfg.max_reflection_rounds == 3
    assert cfg.reflection_temperature == 0.7
    assert cfg.improvement_threshold == 0.01
    assert cfg.use_critic is True
    assert "{response}" in cfg.critique_prompt_template
    assert "{response}" in cfg.revision_prompt_template
    assert "{critique}" in cfg.revision_prompt_template


def test_reflection_step_fields():
    """ReflectionStep should store all required fields correctly."""
    step = ReflectionStep(round=1, response="hello", critique="ok", score=0.5, improved=True)
    assert step.round == 1
    assert step.response == "hello"
    assert step.critique == "ok"
    assert step.score == 0.5
    assert step.improved is True


def test_response_scorer_returns_float(scorer):
    """score_response must return a plain Python float."""
    result = scorer.score_response("Hello", "World")
    assert isinstance(result, float)


def test_response_scorer_batch_length(scorer):
    """score_batch must return a list with the same length as the inputs."""
    prompts = ["A", "B", "C"]
    responses = ["x", "y", "z"]
    scores = scorer.score_batch(prompts, responses)
    assert isinstance(scores, list)
    assert len(scores) == 3
    assert all(isinstance(s, float) for s in scores)


def test_self_critique_generator_critique_is_string(critic):
    """generate_critique must return a non-None string."""
    result = critic.generate_critique("This is a test response.", max_tokens=8)
    assert isinstance(result, str)


def test_self_critique_generator_revision_is_string(critic):
    """generate_revision must return a non-None string."""
    result = critic.generate_revision("This is a test response.", "Could be clearer.", max_tokens=8)
    assert isinstance(result, str)


def test_self_reflection_loop_returns_steps(reflection_loop):
    """reflect must return a (str, list[ReflectionStep]) tuple."""
    best, steps = reflection_loop.reflect("An initial response.", max_new_tokens=8)
    assert isinstance(best, str)
    assert isinstance(steps, list)
    assert len(steps) >= 1
    for step in steps:
        assert isinstance(step, ReflectionStep)


def test_self_reflection_loop_stops_early(small_model, scorer, critic):
    """With improvement_threshold=999.0, the loop should stop after round 1."""
    cfg = ReflectionConfig(max_reflection_rounds=3, improvement_threshold=999.0)
    loop = SelfReflectionLoop(small_model, scorer, critic, cfg)
    _, steps = loop.reflect("Some response.", max_new_tokens=8)
    # With a threshold so high that no real improvement can exceed it, stop after 1 round.
    assert len(steps) == 1


def test_self_reflection_best_of(reflection_loop):
    """best_of_reflection must return one of the provided response strings."""
    responses = ["alpha response", "beta response", "gamma response"]
    best = reflection_loop.best_of_reflection(responses)
    assert isinstance(best, str)
    assert best in responses


def test_compute_self_consistency_identical():
    """Identical responses should yield a consistency score of 1.0."""
    responses = ["hello world", "hello world", "hello world"]
    score = compute_self_consistency_score(responses, encode)
    assert score == pytest.approx(1.0)


def test_compute_self_consistency_disjoint():
    """Completely disjoint token sets should yield a consistency score of 0.0."""
    # ASCII letters A-Z (65-90) vs digits 0-9 (48-57) — no shared bytes.
    responses = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789"]
    score = compute_self_consistency_score(responses, encode)
    assert score == pytest.approx(0.0)


def test_reflection_data_collector_training_pairs():
    """ReflectionDataCollector should produce correct chosen/rejected pairs."""
    collector = ReflectionDataCollector()

    step_good = ReflectionStep(
        round=1, response="good response", critique="ok", score=0.9, improved=True
    )
    step_bad = ReflectionStep(
        round=2, response="bad response", critique="poor", score=0.1, improved=False
    )

    collector.add_step("test prompt", step_good)
    collector.add_step("test prompt", step_bad)

    pairs = collector.get_training_pairs()
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair["prompt"] == "test prompt"
    assert pair["chosen"] == "good response"
    assert pair["rejected"] == "bad response"

    collector.clear()
    assert collector.data == []
    assert collector.get_training_pairs() == []
