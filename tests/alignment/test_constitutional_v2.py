"""Tests for constitutional_v2: multi-principle CAI pipeline."""
from __future__ import annotations

import pytest
import torch

from src.alignment.constitutional_v2 import (
    ConstitutionalPrinciple,
    HARMLESSNESS_PRINCIPLES,
    sample_principles,
    format_critique_prompt,
    format_revision_prompt,
    greedy_generate,
    ConstitutionalReviser,
    compute_revision_similarity,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Fixtures
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


def encode_fn(text: str) -> list[int]:
    """Simple byte-level encoding (values 0-255)."""
    return list(text.encode("utf-8", errors="replace"))[:64]


def decode_fn(ids: list[int]) -> str:
    """Simple byte-level decoding."""
    return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def reviser(small_model):
    return ConstitutionalReviser(
        model=small_model,
        encode_fn=encode_fn,
        decode_fn=decode_fn,
        principles=HARMLESSNESS_PRINCIPLES,
        max_new_tokens=4,
    )


# ---------------------------------------------------------------------------
# Test 1: ConstitutionalPrinciple fields
# ---------------------------------------------------------------------------

def test_constitutional_principle_fields():
    p = ConstitutionalPrinciple(
        name="test_principle",
        critique_prompt="Does {response} look OK?",
        revision_prompt="Fix {response} given {critique}.",
        weight=2.5,
    )
    assert p.name == "test_principle"
    assert "{response}" in p.critique_prompt
    assert "{response}" in p.revision_prompt
    assert "{critique}" in p.revision_prompt
    assert p.weight == 2.5


# ---------------------------------------------------------------------------
# Test 2: ConstitutionalPrinciple default weight
# ---------------------------------------------------------------------------

def test_constitutional_principle_default_weight():
    p = ConstitutionalPrinciple(
        name="default_weight",
        critique_prompt="Check {response}.",
        revision_prompt="Revise {response} with {critique}.",
    )
    assert p.weight == 1.0


# ---------------------------------------------------------------------------
# Test 3: HARMLESSNESS_PRINCIPLES has 3 entries
# ---------------------------------------------------------------------------

def test_harmlessness_principles_count():
    assert len(HARMLESSNESS_PRINCIPLES) == 3


# ---------------------------------------------------------------------------
# Test 4: HARMLESSNESS_PRINCIPLES names are unique and expected
# ---------------------------------------------------------------------------

def test_harmlessness_principles_names():
    names = {p.name for p in HARMLESSNESS_PRINCIPLES}
    assert "no_harm" in names
    assert "no_deception" in names
    assert "helpful" in names


# ---------------------------------------------------------------------------
# Test 5: sample_principles returns correct count
# ---------------------------------------------------------------------------

def test_sample_principles_count():
    sampled = sample_principles(HARMLESSNESS_PRINCIPLES, n=5, seed=0)
    assert len(sampled) == 5


# ---------------------------------------------------------------------------
# Test 6: sample_principles with seed is reproducible
# ---------------------------------------------------------------------------

def test_sample_principles_reproducible():
    s1 = sample_principles(HARMLESSNESS_PRINCIPLES, n=4, seed=99)
    s2 = sample_principles(HARMLESSNESS_PRINCIPLES, n=4, seed=99)
    assert [p.name for p in s1] == [p.name for p in s2]


# ---------------------------------------------------------------------------
# Test 7: sample_principles with different seeds may differ
# ---------------------------------------------------------------------------

def test_sample_principles_different_seeds_may_differ():
    s1 = sample_principles(HARMLESSNESS_PRINCIPLES, n=10, seed=1)
    s2 = sample_principles(HARMLESSNESS_PRINCIPLES, n=10, seed=2)
    assert all(isinstance(p, ConstitutionalPrinciple) for p in s1)
    assert all(isinstance(p, ConstitutionalPrinciple) for p in s2)


# ---------------------------------------------------------------------------
# Test 8: format_critique_prompt fills {response}
# ---------------------------------------------------------------------------

def test_format_critique_prompt_fills_response():
    principle = HARMLESSNESS_PRINCIPLES[0]  # no_harm
    result = format_critique_prompt(principle, response="This is a test response.")
    assert "This is a test response." in result
    assert "{response}" not in result


# ---------------------------------------------------------------------------
# Test 9: format_revision_prompt fills {response} and {critique}
# ---------------------------------------------------------------------------

def test_format_revision_prompt_fills_both():
    principle = HARMLESSNESS_PRINCIPLES[1]  # no_deception
    result = format_revision_prompt(
        principle,
        response="My response.",
        critique="This is deceptive.",
    )
    assert "My response." in result
    assert "This is deceptive." in result
    assert "{response}" not in result
    assert "{critique}" not in result


# ---------------------------------------------------------------------------
# Test 10: greedy_generate returns a string
# ---------------------------------------------------------------------------

def test_greedy_generate_returns_string(small_model):
    output = greedy_generate(
        model=small_model,
        tokenizer_encode=encode_fn,
        tokenizer_decode=decode_fn,
        prompt="Hello world",
        max_new_tokens=4,
    )
    assert isinstance(output, str)


# ---------------------------------------------------------------------------
# Test 11: ConstitutionalReviser.critique_and_revise returns tuple of 2 strings
# ---------------------------------------------------------------------------

def test_critique_and_revise_returns_two_strings(reviser):
    principle = HARMLESSNESS_PRINCIPLES[0]
    result = reviser.critique_and_revise("A sample response.", principle)
    assert isinstance(result, tuple)
    assert len(result) == 2
    critique, revision = result
    assert isinstance(critique, str)
    assert isinstance(revision, str)


# ---------------------------------------------------------------------------
# Test 12: ConstitutionalReviser.run_pipeline returns correct keys
# ---------------------------------------------------------------------------

def test_run_pipeline_correct_keys(reviser):
    result = reviser.run_pipeline("Initial response text.", n_rounds=1)
    assert set(result.keys()) == {"initial", "final", "revisions", "critiques", "n_rounds"}


# ---------------------------------------------------------------------------
# Test 13: run_pipeline n_rounds matches
# ---------------------------------------------------------------------------

def test_run_pipeline_n_rounds_matches(reviser):
    result = reviser.run_pipeline("Some response.", n_rounds=3)
    assert result["n_rounds"] == 3


# ---------------------------------------------------------------------------
# Test 14: run_pipeline revisions list length == n_rounds
# ---------------------------------------------------------------------------

def test_run_pipeline_revisions_length(reviser):
    n = 2
    result = reviser.run_pipeline("Another response.", n_rounds=n)
    assert len(result["revisions"]) == n
    assert len(result["critiques"]) == n


# ---------------------------------------------------------------------------
# Test 15: compute_revision_similarity identical strings -> 1.0
# ---------------------------------------------------------------------------

def test_compute_revision_similarity_identical():
    s = "This is a test sentence for similarity."
    assert compute_revision_similarity(s, s) == 1.0


# ---------------------------------------------------------------------------
# Test 16: compute_revision_similarity range [0, 1]
# ---------------------------------------------------------------------------

def test_compute_revision_similarity_range():
    sim = compute_revision_similarity("hello world", "goodbye moon")
    assert 0.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# Test 17: run_pipeline initial field preserved
# ---------------------------------------------------------------------------

def test_run_pipeline_initial_preserved(reviser):
    initial = "The very first response."
    result = reviser.run_pipeline(initial, n_rounds=1)
    assert result["initial"] == initial
