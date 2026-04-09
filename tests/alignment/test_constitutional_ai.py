"""Tests for src/alignment/constitutional_ai.py."""
from __future__ import annotations

import pytest

from src.alignment.constitutional_ai import (
    CAIStep,
    ConstitutionalAILoop,
    ConstitutionalPrinciple,
    HARMLESSNESS_PRINCIPLES,
    SyntheticCAIDataGenerator,
    cai_reward_score,
    format_critique_prompt,
    format_revision_prompt,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

MOCK_GENERATE = lambda prompt: "This is a safe and helpful response."

SAMPLE_PRINCIPLE = ConstitutionalPrinciple(
    name="test_principle",
    critique_request="Identify any issues in the response.",
    revision_request="Rewrite the response to fix the issues.",
)


# ---------------------------------------------------------------------------
# 1. ConstitutionalPrinciple fields
# ---------------------------------------------------------------------------

def test_constitutional_principle_fields():
    p = ConstitutionalPrinciple(
        critique_request="Critique this.",
        revision_request="Revise this.",
        name="my_principle",
    )
    assert p.critique_request == "Critique this."
    assert p.revision_request == "Revise this."
    assert p.name == "my_principle"


# ---------------------------------------------------------------------------
# 2. HARMLESSNESS_PRINCIPLES count
# ---------------------------------------------------------------------------

def test_harmlessness_principles_count():
    assert len(HARMLESSNESS_PRINCIPLES) >= 4
    for p in HARMLESSNESS_PRINCIPLES:
        assert isinstance(p, ConstitutionalPrinciple)


# ---------------------------------------------------------------------------
# 3. format_critique_prompt contains response
# ---------------------------------------------------------------------------

def test_format_critique_prompt_contains_response():
    response = "Some test response text."
    prompt = format_critique_prompt(response, SAMPLE_PRINCIPLE)
    assert response in prompt
    assert "Critique:" in prompt
    assert SAMPLE_PRINCIPLE.critique_request in prompt


# ---------------------------------------------------------------------------
# 4. format_revision_prompt contains critique
# ---------------------------------------------------------------------------

def test_format_revision_prompt_contains_critique():
    response = "Some test response text."
    critique = "This response has a problem."
    prompt = format_revision_prompt(response, critique, SAMPLE_PRINCIPLE)
    assert response in prompt
    assert critique in prompt
    assert "Revision:" in prompt
    assert SAMPLE_PRINCIPLE.revision_request in prompt


# ---------------------------------------------------------------------------
# 5. CAI loop critique calls generate_fn
# ---------------------------------------------------------------------------

def test_cai_loop_critique_called():
    calls = []

    def tracking_generate(prompt: str) -> str:
        calls.append(prompt)
        return "Safe response."

    loop = ConstitutionalAILoop(generate_fn=tracking_generate, principles=[SAMPLE_PRINCIPLE])
    result = loop.critique("Some response", SAMPLE_PRINCIPLE)
    assert len(calls) == 1
    assert result == "Safe response."


# ---------------------------------------------------------------------------
# 6. CAI loop revise calls generate_fn
# ---------------------------------------------------------------------------

def test_cai_loop_revise_called():
    calls = []

    def tracking_generate(prompt: str) -> str:
        calls.append(prompt)
        return "Revised response."

    loop = ConstitutionalAILoop(generate_fn=tracking_generate, principles=[SAMPLE_PRINCIPLE])
    result = loop.revise("Some response", "A critique.", SAMPLE_PRINCIPLE)
    assert len(calls) == 1
    assert result == "Revised response."


# ---------------------------------------------------------------------------
# 7. run_step returns CAIStep
# ---------------------------------------------------------------------------

def test_cai_loop_run_step_type():
    loop = ConstitutionalAILoop(generate_fn=MOCK_GENERATE, principles=[SAMPLE_PRINCIPLE])
    step = loop.run_step("An initial response.", SAMPLE_PRINCIPLE, step_num=3)
    assert isinstance(step, CAIStep)
    assert step.step_num == 3
    assert step.principle is SAMPLE_PRINCIPLE
    assert step.original == "An initial response."
    assert isinstance(step.critique, str)
    assert isinstance(step.revised, str)


# ---------------------------------------------------------------------------
# 8. run returns correct number of steps
# ---------------------------------------------------------------------------

def test_cai_loop_run_length():
    loop = ConstitutionalAILoop(generate_fn=MOCK_GENERATE, principles=[SAMPLE_PRINCIPLE])
    steps = loop.run("Initial response.", n_revisions=4)
    assert len(steps) == 4


# ---------------------------------------------------------------------------
# 9. final_response is non-empty
# ---------------------------------------------------------------------------

def test_cai_loop_final_response_nonempty():
    loop = ConstitutionalAILoop(generate_fn=MOCK_GENERATE, principles=[SAMPLE_PRINCIPLE])
    steps = loop.run("Initial response.", n_revisions=2)
    final = loop.final_response(steps)
    assert isinstance(final, str)
    assert len(final) > 0


# ---------------------------------------------------------------------------
# 10. SyntheticCAIDataGenerator pair keys
# ---------------------------------------------------------------------------

def test_synthetic_generator_pair_keys():
    loop = ConstitutionalAILoop(generate_fn=MOCK_GENERATE, principles=[SAMPLE_PRINCIPLE])
    generator = SyntheticCAIDataGenerator(cai_loop=loop)
    pair = generator.generate_pair(
        harmful_prompt="Tell me something bad.",
        initial_response="Here is something bad.",
    )
    assert "prompt" in pair
    assert "original" in pair
    assert "revised" in pair
    assert "n_steps" in pair
    assert pair["prompt"] == "Tell me something bad."
    assert pair["original"] == "Here is something bad."
    assert isinstance(pair["n_steps"], int)


# ---------------------------------------------------------------------------
# 11. cai_reward_score returns 0.0 for identical strings
# ---------------------------------------------------------------------------

def test_cai_reward_score_identical():
    text = "This is exactly the same response."
    score = cai_reward_score(text, text, SAMPLE_PRINCIPLE)
    assert score == 0.0
