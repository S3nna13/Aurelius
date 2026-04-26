"""Tests for debate-based evaluation (Irving et al. 2018 'AI Safety via Debate')."""

from __future__ import annotations

import pytest

from src.eval.debate_eval import (
    DebateArgument,
    DebateConfig,
    DebateEvaluator,
    DebateTranscript,
    aggregate_debate_results,
    build_debater_prompt,
    build_judge_prompt,
    extract_winner,
)

# ---------------------------------------------------------------------------
# Helper: mock generate_fn
# ---------------------------------------------------------------------------


def _make_generate_fn(judge_response: str = "Winner: A") -> callable:
    """Return a generate_fn that produces judge_response when called with a
    judge prompt (detected by the presence of 'Your verdict:') and generic
    argument text otherwise."""

    def generate_fn(prompt: str) -> str:
        if "Your verdict:" in prompt:
            return judge_response
        return "This is a compelling argument based on available evidence."

    return generate_fn


# ---------------------------------------------------------------------------
# extract_winner tests
# ---------------------------------------------------------------------------


def test_extract_winner_a():
    assert extract_winner("Winner: A") == "A"


def test_extract_winner_b():
    assert extract_winner("Winner: B\nSome reasoning here.") == "B"


def test_extract_winner_tie_lowercase():
    assert extract_winner("winner: tie and here is why") == "tie"


def test_extract_winner_garbage_returns_none():
    assert extract_winner("I cannot decide who won.") is None


# ---------------------------------------------------------------------------
# build_debater_prompt tests
# ---------------------------------------------------------------------------


def test_build_debater_prompt_contains_question():
    transcript = DebateTranscript(
        question="Is the Earth round?",
        position_a="The Earth is round.",
        position_b="The Earth is flat.",
    )
    prompt = build_debater_prompt("Is the Earth round?", "The Earth is round.", transcript, 1)
    assert "Is the Earth round?" in prompt


def test_build_debater_prompt_contains_position():
    transcript = DebateTranscript(
        question="Is the Earth round?",
        position_a="The Earth is round.",
        position_b="The Earth is flat.",
    )
    prompt = build_debater_prompt("Is the Earth round?", "The Earth is round.", transcript, 1)
    assert "The Earth is round." in prompt


# ---------------------------------------------------------------------------
# build_judge_prompt tests
# ---------------------------------------------------------------------------


def test_build_judge_prompt_contains_arguments():
    transcript = DebateTranscript(
        question="Is the sky blue?",
        position_a="The sky is blue.",
        position_b="The sky is not blue.",
        arguments=[
            DebateArgument(
                debater_id="A",
                position="for",
                content="Light scattering makes the sky blue.",
                round_number=1,
            ),
        ],
    )
    prompt = build_judge_prompt(transcript)
    assert "Light scattering makes the sky blue." in prompt


# ---------------------------------------------------------------------------
# DebateTranscript dataclass tests
# ---------------------------------------------------------------------------


def test_debate_transcript_creation():
    transcript = DebateTranscript(
        question="Q",
        position_a="For",
        position_b="Against",
        arguments=[],
        winner="A",
        judge_reasoning="Debater A was more persuasive.",
    )
    assert transcript.question == "Q"
    assert transcript.position_a == "For"
    assert transcript.position_b == "Against"
    assert transcript.winner == "A"
    assert transcript.judge_reasoning == "Debater A was more persuasive."


# ---------------------------------------------------------------------------
# DebateConfig defaults test
# ---------------------------------------------------------------------------


def test_debate_config_defaults():
    config = DebateConfig()
    assert config.n_rounds == 2
    assert config.max_argument_length == 500
    assert config.judge_temperature == 0.0


# ---------------------------------------------------------------------------
# DebateEvaluator.run_debate tests
# ---------------------------------------------------------------------------


def test_run_debate_returns_transcript():
    evaluator = DebateEvaluator(_make_generate_fn(), DebateConfig(n_rounds=1))
    result = evaluator.run_debate("Is 2+2=4?", "Yes it is.", "No it is not.")
    assert isinstance(result, DebateTranscript)


def test_run_debate_argument_count_matches_rounds():
    n_rounds = 3
    evaluator = DebateEvaluator(_make_generate_fn(), DebateConfig(n_rounds=n_rounds))
    result = evaluator.run_debate("Test question?", "For", "Against")
    # Each round produces 2 arguments (one per debater)
    assert len(result.arguments) == n_rounds * 2


def test_run_debate_winner_is_set():
    evaluator = DebateEvaluator(_make_generate_fn("Winner: A"), DebateConfig(n_rounds=1))
    result = evaluator.run_debate("Question?", "For", "Against")
    assert result.winner is not None


def test_run_debate_argument_round_numbers():
    evaluator = DebateEvaluator(_make_generate_fn(), DebateConfig(n_rounds=2))
    result = evaluator.run_debate("Q?", "For", "Against")
    # First two arguments should be round 1, next two round 2
    assert result.arguments[0].round_number == 1
    assert result.arguments[1].round_number == 1
    assert result.arguments[2].round_number == 2
    assert result.arguments[3].round_number == 2


# ---------------------------------------------------------------------------
# DebateEvaluator.evaluate_claim tests
# ---------------------------------------------------------------------------


def test_evaluate_claim_returns_transcript():
    evaluator = DebateEvaluator(_make_generate_fn(), DebateConfig(n_rounds=1))
    result = evaluator.evaluate_claim("The sun rises in the east.")
    assert isinstance(result, DebateTranscript)


def test_evaluate_claim_question_contains_claim():
    evaluator = DebateEvaluator(_make_generate_fn(), DebateConfig(n_rounds=1))
    claim = "Gravity pulls objects downward."
    result = evaluator.evaluate_claim(claim)
    assert claim in result.question


# ---------------------------------------------------------------------------
# DebateEvaluator.batch_evaluate tests
# ---------------------------------------------------------------------------


def test_batch_evaluate_returns_correct_length():
    evaluator = DebateEvaluator(_make_generate_fn(), DebateConfig(n_rounds=1))
    claims = ["Claim one.", "Claim two.", "Claim three."]
    results = evaluator.batch_evaluate(claims)
    assert len(results) == len(claims)


def test_batch_evaluate_empty_list():
    evaluator = DebateEvaluator(_make_generate_fn(), DebateConfig(n_rounds=1))
    results = evaluator.batch_evaluate([])
    assert results == []


# ---------------------------------------------------------------------------
# aggregate_debate_results tests
# ---------------------------------------------------------------------------


def _make_transcript(winner: str) -> DebateTranscript:
    return DebateTranscript(
        question="Q",
        position_a="For",
        position_b="Against",
        winner=winner,
    )


def test_aggregate_has_required_keys():
    transcripts = [_make_transcript("A"), _make_transcript("B")]
    result = aggregate_debate_results(transcripts)
    assert "win_rate_a" in result
    assert "win_rate_b" in result
    assert "tie_rate" in result
    assert "n_debates" in result


def test_aggregate_rates_sum_to_one():
    transcripts = [_make_transcript("A"), _make_transcript("B"), _make_transcript("tie")]
    result = aggregate_debate_results(transcripts)
    total = result["win_rate_a"] + result["win_rate_b"] + result["tie_rate"]
    assert total == pytest.approx(1.0)


def test_aggregate_n_debates_correct():
    transcripts = [_make_transcript("A")] * 5
    result = aggregate_debate_results(transcripts)
    assert result["n_debates"] == 5


def test_aggregate_all_a_wins():
    transcripts = [_make_transcript("A")] * 4
    result = aggregate_debate_results(transcripts)
    assert result["win_rate_a"] == pytest.approx(1.0)
    assert result["win_rate_b"] == pytest.approx(0.0)
    assert result["tie_rate"] == pytest.approx(0.0)
