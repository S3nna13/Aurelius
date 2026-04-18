"""Tests for src/alignment/debate_framework.py"""

from __future__ import annotations

import math

import torch
import pytest

from src.alignment.debate_framework import (
    DebateConfig,
    DebateEvaluator,
    DebateResult,
    DebateRound,
    DebaterModel,
    JudgeModel,
    SelfPlayDebateTrainer,
)

# ---------------------------------------------------------------------------
# Shared tiny configuration
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 2
SEQ_LEN = 8
BATCH = 2
N_TURNS = 1
MAX_NEW = 4


def _make_debater() -> DebaterModel:
    return DebaterModel(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS)


def _make_judge() -> JudgeModel:
    return JudgeModel(d_model=D_MODEL, vocab_size=VOCAB_SIZE, n_layers=N_LAYERS)


def _make_question_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))


# ---------------------------------------------------------------------------
# DebaterModel tests
# ---------------------------------------------------------------------------


def test_debater_forward_output_shape() -> None:
    """DebaterModel forward should return [B, T, vocab_size]."""
    model = _make_debater()
    ids = _make_question_ids()
    logits = model(ids)
    assert logits.shape == (BATCH, SEQ_LEN, VOCAB_SIZE), (
        f"Expected {(BATCH, SEQ_LEN, VOCAB_SIZE)}, got {logits.shape}"
    )


def test_debater_forward_finite_values() -> None:
    """DebaterModel forward logits should all be finite."""
    model = _make_debater()
    ids = _make_question_ids()
    logits = model(ids)
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"


def test_debater_generate_argument_output_length() -> None:
    """generate_argument output should be longer than the input context."""
    model = _make_debater()
    ids = _make_question_ids()
    gen = model.generate_argument(ids, max_new=MAX_NEW)
    assert gen.shape[1] > ids.shape[1], (
        f"Generated length {gen.shape[1]} not greater than input length {ids.shape[1]}"
    )


def test_debater_generate_argument_exact_length() -> None:
    """generate_argument should add exactly max_new tokens."""
    model = _make_debater()
    ids = _make_question_ids()
    gen = model.generate_argument(ids, max_new=MAX_NEW)
    assert gen.shape[1] == SEQ_LEN + MAX_NEW, (
        f"Expected length {SEQ_LEN + MAX_NEW}, got {gen.shape[1]}"
    )


def test_debater_generate_argument_batch_preserved() -> None:
    """generate_argument should preserve the batch dimension."""
    model = _make_debater()
    ids = _make_question_ids()
    gen = model.generate_argument(ids, max_new=MAX_NEW)
    assert gen.shape[0] == BATCH


# ---------------------------------------------------------------------------
# JudgeModel tests
# ---------------------------------------------------------------------------


def test_judge_forward_output_shape() -> None:
    """JudgeModel forward should return [B, 2]."""
    judge = _make_judge()
    ids = _make_question_ids()
    logits = judge(ids)
    assert logits.shape == (BATCH, 2), (
        f"Expected {(BATCH, 2)}, got {logits.shape}"
    )


def test_judge_forward_finite_values() -> None:
    """JudgeModel forward logits should be finite."""
    judge = _make_judge()
    ids = _make_question_ids()
    logits = judge(ids)
    assert torch.isfinite(logits).all()


def test_judge_score_argument_range() -> None:
    """score_argument should return values strictly in (0, 1)."""
    judge = _make_judge()
    ids = _make_question_ids()
    scores = judge.score_argument(ids)
    assert scores.shape == (BATCH,)
    assert (scores > 0).all() and (scores < 1).all(), (
        f"Scores out of (0,1): {scores}"
    )


def test_judge_scores_sum_to_one() -> None:
    """support score + oppose score (1 - support) should equal 1 for each sample."""
    judge = _make_judge()
    ids = _make_question_ids()
    support_scores = judge.score_argument(ids)
    oppose_scores = 1.0 - support_scores
    sums = support_scores + oppose_scores
    assert torch.allclose(sums, torch.ones(BATCH), atol=1e-6), (
        f"Scores do not sum to 1: {sums}"
    )


# ---------------------------------------------------------------------------
# DebateRound tests
# ---------------------------------------------------------------------------


def test_debate_round_returns_debate_result() -> None:
    """run_round should return a DebateResult instance."""
    debater_a = _make_debater()
    debater_b = _make_debater()
    judge = _make_judge()
    debate = DebateRound(debater_a, debater_b, judge)
    question_ids = _make_question_ids()
    result = debate.run_round(question_ids, n_turns=N_TURNS, max_new=MAX_NEW)
    assert isinstance(result, DebateResult)


def test_debate_result_transcript_longer_than_question() -> None:
    """transcript_ids should be longer than the original question."""
    debater_a = _make_debater()
    debater_b = _make_debater()
    judge = _make_judge()
    debate = DebateRound(debater_a, debater_b, judge)
    question_ids = _make_question_ids()
    result = debate.run_round(question_ids, n_turns=N_TURNS, max_new=MAX_NEW)
    assert result.transcript_ids.shape[1] > question_ids.shape[1], (
        "Transcript is not longer than the question"
    )


def test_debate_result_scores_sum_to_one() -> None:
    """a_score + b_score should be approximately 1.0."""
    debater_a = _make_debater()
    debater_b = _make_debater()
    judge = _make_judge()
    debate = DebateRound(debater_a, debater_b, judge)
    question_ids = _make_question_ids()
    result = debate.run_round(question_ids, n_turns=N_TURNS, max_new=MAX_NEW)
    assert math.isclose(result.a_score + result.b_score, 1.0, abs_tol=1e-5), (
        f"a_score + b_score = {result.a_score + result.b_score} != 1.0"
    )


def test_debate_result_winner_valid_value() -> None:
    """winner should be exactly 'a', 'b', or 'tie'."""
    debater_a = _make_debater()
    debater_b = _make_debater()
    judge = _make_judge()
    debate = DebateRound(debater_a, debater_b, judge)
    question_ids = _make_question_ids()
    result = debate.run_round(question_ids, n_turns=N_TURNS, max_new=MAX_NEW)
    assert result.winner in {"a", "b", "tie"}, (
        f"Unexpected winner value: {result.winner!r}"
    )


def test_debate_result_a_score_in_range() -> None:
    """a_score should be in [0, 1]."""
    debater_a = _make_debater()
    debater_b = _make_debater()
    judge = _make_judge()
    debate = DebateRound(debater_a, debater_b, judge)
    question_ids = _make_question_ids()
    result = debate.run_round(question_ids, n_turns=N_TURNS, max_new=MAX_NEW)
    assert 0.0 <= result.a_score <= 1.0, f"a_score out of range: {result.a_score}"


# ---------------------------------------------------------------------------
# SelfPlayDebateTrainer tests
# ---------------------------------------------------------------------------


def test_trainer_judge_step_returns_finite_scalar() -> None:
    """judge_step should return a finite scalar loss."""
    debater_a = _make_debater()
    debater_b = _make_debater()
    judge = _make_judge()
    trainer = SelfPlayDebateTrainer(
        debater_a, debater_b, judge, lr_debaters=1e-4, lr_judge=1e-3
    )
    ids = _make_question_ids()
    labels = torch.ones(BATCH, dtype=torch.float32)
    loss = trainer.judge_step(ids, labels)
    assert loss.ndim == 0, "judge_step loss should be scalar"
    assert torch.isfinite(loss), f"judge_step loss is not finite: {loss}"


def test_trainer_debater_step_returns_finite_losses() -> None:
    """debater_step should return two finite scalar losses."""
    debater_a = _make_debater()
    debater_b = _make_debater()
    judge = _make_judge()
    trainer = SelfPlayDebateTrainer(
        debater_a, debater_b, judge, lr_debaters=1e-4, lr_judge=1e-3
    )
    question_ids = _make_question_ids()
    reward_signal = torch.rand(BATCH)
    loss_a, loss_b = trainer.debater_step(question_ids, reward_signal, max_new=MAX_NEW)
    assert torch.isfinite(loss_a), f"loss_a is not finite: {loss_a}"
    assert torch.isfinite(loss_b), f"loss_b is not finite: {loss_b}"
    assert loss_a.ndim == 0
    assert loss_b.ndim == 0


def test_trainer_self_play_step_returns_expected_keys() -> None:
    """self_play_step should return a dict with the required keys."""
    debater_a = _make_debater()
    debater_b = _make_debater()
    judge = _make_judge()
    trainer = SelfPlayDebateTrainer(
        debater_a, debater_b, judge, lr_debaters=1e-4, lr_judge=1e-3
    )
    question_ids = _make_question_ids()
    result = trainer.self_play_step(question_ids, n_turns=N_TURNS, max_new=MAX_NEW)
    for key in ("judge_loss", "loss_a", "loss_b", "winner"):
        assert key in result, f"Missing key in self_play_step result: {key!r}"


def test_trainer_self_play_step_losses_finite() -> None:
    """All scalar losses in self_play_step result should be finite."""
    debater_a = _make_debater()
    debater_b = _make_debater()
    judge = _make_judge()
    trainer = SelfPlayDebateTrainer(
        debater_a, debater_b, judge, lr_debaters=1e-4, lr_judge=1e-3
    )
    question_ids = _make_question_ids()
    result = trainer.self_play_step(question_ids, n_turns=N_TURNS, max_new=MAX_NEW)
    for key in ("judge_loss", "loss_a", "loss_b"):
        assert torch.isfinite(result[key]), f"{key} is not finite: {result[key]}"


# ---------------------------------------------------------------------------
# DebateEvaluator tests
# ---------------------------------------------------------------------------


def _make_result(winner: str, a_score: float) -> DebateResult:
    b_score = 1.0 - a_score
    transcript = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN + MAX_NEW))
    return DebateResult(
        transcript_ids=transcript,
        a_score=a_score,
        b_score=b_score,
        winner=winner,
    )


def test_evaluator_win_rate_sums_to_one() -> None:
    """win_rate fractions should sum to 1.0."""
    ev = DebateEvaluator()
    results = [
        _make_result("a", 0.7),
        _make_result("b", 0.3),
        _make_result("a", 0.6),
        _make_result("tie", 0.5),
    ]
    rates = ev.win_rate(results)
    total = rates["a_wins"] + rates["b_wins"] + rates["ties"]
    assert math.isclose(total, 1.0, abs_tol=1e-6), f"win_rate fractions sum to {total}"


def test_evaluator_win_rate_keys_present() -> None:
    """win_rate should return dict with keys a_wins, b_wins, ties."""
    ev = DebateEvaluator()
    results = [_make_result("a", 0.8)]
    rates = ev.win_rate(results)
    assert "a_wins" in rates and "b_wins" in rates and "ties" in rates


def test_evaluator_judge_confidence_in_range() -> None:
    """judge_confidence should be in [0, 1]."""
    ev = DebateEvaluator()
    results = [
        _make_result("a", 0.9),
        _make_result("b", 0.1),
        _make_result("tie", 0.5),
    ]
    conf = ev.judge_confidence(results)
    assert 0.0 <= conf <= 1.0, f"judge_confidence out of [0,1]: {conf}"


def test_evaluator_judge_confidence_certain() -> None:
    """judge_confidence = 1.0 when a_score is always 0.0 or 1.0."""
    ev = DebateEvaluator()
    results = [_make_result("a", 1.0), _make_result("b", 0.0)]
    conf = ev.judge_confidence(results)
    assert math.isclose(conf, 1.0, abs_tol=1e-6), f"Expected 1.0, got {conf}"


def test_evaluator_judge_confidence_random() -> None:
    """judge_confidence = 0.0 when a_score is always exactly 0.5."""
    ev = DebateEvaluator()
    results = [_make_result("tie", 0.5), _make_result("tie", 0.5)]
    conf = ev.judge_confidence(results)
    assert math.isclose(conf, 0.0, abs_tol=1e-6), f"Expected 0.0, got {conf}"


def test_evaluator_argument_diversity_non_negative() -> None:
    """argument_diversity should be >= 0."""
    ev = DebateEvaluator()
    results = [_make_result("a", 0.7), _make_result("b", 0.4)]
    div = ev.argument_diversity(results)
    assert div >= 0.0, f"argument_diversity is negative: {div}"


def test_evaluator_argument_diversity_identical_transcripts() -> None:
    """argument_diversity = 0 for identical transcripts."""
    ev = DebateEvaluator()
    transcript = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN))
    r1 = DebateResult(transcript_ids=transcript.clone(), a_score=0.6, b_score=0.4, winner="a")
    r2 = DebateResult(transcript_ids=transcript.clone(), a_score=0.6, b_score=0.4, winner="a")
    div = ev.argument_diversity([r1, r2])
    assert div == 0.0, f"Expected 0.0 diversity for identical transcripts, got {div}"


def test_evaluator_argument_diversity_single_result() -> None:
    """argument_diversity with a single result should return 0.0."""
    ev = DebateEvaluator()
    results = [_make_result("a", 0.8)]
    div = ev.argument_diversity(results)
    assert div == 0.0


# ---------------------------------------------------------------------------
# DebateConfig tests
# ---------------------------------------------------------------------------


def test_debate_config_defaults() -> None:
    """DebateConfig should have the correct default values."""
    cfg = DebateConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.n_turns == 2
    assert cfg.max_new_tokens == 8
    assert math.isclose(cfg.lr_debaters, 1e-4)
    assert math.isclose(cfg.lr_judge, 1e-3)


def test_debate_config_custom_values() -> None:
    """DebateConfig should accept and store custom values."""
    cfg = DebateConfig(d_model=16, vocab_size=16, n_layers=1)
    assert cfg.d_model == 16
    assert cfg.vocab_size == 16
    assert cfg.n_layers == 1
