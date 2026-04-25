"""Tests for BestOfNOptimizer and ThinkingModeParser."""
from __future__ import annotations

import pytest
from src.reasoning.best_of_n_chain_optimizer import (
    ThinkingBlock,
    ParsedReasoning,
    ThinkingModeParser,
    CandidateResult,
    BestOfNOptimizer,
    BEST_OF_N_REGISTRY,
)


# ---------------------------------------------------------------------------
# ThinkingModeParser
# ---------------------------------------------------------------------------

def test_parser_no_think_blocks_returns_full_text_as_answer():
    parser = ThinkingModeParser()
    result = parser.parse("plain answer without thinking")
    assert result.thinking_blocks == []
    assert result.final_answer == "plain answer without thinking"


def test_parser_no_think_blocks_raw_preserved():
    parser = ThinkingModeParser()
    text = "hello world"
    result = parser.parse(text)
    assert result.raw == text


def test_parser_single_block_extracted():
    parser = ThinkingModeParser()
    result = parser.parse("<think>step one</think>\nfinal")
    assert len(result.thinking_blocks) == 1
    assert result.thinking_blocks[0].content == "step one"


def test_parser_single_block_final_answer():
    parser = ThinkingModeParser()
    result = parser.parse("<think>step one</think>\nfinal answer here")
    assert result.final_answer == "final answer here"


def test_parser_multiple_blocks_count():
    parser = ThinkingModeParser()
    text = "<think>a</think><think>b</think><think>c</think>done"
    result = parser.parse(text)
    assert len(result.thinking_blocks) == 3


def test_parser_multiple_blocks_contents():
    parser = ThinkingModeParser()
    text = "<think>alpha</think><think>beta</think>answer"
    result = parser.parse(text)
    assert result.thinking_blocks[0].content == "alpha"
    assert result.thinking_blocks[1].content == "beta"


def test_parser_multiple_blocks_final_answer():
    parser = ThinkingModeParser()
    result = parser.parse("<think>x</think><think>y</think>the end")
    assert result.final_answer == "the end"


def test_parser_nested_like_text_not_recursed():
    parser = ThinkingModeParser()
    # THINK_RE is non-greedy; outer tags captured, inner literal text preserved
    result = parser.parse("<think>outer <think>inner</think></think>leftover")
    # non-greedy stops at first </think>, so "outer <think>inner" is one block
    assert result.thinking_blocks[0].content == "outer <think>inner"


def test_parser_multiline_think_block():
    parser = ThinkingModeParser()
    text = "<think>\nline1\nline2\n</think>\nanswer"
    result = parser.parse(text)
    assert "line1" in result.thinking_blocks[0].content
    assert result.final_answer == "answer"


def test_parser_start_end_pos_in_think_block():
    parser = ThinkingModeParser()
    text = "<think>hi</think>rest"
    result = parser.parse(text)
    block = result.thinking_blocks[0]
    assert block.start_pos == 0
    assert block.end_pos == len("<think>hi</think>")


def test_strip_thinking_removes_all_blocks():
    parser = ThinkingModeParser()
    text = "<think>a</think>clean<think>b</think>"
    assert parser.strip_thinking(text) == "clean"


def test_strip_thinking_no_blocks_returns_original():
    parser = ThinkingModeParser()
    assert parser.strip_thinking("no blocks here") == "no blocks here"


# ---------------------------------------------------------------------------
# BestOfNOptimizer
# ---------------------------------------------------------------------------

def test_generate_candidates_default_count():
    opt = BestOfNOptimizer(n=5)
    candidates = opt.generate_candidates("test prompt")
    assert len(candidates) == 5


def test_generate_candidates_override_count():
    opt = BestOfNOptimizer(n=5)
    candidates = opt.generate_candidates("prompt", n=3)
    assert len(candidates) == 3


def test_generate_candidates_are_strings():
    opt = BestOfNOptimizer(n=4)
    for c in opt.generate_candidates("q"):
        assert isinstance(c, str)


def test_default_scorer_higher_answer_scores_higher():
    opt = BestOfNOptimizer()
    short = ParsedReasoning(thinking_blocks=[], final_answer="short", raw="short")
    long_ = ParsedReasoning(
        thinking_blocks=[],
        final_answer="a much longer final answer text here",
        raw="a much longer final answer text here",
    )
    assert opt.score(long_) > opt.score(short)


def test_default_scorer_penalises_more_think_blocks():
    opt = BestOfNOptimizer()
    block = ThinkingBlock(content="x", start_pos=0, end_pos=10)
    fewer = ParsedReasoning(thinking_blocks=[block], final_answer="answer", raw="")
    more = ParsedReasoning(thinking_blocks=[block, block, block], final_answer="answer", raw="")
    assert opt.score(fewer) > opt.score(more)


def test_custom_scorer_used():
    custom_calls = []

    def custom(r: ParsedReasoning) -> float:
        custom_calls.append(True)
        return 99.0

    opt = BestOfNOptimizer(n=2, scorer=custom)
    candidates = opt.generate_candidates("p")
    opt.select_best(candidates)
    assert len(custom_calls) == 2


def test_select_best_picks_highest_score():
    opt = BestOfNOptimizer(scorer=lambda r: len(r.final_answer))
    candidates = [
        "<think>t</think>short",
        "<think>t</think>a longer final answer wins",
        "<think>t</think>medium answer here",
    ]
    best = opt.select_best(candidates)
    assert best.candidate_id == 1


def test_select_best_sets_selected_flag():
    opt = BestOfNOptimizer(n=3)
    best = opt.select_best(opt.generate_candidates("q"))
    assert best.selected is True


def test_select_best_returns_candidate_result():
    opt = BestOfNOptimizer(n=2)
    result = opt.select_best(opt.generate_candidates("q"))
    assert isinstance(result, CandidateResult)


def test_optimize_end_to_end_returns_candidate_result():
    opt = BestOfNOptimizer(n=4)
    result = opt.optimize("what is 2 + 2?")
    assert isinstance(result, CandidateResult)
    assert result.selected is True


def test_optimize_candidate_id_in_range():
    n = 6
    opt = BestOfNOptimizer(n=n)
    result = opt.optimize("prompt")
    assert 0 <= result.candidate_id < n


def test_candidate_result_score_is_float():
    opt = BestOfNOptimizer(n=2)
    result = opt.optimize("q")
    assert isinstance(result.score, float)


def test_candidate_result_default_selected_false():
    parser = ThinkingModeParser()
    parsed = parser.parse("hello")
    cr = CandidateResult(candidate_id=0, reasoning=parsed, score=1.0)
    assert cr.selected is False


def test_registry_contains_default():
    assert "default" in BEST_OF_N_REGISTRY
    assert BEST_OF_N_REGISTRY["default"] is BestOfNOptimizer
