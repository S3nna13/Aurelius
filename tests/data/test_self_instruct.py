"""Tests for src/data/self_instruct.py — Self-Instruct pipeline."""

from __future__ import annotations

import pytest

from src.data.self_instruct import (
    Instruction,
    InstructionPool,
    SelfInstructPipeline,
    build_instruction_prompt,
    build_instance_prompt,
    filter_instruction,
    is_classification_task,
    make_seed_instructions,
    parse_generated_instructions,
    rouge_l_similarity,
)


# ---------------------------------------------------------------------------
# Mock generate_fn
# ---------------------------------------------------------------------------

def mock_generate_fn(prompt: str) -> str:
    """Returns a numbered list of three instructions."""
    return (
        "1. Write a poem about cats.\n"
        "2. Summarize this document in one paragraph.\n"
        "3. Translate the following text to French."
    )


# ---------------------------------------------------------------------------
# ROUGE-L tests
# ---------------------------------------------------------------------------

def test_rouge_l_identical():
    """rouge_l_similarity of identical strings should be ~1.0."""
    score = rouge_l_similarity("hello world", "hello world")
    assert abs(score - 1.0) < 1e-6


def test_rouge_l_dissimilar():
    """rouge_l_similarity of completely different strings should be < 0.5."""
    score = rouge_l_similarity("hello world", "goodbye moon")
    assert score < 0.5


def test_rouge_l_empty_strings():
    """rouge_l_similarity with both empty strings should not crash and return 1.0."""
    score = rouge_l_similarity("", "")
    assert score == 1.0


def test_rouge_l_one_empty():
    """rouge_l_similarity with one empty string should return 0.0."""
    assert rouge_l_similarity("", "hello") == 0.0
    assert rouge_l_similarity("hello", "") == 0.0


# ---------------------------------------------------------------------------
# filter_instruction tests
# ---------------------------------------------------------------------------

def test_filter_passes_novel_instruction():
    """A clearly novel, well-formed instruction should pass all filters."""
    existing = ["Write a poem about spring.", "Summarize this article."]
    novel = "Explain the theory of relativity in simple language."
    assert filter_instruction(novel, existing) is True


def test_filter_rejects_too_short():
    """An instruction shorter than min_length characters should be rejected."""
    assert filter_instruction("Hi", []) is False


def test_filter_rejects_similar_instruction():
    """An instruction with ROUGE-L >= 0.7 vs an existing one should be rejected."""
    existing = ["Write a poem about the ocean."]
    # Very close to the existing instruction — should be rejected
    duplicate = "Write a poem about the ocean."
    assert filter_instruction(duplicate, existing) is False


def test_filter_rejects_too_long():
    """An instruction longer than max_length characters should be rejected."""
    long_instruction = "x " * 300  # 600 chars
    assert filter_instruction(long_instruction, []) is False


# ---------------------------------------------------------------------------
# is_classification_task tests
# ---------------------------------------------------------------------------

def test_is_classification_positive():
    """Sentiment classification question should be detected as classification."""
    assert is_classification_task("Is this review positive or negative?") is True


def test_is_classification_negative():
    """A creative writing instruction should NOT be classified as classification."""
    assert is_classification_task("Write a poem about summer") is False


def test_is_classification_classify_keyword():
    """Instructions containing 'classify' should be detected as classification."""
    assert is_classification_task("Classify the following text into one of three categories.") is True


# ---------------------------------------------------------------------------
# parse_generated_instructions tests
# ---------------------------------------------------------------------------

def test_parse_numbered_list():
    """parse_generated_instructions should correctly split a numbered list."""
    raw = (
        "1. Write a poem about cats.\n"
        "2. Summarize this document.\n"
        "3. Translate to French."
    )
    result = parse_generated_instructions(raw)
    assert len(result) == 3
    assert result[0] == "Write a poem about cats."
    assert result[1] == "Summarize this document."
    assert result[2] == "Translate to French."


def test_parse_empty_string():
    """parse_generated_instructions on empty string should return empty list."""
    result = parse_generated_instructions("")
    assert result == []


# ---------------------------------------------------------------------------
# build_instruction_prompt tests
# ---------------------------------------------------------------------------

def test_build_instruction_prompt_nonempty():
    """build_instruction_prompt should return a non-empty string containing instructions."""
    seeds = make_seed_instructions()[:8]
    prompt = build_instruction_prompt(seeds, n_examples=8)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    # Should contain at least one of the seed instruction texts
    assert any(seed.instruction in prompt for seed in seeds)


def test_build_instruction_prompt_has_numbering():
    """build_instruction_prompt should include numbered examples."""
    seeds = make_seed_instructions()[:3]
    prompt = build_instruction_prompt(seeds, n_examples=3)
    assert "1." in prompt
    assert "2." in prompt


def test_build_instance_prompt_nonempty():
    """build_instance_prompt should return a non-empty string."""
    instr = Instruction(instruction="Write a haiku about the moon.", source="seed")
    prompt = build_instance_prompt(instr)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert "haiku" in prompt


# ---------------------------------------------------------------------------
# InstructionPool tests
# ---------------------------------------------------------------------------

def test_pool_deduplication():
    """InstructionPool.add() should reject instructions too similar to existing ones."""
    seeds = make_seed_instructions()
    pool = InstructionPool(seeds, rouge_threshold=0.7)
    initial_size = len(pool)

    # Attempt to add something nearly identical to an existing seed
    duplicate = Instruction(instruction="Write a short poem about the changing seasons.", source="generated")
    added = pool.add(duplicate)
    assert added is False
    assert len(pool) == initial_size


def test_pool_add_novel():
    """InstructionPool.add() should accept a clearly novel instruction."""
    seeds = make_seed_instructions()
    pool = InstructionPool(seeds, rouge_threshold=0.7)
    initial_size = len(pool)

    novel = Instruction(
        instruction="Describe the mating rituals of deep-sea anglerfish in scientific detail.",
        source="generated",
    )
    added = pool.add(novel)
    assert added is True
    assert len(pool) == initial_size + 1


def test_pool_sample_returns_n():
    """InstructionPool.sample() should return exactly n items when pool is large enough."""
    seeds = make_seed_instructions()  # 20 seeds
    pool = InstructionPool(seeds)
    sampled = pool.sample(n=8)
    assert len(sampled) == 8


def test_pool_sample_does_not_exceed_pool_size():
    """InstructionPool.sample() with n > pool size should return all items."""
    small_seeds = make_seed_instructions()[:3]
    pool = InstructionPool(small_seeds)
    sampled = pool.sample(n=10)
    assert len(sampled) == 3


def test_pool_len():
    """InstructionPool.__len__ should return the correct count."""
    seeds = make_seed_instructions()
    pool = InstructionPool(seeds)
    assert len(pool) == len(seeds)


# ---------------------------------------------------------------------------
# SelfInstructPipeline tests
# ---------------------------------------------------------------------------

def test_pipeline_run_returns_instructions():
    """SelfInstructPipeline.run() should return a list of Instruction objects."""
    seeds = make_seed_instructions()
    pipeline = SelfInstructPipeline(
        seed_instructions=seeds,
        generate_fn=mock_generate_fn,
        rouge_threshold=0.7,
        n_few_shot=8,
        max_instructions=50,
    )
    results = pipeline.run(n_iterations=2, n_per_iter=5)
    assert isinstance(results, list)
    # All returned items must be Instruction instances
    for item in results:
        assert isinstance(item, Instruction)


def test_pipeline_export_sft_dataset():
    """SelfInstructPipeline.export_sft_dataset() should return dicts with correct keys."""
    seeds = make_seed_instructions()
    pipeline = SelfInstructPipeline(
        seed_instructions=seeds,
        generate_fn=mock_generate_fn,
        rouge_threshold=0.7,
        n_few_shot=8,
        max_instructions=50,
    )
    pipeline.run(n_iterations=1, n_per_iter=3)
    dataset = pipeline.export_sft_dataset()

    assert isinstance(dataset, list)
    assert len(dataset) > 0
    for record in dataset:
        assert "instruction" in record
        assert "input" in record
        assert "output" in record
        assert isinstance(record["instruction"], str)
        assert isinstance(record["input"], str)
        assert isinstance(record["output"], str)


def test_pipeline_pool_size_includes_seeds():
    """pipeline.pool_size should start at the seed count."""
    seeds = make_seed_instructions()
    pipeline = SelfInstructPipeline(
        seed_instructions=seeds,
        generate_fn=mock_generate_fn,
    )
    assert pipeline.pool_size == len(seeds)


def test_pipeline_generated_source_label():
    """All instructions added by the pipeline should have source='generated'."""
    seeds = make_seed_instructions()
    pipeline = SelfInstructPipeline(
        seed_instructions=seeds,
        generate_fn=mock_generate_fn,
        rouge_threshold=0.7,
        max_instructions=50,
    )
    generated = pipeline.run(n_iterations=2, n_per_iter=5)
    for instr in generated:
        assert instr.source == "generated"


# ---------------------------------------------------------------------------
# make_seed_instructions tests
# ---------------------------------------------------------------------------

def test_make_seed_instructions_count():
    """make_seed_instructions should return exactly 20 seeds."""
    seeds = make_seed_instructions()
    assert len(seeds) == 20


def test_make_seed_instructions_all_seed_source():
    """All seed instructions should have source='seed'."""
    seeds = make_seed_instructions()
    for s in seeds:
        assert s.source == "seed"
