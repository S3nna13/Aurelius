"""Tests for few_shot_prompting module."""

import pytest

from src.data.few_shot_prompting import (
    FewShotConfig,
    FewShotExample,
    FewShotPromptBuilder,
    compute_example_similarity,
    estimate_prompt_tokens,
    format_example,
    format_few_shot_prompt,
    select_examples_by_similarity,
    select_examples_diverse,
    select_examples_random,
    truncate_prompt_to_fit,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Tiny model config
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)


def byte_encode(s: str) -> list[int]:
    """Byte-level encode: UTF-8 bytes, truncated to 256."""
    return list(s.encode("utf-8", errors="replace"))[:256]


@pytest.fixture(scope="module")
def tiny_model():
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


@pytest.fixture
def sample_examples():
    return [
        FewShotExample(input_text="What is the capital of France?", output_text="Paris"),
        FewShotExample(input_text="What is 2+2?", output_text="4"),
        FewShotExample(input_text="Name a planet.", output_text="Mars"),
        FewShotExample(input_text="What color is the sky?", output_text="Blue"),
        FewShotExample(input_text="Who wrote Hamlet?", output_text="Shakespeare"),
        FewShotExample(input_text="What is water made of?", output_text="H2O"),
    ]


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = FewShotConfig()
    assert cfg.n_shots == 5
    assert cfg.selection_strategy == "random"


# ---------------------------------------------------------------------------
# 2. test_format_example_template
# ---------------------------------------------------------------------------


def test_format_example_template():
    ex = FewShotExample(input_text="hello", output_text="world")
    result = format_example(ex, "Q: {input}\nA: {output}")
    assert "hello" in result
    assert "world" in result


# ---------------------------------------------------------------------------
# 3. test_format_few_shot_prompt_contains_examples
# ---------------------------------------------------------------------------


def test_format_few_shot_prompt_contains_examples(sample_examples):
    cfg = FewShotConfig(n_shots=2, instruction="")
    selected = sample_examples[:2]
    prompt = format_few_shot_prompt(selected, "Test query", cfg)
    for ex in selected:
        assert ex.input_text in prompt
        assert ex.output_text in prompt


# ---------------------------------------------------------------------------
# 4. test_format_few_shot_prompt_with_instruction
# ---------------------------------------------------------------------------


def test_format_few_shot_prompt_with_instruction(sample_examples):
    cfg = FewShotConfig(n_shots=1, instruction="Be helpful.")
    selected = sample_examples[:1]
    prompt = format_few_shot_prompt(selected, "Test query", cfg)
    assert prompt.startswith("Be helpful.")


# ---------------------------------------------------------------------------
# 5. test_compute_similarity_identical
# ---------------------------------------------------------------------------


def test_compute_similarity_identical():
    text = "hello world"
    ex = FewShotExample(input_text=text, output_text="")
    sim = compute_example_similarity(text, ex)
    assert sim == 1.0


# ---------------------------------------------------------------------------
# 6. test_compute_similarity_disjoint
# ---------------------------------------------------------------------------


def test_compute_similarity_disjoint():
    ex = FewShotExample(input_text="aaaa", output_text="")
    # query uses only 'b', example uses only 'a' — no shared chars
    sim = compute_example_similarity("bbbb", ex)
    assert sim == 0.0


# ---------------------------------------------------------------------------
# 7. test_select_random_count
# ---------------------------------------------------------------------------


def test_select_random_count(sample_examples):
    selected = select_examples_random(sample_examples, n=3, seed=0)
    assert len(selected) == 3


# ---------------------------------------------------------------------------
# 8. test_select_random_reproducible
# ---------------------------------------------------------------------------


def test_select_random_reproducible(sample_examples):
    s1 = select_examples_random(sample_examples, n=3, seed=42)
    s2 = select_examples_random(sample_examples, n=3, seed=42)
    assert [e.input_text for e in s1] == [e.input_text for e in s2]


# ---------------------------------------------------------------------------
# 9. test_select_by_similarity_count
# ---------------------------------------------------------------------------


def test_select_by_similarity_count(sample_examples):
    selected = select_examples_by_similarity(sample_examples, query="What is the capital?", n=3)
    assert len(selected) == 3


# ---------------------------------------------------------------------------
# 10. test_select_by_similarity_top
# ---------------------------------------------------------------------------


def test_select_by_similarity_top(sample_examples):
    # "What is the capital of France?" is identical to sample_examples[0].input_text
    query = "What is the capital of France?"
    selected = select_examples_by_similarity(sample_examples, query=query, n=2)
    input_texts = [e.input_text for e in selected]
    assert "What is the capital of France?" in input_texts


# ---------------------------------------------------------------------------
# 11. test_select_diverse_count
# ---------------------------------------------------------------------------


def test_select_diverse_count(sample_examples):
    selected = select_examples_diverse(sample_examples, query="What is Paris?", n=3)
    assert len(selected) == 3


# ---------------------------------------------------------------------------
# 12. test_estimate_prompt_tokens
# ---------------------------------------------------------------------------


def test_estimate_prompt_tokens():
    prompt = "This is a sample prompt with several words."
    count = estimate_prompt_tokens(prompt)
    assert count > 0


# ---------------------------------------------------------------------------
# 13. test_truncate_from_start
# ---------------------------------------------------------------------------


def test_truncate_from_start():
    prompt = "one two three four five six seven eight nine ten"
    truncated = truncate_prompt_to_fit(prompt, max_tokens=5, truncate_from="start")
    assert len(truncated.split()) <= 5
    assert len(truncated) < len(prompt)


# ---------------------------------------------------------------------------
# 14. test_truncate_from_end
# ---------------------------------------------------------------------------


def test_truncate_from_end():
    prompt = "one two three four five six seven eight nine ten"
    truncated = truncate_prompt_to_fit(prompt, max_tokens=5, truncate_from="end")
    assert len(truncated.split()) <= 5
    assert len(truncated) < len(prompt)


# ---------------------------------------------------------------------------
# 15. test_prompt_builder_build
# ---------------------------------------------------------------------------


def test_prompt_builder_build(sample_examples):
    cfg = FewShotConfig(n_shots=2, selection_strategy="random")
    builder = FewShotPromptBuilder(cfg)
    builder.add_examples(sample_examples)
    result = builder.build("What is 3+3?", seed=7)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 16. test_prompt_builder_evaluate_quality_keys
# ---------------------------------------------------------------------------


def test_prompt_builder_evaluate_quality_keys(tiny_model, sample_examples):
    cfg = FewShotConfig(n_shots=2, selection_strategy="random")
    builder = FewShotPromptBuilder(cfg)
    builder.add_examples(sample_examples)
    prompt = builder.build("What color is grass?", seed=0)
    result = builder.evaluate_prompt_quality(
        tiny_model,
        byte_encode,
        prompt,
        "Green",
    )
    assert "perplexity" in result
    assert "token_count" in result
