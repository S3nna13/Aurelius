"""Tests for src/data/backtranslation.py"""

from src.data.backtranslation import (
    BacktranslationConfig,
    BacktranslationPipeline,
    BacktranslationSample,
    build_backtranslation_prompt,
    deduplicate_samples,
    score_instruction_quality,
)

# ---------------------------------------------------------------------------
# Mock generate_fn
# ---------------------------------------------------------------------------


def mock_generate(prompt: str) -> str:
    return "Explain how machine learning models are trained with gradient descent."


def mock_generate_short(prompt: str) -> str:
    return "Hi"


# ---------------------------------------------------------------------------
# BacktranslationConfig
# ---------------------------------------------------------------------------


def test_config_defaults():
    cfg = BacktranslationConfig()
    assert cfg.source_language == "en"
    assert cfg.min_source_length == 20
    assert cfg.n_instructions_per_source == 1
    assert cfg.filter_low_quality is True


def test_config_custom():
    cfg = BacktranslationConfig(n_instructions_per_source=3, quality_threshold=0.5)
    assert cfg.n_instructions_per_source == 3
    assert cfg.quality_threshold == 0.5


# ---------------------------------------------------------------------------
# build_backtranslation_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_contains_source():
    text = "Machine learning is a subset of artificial intelligence."
    prompt = build_backtranslation_prompt(text)
    assert text in prompt


def test_build_prompt_different_templates():
    text = "Sample text for testing purposes here."
    p0 = build_backtranslation_prompt(text, template_idx=0)
    p1 = build_backtranslation_prompt(text, template_idx=1)
    assert p0 != p1


def test_build_prompt_template_wraps():
    # Template idx wraps around
    text = "Some text."
    p0 = build_backtranslation_prompt(text, template_idx=0)
    p3 = build_backtranslation_prompt(text, template_idx=3)  # wraps to 0
    assert p0 == p3


# ---------------------------------------------------------------------------
# score_instruction_quality
# ---------------------------------------------------------------------------


def test_score_quality_returns_float():
    q = score_instruction_quality("Explain how neural networks work.", "Neural networks learn.")
    assert isinstance(q, float)


def test_score_quality_in_range():
    q = score_instruction_quality(
        "How do transformers process text sequences?", "Transformers use attention."
    )
    assert 0.0 <= q <= 1.0


def test_score_quality_empty_instruction_zero():
    q = score_instruction_quality("", "some source text here")
    assert q == 0.0


def test_score_quality_question_mark_boosts():
    q_question = score_instruction_quality("How does gradient descent work?", "gradient descent")
    q_no_question = score_instruction_quality("gradient descent things", "gradient descent")
    assert q_question >= q_no_question


def test_score_quality_very_short_instruction_low():
    q = score_instruction_quality("Hi", "Some longer source text to compare against.")
    assert q < 0.5


# ---------------------------------------------------------------------------
# deduplicate_samples
# ---------------------------------------------------------------------------


def _make_sample(instruction: str) -> BacktranslationSample:
    return BacktranslationSample(
        instruction=instruction,
        response="some response",
        source_text="some response",
    )


def test_deduplicate_removes_identical():
    s1 = _make_sample("Explain how neural networks learn.")
    s2 = _make_sample("Explain how neural networks learn.")
    result = deduplicate_samples([s1, s2])
    assert len(result) == 1


def test_deduplicate_keeps_different():
    s1 = _make_sample("Explain how neural networks learn from data.")
    s2 = _make_sample("What is the capital city of France and why?")
    result = deduplicate_samples([s1, s2])
    assert len(result) == 2


def test_deduplicate_empty_input():
    result = deduplicate_samples([])
    assert result == []


# ---------------------------------------------------------------------------
# BacktranslationPipeline
# ---------------------------------------------------------------------------

SOURCE_TEXT = (
    "Gradient descent is an optimization algorithm used in machine learning. "
    "It iteratively adjusts model parameters to minimize a loss function by "
    "moving in the direction of the negative gradient."
)


def test_pipeline_process_source_returns_list():
    pipeline = BacktranslationPipeline(mock_generate)
    samples = pipeline.process_source(SOURCE_TEXT)
    assert isinstance(samples, list)


def test_pipeline_process_source_sample_type():
    pipeline = BacktranslationPipeline(mock_generate)
    samples = pipeline.process_source(SOURCE_TEXT)
    if samples:
        assert isinstance(samples[0], BacktranslationSample)


def test_pipeline_short_source_filtered():
    pipeline = BacktranslationPipeline(mock_generate)
    samples = pipeline.process_source("Too short.")  # < min_source_length
    assert len(samples) == 0


def test_pipeline_run_returns_list():
    pipeline = BacktranslationPipeline(mock_generate)
    result = pipeline.run([SOURCE_TEXT])
    assert isinstance(result, list)


def test_pipeline_run_multiple_sources():
    texts = [SOURCE_TEXT, SOURCE_TEXT + " Additional context here for second source."]
    pipeline = BacktranslationPipeline(mock_generate)
    result = pipeline.run(texts, deduplicate=False)
    assert len(result) <= len(texts)  # may be filtered by quality


def test_pipeline_get_stats_keys():
    pipeline = BacktranslationPipeline(mock_generate)
    samples = pipeline.run([SOURCE_TEXT])
    stats = pipeline.get_stats(samples)
    assert "n_samples" in stats
    assert "mean_quality" in stats
    assert "mean_instruction_len" in stats
    assert "mean_response_len" in stats


def test_pipeline_get_stats_empty():
    pipeline = BacktranslationPipeline(mock_generate)
    stats = pipeline.get_stats([])
    assert stats["n_samples"] == 0.0


def test_pipeline_sample_is_synthetic():
    pipeline = BacktranslationPipeline(mock_generate)
    samples = pipeline.run([SOURCE_TEXT])
    for s in samples:
        assert s.is_synthetic is True
