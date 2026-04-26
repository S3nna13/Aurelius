"""Tests for src/data/data_flywheel.py — 15 tests."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.data.data_flywheel import (
    DataBuffer,
    DataFlywheelGenerator,
    FlywheelConfig,
    GeneratedSample,
    QualityFilter,
)
from src.model.config import AureliusConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(**kw) -> FlywheelConfig:
    return FlywheelConfig(**kw)


def _make_sample(
    prompt="hello world", response="Hello! This is a response.", **kw
) -> GeneratedSample:
    return GeneratedSample(prompt=prompt, response=response, **kw)


def _accepted_sample(**kw) -> GeneratedSample:
    s = _make_sample(**kw)
    s.accepted = True
    s.quality_score = 0.8
    s.diversity_score = 0.9
    return s


# ---------------------------------------------------------------------------
# Tiny model for integration tests
# ---------------------------------------------------------------------------


class TinyModel(nn.Module):
    """Minimal language model matching the AureliusTransformer API."""

    def __init__(self, vocab_size: int = 256) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = AureliusConfig(
            n_layers=2,
            d_model=64,
            n_heads=2,
            n_kv_heads=2,
            head_dim=32,
            d_ff=128,
            vocab_size=vocab_size,
            max_seq_len=512,
        )
        self.embed = nn.Embedding(vocab_size, 64)
        self.lm_head = nn.Linear(64, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        x = self.embed(input_ids)  # (B, S, 64)
        logits = self.lm_head(x)  # (B, S, V)
        loss = logits.mean()
        past_kv = None
        return loss, logits, past_kv


def _make_tiny_model() -> TinyModel:
    return TinyModel(vocab_size=256)


def _enc(text: str) -> list[int]:
    """Simple byte-level encoder."""
    return list(text.encode("utf-8", errors="replace"))


def _dec(ids: list[int]) -> str:
    """Simple byte-level decoder."""
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


def _make_flywheel(seed_prompts=None, config=None) -> DataFlywheelGenerator:
    model = _make_tiny_model()
    cfg = config or FlywheelConfig(
        n_generate_per_step=4, min_quality_score=0.0, diversity_threshold=0.0
    )
    prompts = seed_prompts or ["Tell me about science.", "What is math?"]
    return DataFlywheelGenerator(model, cfg, _enc, _dec, prompts)


# ===========================================================================
# 1. FlywheelConfig defaults
# ===========================================================================


def test_flywheel_config_defaults():
    cfg = FlywheelConfig()
    assert cfg.min_quality_score == 0.5
    assert cfg.max_buffer_size == 10000
    assert cfg.diversity_threshold == 0.3
    assert cfg.n_generate_per_step == 16
    assert cfg.temperature == 0.8


# ===========================================================================
# 2. GeneratedSample fields and defaults
# ===========================================================================


def test_generated_sample_defaults():
    s = GeneratedSample(prompt="hi", response="hello")
    assert s.prompt == "hi"
    assert s.response == "hello"
    assert s.quality_score == 0.0
    assert s.diversity_score == 0.0
    assert s.accepted is False
    assert s.generation_step == 0


# ===========================================================================
# 3. QualityFilter.score_length returns [0, 1]
# ===========================================================================


def test_score_length_range():
    cfg = FlywheelConfig()
    qf = QualityFilter(cfg)
    texts = [
        "",
        "hi",
        "a" * 10,
        "a" * 50,
        "a" * 200,
        "a" * 500,
        "a" * 2000,
        "a" * 10000,
    ]
    for t in texts:
        score = qf.score_length(t)
        assert 0.0 <= score <= 1.0, f"score_length out of range for len={len(t)}: {score}"


# ===========================================================================
# 4. QualityFilter.score_length penalizes very short strings
# ===========================================================================


def test_score_length_penalizes_short():
    cfg = FlywheelConfig()
    qf = QualityFilter(cfg)
    short_score = qf.score_length("hi")  # 2 chars — below 10
    optimal_score = qf.score_length("a" * 100)  # within [50, 500]
    assert short_score < optimal_score
    assert short_score == 0.0


# ===========================================================================
# 5. QualityFilter.score_diversity returns 1.0 with no existing
# ===========================================================================


def test_score_diversity_no_existing():
    cfg = FlywheelConfig()
    qf = QualityFilter(cfg)
    score = qf.score_diversity("some text here", existing=[])
    assert score == 1.0


# ===========================================================================
# 6. QualityFilter.score_diversity returns lower score for duplicate
# ===========================================================================


def test_score_diversity_duplicate_is_lower():
    cfg = FlywheelConfig()
    qf = QualityFilter(cfg)
    text = "The quick brown fox jumps over the lazy dog"
    # exact duplicate should have high similarity → low diversity
    dup_score = qf.score_diversity(text, existing=[text])
    # completely different text
    diff_score = qf.score_diversity("xyz987", existing=[text])
    assert dup_score < diff_score
    assert dup_score < 0.5


# ===========================================================================
# 7. QualityFilter.score_coherence returns [0, 1]
# ===========================================================================


def test_score_coherence_range():
    cfg = FlywheelConfig()
    qf = QualityFilter(cfg)
    cases = [
        ("", ""),
        ("hello world", ""),
        ("hello world", "Hello! World is great."),
        ("science", "Mathematics is fascinating and complex."),
        ("a b c", "a b c d e f."),
    ]
    for prompt, response in cases:
        score = qf.score_coherence(prompt, response)
        assert 0.0 <= score <= 1.0, (
            f"coherence out of range: prompt={prompt!r} response={response!r} score={score}"
        )


# ===========================================================================
# 8. QualityFilter.filter sets accepted field
# ===========================================================================


def test_quality_filter_sets_accepted():
    cfg = FlywheelConfig(min_quality_score=0.0, diversity_threshold=0.0)
    qf = QualityFilter(cfg)

    samples = [
        GeneratedSample(
            prompt="hello world", response="Hello! This is a well-written response about the world."
        ),
        GeneratedSample(prompt="foo", response=""),  # empty response — should fail coherence
    ]
    result = qf.filter(samples, existing_texts=[])

    # All samples returned
    assert len(result) == 2
    # accepted fields are booleans
    for s in result:
        assert isinstance(s.accepted, bool)
        assert 0.0 <= s.quality_score <= 1.0
        assert 0.0 <= s.diversity_score <= 1.0

    # Good sample should be accepted (thresholds at 0)
    assert result[0].accepted is True
    # Empty response → not accepted
    assert result[1].accepted is False


# ===========================================================================
# 9. DataBuffer.add increases size
# ===========================================================================


def test_data_buffer_add_increases_size():
    cfg = FlywheelConfig()
    buf = DataBuffer(cfg)
    assert len(buf) == 0

    samples = [_accepted_sample() for _ in range(5)]
    n_added = buf.add(samples)

    assert n_added == 5
    assert len(buf) == 5


# ===========================================================================
# 10. DataBuffer.add respects max_buffer_size
# ===========================================================================


def test_data_buffer_max_size():
    cfg = FlywheelConfig(max_buffer_size=3)
    buf = DataBuffer(cfg)

    # Add 5 accepted samples — should keep only last 3
    samples = [_accepted_sample(prompt=f"p{i}", response=f"Response number {i}!") for i in range(5)]
    buf.add(samples)

    assert len(buf) == 3


# ===========================================================================
# 11. DataBuffer.sample returns correct count
# ===========================================================================


def test_data_buffer_sample_count():
    cfg = FlywheelConfig()
    buf = DataBuffer(cfg)
    samples = [
        _accepted_sample(prompt=f"p{i}", response=f"Response {i} is great.") for i in range(10)
    ]
    buf.add(samples)

    drawn = buf.sample(7)
    assert len(drawn) == 7

    # With replacement: can request more than buffer size
    drawn_large = buf.sample(20)
    assert len(drawn_large) == 20


# ===========================================================================
# 12. DataBuffer.stats returns required keys
# ===========================================================================


def test_data_buffer_stats_keys():
    cfg = FlywheelConfig()
    buf = DataBuffer(cfg)

    # Empty buffer
    empty_stats = buf.stats()
    assert {"size", "mean_quality", "mean_diversity"} == set(empty_stats.keys())
    assert empty_stats["size"] == 0

    # Non-empty buffer
    samples = [_accepted_sample() for _ in range(4)]
    buf.add(samples)
    stats = buf.stats()
    assert {"size", "mean_quality", "mean_diversity"} == set(stats.keys())
    assert stats["size"] == 4
    assert 0.0 <= stats["mean_quality"] <= 1.0
    assert 0.0 <= stats["mean_diversity"] <= 1.0


# ===========================================================================
# 13. DataFlywheelGenerator.run_step returns required keys
# ===========================================================================


def test_run_step_returns_required_keys():
    fw = _make_flywheel()
    result = fw.run_step()

    required = {"generated", "accepted", "buffer_size", "acceptance_rate"}
    assert required == set(result.keys()), f"Missing keys: {required - set(result.keys())}"

    assert isinstance(result["generated"], int)
    assert isinstance(result["accepted"], int)
    assert isinstance(result["buffer_size"], int)
    assert 0.0 <= result["acceptance_rate"] <= 1.0
    assert result["generated"] >= 0
    assert result["accepted"] >= 0
    assert result["buffer_size"] >= 0


# ===========================================================================
# 14. DataFlywheelGenerator.get_training_data returns list of tuples
# ===========================================================================


def test_get_training_data_returns_tuples():
    # Use permissive thresholds so samples get accepted
    cfg = FlywheelConfig(
        n_generate_per_step=4,
        min_quality_score=0.0,
        diversity_threshold=0.0,
    )
    fw = _make_flywheel(config=cfg)

    # Run a step to populate buffer; manually inject accepted samples if needed
    fw.run_step()

    # Manually inject samples to guarantee non-empty buffer
    accepted = _accepted_sample(prompt="test", response="This is a response.")
    fw._buffer._buffer.append(accepted)

    data = fw.get_training_data(3)
    assert isinstance(data, list)
    for item in data:
        assert isinstance(item, tuple)
        assert len(item) == 2
        assert isinstance(item[0], str)
        assert isinstance(item[1], str)


# ===========================================================================
# 15. DataBuffer.export returns (prompt, response) tuples
# ===========================================================================


def test_data_buffer_export():
    cfg = FlywheelConfig()
    buf = DataBuffer(cfg)

    samples = [
        _accepted_sample(prompt="prompt_a", response="Response A is good."),
        _accepted_sample(prompt="prompt_b", response="Response B is fine."),
    ]
    buf.add(samples)

    exported = buf.export()
    assert isinstance(exported, list)
    assert len(exported) == 2
    for item in exported:
        assert isinstance(item, tuple)
        assert len(item) == 2
        prompt, response = item
        assert isinstance(prompt, str)
        assert isinstance(response, str)

    prompts = [p for p, _ in exported]
    assert "prompt_a" in prompts
    assert "prompt_b" in prompts
