"""
Tests for src/agent/context_compressor.py  (≥28 tests)
"""

import math
import pytest
from src.agent.context_compressor import (
    Turn,
    CompressionConfig,
    ContextCompressor,
    CONTEXT_COMPRESSOR_REGISTRY,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def compressor():
    return ContextCompressor()


def make_turns(n: int, words_each: int = 10) -> list[Turn]:
    """Create n turns each with `words_each` words."""
    return [Turn(role="user", content=" ".join([f"word{i}"] * words_each)) for i in range(n)]


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------

def test_registry_key_exists():
    assert "default" in CONTEXT_COMPRESSOR_REGISTRY


def test_registry_value_is_context_compressor():
    assert CONTEXT_COMPRESSOR_REGISTRY["default"] is ContextCompressor


# ---------------------------------------------------------------------------
# Turn dataclass
# ---------------------------------------------------------------------------

def test_turn_token_estimate_auto_set():
    t = Turn(role="user", content="hello world foo")
    assert t.token_estimate == 3


def test_turn_token_estimate_explicit():
    t = Turn(role="user", content="hello world", token_estimate=99)
    assert t.token_estimate == 99


def test_turn_token_estimate_zero_triggers_auto():
    t = Turn(role="assistant", content="one two three four", token_estimate=0)
    assert t.token_estimate == 4


def test_turn_role_stored():
    t = Turn(role="system", content="hi")
    assert t.role == "system"


def test_turn_content_stored():
    t = Turn(role="user", content="hello")
    assert t.content == "hello"


# ---------------------------------------------------------------------------
# CompressionConfig
# ---------------------------------------------------------------------------

def test_compression_config_defaults():
    cfg = CompressionConfig()
    assert cfg.max_tokens == 4096
    assert cfg.keep_recent == 4
    assert cfg.summary_ratio == 0.3


def test_compression_config_frozen():
    cfg = CompressionConfig()
    with pytest.raises((AttributeError, TypeError)):
        cfg.max_tokens = 999  # type: ignore[misc]


def test_compression_config_custom():
    cfg = CompressionConfig(max_tokens=512, keep_recent=2, summary_ratio=0.5)
    assert cfg.max_tokens == 512
    assert cfg.keep_recent == 2
    assert cfg.summary_ratio == 0.5


# ---------------------------------------------------------------------------
# estimate_tokens()
# ---------------------------------------------------------------------------

def test_estimate_tokens_empty(compressor):
    assert compressor.estimate_tokens([]) == 0


def test_estimate_tokens_single_turn(compressor):
    t = Turn(role="user", content="a b c")
    assert compressor.estimate_tokens([t]) == 3


def test_estimate_tokens_multiple_turns(compressor):
    turns = [
        Turn(role="user", content="a b c"),       # 3
        Turn(role="assistant", content="d e f g"), # 4
    ]
    assert compressor.estimate_tokens(turns) == 7


def test_estimate_tokens_uses_token_estimate_field(compressor):
    turns = [Turn(role="user", content="ignored", token_estimate=100)]
    assert compressor.estimate_tokens(turns) == 100


# ---------------------------------------------------------------------------
# compress() — under limit
# ---------------------------------------------------------------------------

def test_compress_under_limit_returns_unchanged(compressor):
    turns = [Turn(role="user", content="hi")]  # 1 token << 4096
    result = compressor.compress(turns)
    assert len(result) == len(turns)
    assert result[0].content == "hi"


def test_compress_under_limit_same_identity_content(compressor):
    turns = make_turns(2, words_each=5)
    result = compressor.compress(turns)
    assert [t.content for t in result] == [t.content for t in turns]


def test_compress_exactly_at_limit_returns_unchanged():
    cfg = CompressionConfig(max_tokens=10, keep_recent=4)
    compressor = ContextCompressor(config=cfg)
    turns = [Turn(role="user", content=" ".join(["w"] * 10))]  # exactly 10
    result = compressor.compress(turns)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# compress() — over limit
# ---------------------------------------------------------------------------

def test_compress_over_limit_keeps_last_keep_recent():
    cfg = CompressionConfig(max_tokens=5, keep_recent=2, summary_ratio=0.5)
    compressor = ContextCompressor(config=cfg)
    turns = make_turns(6, words_each=5)  # 30 tokens >> 5
    result = compressor.compress(turns)
    # first element is summary, then last 2 original turns
    recent_contents = [t.content for t in result[1:]]
    expected = [turns[-2].content, turns[-1].content]
    assert recent_contents == expected


def test_compress_over_limit_creates_summary_turn():
    cfg = CompressionConfig(max_tokens=5, keep_recent=2, summary_ratio=0.5)
    compressor = ContextCompressor(config=cfg)
    turns = make_turns(6, words_each=5)
    result = compressor.compress(turns)
    assert result[0].role == "summary"


def test_compress_summary_role_is_summary():
    cfg = CompressionConfig(max_tokens=5, keep_recent=1, summary_ratio=0.5)
    compressor = ContextCompressor(config=cfg)
    turns = make_turns(5, words_each=5)
    result = compressor.compress(turns)
    assert result[0].role == "summary"


def test_compress_summary_content_not_empty():
    cfg = CompressionConfig(max_tokens=5, keep_recent=2, summary_ratio=0.3)
    compressor = ContextCompressor(config=cfg)
    turns = make_turns(6, words_each=10)
    result = compressor.compress(turns)
    assert result[0].content.strip() != ""


def test_compress_result_starts_with_summary_then_recent():
    cfg = CompressionConfig(max_tokens=5, keep_recent=3, summary_ratio=0.5)
    compressor = ContextCompressor(config=cfg)
    turns = make_turns(8, words_each=5)
    result = compressor.compress(turns)
    assert result[0].role == "summary"
    for t in result[1:]:
        assert t.role != "summary"


def test_compress_summary_every_nth_word():
    """Check the summary picks every Nth word where N=ceil(1/ratio)."""
    cfg = CompressionConfig(max_tokens=2, keep_recent=0, summary_ratio=0.5)
    compressor = ContextCompressor(config=cfg)
    # N = ceil(1/0.5) = 2 → every other word
    t = Turn(role="user", content="w0 w1 w2 w3 w4")
    result = compressor.compress([t])
    # should pick w0, w2, w4
    assert result[0].content == "w0 w2 w4"


# ---------------------------------------------------------------------------
# compression_ratio()
# ---------------------------------------------------------------------------

def test_compression_ratio_unchanged_is_one(compressor):
    turns = [Turn(role="user", content="hi")]
    compressed = compressor.compress(turns)
    ratio = compressor.compression_ratio(turns, compressed)
    assert ratio == pytest.approx(1.0)


def test_compression_ratio_greater_than_one_when_compressed():
    cfg = CompressionConfig(max_tokens=5, keep_recent=2, summary_ratio=0.3)
    compressor = ContextCompressor(config=cfg)
    turns = make_turns(10, words_each=20)
    compressed = compressor.compress(turns)
    ratio = compressor.compression_ratio(turns, compressed)
    assert ratio > 1.0


def test_compression_ratio_uses_token_estimates(compressor):
    original = [Turn(role="user", content="x", token_estimate=100)]
    compressed = [Turn(role="summary", content="x", token_estimate=10)]
    ratio = compressor.compression_ratio(original, compressed)
    assert ratio == pytest.approx(10.0)


def test_compression_ratio_empty_compressed_denominator():
    """When compressed is empty, denominator clamps to 1."""
    cfg = CompressionConfig(max_tokens=9999)
    compressor = ContextCompressor(config=cfg)
    original = [Turn(role="user", content="a b c")]
    ratio = compressor.compression_ratio(original, [])
    assert ratio == pytest.approx(3.0)
