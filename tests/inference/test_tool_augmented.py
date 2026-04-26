"""Tests for src/inference/tool_augmented.py"""

from __future__ import annotations

import time

import pytest
import torch

from src.inference.tool_augmented import (
    ToolAugmentedGenerator,
    ToolCache,
    ToolCall,
    ToolConfig,
    ToolResult,
    ToolSpec,
    create_calculator_tool,
    format_tool_results,
    parse_tool_calls_json,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_config():
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=512,
    )


@pytest.fixture
def model(small_config):
    torch.manual_seed(0)
    m = AureliusTransformer(small_config)
    m.eval()
    return m


def _encode(text: str) -> list[int]:
    return list(text.encode("utf-8", errors="replace"))[:200]


def _decode(ids: list[int]) -> str:
    return bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. test_tool_config_defaults
# ---------------------------------------------------------------------------


def test_tool_config_defaults():
    cfg = ToolConfig()
    assert cfg.max_tool_calls == 10
    assert cfg.timeout_seconds == 5.0
    assert cfg.cache_results is True
    assert cfg.retry_on_error is True
    assert cfg.max_retries == 2
    assert cfg.parallel_calls is False


# ---------------------------------------------------------------------------
# 2. test_tool_spec_fields
# ---------------------------------------------------------------------------


def test_tool_spec_fields():
    spec = ToolSpec(
        name="my_tool",
        description="Does something",
        parameters={"x": {"type": "integer"}},
        returns_type="number",
    )
    assert spec.name == "my_tool"
    assert spec.description == "Does something"
    assert "x" in spec.parameters
    assert spec.returns_type == "number"


# ---------------------------------------------------------------------------
# 3. test_tool_call_fields
# ---------------------------------------------------------------------------


def test_tool_call_fields():
    ts = time.time()
    call = ToolCall(
        tool_name="calculator",
        arguments={"expression": "1+1"},
        call_id="abc-123",
        timestamp=ts,
    )
    assert call.tool_name == "calculator"
    assert call.arguments == {"expression": "1+1"}
    assert call.call_id == "abc-123"
    assert call.timestamp == ts


# ---------------------------------------------------------------------------
# 4. test_parse_tool_calls_valid_json
# ---------------------------------------------------------------------------


def test_parse_tool_calls_valid_json():
    text = 'Some preamble [{"tool": "calculator", "args": {"expression": "3*3"}}] trailing'
    calls = parse_tool_calls_json(text)
    assert len(calls) == 1
    assert calls[0].tool_name == "calculator"
    assert calls[0].arguments == {"expression": "3*3"}
    assert calls[0].call_id  # non-empty UUID
    assert calls[0].timestamp > 0


# ---------------------------------------------------------------------------
# 5. test_parse_tool_calls_empty_text
# ---------------------------------------------------------------------------


def test_parse_tool_calls_empty_text():
    assert parse_tool_calls_json("") == []
    assert parse_tool_calls_json("No JSON here at all.") == []
    assert parse_tool_calls_json("[{not: valid}]") == []


# ---------------------------------------------------------------------------
# 6. test_format_tool_results_success
# ---------------------------------------------------------------------------


def test_format_tool_results_success():
    result = ToolResult(
        call_id="c1",
        tool_name="calculator",
        result="4",
        success=True,
        error_message=None,
        execution_time=0.01,
    )
    formatted = format_tool_results([result])
    assert "Tool calculator returned: 4" in formatted


# ---------------------------------------------------------------------------
# 7. test_format_tool_results_error
# ---------------------------------------------------------------------------


def test_format_tool_results_error():
    result = ToolResult(
        call_id="c2",
        tool_name="bad_tool",
        result="",
        success=False,
        error_message="Something broke",
        execution_time=0.0,
    )
    formatted = format_tool_results([result])
    assert "Tool bad_tool failed: Something broke" in formatted


# ---------------------------------------------------------------------------
# 8. test_tool_cache_put_get
# ---------------------------------------------------------------------------


def test_tool_cache_put_get():
    cache = ToolCache()
    cache.put("calc", {"expression": "1+1"}, "2")
    retrieved = cache.get("calc", {"expression": "1+1"})
    assert retrieved == "2"


# ---------------------------------------------------------------------------
# 9. test_tool_cache_miss_returns_none
# ---------------------------------------------------------------------------


def test_tool_cache_miss_returns_none():
    cache = ToolCache()
    result = cache.get("nonexistent_tool", {"x": 1})
    assert result is None


# ---------------------------------------------------------------------------
# 10. test_tool_cache_stats_keys
# ---------------------------------------------------------------------------


def test_tool_cache_stats_keys():
    cache = ToolCache()
    cache.put("t", {}, "v")
    cache.get("t", {})  # hit
    cache.get("t", {"x": 1})  # miss
    stats = cache.stats()
    assert "size" in stats
    assert "hits" in stats
    assert "misses" in stats
    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 1


# ---------------------------------------------------------------------------
# 11. test_tool_augmented_generator_register
# ---------------------------------------------------------------------------


def test_tool_augmented_generator_register(model):
    cfg = ToolConfig(cache_results=False)
    gen = ToolAugmentedGenerator(
        model=model,
        tokenizer_encode=_encode,
        tokenizer_decode=_decode,
        config=cfg,
    )
    spec, fn = create_calculator_tool()
    gen.register_tool(spec, fn)
    tools = gen.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "calculator"


# ---------------------------------------------------------------------------
# 12. test_create_calculator_tool_addition
# ---------------------------------------------------------------------------


def test_create_calculator_tool_addition():
    spec, fn = create_calculator_tool()
    assert spec.name == "calculator"
    result = fn(expression="2+2")
    assert result == "4"
