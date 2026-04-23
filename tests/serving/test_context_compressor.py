"""Tests for src/serving/context_compressor.py (~45 tests)."""

import pytest
from src.serving.context_compressor import (
    CompressedTurn,
    CompressionStrategy,
    ContextCompressor,
    CONTEXT_COMPRESSOR_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_turns(*roles_contents):
    """Build a list of turn dicts from (role, content) pairs."""
    return [{"role": r, "content": c} for r, c in roles_contents]


def simple_turns(n: int, role: str = "user") -> list[dict]:
    return [{"role": role, "content": f"msg {i}"} for i in range(n)]


# ---------------------------------------------------------------------------
# CompressionStrategy enum
# ---------------------------------------------------------------------------

class TestCompressionStrategyEnum:
    def test_truncate_oldest_value(self):
        assert CompressionStrategy.TRUNCATE_OLDEST == "truncate_oldest"

    def test_summarize_middle_value(self):
        assert CompressionStrategy.SUMMARIZE_MIDDLE == "summarize_middle"

    def test_drop_tool_results_value(self):
        assert CompressionStrategy.DROP_TOOL_RESULTS == "drop_tool_results"

    def test_is_str_subclass(self):
        assert isinstance(CompressionStrategy.TRUNCATE_OLDEST, str)

    def test_three_members(self):
        assert len(list(CompressionStrategy)) == 3


# ---------------------------------------------------------------------------
# CompressedTurn dataclass
# ---------------------------------------------------------------------------

class TestCompressedTurn:
    def test_basic_construction(self):
        t = CompressedTurn(role="user", content="hello")
        assert t.role == "user"
        assert t.content == "hello"

    def test_was_compressed_default_false(self):
        t = CompressedTurn(role="assistant", content="hi")
        assert t.was_compressed is False

    def test_was_compressed_can_be_set(self):
        t = CompressedTurn(role="system", content="[summarized]", was_compressed=True)
        assert t.was_compressed is True


# ---------------------------------------------------------------------------
# TRUNCATE_OLDEST strategy
# ---------------------------------------------------------------------------

class TestTruncateOldest:
    def _compressor(self, max_turns=3):
        return ContextCompressor(
            max_turns=max_turns,
            strategy=CompressionStrategy.TRUNCATE_OLDEST,
        )

    def test_returns_list_of_compressed_turns(self):
        c = self._compressor()
        result = c.compress(simple_turns(2))
        assert all(isinstance(t, CompressedTurn) for t in result)

    def test_short_conversation_returned_whole(self):
        c = self._compressor(max_turns=10)
        turns = simple_turns(5)
        result = c.compress(turns)
        assert len(result) == 5

    def test_keeps_last_max_turns_non_system(self):
        c = self._compressor(max_turns=3)
        turns = simple_turns(6)
        result = c.compress(turns)
        assert len(result) == 3
        assert result[-1].content == "msg 5"
        assert result[0].content == "msg 3"

    def test_system_turn_always_preserved(self):
        c = self._compressor(max_turns=2)
        turns = [
            {"role": "system", "content": "You are helpful."},
        ] + simple_turns(5)
        result = c.compress(turns)
        assert result[0].role == "system"
        assert result[0].content == "You are helpful."

    def test_system_turn_not_counted_in_max_turns(self):
        c = self._compressor(max_turns=2)
        turns = [{"role": "system", "content": "sys"}] + simple_turns(5)
        result = c.compress(turns)
        # 1 system + 2 non-system = 3 total
        assert len(result) == 3

    def test_no_system_turn_in_input(self):
        c = self._compressor(max_turns=3)
        turns = simple_turns(10)
        result = c.compress(turns)
        assert len(result) == 3

    def test_was_compressed_false_on_kept_turns(self):
        c = self._compressor(max_turns=10)
        turns = simple_turns(5)
        result = c.compress(turns)
        assert all(not t.was_compressed for t in result)

    def test_exact_max_turns_returns_all(self):
        c = self._compressor(max_turns=5)
        turns = simple_turns(5)
        result = c.compress(turns)
        assert len(result) == 5

    def test_multiple_system_turns_all_preserved(self):
        c = self._compressor(max_turns=2)
        turns = [
            {"role": "system", "content": "sys1"},
            {"role": "system", "content": "sys2"},
        ] + simple_turns(5)
        result = c.compress(turns)
        sys_turns = [t for t in result if t.role == "system"]
        assert len(sys_turns) == 2


# ---------------------------------------------------------------------------
# SUMMARIZE_MIDDLE strategy
# ---------------------------------------------------------------------------

class TestSummarizeMiddle:
    def _compressor(self, max_turns=6):
        return ContextCompressor(
            max_turns=max_turns,
            strategy=CompressionStrategy.SUMMARIZE_MIDDLE,
        )

    def test_returns_compressed_turns(self):
        c = self._compressor()
        result = c.compress(simple_turns(10))
        assert all(isinstance(t, CompressedTurn) for t in result)

    def test_middle_replaced_by_single_summary_turn(self):
        c = self._compressor(max_turns=6)
        # 10 turns → keep first 2, last 4, summarize 4 middle → summary turn
        turns = simple_turns(10)
        result = c.compress(turns)
        summary_turns = [t for t in result if t.was_compressed]
        assert len(summary_turns) == 1

    def test_summary_turn_content_format(self):
        c = self._compressor(max_turns=6)
        turns = simple_turns(10)
        result = c.compress(turns)
        summary = next(t for t in result if t.was_compressed)
        assert "summarized" in summary.content
        assert "4" in summary.content  # 10 - 2 - 4 = 4 middle turns

    def test_summary_turn_was_compressed_true(self):
        c = self._compressor(max_turns=6)
        result = c.compress(simple_turns(10))
        summary = next(t for t in result if t.was_compressed)
        assert summary.was_compressed is True

    def test_summary_turn_role_is_system(self):
        c = self._compressor(max_turns=6)
        result = c.compress(simple_turns(10))
        summary = next(t for t in result if t.was_compressed)
        assert summary.role == "system"

    def test_short_conversation_no_summary(self):
        c = self._compressor(max_turns=6)
        turns = simple_turns(4)
        result = c.compress(turns)
        assert not any(t.was_compressed for t in result)
        assert len(result) == 4

    def test_first_two_turns_preserved(self):
        c = self._compressor(max_turns=6)
        turns = simple_turns(10)
        result = c.compress(turns)
        assert result[0].content == "msg 0"
        assert result[1].content == "msg 1"

    def test_last_turns_preserved(self):
        c = self._compressor(max_turns=6)
        turns = simple_turns(10)
        result = c.compress(turns)
        non_summary = [t for t in result if not t.was_compressed]
        assert non_summary[-1].content == "msg 9"

    def test_exact_fit_no_summary(self):
        c = self._compressor(max_turns=5)
        # first 2 + last 3 = 5; exact fit with 5 turns
        turns = simple_turns(5)
        result = c.compress(turns)
        assert not any(t.was_compressed for t in result)


# ---------------------------------------------------------------------------
# DROP_TOOL_RESULTS strategy
# ---------------------------------------------------------------------------

class TestDropToolResults:
    def _compressor(self, max_turns=20):
        return ContextCompressor(
            max_turns=max_turns,
            strategy=CompressionStrategy.DROP_TOOL_RESULTS,
        )

    def test_tool_turns_removed(self):
        c = self._compressor()
        turns = [
            {"role": "user", "content": "hi"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "ok"},
        ]
        result = c.compress(turns)
        assert all(t.role != "tool" for t in result)

    def test_non_tool_turns_preserved(self):
        c = self._compressor()
        turns = [
            {"role": "user", "content": "u"},
            {"role": "tool", "content": "t"},
            {"role": "assistant", "content": "a"},
        ]
        result = c.compress(turns)
        assert len(result) == 2
        assert result[0].role == "user"
        assert result[1].role == "assistant"

    def test_no_tool_turns_returns_all(self):
        c = self._compressor()
        turns = simple_turns(5)
        result = c.compress(turns)
        assert len(result) == 5

    def test_all_tool_turns_returns_empty(self):
        c = self._compressor()
        turns = [{"role": "tool", "content": f"r{i}"} for i in range(5)]
        result = c.compress(turns)
        assert result == []

    def test_respects_max_turns_after_drop(self):
        c = self._compressor(max_turns=3)
        turns = (
            [{"role": "tool", "content": "t"}]
            + [{"role": "user", "content": f"u{i}"} for i in range(6)]
        )
        result = c.compress(turns)
        assert len(result) == 3

    def test_was_compressed_false_on_kept_turns(self):
        c = self._compressor()
        turns = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        result = c.compress(turns)
        assert all(not t.was_compressed for t in result)


# ---------------------------------------------------------------------------
# compression_ratio
# ---------------------------------------------------------------------------

class TestCompressionRatio:
    def _compressor(self):
        return ContextCompressor(max_turns=3, strategy=CompressionStrategy.TRUNCATE_OLDEST)

    def test_ratio_one_when_no_compression(self):
        c = self._compressor()
        turns = simple_turns(3)
        compressed = c.compress(turns)
        assert c.compression_ratio(turns, compressed) == 1.0

    def test_ratio_less_than_one_when_compressed(self):
        c = self._compressor()
        turns = simple_turns(10)
        compressed = c.compress(turns)
        ratio = c.compression_ratio(turns, compressed)
        assert 0.0 < ratio < 1.0

    def test_ratio_correct_value(self):
        c = self._compressor()
        turns = simple_turns(6)
        compressed = c.compress(turns)  # keeps 3
        ratio = c.compression_ratio(turns, compressed)
        assert ratio == pytest.approx(3 / 6)

    def test_ratio_empty_original_returns_one(self):
        c = self._compressor()
        ratio = c.compression_ratio([], [])
        assert ratio == 1.0

    def test_ratio_is_float(self):
        c = self._compressor()
        turns = simple_turns(6)
        compressed = c.compress(turns)
        assert isinstance(c.compression_ratio(turns, compressed), float)


# ---------------------------------------------------------------------------
# CONTEXT_COMPRESSOR_REGISTRY
# ---------------------------------------------------------------------------

class TestContextCompressorRegistry:
    def test_has_default(self):
        assert "default" in CONTEXT_COMPRESSOR_REGISTRY

    def test_has_aggressive(self):
        assert "aggressive" in CONTEXT_COMPRESSOR_REGISTRY

    def test_has_tool_drop(self):
        assert "tool_drop" in CONTEXT_COMPRESSOR_REGISTRY

    def test_default_is_context_compressor(self):
        assert isinstance(CONTEXT_COMPRESSOR_REGISTRY["default"], ContextCompressor)

    def test_aggressive_max_turns_8(self):
        assert CONTEXT_COMPRESSOR_REGISTRY["aggressive"].max_turns == 8

    def test_aggressive_strategy_truncate_oldest(self):
        assert (
            CONTEXT_COMPRESSOR_REGISTRY["aggressive"].strategy
            == CompressionStrategy.TRUNCATE_OLDEST
        )

    def test_default_max_turns_20(self):
        assert CONTEXT_COMPRESSOR_REGISTRY["default"].max_turns == 20

    def test_tool_drop_strategy(self):
        assert (
            CONTEXT_COMPRESSOR_REGISTRY["tool_drop"].strategy
            == CompressionStrategy.DROP_TOOL_RESULTS
        )

    def test_tool_drop_max_turns_20(self):
        assert CONTEXT_COMPRESSOR_REGISTRY["tool_drop"].max_turns == 20
