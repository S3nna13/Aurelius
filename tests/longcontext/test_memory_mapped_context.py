"""Tests for ContextChunk and MemoryMappedContext."""

from __future__ import annotations

from src.longcontext.memory_mapped_context import ContextChunk, MemoryMappedContext

# ===========================================================================
# ContextChunk fields and defaults
# ===========================================================================


def test_chunk_default_chunk_id_is_str():
    c = ContextChunk()
    assert isinstance(c.chunk_id, str)


def test_chunk_default_chunk_id_length():
    c = ContextChunk()
    assert len(c.chunk_id) == 8


def test_chunk_unique_ids():
    ids = {ContextChunk().chunk_id for _ in range(50)}
    assert len(ids) == 50


def test_chunk_default_start_pos():
    assert ContextChunk().start_pos == 0


def test_chunk_default_end_pos():
    assert ContextChunk().end_pos == 0


def test_chunk_default_token_ids():
    assert ContextChunk().token_ids == []


def test_chunk_default_compressed_false():
    assert ContextChunk().compressed is False


def test_chunk_custom_fields():
    c = ContextChunk(
        chunk_id="abc12345", start_pos=10, end_pos=20, token_ids=[1, 2, 3], compressed=True
    )
    assert c.chunk_id == "abc12345"
    assert c.start_pos == 10
    assert c.end_pos == 20
    assert c.token_ids == [1, 2, 3]
    assert c.compressed is True


# ===========================================================================
# MemoryMappedContext.append_tokens
# ===========================================================================


def test_append_tokens_returns_list():
    ctx = MemoryMappedContext(chunk_size=4)
    result = ctx.append_tokens([0, 1, 2, 3])
    assert isinstance(result, list)


def test_append_tokens_single_chunk():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3])
    assert len(chunks) == 1


def test_append_tokens_two_chunks():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3, 4, 5, 6, 7])
    assert len(chunks) == 2


def test_append_tokens_partial_chunk():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1])
    assert len(chunks) == 1
    assert chunks[0].token_ids == [0, 1]


def test_append_tokens_empty_returns_empty():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([])
    assert chunks == []


def test_append_tokens_chunk_ids_are_str():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3])
    assert isinstance(chunks[0].chunk_id, str)


def test_append_tokens_returns_context_chunk_instances():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3])
    assert all(isinstance(c, ContextChunk) for c in chunks)


# ===========================================================================
# Correct start_pos / end_pos
# ===========================================================================


def test_chunk_start_pos_zero_first():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3])
    assert chunks[0].start_pos == 0


def test_chunk_end_pos_first():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3])
    assert chunks[0].end_pos == 4


def test_chunk_positions_second_chunk():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3, 4, 5, 6, 7])
    assert chunks[1].start_pos == 4
    assert chunks[1].end_pos == 8


def test_chunk_positions_across_two_appends():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([0, 1, 2, 3])
    c2 = ctx.append_tokens([4, 5, 6, 7])
    assert c2[0].start_pos == 4
    assert c2[0].end_pos == 8


def test_chunk_positions_partial_then_full():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([0, 1])
    c2 = ctx.append_tokens([2, 3, 4, 5])
    assert c2[0].start_pos == 2
    assert c2[0].end_pos == 6


# ===========================================================================
# total_tokens
# ===========================================================================


def test_total_tokens_starts_at_zero():
    ctx = MemoryMappedContext()
    assert ctx.total_tokens() == 0


def test_total_tokens_after_one_append():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([0, 1, 2, 3])
    assert ctx.total_tokens() == 4


def test_total_tokens_cumulative():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([0, 1, 2, 3])
    ctx.append_tokens([4, 5])
    assert ctx.total_tokens() == 6


def test_total_tokens_multiple_appends():
    ctx = MemoryMappedContext(chunk_size=4)
    for _ in range(5):
        ctx.append_tokens([0, 1, 2, 3])
    assert ctx.total_tokens() == 20


# ===========================================================================
# chunk_count
# ===========================================================================


def test_chunk_count_starts_at_zero():
    ctx = MemoryMappedContext()
    assert ctx.chunk_count() == 0


def test_chunk_count_increments():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([0, 1, 2, 3])
    assert ctx.chunk_count() == 1
    ctx.append_tokens([4, 5, 6, 7])
    assert ctx.chunk_count() == 2


def test_chunk_count_multiple_chunks_per_append():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens(list(range(12)))  # 3 chunks
    assert ctx.chunk_count() == 3


# ===========================================================================
# get_range
# ===========================================================================


def test_get_range_exact_chunk():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([10, 20, 30, 40])
    assert ctx.get_range(0, 4) == [10, 20, 30, 40]


def test_get_range_partial():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([10, 20, 30, 40])
    assert ctx.get_range(1, 3) == [20, 30]


def test_get_range_spanning_chunks():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([0, 1, 2, 3, 4, 5, 6, 7])
    assert ctx.get_range(2, 6) == [2, 3, 4, 5]


def test_get_range_empty():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([0, 1, 2, 3])
    assert ctx.get_range(4, 4) == []


def test_get_range_out_of_bounds():
    ctx = MemoryMappedContext(chunk_size=4)
    ctx.append_tokens([0, 1, 2, 3])
    assert ctx.get_range(0, 10) == [0, 1, 2, 3]


# ===========================================================================
# get_chunk
# ===========================================================================


def test_get_chunk_returns_chunk():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3])
    cid = chunks[0].chunk_id
    result = ctx.get_chunk(cid)
    assert isinstance(result, ContextChunk)
    assert result.chunk_id == cid


def test_get_chunk_none_for_unknown():
    ctx = MemoryMappedContext(chunk_size=4)
    assert ctx.get_chunk("nonexistent") is None


def test_get_chunk_correct_tokens():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([100, 200, 300, 400])
    cid = chunks[0].chunk_id
    result = ctx.get_chunk(cid)
    assert result.token_ids == [100, 200, 300, 400]


# ===========================================================================
# compress_chunk
# ===========================================================================


def test_compress_chunk_returns_true_for_valid():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3])
    assert ctx.compress_chunk(chunks[0].chunk_id) is True


def test_compress_chunk_returns_false_for_unknown():
    ctx = MemoryMappedContext(chunk_size=4)
    assert ctx.compress_chunk("badid123") is False


def test_compress_chunk_sets_compressed_true():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3])
    cid = chunks[0].chunk_id
    assert ctx.get_chunk(cid).compressed is False
    ctx.compress_chunk(cid)
    assert ctx.get_chunk(cid).compressed is True


def test_compress_chunk_idempotent():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3])
    cid = chunks[0].chunk_id
    ctx.compress_chunk(cid)
    assert ctx.compress_chunk(cid) is True
    assert ctx.get_chunk(cid).compressed is True


def test_compress_chunk_does_not_affect_other_chunks():
    ctx = MemoryMappedContext(chunk_size=4)
    chunks = ctx.append_tokens([0, 1, 2, 3, 4, 5, 6, 7])
    ctx.compress_chunk(chunks[0].chunk_id)
    assert ctx.get_chunk(chunks[1].chunk_id).compressed is False
