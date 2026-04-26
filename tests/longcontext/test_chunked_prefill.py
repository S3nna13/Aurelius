"""Unit tests for ChunkedPrefill scheduler."""

from __future__ import annotations

import pytest
import torch

from src.longcontext.chunked_prefill import (
    ChunkedPrefill,
    ChunkedPrefillConfig,
)

B = 2
S = 32
CHUNK = 8


def _ids(seq_len: int = S, batch: int = B, dtype: torch.dtype = torch.long) -> torch.Tensor:
    return torch.arange(batch * seq_len, dtype=dtype).reshape(batch, seq_len)


def test_iter_chunks_divisible_yields_exact_number():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=0))
    ids = _ids()
    chunks = list(sched.iter_chunks(ids))
    assert len(chunks) == S // CHUNK
    for i, (start, end, chunk) in enumerate(chunks):
        assert start == i * CHUNK
        assert end == (i + 1) * CHUNK
        assert chunk.shape == (B, CHUNK)


def test_iter_chunks_remainder_produces_smaller_final_chunk():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=5, overlap=0))
    ids = _ids(seq_len=13)  # 13 = 5 + 5 + 3
    chunks = list(sched.iter_chunks(ids))
    assert len(chunks) == 3
    assert chunks[0][2].shape == (B, 5)
    assert chunks[1][2].shape == (B, 5)
    assert chunks[2][2].shape == (B, 3)
    assert chunks[-1] == (10, 13, chunks[-1][2])


def test_iter_chunks_overlap_produces_overlapping_windows():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=3))
    ids = _ids()
    chunks = list(sched.iter_chunks(ids))
    # stride = 5; starts: 0, 5, 10, 15, 20, 25 (end=32 < 33, next start 30 -> end 32)
    starts = [s for (s, _e, _c) in chunks]
    assert starts[0] == 0
    for prev, nxt in zip(starts, starts[1:]):
        assert nxt - prev == CHUNK - 3
    # Verify overlap: last 3 of chunk i == first 3 of chunk i+1 where both full.
    a = chunks[0][2]
    b = chunks[1][2]
    assert torch.equal(a[:, -3:], b[:, :3])


def test_run_chunk_fn_identity_returns_input_unchanged():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=0))
    ids = _ids()
    out = sched.run_chunk_fn(ids, chunk_fn=lambda x: x, concat_dim=1)
    assert out.shape == ids.shape
    assert torch.equal(out, ids)


def test_run_chunk_fn_last_token_returns_last_per_chunk():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=0))
    ids = _ids()

    def last_token(x: torch.Tensor) -> torch.Tensor:
        return x[:, -1:]

    out = sched.run_chunk_fn(ids, chunk_fn=last_token, concat_dim=1)
    assert out.shape == (B, S // CHUNK)
    expected = torch.stack([ids[:, CHUNK - 1 + i * CHUNK] for i in range(S // CHUNK)], dim=1)
    assert torch.equal(out, expected)


def test_run_chunk_fn_concat_dim_default_is_one():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=0))
    ids = _ids()
    out_default = sched.run_chunk_fn(ids, chunk_fn=lambda x: x)
    out_explicit = sched.run_chunk_fn(ids, chunk_fn=lambda x: x, concat_dim=1)
    assert torch.equal(out_default, out_explicit)
    assert out_default.shape == ids.shape


def test_chunk_size_ge_seq_len_yields_single_chunk():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=S * 4, overlap=0))
    ids = _ids()
    chunks = list(sched.iter_chunks(ids))
    assert len(chunks) == 1
    start, end, chunk = chunks[0]
    assert start == 0 and end == S
    assert torch.equal(chunk, ids)


def test_chunk_size_non_positive_raises():
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkedPrefill(ChunkedPrefillConfig(chunk_size=0, overlap=0))
    with pytest.raises(ValueError, match="chunk_size"):
        ChunkedPrefill(ChunkedPrefillConfig(chunk_size=-4, overlap=0))


def test_overlap_ge_chunk_size_raises():
    with pytest.raises(ValueError, match="overlap"):
        ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=CHUNK))
    with pytest.raises(ValueError, match="overlap"):
        ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=CHUNK + 1))


def test_determinism_same_output_for_same_input():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=2))
    ids = _ids()

    def fn(x):
        return x.float().mul(2.0)

    a = sched.run_chunk_fn(ids, chunk_fn=fn, concat_dim=1)
    b = sched.run_chunk_fn(ids, chunk_fn=fn, concat_dim=1)
    assert torch.equal(a, b)


def test_batch_dim_preserved():
    for batch in (1, 2, 5):
        sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=0))
        ids = _ids(batch=batch)
        out = sched.run_chunk_fn(ids, chunk_fn=lambda x: x)
        assert out.shape[0] == batch
        for _s, _e, c in sched.iter_chunks(ids):
            assert c.shape[0] == batch


def test_dtype_preserved():
    for dt in (torch.long, torch.int32, torch.float32):
        sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=0))
        ids = _ids(dtype=dt)
        out = sched.run_chunk_fn(ids, chunk_fn=lambda x: x)
        assert out.dtype == dt
        for _s, _e, c in sched.iter_chunks(ids):
            assert c.dtype == dt


def test_iter_chunks_requires_2d_input():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=0))
    with pytest.raises(ValueError):
        list(sched.iter_chunks(torch.arange(10)))


def test_config_defaults():
    cfg = ChunkedPrefillConfig()
    assert cfg.chunk_size == 512
    assert cfg.overlap == 0


def test_chunk_fn_non_tensor_return_raises():
    sched = ChunkedPrefill(ChunkedPrefillConfig(chunk_size=CHUNK, overlap=0))
    ids = _ids()
    with pytest.raises(TypeError):
        sched.run_chunk_fn(ids, chunk_fn=lambda x: x.tolist())
