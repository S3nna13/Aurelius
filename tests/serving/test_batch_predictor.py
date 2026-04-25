"""Tests for batch_predictor.py."""

from __future__ import annotations

import threading
import time

import pytest
import torch

from src.serving.batch_predictor import BatchPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _identity_backend(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return tensors


def _double_backend(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    return [t * 2 for t in tensors]


def _slow_backend(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    time.sleep(0.05)
    return tensors


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_init(self):
        bp = BatchPredictor(backend=_identity_backend)
        assert bp._max_batch_size == 8

    def test_custom_init(self):
        bp = BatchPredictor(backend=_identity_backend, max_batch_size=4, max_wait_ms=20.0)
        assert bp._max_batch_size == 4
        assert bp._max_wait_s == 0.02

    def test_max_batch_size_too_small_raises(self):
        with pytest.raises(ValueError):
            BatchPredictor(backend=_identity_backend, max_batch_size=0)

    def test_max_batch_size_too_large_raises(self):
        with pytest.raises(ValueError):
            BatchPredictor(backend=_identity_backend, max_batch_size=2048)

    def test_invalid_backend_raises(self):
        with pytest.raises(TypeError):
            BatchPredictor(backend=None)  # type: ignore[arg-type]

    def test_negative_wait_raises(self):
        with pytest.raises(ValueError):
            BatchPredictor(backend=_identity_backend, max_wait_ms=-1.0)


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


class TestSubmit:
    def test_submit_returns_future_id(self):
        bp = BatchPredictor(backend=_identity_backend)
        fid = bp.submit("req-1", torch.tensor([1.0]))
        assert isinstance(fid, str)
        assert len(fid) > 0
        bp.shutdown()

    def test_submit_validates_request_id_type(self):
        bp = BatchPredictor(backend=_identity_backend)
        with pytest.raises(TypeError):
            bp.submit(123, torch.tensor([1.0]))  # type: ignore[arg-type]
        bp.shutdown()

    def test_submit_validates_input_tensor_type(self):
        bp = BatchPredictor(backend=_identity_backend)
        with pytest.raises(TypeError):
            bp.submit("req-1", [1.0])  # type: ignore[arg-type]
        bp.shutdown()


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------


class TestBatching:
    def test_batching_collects_multiple_requests(self):
        bp = BatchPredictor(backend=_identity_backend, max_batch_size=4, max_wait_ms=100_000.0)
        fids = [bp.submit(f"req-{i}", torch.tensor([float(i)])) for i in range(4)]
        results = bp.collect_results(fids, timeout=5.0)
        assert len(results) == 4
        for i, fid in enumerate(fids):
            assert torch.equal(results[fid], torch.tensor([float(i)]))
        bp.shutdown()

    def test_batch_triggered_by_max_wait(self):
        bp = BatchPredictor(backend=_identity_backend, max_batch_size=10, max_wait_ms=50.0)
        fid = bp.submit("req-1", torch.tensor([1.0]))
        time.sleep(0.06)
        results = bp.collect_results([fid], timeout=5.0)
        assert torch.equal(results[fid], torch.tensor([1.0]))
        bp.shutdown()

    def test_backend_transform_applied(self):
        bp = BatchPredictor(backend=_double_backend, max_batch_size=2, max_wait_ms=100_000.0)
        fids = [bp.submit(f"req-{i}", torch.tensor([float(i)])) for i in range(2)]
        results = bp.collect_results(fids, timeout=5.0)
        for i, fid in enumerate(fids):
            assert torch.equal(results[fid], torch.tensor([float(i) * 2]))
        bp.shutdown()

    def test_multiple_batches(self):
        bp = BatchPredictor(backend=_identity_backend, max_batch_size=2, max_wait_ms=100_000.0)
        fids = [bp.submit(f"req-{i}", torch.tensor([float(i)])) for i in range(5)]
        bp.flush()
        results = bp.collect_results(fids, timeout=5.0)
        assert len(results) == 5
        bp.shutdown()


# ---------------------------------------------------------------------------
# Flush
# ---------------------------------------------------------------------------


class TestFlush:
    def test_flush_forces_immediate_processing(self):
        bp = BatchPredictor(backend=_identity_backend, max_batch_size=10, max_wait_ms=100_000.0)
        fids = [bp.submit(f"req-{i}", torch.tensor([float(i)])) for i in range(3)]
        bp.flush()
        results = bp.collect_results(fids, timeout=5.0)
        assert len(results) == 3
        bp.shutdown()

    def test_flush_empty_queue_is_noop(self):
        bp = BatchPredictor(backend=_identity_backend)
        bp.flush()
        bp.shutdown()


# ---------------------------------------------------------------------------
# Collect results timeout
# ---------------------------------------------------------------------------


class TestCollectResultsTimeout:
    def test_timeout_on_unknown_future_id(self):
        bp = BatchPredictor(backend=_identity_backend)
        with pytest.raises(KeyError):
            bp.collect_results(["nonexistent"], timeout=0.1)
        bp.shutdown()

    def test_timeout_when_backend_is_slow(self):
        bp = BatchPredictor(backend=_slow_backend, max_batch_size=10, max_wait_ms=100_000.0)
        fid = bp.submit("req-1", torch.tensor([1.0]))
        with pytest.raises(TimeoutError):
            bp.collect_results([fid], timeout=0.01)
        bp.shutdown()

    def test_partial_results_not_returned_on_timeout(self):
        bp = BatchPredictor(backend=_identity_backend, max_batch_size=10, max_wait_ms=100_000.0)
        fid1 = bp.submit("req-1", torch.tensor([1.0]))
        bp.flush()
        bp.collect_results([fid1], timeout=2.0)
        fid2 = bp.submit("req-2", torch.tensor([2.0]))
        with pytest.raises(TimeoutError):
            bp.collect_results([fid2], timeout=0.05)
        assert fid2 in bp._events
        bp.shutdown()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_submissions(self):
        bp = BatchPredictor(backend=_double_backend, max_batch_size=4, max_wait_ms=100_000.0)
        fids: list[str] = []
        lock = threading.Lock()

        def worker(i: int) -> None:
            fid = bp.submit(f"req-{i}", torch.tensor([float(i)]))
            with lock:
                fids.append(fid)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(fids) == 20
        bp.flush()
        results = bp.collect_results(fids, timeout=10.0)
        assert len(results) == 20
        for i, fid in enumerate(fids):
            assert torch.equal(results[fid], torch.tensor([float(i) * 2]))
        bp.shutdown()
