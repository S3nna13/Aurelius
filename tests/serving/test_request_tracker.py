"""Tests for request tracker."""

from __future__ import annotations

import time

from src.serving.request_tracker import RequestTracker


class TestRequestTracker:
    def test_start_and_finish(self):
        rt = RequestTracker()
        req_id = rt.start("POST", "/api/chat")
        rt.finish(req_id, status=200)
        assert rt.active_count() == 0
        assert len(rt.completed()) == 1

    def test_active_count(self):
        rt = RequestTracker()
        rt.start()
        rt.start()
        assert rt.active_count() == 2

    def test_completed_request_status(self):
        rt = RequestTracker()
        req_id = rt.start()
        rt.finish(req_id, status=503, reason="overloaded")
        assert rt.completed()[0].status == 503

    def test_duration_positive(self):
        rt = RequestTracker()
        req_id = rt.start()
        time.sleep(0.01)
        rt.finish(req_id)
        assert rt.completed()[0].duration_ms() > 0.0

    def test_latency_stats(self):
        rt = RequestTracker()
        for _ in range(3):
            req_id = rt.start()
            rt.finish(req_id)
        stats = rt.latency_stats()
        assert stats["avg_ms"] > 0.0
        assert stats["p50_ms"] > 0.0

    def test_finish_unknown_does_nothing(self):
        rt = RequestTracker()
        rt.finish("nonexistent")
        assert len(rt.completed()) == 0
