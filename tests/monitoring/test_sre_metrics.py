"""Tests for sre_metrics module."""

import threading

import pytest

from src.monitoring.sre_metrics import Counter, Histogram, SREMetricsCollector


class TestCounter:
    """Tests for Counter class."""

    def test_init_zero(self):
        """Test counter initializes at zero."""
        c = Counter()
        assert c.get() == 0

    def test_increment(self):
        """Test increment increases count."""
        c = Counter()
        c.increment()
        assert c.get() == 1

    def test_increment_by_amount(self):
        """Test increment with custom amount."""
        c = Counter()
        c.increment(5)
        assert c.get() == 5

    def test_increment_multiple(self):
        """Test multiple increments."""
        c = Counter()
        for _ in range(10):
            c.increment()
        assert c.get() == 10

    def test_reset(self):
        """Test reset sets count to zero."""
        c = Counter()
        c.increment(100)
        c.reset()
        assert c.get() == 0


class TestHistogram:
    """Tests for Histogram class."""

    def test_init_empty(self):
        """Test histogram initializes empty."""
        h = Histogram()
        assert len(h) == 0

    def test_record_single(self):
        """Test recording a single value."""
        h = Histogram()
        h.record(1.5)
        assert len(h) == 1

    def test_record_many(self):
        """Test recording multiple values at once."""
        h = Histogram()
        h.record_many([1.0, 2.0, 3.0])
        assert len(h) == 3

    def test_percentile_single_value(self):
        """Test percentile with single value."""
        h = Histogram()
        h.record(100.0)
        assert h.get_percentile(50) == 100.0
        assert h.get_percentile(99) == 100.0

    def test_percentile_empty(self):
        """Test percentile with no data."""
        h = Histogram()
        assert h.get_percentile(50) is None

    def test_percentile_p50(self):
        """Test median calculation."""
        h = Histogram()
        h.record_many([1.0, 2.0, 3.0, 4.0, 5.0])
        assert h.get_percentile(50) == 3.0

    def test_percentile_p90(self):
        """Test p90 calculation with interpolation."""
        h = Histogram()
        h.record_many([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # With 10 values, p90 = index 8.1, which interpolates between 9.0 and 10.0
        p90 = h.get_percentile(90)
        assert 9.0 <= p90 <= 10.0

    def test_percentile_p99(self):
        """Test p99 calculation."""
        h = Histogram()
        values = list(range(1, 101))
        h.record_many(values)
        p99 = h.get_percentile(99)
        assert p99 is not None
        assert 99.0 <= p99 <= 100.0

    def test_percentile_interpolation(self):
        """Test linear interpolation for percentiles."""
        h = Histogram()
        h.record_many([1.0, 2.0])
        # p75 should interpolate between 1.0 and 2.0
        p75 = h.get_percentile(75)
        assert p75 is not None
        assert 1.5 <= p75 <= 2.0

    def test_clear(self):
        """Test clearing histogram."""
        h = Histogram()
        h.record(100.0)
        h.clear()
        assert len(h) == 0
        assert h.get_percentile(50) is None

    def test_thread_safety(self):
        """Test histogram is thread-safe."""
        h = Histogram()
        threads = []
        for i in range(10):

            def record_values():
                for j in range(100):
                    h.record(float(i * 100 + j))

            t = threading.Thread(target=record_values)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(h) == 1000


class TestSREMetricsCollector:
    """Tests for SREMetricsCollector class."""

    def test_init_default(self):
        """Test collector initializes with defaults."""
        c = SREMetricsCollector()
        assert c.name == "default"
        assert c.get_request_count() == 0
        assert c.get_error_count() == 0
        assert c.get_error_rate() == 0.0

    def test_init_with_name(self):
        """Test collector with custom name."""
        c = SREMetricsCollector(name="test-collector")
        assert c.name == "test-collector"

    def test_record_request_success(self):
        """Test recording a successful request."""
        c = SREMetricsCollector()
        c.record_request(latency_ms=100.0, success=True)
        assert c.get_request_count() == 1
        assert c.get_error_count() == 0

    def test_record_request_failure(self):
        """Test recording a failed request."""
        c = SREMetricsCollector()
        c.record_request(latency_ms=100.0, success=False)
        assert c.get_request_count() == 1
        assert c.get_error_count() == 1

    def test_record_error(self):
        """Test recording an error."""
        c = SREMetricsCollector()
        c.record_error("timeout")
        assert c.get_error_count() == 1
        assert c.get_error_types() == {"timeout": 1}

    def test_record_error_multiple_types(self):
        """Test recording multiple error types."""
        c = SREMetricsCollector()
        c.record_error("timeout")
        c.record_error("timeout")
        c.record_error("connection")
        assert c.get_error_types() == {"timeout": 2, "connection": 1}

    def test_record_traffic(self):
        """Test recording traffic."""
        c = SREMetricsCollector()
        c.record_traffic(5)
        assert c.get_traffic_count() == 5

    def test_record_saturation(self):
        """Test recording saturation."""
        c = SREMetricsCollector()
        c.record_saturation(0.75)
        stats = c.get_saturation_stats()
        assert stats["peak"] == 0.75

    def test_record_saturation_multiple(self):
        """Test recording multiple saturation values."""
        c = SREMetricsCollector()
        c.record_saturation(0.5)
        c.record_saturation(0.8)
        c.record_saturation(0.6)
        stats = c.get_saturation_stats()
        assert stats["peak"] == 0.8
        assert stats["mean"] is not None
        assert abs(stats["mean"] - 0.633) < 0.01

    def test_get_error_rate_no_requests(self):
        """Test error rate with no requests."""
        c = SREMetricsCollector()
        assert c.get_error_rate() == 0.0

    def test_get_error_rate_with_errors(self):
        """Test error rate calculation."""
        c = SREMetricsCollector()
        c.record_request(100.0, success=True)
        c.record_request(100.0, success=True)
        c.record_request(100.0, success=False)
        assert c.get_error_rate() == pytest.approx(1 / 3)

    def test_get_percentile_no_data(self):
        """Test percentile with no data."""
        c = SREMetricsCollector()
        assert c.get_percentile(50) is None
        assert c.get_percentile(99) is None

    def test_get_percentile_single_request(self):
        """Test percentile with single request."""
        c = SREMetricsCollector()
        c.record_request(200.0, success=True)
        assert c.get_percentile(50) == 200.0
        assert c.get_percentile(99) == 200.0

    def test_get_percentile_multiple_requests(self):
        """Test percentile with multiple requests."""
        c = SREMetricsCollector()
        for i in range(1, 101):
            c.record_request(float(i), success=True)
        p50 = c.get_percentile(50)
        p90 = c.get_percentile(90)
        p99 = c.get_percentile(99)
        assert p50 is not None
        assert p90 is not None
        assert p99 is not None
        assert 50 <= p50 <= 51
        assert 90 <= p90 <= 91
        assert 99 <= p99 <= 100

    def test_get_latency_stats(self):
        """Test latency statistics."""
        c = SREMetricsCollector()
        for i in range(1, 101):
            c.record_request(float(i), success=True)
        stats = c.get_latency_stats()
        assert stats["p50"] is not None
        assert stats["p90"] is not None
        assert stats["p95"] is not None
        assert stats["p99"] is not None
        assert stats["p99_9"] is not None

    def test_health_score_no_data(self):
        """Test health score with no data."""
        c = SREMetricsCollector()
        assert c.get_health_score() == 1.0

    def test_health_score_all_healthy(self):
        """Test health score when all signals healthy."""
        c = SREMetricsCollector()
        c.record_request(100.0, success=True)  # Fast
        c.record_saturation(0.5)  # Low saturation
        score = c.get_health_score()
        assert score == 1.0

    def test_health_score_latency_degraded(self):
        """Test health score with degraded latency."""
        c = SREMetricsCollector()
        # Record very slow requests
        for _ in range(100):
            c.record_request(500.0, success=True)
        score = c.get_latency_health_score()
        assert score < 1.0
        assert score > 0.0

    def test_health_score_error_degraded(self):
        """Test health score with high error rate."""
        c = SREMetricsCollector()
        # Record 20 requests, 15 errors (75% error rate)
        for _ in range(15):
            c.record_request(100.0, success=False)
        for _ in range(5):
            c.record_request(100.0, success=True)
        score = c.get_error_health_score()
        assert score < 1.0

    def test_health_score_saturation_degraded(self):
        """Test health score with high saturation."""
        c = SREMetricsCollector()
        c.record_saturation(0.95)  # Very high saturation
        score = c.get_saturation_health_score()
        assert score < 1.0

    def test_get_summary(self):
        """Test getting complete metrics summary."""
        c = SREMetricsCollector(name="test")
        c.record_request(100.0, success=True)
        c.record_request(200.0, success=False)
        summary = c.get_summary()
        assert summary["name"] == "test"
        assert summary["requests"] == 2
        assert summary["errors"] == 1
        assert summary["error_rate"] == 0.5
        assert "latency" in summary
        assert "saturation" in summary
        assert "health_scores" in summary

    def test_reset(self):
        """Test reset clears all metrics."""
        c = SREMetricsCollector()
        c.record_request(100.0, success=True)
        c.record_error("timeout")
        c.record_saturation(0.8)
        c.reset()
        assert c.get_request_count() == 0
        assert c.get_error_count() == 0
        assert c.get_error_types() == {}
        assert c.get_saturation_stats()["peak"] is None

    def test_reset_counters(self):
        """Test reset_counters only clears counters."""
        c = SREMetricsCollector()
        c.record_request(100.0, success=True)
        c.record_saturation(0.8)
        c.reset_counters()
        assert c.get_request_count() == 0
        # Histogram should still have data
        assert len(c._saturation_histogram) == 1

    def test_repr(self):
        """Test string representation."""
        c = SREMetricsCollector(name="test")
        r = repr(c)
        assert "SREMetricsCollector" in r
        assert "test" in r

    def test_latency_histogram_values(self):
        """Test latency values recorded correctly."""
        c = SREMetricsCollector()
        c.record_request(100.0, success=True)
        c.record_request(200.0, success=True)
        c.record_request(300.0, success=True)
        assert c.get_percentile(50) == 200.0

    def test_saturation_threshold(self):
        """Test saturation threshold constant."""
        c = SREMetricsCollector()
        assert c.SATURATION_THRESHOLD == 0.8

    def test_error_rate_threshold(self):
        """Test error rate threshold constant."""
        c = SREMetricsCollector()
        assert c.ERROR_RATE_THRESHOLD == 0.05

    def test_latency_slo_threshold(self):
        """Test latency SLO constant."""
        c = SREMetricsCollector()
        assert c.LATENCY_SLO_MS == 200.0

    def test_thread_safety(self):
        """Test collector is thread-safe."""
        c = SREMetricsCollector()
        threads = []
        for i in range(10):

            def record_data():
                for j in range(100):
                    c.record_request(float(j), success=(j % 10 != 0))

            t = threading.Thread(target=record_data)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert c.get_request_count() == 1000
        assert c.get_error_count() == 100

    def test_concurrent_record_request_and_collect(self):
        """Test concurrent recording and collecting doesn't crash."""
        import threading

        c = SREMetricsCollector()
        errors = []

        def record_requests():
            try:
                for _ in range(500):
                    c.record_request(100.0, success=True)
            except Exception as e:
                errors.append(e)

        def collect_stats():
            try:
                for _ in range(500):
                    c.get_health_score()
                    c.get_percentile(90)
                    c.get_summary()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=record_requests)
        t2 = threading.Thread(target=collect_stats)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert len(errors) == 0


class TestSREMetricsCollectorIntegration:
    """Integration tests for SREMetricsCollector."""

    def test_realistic_web_request_pattern(self):
        """Test with realistic web request patterns."""
        c = SREMetricsCollector(name="web-service")

        # Simulate 1000 requests with p90 latency around 150ms
        for i in range(900):
            latency = 50 + (i % 100) * 1.0  # 50-150ms
            c.record_request(latency, success=True)

        # 100 slow requests (errors/timeouts)
        for _ in range(100):
            c.record_request(500.0, success=False)
            # Note: record_request with success=False already records the error

        assert c.get_request_count() == 1000
        assert c.get_error_count() == 100
        assert 0.09 <= c.get_error_rate() <= 0.11

        p90 = c.get_percentile(90)
        assert p90 is not None
        assert 100 <= p90 <= 200

        health = c.get_health_score()
        assert 0.0 < health < 1.0

    def test_healthy_service(self):
        """Test metrics for a healthy service."""
        c = SREMetricsCollector(name="healthy-service")

        for _ in range(1000):
            c.record_request(50.0, success=True)
            c.record_saturation(0.4)

        assert c.get_health_score() == 1.0
        assert c.get_latency_health_score() == 1.0
        assert c.get_error_health_score() == 1.0
        assert c.get_saturation_health_score() == 1.0

    def test_critical_service(self):
        """Test metrics for a critical/failing service."""
        c = SREMetricsCollector(name="failing-service")

        for _ in range(100):
            c.record_request(1000.0, success=False)
            c.record_error("server_crash")
            c.record_saturation(0.99)

        assert c.get_health_score() < 0.5

    def test_empty_service(self):
        """Test metrics for a service with no traffic."""
        c = SREMetricsCollector(name="empty-service")
        summary = c.get_summary()

        assert summary["requests"] == 0
        assert summary["errors"] == 0
        assert summary["error_rate"] == 0.0
        assert c.get_health_score() == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
