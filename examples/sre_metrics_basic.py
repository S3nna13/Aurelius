# Example: collecting SRE golden signals
# 
"""
Example: instrument code with SRE metrics (Aurelius SREMetricsCollector).

Usage:
    python -m examples.sre_metrics_basic
"""

from src.monitoring.sre_metrics import SREMetricsCollector

if __name__ == "__main__":
    metrics = SREMetricsCollector(service="aurelius-api")

    # Simulate some requests
    for i in range(10):
        latency = 50 + i * 10  # ms
        success = (i % 3 != 0)  # occasional failures
        metrics.record_request(latency_ms=latency, success=success)

    # Record an error
    metrics.record_error(error_type="timeout")

    # Record traffic burst
    metrics.record_traffic(count=100)

    # Record saturation
    metrics.record_saturation(utilization=0.85)

    # Print summary
    print("=== SRE Metrics Summary ===")
    print(f"  Total Requests: {metrics.total_requests}")
    print(f"  Error Rate: {metrics.error_rate:.2%}")
    print(f"  P50 Latency: {metrics.p50_latency_ms:.1f} ms")
    print(f"  P99 Latency: {metrics.p99_latency_ms:.1f} ms")
    print(f"  Saturation: {metrics.saturation:.1%}")