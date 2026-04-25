import pytest

from src.runtime.resource_governor import (
    RESOURCE_GOVERNOR_REGISTRY,
    GovernorDecision,
    ResourceGovernor,
    ResourceLimit,
    ResourceSnapshot,
)


class TestResourceLimit:
    def test_defaults(self):
        lim = ResourceLimit()
        assert lim.max_memory_mb == 8192.0
        assert lim.max_cpu_percent == 80.0
        assert lim.max_concurrent_requests == 100
        assert lim.max_queue_depth == 1000

    def test_custom(self):
        lim = ResourceLimit(max_memory_mb=1000.0, max_cpu_percent=50.0, max_concurrent_requests=10, max_queue_depth=50)
        assert lim.max_memory_mb == 1000.0
        assert lim.max_cpu_percent == 50.0
        assert lim.max_concurrent_requests == 10
        assert lim.max_queue_depth == 50


class TestResourceSnapshot:
    def test_fields(self):
        s = ResourceSnapshot(timestamp_s=1.0, memory_mb=100.0, cpu_percent=50.0, active_requests=3, queue_depth=5)
        assert s.timestamp_s == 1.0
        assert s.memory_mb == 100.0
        assert s.cpu_percent == 50.0
        assert s.active_requests == 3
        assert s.queue_depth == 5

    def test_frozen(self):
        s = ResourceSnapshot(1.0, 1.0, 1.0, 1, 1)
        with pytest.raises(Exception):
            s.memory_mb = 5.0  # type: ignore[misc]


class TestGovernorDecision:
    def test_members(self):
        assert GovernorDecision.ALLOW.value == "allow"
        assert GovernorDecision.THROTTLE.value == "throttle"
        assert GovernorDecision.REJECT.value == "reject"


class TestResourceGovernor:
    def test_default_limits(self):
        g = ResourceGovernor()
        assert g.limits.max_memory_mb == 8192.0

    def test_custom_limits(self):
        lim = ResourceLimit(max_memory_mb=500.0)
        g = ResourceGovernor(limits=lim)
        assert g.limits.max_memory_mb == 500.0

    def test_allow_baseline_no_update(self):
        g = ResourceGovernor()
        assert g.decide() is GovernorDecision.ALLOW

    def test_allow_baseline(self):
        g = ResourceGovernor()
        g.update(memory_mb=100.0, cpu_percent=10.0, active_requests=1, queue_depth=1)
        assert g.decide() is GovernorDecision.ALLOW

    def test_reject_on_memory_breach(self):
        g = ResourceGovernor(ResourceLimit(max_memory_mb=1000.0))
        g.update(memory_mb=1500.0, cpu_percent=1.0, active_requests=1, queue_depth=1)
        assert g.decide() is GovernorDecision.REJECT

    def test_reject_on_queue_breach(self):
        g = ResourceGovernor(ResourceLimit(max_queue_depth=10))
        g.update(memory_mb=100.0, cpu_percent=1.0, active_requests=1, queue_depth=20)
        assert g.decide() is GovernorDecision.REJECT

    def test_throttle_on_cpu(self):
        g = ResourceGovernor(ResourceLimit(max_cpu_percent=80.0))
        g.update(memory_mb=100.0, cpu_percent=90.0, active_requests=1, queue_depth=1)
        assert g.decide() is GovernorDecision.THROTTLE

    def test_throttle_on_active_requests(self):
        g = ResourceGovernor(ResourceLimit(max_concurrent_requests=10))
        g.update(memory_mb=100.0, cpu_percent=1.0, active_requests=20, queue_depth=1)
        assert g.decide() is GovernorDecision.THROTTLE

    def test_reject_trumps_throttle(self):
        g = ResourceGovernor(ResourceLimit(max_memory_mb=1000.0, max_cpu_percent=50.0))
        g.update(memory_mb=2000.0, cpu_percent=90.0, active_requests=1, queue_depth=1)
        assert g.decide() is GovernorDecision.REJECT

    def test_headroom_no_update(self):
        g = ResourceGovernor()
        hr = g.headroom()
        assert hr["memory"] == 100.0
        assert hr["cpu"] == 100.0
        assert hr["active_requests"] == 100.0
        assert hr["queue_depth"] == 100.0

    def test_headroom_memory_pct(self):
        g = ResourceGovernor(ResourceLimit(max_memory_mb=1000.0))
        g.update(memory_mb=250.0, cpu_percent=0.0, active_requests=0, queue_depth=0)
        hr = g.headroom()
        assert hr["memory"] == pytest.approx(75.0)

    def test_headroom_cpu_pct(self):
        g = ResourceGovernor(ResourceLimit(max_cpu_percent=100.0))
        g.update(memory_mb=0.0, cpu_percent=30.0, active_requests=0, queue_depth=0)
        hr = g.headroom()
        assert hr["cpu"] == pytest.approx(70.0)

    def test_headroom_active_requests(self):
        g = ResourceGovernor(ResourceLimit(max_concurrent_requests=100))
        g.update(memory_mb=0.0, cpu_percent=0.0, active_requests=25, queue_depth=0)
        hr = g.headroom()
        assert hr["active_requests"] == pytest.approx(75.0)

    def test_headroom_queue_depth(self):
        g = ResourceGovernor(ResourceLimit(max_queue_depth=1000))
        g.update(memory_mb=0.0, cpu_percent=0.0, active_requests=0, queue_depth=100)
        hr = g.headroom()
        assert hr["queue_depth"] == pytest.approx(90.0)

    def test_headroom_clamped_at_zero(self):
        g = ResourceGovernor(ResourceLimit(max_memory_mb=100.0))
        g.update(memory_mb=500.0, cpu_percent=0.0, active_requests=0, queue_depth=0)
        hr = g.headroom()
        assert hr["memory"] == 0.0

    def test_is_overloaded_false(self):
        g = ResourceGovernor()
        g.update(memory_mb=1.0, cpu_percent=1.0, active_requests=1, queue_depth=1)
        assert g.is_overloaded() is False

    def test_is_overloaded_true_throttle(self):
        g = ResourceGovernor(ResourceLimit(max_cpu_percent=50.0))
        g.update(memory_mb=1.0, cpu_percent=90.0, active_requests=1, queue_depth=1)
        assert g.is_overloaded() is True

    def test_is_overloaded_true_reject(self):
        g = ResourceGovernor(ResourceLimit(max_memory_mb=100.0))
        g.update(memory_mb=500.0, cpu_percent=1.0, active_requests=1, queue_depth=1)
        assert g.is_overloaded() is True

    def test_history_empty(self):
        g = ResourceGovernor()
        assert g.history() == []

    def test_history_last_n(self):
        g = ResourceGovernor()
        for i in range(20):
            g.update(memory_mb=float(i), cpu_percent=0.0, active_requests=0, queue_depth=0)
        last5 = g.history(n=5)
        assert len(last5) == 5
        assert [s.memory_mb for s in last5] == [15.0, 16.0, 17.0, 18.0, 19.0]

    def test_history_default_n(self):
        g = ResourceGovernor()
        for i in range(20):
            g.update(memory_mb=float(i), cpu_percent=0.0, active_requests=0, queue_depth=0)
        assert len(g.history()) == 10

    def test_history_zero_n(self):
        g = ResourceGovernor()
        g.update(1.0, 1.0, 1, 1)
        assert g.history(n=0) == []

    def test_history_snapshots_are_snapshots(self):
        g = ResourceGovernor()
        g.update(memory_mb=1.0, cpu_percent=2.0, active_requests=3, queue_depth=4)
        h = g.history()
        assert isinstance(h[0], ResourceSnapshot)
        assert h[0].memory_mb == 1.0

    def test_update_sets_latest(self):
        g = ResourceGovernor()
        g.update(1.0, 2.0, 3, 4)
        assert g._latest is not None
        assert g._latest.memory_mb == 1.0

    def test_update_multiple_records_latest(self):
        g = ResourceGovernor()
        g.update(1.0, 2.0, 3, 4)
        g.update(5.0, 6.0, 7, 8)
        assert g._latest.memory_mb == 5.0


class TestRegistry:
    def test_default_present(self):
        assert "default" in RESOURCE_GOVERNOR_REGISTRY

    def test_default_constructs(self):
        cls = RESOURCE_GOVERNOR_REGISTRY["default"]
        assert isinstance(cls(), ResourceGovernor)
