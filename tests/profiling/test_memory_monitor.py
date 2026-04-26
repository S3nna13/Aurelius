from src.profiling.memory_monitor import (
    MEMORY_MONITOR_REGISTRY,
    MemoryMonitor,
    MemorySnapshot,
    WatermarkConfig,
)


def test_snapshot_returns_memory_snapshot():
    m = MemoryMonitor()
    snap = m.snapshot()
    assert isinstance(snap, MemorySnapshot)


def test_snapshot_label_preserved():
    m = MemoryMonitor()
    snap = m.snapshot("before")
    assert snap.label == "before"


def test_snapshot_timestamp_nonneg():
    m = MemoryMonitor()
    snap = m.snapshot()
    assert snap.timestamp_s >= 0


def test_snapshot_fields_nonneg():
    m = MemoryMonitor()
    snap = m.snapshot()
    assert snap.allocated_mb >= 0
    assert snap.reserved_mb >= 0
    assert snap.peak_mb >= 0


def test_default_watermark_values():
    cfg = WatermarkConfig()
    assert cfg.warn_mb == 4096.0
    assert cfg.critical_mb == 8192.0


def test_check_watermarks_empty_below_threshold():
    m = MemoryMonitor()
    snap = MemorySnapshot(0.0, 10.0, 10.0, 10.0, "small")
    out = m.check_watermarks(snap)
    assert out == []


def test_check_watermarks_warn_triggered():
    m = MemoryMonitor()
    snap = MemorySnapshot(0.0, 5000.0, 100.0, 5000.0, "")
    out = m.check_watermarks(snap)
    assert any("WARN" in s for s in out)


def test_check_watermarks_critical_triggered():
    m = MemoryMonitor()
    snap = MemorySnapshot(0.0, 10000.0, 100.0, 10000.0, "")
    out = m.check_watermarks(snap)
    assert any("CRITICAL" in s for s in out)


def test_check_watermarks_critical_not_warn():
    m = MemoryMonitor()
    snap = MemorySnapshot(0.0, 10000.0, 100.0, 10000.0, "")
    out = m.check_watermarks(snap)
    assert not any("WARN:" in s for s in out if "CRITICAL" not in s)


def test_check_watermarks_reserved_critical():
    m = MemoryMonitor()
    snap = MemorySnapshot(0.0, 10.0, 10000.0, 10.0, "")
    out = m.check_watermarks(snap)
    assert any("reserved_mb" in s and "CRITICAL" in s for s in out)


def test_set_watermark_updates():
    m = MemoryMonitor()
    m.set_watermark(WatermarkConfig(warn_mb=1.0, critical_mb=2.0))
    snap = MemorySnapshot(0.0, 1.5, 0.0, 1.5, "")
    out = m.check_watermarks(snap)
    assert any("WARN" in s for s in out)


def test_record_appends_to_history():
    m = MemoryMonitor()
    m.record("a")
    m.record("b")
    assert len(m.history()) == 2


def test_record_returns_snapshot():
    m = MemoryMonitor()
    s = m.record("x")
    assert isinstance(s, MemorySnapshot)
    assert s.label == "x"


def test_history_is_copy():
    m = MemoryMonitor()
    m.record("a")
    h = m.history()
    h.clear()
    assert len(m.history()) == 1


def test_history_initially_empty():
    m = MemoryMonitor()
    assert m.history() == []


def test_peak_snapshot_none_when_empty():
    m = MemoryMonitor()
    assert m.peak_snapshot() is None


def test_peak_snapshot_returns_highest():
    m = MemoryMonitor()
    # inject manually via internal list
    m._history.append(MemorySnapshot(0.0, 100.0, 0.0, 100.0, "a"))
    m._history.append(MemorySnapshot(0.0, 500.0, 0.0, 500.0, "b"))
    m._history.append(MemorySnapshot(0.0, 200.0, 0.0, 200.0, "c"))
    peak = m.peak_snapshot()
    assert peak is not None
    assert peak.label == "b"


def test_reset_clears_history():
    m = MemoryMonitor()
    m.record("a")
    m.record("b")
    m.reset()
    assert m.history() == []


def test_reset_allows_new_records():
    m = MemoryMonitor()
    m.record("a")
    m.reset()
    m.record("b")
    assert len(m.history()) == 1


def test_memory_snapshot_frozen():
    s = MemorySnapshot(0.0, 1.0, 1.0, 1.0, "x")
    try:
        s.allocated_mb = 99.0  # type: ignore
    except Exception:
        return
    assert False, "expected frozen"


def test_memory_snapshot_default_label():
    s = MemorySnapshot(0.0, 1.0, 1.0, 1.0)
    assert s.label == ""


def test_watermark_exact_threshold_warn():
    m = MemoryMonitor()
    snap = MemorySnapshot(0.0, 4096.0, 0.0, 4096.0, "")
    out = m.check_watermarks(snap)
    assert any("WARN" in s for s in out)


def test_watermark_exact_threshold_critical():
    m = MemoryMonitor()
    snap = MemorySnapshot(0.0, 8192.0, 0.0, 8192.0, "")
    out = m.check_watermarks(snap)
    assert any("CRITICAL" in s for s in out)


def test_registry_key():
    assert "default" in MEMORY_MONITOR_REGISTRY
    assert MEMORY_MONITOR_REGISTRY["default"] is MemoryMonitor


def test_custom_watermark_constructor():
    m = MemoryMonitor(WatermarkConfig(warn_mb=10.0, critical_mb=20.0))
    snap = MemorySnapshot(0.0, 15.0, 0.0, 15.0, "")
    out = m.check_watermarks(snap)
    assert any("WARN" in s for s in out)


def test_snapshot_ordering_monotonic():
    m = MemoryMonitor()
    a = m.snapshot()
    b = m.snapshot()
    assert b.timestamp_s >= a.timestamp_s
