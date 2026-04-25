"""Tests for src/monitoring/slo_tracker.py"""
import pytest

from src.monitoring.slo_tracker import (
    SLODefinition,
    SLOStatus,
    SLOTracker,
    SLOType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fresh() -> SLOTracker:
    return SLOTracker()


def latency_slo(target: float = 0.2, name: str = "latency") -> SLODefinition:
    return SLODefinition(name=name, slo_type=SLOType.LATENCY, target=target, window_seconds=3600.0)


def availability_slo(target: float = 0.99, name: str = "avail") -> SLODefinition:
    return SLODefinition(name=name, slo_type=SLOType.AVAILABILITY, target=target, window_seconds=3600.0)


# ---------------------------------------------------------------------------
# SLOType enum
# ---------------------------------------------------------------------------

def test_slo_type_values():
    assert SLOType.LATENCY == "LATENCY"
    assert SLOType.AVAILABILITY == "AVAILABILITY"
    assert SLOType.ERROR_RATE == "ERROR_RATE"
    assert SLOType.THROUGHPUT == "THROUGHPUT"


# ---------------------------------------------------------------------------
# define_slo / record
# ---------------------------------------------------------------------------

def test_define_and_record():
    st = fresh()
    slo = latency_slo()
    st.define_slo(slo)
    st.record("latency", 0.1)
    status = st.evaluate("latency")
    assert status.current_value == pytest.approx(0.1)


def test_evaluate_unknown_slo_raises():
    st = fresh()
    with pytest.raises(KeyError):
        st.evaluate("nonexistent")


# ---------------------------------------------------------------------------
# Latency compliance
# ---------------------------------------------------------------------------

def test_latency_compliant_when_below_target():
    st = fresh()
    st.define_slo(latency_slo(target=0.5))
    st.record("latency", 0.1)
    st.record("latency", 0.2)
    status = st.evaluate("latency")
    assert status.compliant is True


def test_latency_noncompliant_when_above_target():
    st = fresh()
    st.define_slo(latency_slo(target=0.1))
    st.record("latency", 0.5)
    status = st.evaluate("latency")
    assert status.compliant is False


def test_error_rate_compliance():
    st = fresh()
    slo = SLODefinition(name="err", slo_type=SLOType.ERROR_RATE, target=0.05)
    st.define_slo(slo)
    st.record("err", 0.01)
    assert st.evaluate("err").compliant is True
    st.record("err", 0.9)
    # Mean is now > 0.05
    assert st.evaluate("err").compliant is False


# ---------------------------------------------------------------------------
# Availability compliance
# ---------------------------------------------------------------------------

def test_availability_compliant():
    st = fresh()
    st.define_slo(availability_slo(target=0.99))
    st.record("avail", 1.0)
    status = st.evaluate("avail")
    assert status.compliant is True


def test_availability_noncompliant():
    st = fresh()
    st.define_slo(availability_slo(target=0.99))
    st.record("avail", 0.90)
    status = st.evaluate("avail")
    assert status.compliant is False


def test_throughput_compliance():
    st = fresh()
    slo = SLODefinition(name="tput", slo_type=SLOType.THROUGHPUT, target=100.0)
    st.define_slo(slo)
    st.record("tput", 120.0)
    assert st.evaluate("tput").compliant is True
    st2 = fresh()
    st2.define_slo(SLODefinition(name="tput", slo_type=SLOType.THROUGHPUT, target=100.0))
    st2.record("tput", 80.0)
    assert st2.evaluate("tput").compliant is False


# ---------------------------------------------------------------------------
# burn_rate
# ---------------------------------------------------------------------------

def test_burn_rate_non_negative():
    st = fresh()
    st.define_slo(availability_slo())
    st.record("avail", 0.9)
    status = st.evaluate("avail")
    assert status.burn_rate >= 0.0


def test_burn_rate_zero_when_fully_compliant():
    st = fresh()
    st.define_slo(availability_slo(target=0.99))
    # All samples >= target
    for _ in range(10):
        st.record("avail", 1.0)
    status = st.evaluate("avail")
    assert status.burn_rate == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# get_all_status
# ---------------------------------------------------------------------------

def test_get_all_status():
    st = fresh()
    st.define_slo(latency_slo(name="l"))
    st.define_slo(availability_slo(name="a"))
    st.record("l", 0.1)
    st.record("a", 1.0)
    statuses = st.get_all_status()
    assert len(statuses) == 2
    names = {s.definition.name for s in statuses}
    assert names == {"l", "a"}


def test_no_data_returns_zero_mean():
    st = fresh()
    st.define_slo(latency_slo())
    status = st.evaluate("latency")
    assert status.current_value == pytest.approx(0.0)
    # 0 mean <= 0.2 target -> compliant
    assert status.compliant is True
