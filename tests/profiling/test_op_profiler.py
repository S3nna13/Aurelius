import time

from src.profiling.op_profiler import OP_PROFILER_REGISTRY, OpProfiler


def test_profile_records_time():
    p = OpProfiler()
    with p.profile("op_a"):
        time.sleep(0.001)
    recs = p.records()
    assert len(recs) == 1
    assert recs[0].name == "op_a"
    assert recs[0].elapsed_ms > 0


def test_profile_accumulates_calls():
    p = OpProfiler()
    for _ in range(3):
        with p.profile("op_b"):
            pass
    recs = p.records()
    assert len(recs) == 1
    assert recs[0].call_count == 3


def test_profile_multiple_ops():
    p = OpProfiler()
    with p.profile("fast"):
        pass
    with p.profile("slow"):
        time.sleep(0.005)
    recs = p.records()
    assert len(recs) == 2
    assert recs[0].elapsed_ms >= recs[1].elapsed_ms


def test_top_k_returns_k():
    p = OpProfiler()
    for name in ["a", "b", "c", "d"]:
        with p.profile(name):
            pass
    top = p.top_k(2)
    assert len(top) == 2


def test_top_k_fewer_than_k():
    p = OpProfiler()
    with p.profile("only"):
        pass
    top = p.top_k(5)
    assert len(top) == 1


def test_total_ms():
    p = OpProfiler()
    with p.profile("x"):
        pass
    with p.profile("y"):
        pass
    assert p.total_ms() > 0


def test_reset_clears():
    p = OpProfiler()
    with p.profile("op"):
        pass
    p.reset()
    assert p.records() == []
    assert p.total_ms() == 0.0


def test_report_contains_name():
    p = OpProfiler()
    with p.profile("my_operation"):
        pass
    report = p.report()
    assert "my_operation" in report


def test_report_empty():
    p = OpProfiler()
    report = p.report()
    assert "No records" in report


def test_registry_key():
    assert "default" in OP_PROFILER_REGISTRY
    assert OP_PROFILER_REGISTRY["default"] is OpProfiler
