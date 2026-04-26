"""Tests for federated evaluator."""

from __future__ import annotations

import statistics

from src.federation.federated_evaluator import (
    FEDERATED_EVALUATOR_REGISTRY,
    ClientEvalResult,
    EvalTask,
    FederatedEvalResult,
    FederatedEvaluator,
)


def _make_task(**kw) -> EvalTask:
    defaults: dict = dict(dataset_name="ds", metric_name="acc")
    defaults.update(kw)
    return EvalTask(**defaults)


def test_eval_task_defaults() -> None:
    t = _make_task()
    assert t.split == "test"
    assert t.max_samples == 100
    assert len(t.task_id) == 8


def test_eval_task_unique_ids() -> None:
    assert _make_task().task_id != _make_task().task_id


def test_eval_task_explicit_id() -> None:
    t = _make_task(task_id="aabbccdd")
    assert t.task_id == "aabbccdd"


def test_eval_task_frozen() -> None:
    t = _make_task()
    try:
        t.dataset_name = "other"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("EvalTask should be frozen")


def test_register_client_idempotent() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.register_client("c1")
    assert e.pending_clients("nosuch") == ["c1"]


def test_register_multiple_clients() -> None:
    e = FederatedEvaluator()
    e.register_client("a")
    e.register_client("b")
    e.register_client("c")
    t = _make_task(task_id="t0000001")
    e.submit_task(t)
    assert set(e.pending_clients("t0000001")) == {"a", "b", "c"}


def test_submit_task_stores() -> None:
    e = FederatedEvaluator()
    t = _make_task(task_id="t0000002")
    e.submit_task(t)
    e.register_client("c1")
    assert e.pending_clients("t0000002") == ["c1"]


def test_receive_result_reduces_pending() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.register_client("c2")
    e.submit_task(_make_task(task_id="t0000003"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t0000003",
            metric_value=0.8,
            num_samples=10,
        )
    )
    assert e.pending_clients("t0000003") == ["c2"]


def test_aggregate_returns_none_when_incomplete() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.register_client("c2")
    e.submit_task(_make_task(task_id="t0000004"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t0000004",
            metric_value=0.7,
            num_samples=5,
        )
    )
    assert e.aggregate("t0000004") is None


def test_aggregate_returns_none_unknown_task() -> None:
    e = FederatedEvaluator()
    assert e.aggregate("missing") is None


def test_aggregate_returns_none_no_clients() -> None:
    e = FederatedEvaluator()
    e.submit_task(_make_task(task_id="t0000005"))
    assert e.aggregate("t0000005") is None


def test_aggregate_returns_result_when_complete() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.register_client("c2")
    e.submit_task(_make_task(task_id="t0000006"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t0000006",
            metric_value=0.8,
            num_samples=10,
        )
    )
    e.receive_result(
        ClientEvalResult(
            client_id="c2",
            task_id="t0000006",
            metric_value=0.6,
            num_samples=10,
        )
    )
    res = e.aggregate("t0000006")
    assert isinstance(res, FederatedEvalResult)


def test_aggregate_weighted_average() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.register_client("c2")
    e.submit_task(_make_task(task_id="t0000007"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t0000007",
            metric_value=1.0,
            num_samples=90,
        )
    )
    e.receive_result(
        ClientEvalResult(
            client_id="c2",
            task_id="t0000007",
            metric_value=0.0,
            num_samples=10,
        )
    )
    res = e.aggregate("t0000007")
    assert res is not None
    # weighted avg = (1.0*90 + 0.0*10) / 100 = 0.9
    assert abs(res.aggregated_metric - 0.9) < 1e-9


def test_aggregate_equal_weights() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.register_client("c2")
    e.submit_task(_make_task(task_id="t0000008"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t0000008",
            metric_value=0.8,
            num_samples=10,
        )
    )
    e.receive_result(
        ClientEvalResult(
            client_id="c2",
            task_id="t0000008",
            metric_value=0.4,
            num_samples=10,
        )
    )
    res = e.aggregate("t0000008")
    assert res is not None
    assert abs(res.aggregated_metric - 0.6) < 1e-9


def test_aggregate_std_dev() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.register_client("c2")
    e.submit_task(_make_task(task_id="t0000009"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t0000009",
            metric_value=0.8,
            num_samples=10,
        )
    )
    e.receive_result(
        ClientEvalResult(
            client_id="c2",
            task_id="t0000009",
            metric_value=0.4,
            num_samples=10,
        )
    )
    res = e.aggregate("t0000009")
    assert res is not None
    expected = statistics.pstdev([0.8, 0.4])
    assert abs(res.std_dev - expected) < 1e-9


def test_aggregate_single_client_std_dev_zero() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.submit_task(_make_task(task_id="t000000a"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t000000a",
            metric_value=0.5,
            num_samples=10,
        )
    )
    res = e.aggregate("t000000a")
    assert res is not None
    assert res.std_dev == 0.0


def test_aggregate_num_clients() -> None:
    e = FederatedEvaluator()
    for cid in ("a", "b", "c"):
        e.register_client(cid)
    e.submit_task(_make_task(task_id="t000000b"))
    for cid in ("a", "b", "c"):
        e.receive_result(
            ClientEvalResult(
                client_id=cid,
                task_id="t000000b",
                metric_value=0.5,
                num_samples=4,
            )
        )
    res = e.aggregate("t000000b")
    assert res is not None
    assert res.num_clients == 3


def test_aggregate_preserves_task_id() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.submit_task(_make_task(task_id="t000000c"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t000000c",
            metric_value=0.5,
            num_samples=1,
        )
    )
    res = e.aggregate("t000000c")
    assert res is not None
    assert res.task_id == "t000000c"


def test_aggregate_zero_samples_fallback() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.register_client("c2")
    e.submit_task(_make_task(task_id="t000000d"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t000000d",
            metric_value=0.4,
            num_samples=0,
        )
    )
    e.receive_result(
        ClientEvalResult(
            client_id="c2",
            task_id="t000000d",
            metric_value=0.6,
            num_samples=0,
        )
    )
    res = e.aggregate("t000000d")
    assert res is not None
    assert abs(res.aggregated_metric - 0.5) < 1e-9


def test_completed_tasks_empty() -> None:
    assert FederatedEvaluator().completed_tasks() == []


def test_completed_tasks_after_aggregate() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.submit_task(_make_task(task_id="t000000e"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t000000e",
            metric_value=0.5,
            num_samples=1,
        )
    )
    e.aggregate("t000000e")
    assert "t000000e" in e.completed_tasks()


def test_completed_tasks_no_duplicate() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.submit_task(_make_task(task_id="t000000f"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t000000f",
            metric_value=0.5,
            num_samples=1,
        )
    )
    e.aggregate("t000000f")
    e.aggregate("t000000f")
    assert e.completed_tasks().count("t000000f") == 1


def test_pending_clients_empty_unknown_task() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    assert e.pending_clients("missing") == ["c1"]


def test_client_eval_result_frozen() -> None:
    r = ClientEvalResult(client_id="c1", task_id="t", metric_value=0.5, num_samples=1)
    try:
        r.metric_value = 0.6  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("ClientEvalResult should be frozen")


def test_federated_eval_result_frozen() -> None:
    r = FederatedEvalResult(
        task_id="t",
        aggregated_metric=0.5,
        client_results=[],
        num_clients=0,
        std_dev=0.0,
    )
    try:
        r.num_clients = 1  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("FederatedEvalResult should be frozen")


def test_registry_has_default() -> None:
    assert "default" in FEDERATED_EVALUATOR_REGISTRY
    assert FEDERATED_EVALUATOR_REGISTRY["default"] is FederatedEvaluator


def test_client_eval_result_error_default() -> None:
    r = ClientEvalResult(client_id="c1", task_id="t", metric_value=0.5, num_samples=1)
    assert r.error == ""


def test_receive_result_overwrites_same_client() -> None:
    e = FederatedEvaluator()
    e.register_client("c1")
    e.submit_task(_make_task(task_id="t0000010"))
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t0000010",
            metric_value=0.1,
            num_samples=1,
        )
    )
    e.receive_result(
        ClientEvalResult(
            client_id="c1",
            task_id="t0000010",
            metric_value=0.9,
            num_samples=1,
        )
    )
    res = e.aggregate("t0000010")
    assert res is not None
    assert abs(res.aggregated_metric - 0.9) < 1e-9
