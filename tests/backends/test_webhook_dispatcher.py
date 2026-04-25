import asyncio
import pytest
from src.backends.webhook_dispatcher import (
    CircuitBreaker,
    CircuitState,
    DispatchResult,
    WebhookDispatcher,
    WebhookEndpoint,
)


def make_ep(url="http://example.com/hook", retries=3):
    return WebhookEndpoint(url=url, max_retries=retries, timeout_s=1.0)


# --- CircuitBreaker unit tests ---

def test_circuit_starts_closed():
    cb = CircuitBreaker()
    assert cb.state == CircuitState.CLOSED


def test_circuit_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=3)
    for _ in range(3):
        cb.record_failure()
    assert cb.state == CircuitState.OPEN


def test_circuit_is_open_returns_true_when_open():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure()
    assert cb.is_open() is True


def test_circuit_resets_on_success():
    cb = CircuitBreaker(failure_threshold=2)
    cb.record_failure()
    cb.record_failure()
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.is_open() is False


def test_circuit_half_open_after_recovery_timeout():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_s=60.0)
    cb.record_failure()
    assert cb._state == CircuitState.OPEN
    cb._recovery_timeout_s = 0.0
    assert cb.is_open() is False
    assert cb.state == CircuitState.HALF_OPEN


def test_circuit_does_not_open_below_threshold():
    cb = CircuitBreaker(failure_threshold=5)
    for _ in range(4):
        cb.record_failure()
    assert cb.state == CircuitState.CLOSED


# --- WebhookDispatcher structural tests ---

def test_add_endpoint():
    d = WebhookDispatcher()
    ep = make_ep("http://a.com/hook")
    d.add_endpoint(ep)
    assert d.get_circuit_state("http://a.com/hook") == CircuitState.CLOSED


def test_remove_endpoint():
    d = WebhookDispatcher()
    ep = make_ep("http://b.com/hook")
    d.add_endpoint(ep)
    d.remove_endpoint("http://b.com/hook")
    with pytest.raises(KeyError):
        d.get_circuit_state("http://b.com/hook")


def test_dispatch_unknown_url_returns_error():
    d = WebhookDispatcher()
    result = asyncio.run(d.dispatch("http://unknown.com", {}))
    assert result.success is False
    assert result.error is not None


def test_dispatch_success():
    d = WebhookDispatcher([make_ep("http://c.com/hook")])
    result = asyncio.run(d.dispatch("http://c.com/hook", {"event": "test"}))
    assert result.success is True
    assert result.status_code == 200
    assert result.attempts >= 1


def test_dispatch_sets_url_on_result():
    url = "http://d.com/hook"
    d = WebhookDispatcher([make_ep(url)])
    result = asyncio.run(d.dispatch(url, {}))
    assert result.url == url


def test_broadcast_all_endpoints():
    d = WebhookDispatcher([
        make_ep("http://e1.com/hook"),
        make_ep("http://e2.com/hook"),
        make_ep("http://e3.com/hook"),
    ])
    results = asyncio.run(d.broadcast({"event": "broadcast"}))
    assert len(results) == 3
    assert all(r.success for r in results)


def test_broadcast_empty_returns_empty():
    d = WebhookDispatcher()
    results = asyncio.run(d.broadcast({"x": 1}))
    assert results == []


def test_circuit_open_blocks_dispatch():
    url = "http://f.com/hook"
    d = WebhookDispatcher([make_ep(url)])
    cb = d._breakers[url]
    for _ in range(5):
        cb.record_failure()
    result = asyncio.run(d.dispatch(url, {}))
    assert result.success is False
    assert "circuit open" in (result.error or "")


def test_reset_circuit():
    url = "http://g.com/hook"
    d = WebhookDispatcher([make_ep(url)])
    cb = d._breakers[url]
    for _ in range(5):
        cb.record_failure()
    d.reset_circuit(url)
    assert d.get_circuit_state(url) == CircuitState.CLOSED


def test_reset_circuit_unknown_url_raises():
    d = WebhookDispatcher()
    with pytest.raises(KeyError):
        d.reset_circuit("http://nope.com")


def test_get_circuit_state_unknown_url_raises():
    d = WebhookDispatcher()
    with pytest.raises(KeyError):
        d.get_circuit_state("http://nope.com")


def test_constructor_accepts_endpoint_list():
    eps = [make_ep(f"http://ep{i}.com/hook") for i in range(4)]
    d = WebhookDispatcher(eps)
    assert len(d._endpoints) == 4


def test_dispatch_result_dataclass():
    r = DispatchResult(url="http://x.com", success=True, status_code=200, attempts=1)
    assert r.error is None


def test_endpoint_defaults():
    ep = WebhookEndpoint(url="http://z.com/hook")
    assert ep.secret_header == ""
    assert ep.timeout_s == 5.0
    assert ep.max_retries == 3
