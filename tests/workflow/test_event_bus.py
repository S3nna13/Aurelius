from src.workflow.event_bus import Event, EventBus, Subscription, EVENT_BUS_REGISTRY


def test_registry_has_default():
    assert "default" in EVENT_BUS_REGISTRY
    assert EVENT_BUS_REGISTRY["default"] is EventBus


def test_subscribe_returns_id():
    bus = EventBus()
    sid = bus.subscribe("e", lambda e: None)
    assert isinstance(sid, str)
    assert len(sid) == 8


def test_unique_sub_ids():
    bus = EventBus()
    a = bus.subscribe("x", lambda e: None)
    b = bus.subscribe("x", lambda e: None)
    assert a != b


def test_publish_calls_matching_handler():
    bus = EventBus()
    got = []
    bus.subscribe("tick", lambda e: got.append(e.payload))
    n = bus.publish_named("tick", {"n": 1})
    assert n == 1
    assert got == [{"n": 1}]


def test_publish_no_match():
    bus = EventBus()
    got = []
    bus.subscribe("foo", lambda e: got.append(e))
    n = bus.publish_named("bar", {})
    assert n == 0
    assert got == []


def test_unsubscribe():
    bus = EventBus()
    got = []
    sid = bus.subscribe("x", lambda e: got.append(1))
    assert bus.unsubscribe(sid) is True
    bus.publish_named("x", {})
    assert got == []


def test_unsubscribe_unknown():
    bus = EventBus()
    assert bus.unsubscribe("nope") is False


def test_wildcard_matches_all():
    bus = EventBus()
    got = []
    bus.subscribe("*", lambda e: got.append(e.name))
    bus.publish_named("a", {})
    bus.publish_named("b", {})
    bus.publish_named("c", {})
    assert got == ["a", "b", "c"]


def test_exact_pattern_only_matches_exact():
    bus = EventBus()
    got = []
    bus.subscribe("foo", lambda e: got.append(e.name))
    bus.publish_named("foo", {})
    bus.publish_named("foobar", {})
    assert got == ["foo"]


def test_max_deliveries_limits_and_removes():
    bus = EventBus()
    got = []
    bus.subscribe("x", lambda e: got.append(1), max_deliveries=2)
    bus.publish_named("x", {})
    bus.publish_named("x", {})
    bus.publish_named("x", {})
    assert got == [1, 1]
    assert len(bus.active_subscriptions()) == 0


def test_max_deliveries_unlimited():
    bus = EventBus()
    got = []
    bus.subscribe("x", lambda e: got.append(1), max_deliveries=-1)
    for _ in range(5):
        bus.publish_named("x", {})
    assert len(got) == 5


def test_publish_named_returns_count():
    bus = EventBus()
    bus.subscribe("x", lambda e: None)
    bus.subscribe("x", lambda e: None)
    assert bus.publish_named("x", {}) == 2


def test_publish_event_object():
    bus = EventBus()
    got = []
    bus.subscribe("x", lambda e: got.append(e))
    ev = Event(name="x", payload={"a": 1}, source="test")
    bus.publish(ev)
    assert got[0].source == "test"
    assert got[0].payload == {"a": 1}


def test_active_subscriptions_list():
    bus = EventBus()
    bus.subscribe("a", lambda e: None)
    bus.subscribe("b", lambda e: None)
    assert len(bus.active_subscriptions()) == 2


def test_active_subscriptions_empty():
    bus = EventBus()
    assert bus.active_subscriptions() == []


def test_event_log_records():
    bus = EventBus()
    bus.publish_named("a", {})
    bus.publish_named("b", {})
    log = bus.event_log()
    assert [e.name for e in log] == ["a", "b"]


def test_event_log_returns_copy():
    bus = EventBus()
    bus.publish_named("a", {})
    log = bus.event_log()
    log.clear()
    assert len(bus.event_log()) == 1


def test_clear_log():
    bus = EventBus()
    bus.publish_named("a", {})
    bus.clear_log()
    assert bus.event_log() == []


def test_log_caps_at_max():
    bus = EventBus()
    for i in range(EventBus.MAX_LOG + 50):
        bus.publish_named("x", {"i": i})
    log = bus.event_log()
    assert len(log) == EventBus.MAX_LOG
    assert log[-1].payload["i"] == EventBus.MAX_LOG + 49


def test_subscription_delivered_count():
    bus = EventBus()
    sid = bus.subscribe("x", lambda e: None)
    bus.publish_named("x", {})
    bus.publish_named("x", {})
    subs = {s.sub_id: s for s in bus.active_subscriptions()}
    assert subs[sid].delivered == 2


def test_event_frozen():
    ev = Event(name="a", payload={})
    try:
        ev.name = "b"  # type: ignore
    except Exception:
        return
    assert False


def test_event_default_timestamp_positive():
    ev = Event(name="a", payload={})
    assert ev.timestamp_s > 0


def test_subscription_dataclass_fields():
    s = Subscription(sub_id="x", event_pattern="p", handler=lambda e: None)
    assert s.max_deliveries == -1
    assert s.delivered == 0


def test_multiple_handlers_all_called():
    bus = EventBus()
    got = []
    bus.subscribe("x", lambda e: got.append("a"))
    bus.subscribe("x", lambda e: got.append("b"))
    bus.subscribe("*", lambda e: got.append("c"))
    bus.publish_named("x", {})
    assert sorted(got) == ["a", "b", "c"]


def test_publish_to_empty_bus():
    bus = EventBus()
    assert bus.publish_named("x", {}) == 0


def test_source_defaults_empty():
    bus = EventBus()
    got = []
    bus.subscribe("x", lambda e: got.append(e.source))
    bus.publish_named("x", {})
    assert got == [""]
