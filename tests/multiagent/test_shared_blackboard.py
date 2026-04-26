"""Tests for src/multiagent/shared_blackboard.py."""

from __future__ import annotations

from src.multiagent.shared_blackboard import (
    SHARED_BLACKBOARD_REGISTRY,
    Blackboard,
    BlackboardEntry,
)

# ---------------------------------------------------------------------------
# BlackboardEntry dataclass
# ---------------------------------------------------------------------------


def test_entry_fields_stored():
    entry = BlackboardEntry(key="x", value=42, author="agent-1", timestamp=1.0, version=1)
    assert entry.key == "x"
    assert entry.value == 42
    assert entry.author == "agent-1"
    assert entry.timestamp == 1.0
    assert entry.version == 1


def test_entry_default_version():
    entry = BlackboardEntry(key="y", value="hello", author="a", timestamp=0.5)
    assert entry.version == 1


# ---------------------------------------------------------------------------
# Blackboard.write
# ---------------------------------------------------------------------------


def test_write_creates_entry():
    bb = Blackboard()
    entry = bb.write("alpha", 10, "bot")
    assert isinstance(entry, BlackboardEntry)
    assert entry.key == "alpha"
    assert entry.value == 10
    assert entry.author == "bot"


def test_write_stores_author():
    bb = Blackboard()
    entry = bb.write("k", "v", "writer-7")
    assert entry.author == "writer-7"


def test_write_sets_timestamp_positive():
    bb = Blackboard()
    entry = bb.write("t", True, "a")
    assert entry.timestamp > 0


def test_write_initial_version_is_1():
    bb = Blackboard()
    entry = bb.write("key1", "val", "author")
    assert entry.version == 1


def test_write_increments_version_on_overwrite():
    bb = Blackboard()
    bb.write("key1", "first", "a")
    entry2 = bb.write("key1", "second", "b")
    assert entry2.version == 2


def test_write_version_increments_multiple_times():
    bb = Blackboard()
    for i in range(1, 6):
        entry = bb.write("counter", i, "agent")
    assert entry.version == 5


def test_write_different_keys_independent_versions():
    bb = Blackboard()
    bb.write("a", 1, "x")
    e2 = bb.write("b", 2, "x")
    bb.write("a", 3, "x")
    assert e2.version == 1
    assert bb.read("a").version == 2


# ---------------------------------------------------------------------------
# Blackboard.read
# ---------------------------------------------------------------------------


def test_read_returns_entry():
    bb = Blackboard()
    bb.write("foo", "bar", "agent")
    entry = bb.read("foo")
    assert entry is not None
    assert entry.value == "bar"


def test_read_missing_key_returns_none():
    bb = Blackboard()
    assert bb.read("nonexistent") is None


def test_read_returns_latest_value():
    bb = Blackboard()
    bb.write("x", 1, "a")
    bb.write("x", 99, "b")
    assert bb.read("x").value == 99


# ---------------------------------------------------------------------------
# Blackboard.read_all
# ---------------------------------------------------------------------------


def test_read_all_returns_all_entries():
    bb = Blackboard()
    bb.write("a", 1, "x")
    bb.write("b", 2, "y")
    all_entries = bb.read_all()
    assert set(all_entries.keys()) == {"a", "b"}


def test_read_all_returns_dict():
    bb = Blackboard()
    assert isinstance(bb.read_all(), dict)


def test_read_all_empty_blackboard():
    bb = Blackboard()
    assert bb.read_all() == {}


# ---------------------------------------------------------------------------
# Blackboard.delete
# ---------------------------------------------------------------------------


def test_delete_existing_key_returns_true():
    bb = Blackboard()
    bb.write("z", 0, "agent")
    assert bb.delete("z") is True


def test_delete_removes_entry():
    bb = Blackboard()
    bb.write("z", 0, "agent")
    bb.delete("z")
    assert bb.read("z") is None


def test_delete_missing_key_returns_false():
    bb = Blackboard()
    assert bb.delete("ghost") is False


# ---------------------------------------------------------------------------
# Blackboard.subscribe / _notify
# ---------------------------------------------------------------------------


def test_subscribe_callback_triggered_on_write():
    bb = Blackboard()
    received: list[BlackboardEntry] = []
    bb.subscribe("evt", received.append)
    entry = bb.write("evt", "data", "sender")
    assert len(received) == 1
    assert received[0] is entry


def test_subscribe_callback_not_triggered_for_other_keys():
    bb = Blackboard()
    received: list[BlackboardEntry] = []
    bb.subscribe("alpha", received.append)
    bb.write("beta", 42, "agent")
    assert received == []


def test_subscribe_multiple_callbacks_all_called():
    bb = Blackboard()
    calls_a: list[BlackboardEntry] = []
    calls_b: list[BlackboardEntry] = []
    bb.subscribe("shared", calls_a.append)
    bb.subscribe("shared", calls_b.append)
    bb.write("shared", "hello", "author")
    assert len(calls_a) == 1
    assert len(calls_b) == 1


def test_subscribe_callback_called_on_each_write():
    bb = Blackboard()
    received: list[BlackboardEntry] = []
    bb.subscribe("k", received.append)
    bb.write("k", 1, "a")
    bb.write("k", 2, "a")
    assert len(received) == 2


# ---------------------------------------------------------------------------
# Blackboard.keys
# ---------------------------------------------------------------------------


def test_keys_returns_sorted_list():
    bb = Blackboard()
    bb.write("zebra", 1, "a")
    bb.write("apple", 2, "b")
    bb.write("mango", 3, "c")
    assert bb.keys() == ["apple", "mango", "zebra"]


def test_keys_empty_blackboard():
    bb = Blackboard()
    assert bb.keys() == []


# ---------------------------------------------------------------------------
# Blackboard.version_of
# ---------------------------------------------------------------------------


def test_version_of_missing_key_returns_0():
    bb = Blackboard()
    assert bb.version_of("missing") == 0


def test_version_of_existing_key():
    bb = Blackboard()
    bb.write("v", 1, "a")
    bb.write("v", 2, "a")
    assert bb.version_of("v") == 2


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in SHARED_BLACKBOARD_REGISTRY


def test_registry_default_is_blackboard_class():
    assert SHARED_BLACKBOARD_REGISTRY["default"] is Blackboard
