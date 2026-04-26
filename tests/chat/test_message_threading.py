"""Tests for message_threading.py"""

from src.chat.message_threading import (
    MessageThreader,
    Thread,
    ThreadStatus,
)

# --- ThreadStatus enum ---


def test_thread_status_active():
    assert ThreadStatus.ACTIVE == "active"


def test_thread_status_resolved():
    assert ThreadStatus.RESOLVED == "resolved"


def test_thread_status_archived():
    assert ThreadStatus.ARCHIVED == "archived"


def test_thread_status_count():
    assert len(ThreadStatus) == 3


def test_thread_status_is_str():
    assert isinstance(ThreadStatus.ACTIVE, str)


# --- Thread dataclass ---


def test_thread_auto_id():
    t = Thread()
    assert t.thread_id is not None
    assert len(t.thread_id) == 8


def test_thread_unique_ids():
    t1 = Thread()
    t2 = Thread()
    assert t1.thread_id != t2.thread_id


def test_thread_default_parent_id_none():
    t = Thread()
    assert t.parent_id is None


def test_thread_default_status_active():
    t = Thread()
    assert t.status == ThreadStatus.ACTIVE


def test_thread_default_message_ids_empty():
    t = Thread()
    assert t.message_ids == []


def test_thread_default_metadata_empty():
    t = Thread()
    assert t.metadata == {}


def test_thread_custom_parent_id():
    t = Thread(parent_id="abc")
    assert t.parent_id == "abc"


def test_thread_message_ids_independent():
    t1 = Thread()
    t2 = Thread()
    t1.message_ids.append("x")
    assert t2.message_ids == []


# --- MessageThreader ---


def test_threader_create_thread_returns_thread():
    mt = MessageThreader()
    t = mt.create_thread()
    assert isinstance(t, Thread)


def test_threader_create_thread_no_parent():
    mt = MessageThreader()
    t = mt.create_thread()
    assert t.parent_id is None


def test_threader_create_thread_with_parent():
    mt = MessageThreader()
    t = mt.create_thread(parent_id="parent123")
    assert t.parent_id == "parent123"


def test_threader_create_thread_with_metadata():
    mt = MessageThreader()
    t = mt.create_thread(topic="bugs")
    assert t.metadata.get("topic") == "bugs"


def test_threader_add_message_valid_thread():
    mt = MessageThreader()
    t = mt.create_thread()
    result = mt.add_message(t.thread_id, "msg1")
    assert result is True


def test_threader_add_message_invalid_thread():
    mt = MessageThreader()
    result = mt.add_message("nonexistent", "msg1")
    assert result is False


def test_threader_add_message_increases_count():
    mt = MessageThreader()
    t = mt.create_thread()
    mt.add_message(t.thread_id, "msg1")
    assert mt.message_count() == 1


def test_threader_add_message_appears_in_thread():
    mt = MessageThreader()
    t = mt.create_thread()
    mt.add_message(t.thread_id, "msg1")
    assert "msg1" in t.message_ids


def test_threader_message_count_across_threads():
    mt = MessageThreader()
    t1 = mt.create_thread()
    t2 = mt.create_thread()
    mt.add_message(t1.thread_id, "a")
    mt.add_message(t2.thread_id, "b")
    mt.add_message(t2.thread_id, "c")
    assert mt.message_count() == 3


def test_threader_reply_to_returns_thread():
    mt = MessageThreader()
    parent = mt.create_thread()
    child = mt.reply_to(parent.thread_id, "msg1")
    assert isinstance(child, Thread)


def test_threader_reply_to_sets_parent_id():
    mt = MessageThreader()
    parent = mt.create_thread()
    child = mt.reply_to(parent.thread_id, "msg1")
    assert child.parent_id == parent.thread_id


def test_threader_reply_to_adds_message():
    mt = MessageThreader()
    parent = mt.create_thread()
    child = mt.reply_to(parent.thread_id, "msg1")
    assert "msg1" in child.message_ids


def test_threader_reply_to_child_different_id():
    mt = MessageThreader()
    parent = mt.create_thread()
    child = mt.reply_to(parent.thread_id, "msg1")
    assert child.thread_id != parent.thread_id


def test_threader_get_thread_existing():
    mt = MessageThreader()
    t = mt.create_thread()
    found = mt.get_thread(t.thread_id)
    assert found is t


def test_threader_get_thread_nonexistent():
    mt = MessageThreader()
    assert mt.get_thread("ghost") is None


def test_threader_children_direct():
    mt = MessageThreader()
    parent = mt.create_thread()
    child1 = mt.reply_to(parent.thread_id, "m1")
    child2 = mt.reply_to(parent.thread_id, "m2")
    kids = mt.children(parent.thread_id)
    assert child1 in kids
    assert child2 in kids


def test_threader_children_excludes_grandchildren():
    mt = MessageThreader()
    parent = mt.create_thread()
    child = mt.reply_to(parent.thread_id, "m1")
    grandchild = mt.reply_to(child.thread_id, "m2")
    kids = mt.children(parent.thread_id)
    assert grandchild not in kids


def test_threader_children_empty_for_no_children():
    mt = MessageThreader()
    t = mt.create_thread()
    assert mt.children(t.thread_id) == []


def test_threader_resolve_returns_true():
    mt = MessageThreader()
    t = mt.create_thread()
    assert mt.resolve(t.thread_id) is True


def test_threader_resolve_sets_status():
    mt = MessageThreader()
    t = mt.create_thread()
    mt.resolve(t.thread_id)
    assert t.status == ThreadStatus.RESOLVED


def test_threader_resolve_nonexistent():
    mt = MessageThreader()
    assert mt.resolve("ghost") is False


def test_threader_archive_returns_true():
    mt = MessageThreader()
    t = mt.create_thread()
    assert mt.archive(t.thread_id) is True


def test_threader_archive_sets_status():
    mt = MessageThreader()
    t = mt.create_thread()
    mt.archive(t.thread_id)
    assert t.status == ThreadStatus.ARCHIVED


def test_threader_archive_nonexistent():
    mt = MessageThreader()
    assert mt.archive("ghost") is False


def test_threader_active_threads_excludes_resolved():
    mt = MessageThreader()
    t1 = mt.create_thread()
    t2 = mt.create_thread()
    mt.resolve(t1.thread_id)
    active = mt.active_threads()
    assert t1 not in active
    assert t2 in active


def test_threader_active_threads_excludes_archived():
    mt = MessageThreader()
    t1 = mt.create_thread()
    t2 = mt.create_thread()
    mt.archive(t1.thread_id)
    active = mt.active_threads()
    assert t1 not in active
    assert t2 in active


def test_threader_active_threads_all_active():
    mt = MessageThreader()
    t1 = mt.create_thread()
    t2 = mt.create_thread()
    active = mt.active_threads()
    assert t1 in active
    assert t2 in active


def test_threader_message_count_zero_initially():
    mt = MessageThreader()
    assert mt.message_count() == 0


def test_threader_multiple_messages_same_thread():
    mt = MessageThreader()
    t = mt.create_thread()
    mt.add_message(t.thread_id, "a")
    mt.add_message(t.thread_id, "b")
    mt.add_message(t.thread_id, "c")
    assert mt.message_count() == 3


def test_threader_reply_to_included_in_message_count():
    mt = MessageThreader()
    parent = mt.create_thread()
    mt.reply_to(parent.thread_id, "reply_msg")
    assert mt.message_count() == 1
