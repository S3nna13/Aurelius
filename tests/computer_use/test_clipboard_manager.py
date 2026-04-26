"""
test_clipboard_manager.py
Tests for clipboard_manager.py  (>=28 tests)
"""

import time

from src.computer_use.clipboard_manager import (
    CLIPBOARD_MANAGER_REGISTRY,
    REGISTRY,
    ClipboardEntry,
    ClipboardManager,
)

# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------


def test_registry_contains_default():
    assert "default" in CLIPBOARD_MANAGER_REGISTRY


def test_registry_alias():
    assert REGISTRY is CLIPBOARD_MANAGER_REGISTRY


def test_registry_default_is_class():
    assert CLIPBOARD_MANAGER_REGISTRY["default"] is ClipboardManager


# ---------------------------------------------------------------------------
# ClipboardEntry — timestamp auto-set
# ---------------------------------------------------------------------------


def test_entry_auto_timestamp():
    before = time.monotonic()
    e = ClipboardEntry(content="hello")
    after = time.monotonic()
    assert before <= e.timestamp <= after


def test_entry_explicit_timestamp_preserved():
    e = ClipboardEntry(content="hi", timestamp=999.0)
    assert e.timestamp == 999.0


def test_entry_defaults():
    e = ClipboardEntry(content="data")
    assert e.content_type == "text"
    assert e.source == ""


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


def test_copy_adds_entry():
    cm = ClipboardManager()
    e = cm.copy("hello")
    assert len(cm) == 1
    assert e.content == "hello"


def test_copy_returns_entry():
    cm = ClipboardManager()
    e = cm.copy("world", content_type="html", source="browser")
    assert e.content_type == "html"
    assert e.source == "browser"


def test_copy_multiple_newest_first():
    cm = ClipboardManager()
    cm.copy("first")
    cm.copy("second")
    assert cm.paste().content == "second"


def test_copy_increments_total():
    cm = ClipboardManager()
    cm.copy("a")
    cm.copy("b")
    assert cm.stats()["total_copies"] == 2


# ---------------------------------------------------------------------------
# paste
# ---------------------------------------------------------------------------


def test_paste_returns_latest():
    cm = ClipboardManager()
    cm.copy("old")
    cm.copy("new")
    assert cm.paste().content == "new"


def test_paste_empty_returns_none():
    cm = ClipboardManager()
    assert cm.paste() is None


def test_paste_does_not_remove_entry():
    cm = ClipboardManager()
    cm.copy("keep")
    cm.paste()
    assert len(cm) == 1


# ---------------------------------------------------------------------------
# paste_nth
# ---------------------------------------------------------------------------


def test_paste_nth_zero_is_latest():
    cm = ClipboardManager()
    cm.copy("first")
    cm.copy("second")
    assert cm.paste_nth(0).content == "second"


def test_paste_nth_one_is_second():
    cm = ClipboardManager()
    cm.copy("first")
    cm.copy("second")
    assert cm.paste_nth(1).content == "first"


def test_paste_nth_out_of_range_none():
    cm = ClipboardManager()
    cm.copy("only")
    assert cm.paste_nth(5) is None


def test_paste_nth_negative_none():
    cm = ClipboardManager()
    cm.copy("x")
    assert cm.paste_nth(-1) is None


def test_paste_nth_empty_none():
    cm = ClipboardManager()
    assert cm.paste_nth(0) is None


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_finds_substring():
    cm = ClipboardManager()
    cm.copy("hello world")
    cm.copy("goodbye")
    results = cm.search("hello")
    assert len(results) == 1
    assert results[0].content == "hello world"


def test_search_case_insensitive():
    cm = ClipboardManager()
    cm.copy("Hello World")
    results = cm.search("hello")
    assert len(results) == 1


def test_search_no_match():
    cm = ClipboardManager()
    cm.copy("apple")
    assert cm.search("banana") == []


def test_search_sorted_newest_first():
    cm = ClipboardManager()
    cm.copy("foo bar 1")
    cm.copy("foo bar 2")
    results = cm.search("foo")
    assert results[0].content == "foo bar 2"
    assert results[1].content == "foo bar 1"


def test_search_empty_query_matches_all():
    cm = ClipboardManager()
    cm.copy("abc")
    cm.copy("def")
    results = cm.search("")
    assert len(results) == 2


# ---------------------------------------------------------------------------
# clear_history
# ---------------------------------------------------------------------------


def test_clear_history_returns_count():
    cm = ClipboardManager()
    cm.copy("a")
    cm.copy("b")
    cm.copy("c")
    count = cm.clear_history()
    assert count == 3


def test_clear_history_empties():
    cm = ClipboardManager()
    cm.copy("x")
    cm.clear_history()
    assert len(cm) == 0


def test_clear_history_empty_manager():
    cm = ClipboardManager()
    assert cm.clear_history() == 0


def test_clear_history_does_not_reset_total_copies():
    cm = ClipboardManager()
    cm.copy("a")
    cm.copy("b")
    cm.clear_history()
    assert cm.stats()["total_copies"] == 2


# ---------------------------------------------------------------------------
# __len__
# ---------------------------------------------------------------------------


def test_len_empty():
    cm = ClipboardManager()
    assert len(cm) == 0


def test_len_after_copies():
    cm = ClipboardManager()
    cm.copy("one")
    cm.copy("two")
    assert len(cm) == 2


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


def test_stats_by_type():
    cm = ClipboardManager()
    cm.copy("text1", content_type="text")
    cm.copy("html1", content_type="html")
    cm.copy("text2", content_type="text")
    s = cm.stats()
    assert s["by_type"]["text"] == 2
    assert s["by_type"]["html"] == 1


def test_stats_unique_content():
    cm = ClipboardManager()
    cm.copy("dup")
    cm.copy("dup")
    cm.copy("unique")
    s = cm.stats()
    assert s["unique_content"] == 2


def test_stats_total_copies():
    cm = ClipboardManager()
    for i in range(5):
        cm.copy(f"item-{i}")
    assert cm.stats()["total_copies"] == 5


def test_stats_empty():
    cm = ClipboardManager()
    s = cm.stats()
    assert s["total_copies"] == 0
    assert s["unique_content"] == 0
    assert s["by_type"] == {}


# ---------------------------------------------------------------------------
# max_history eviction
# ---------------------------------------------------------------------------


def test_max_history_eviction():
    cm = ClipboardManager(max_history=3)
    cm.copy("a")
    cm.copy("b")
    cm.copy("c")
    cm.copy("d")  # evicts "a"
    assert len(cm) == 3
    contents = [cm.paste_nth(i).content for i in range(3)]
    assert "a" not in contents
    assert "d" in contents


def test_max_history_default_fifty():
    cm = ClipboardManager()
    for i in range(60):
        cm.copy(f"item-{i}")
    assert len(cm) == 50


def test_max_history_one():
    cm = ClipboardManager(max_history=1)
    cm.copy("first")
    cm.copy("second")
    assert len(cm) == 1
    assert cm.paste().content == "second"
