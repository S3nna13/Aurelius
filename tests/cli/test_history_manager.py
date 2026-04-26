"""Tests for history_manager."""

from __future__ import annotations

import json
import os
import threading

import pytest

from src.cli.history_manager import HistoryError, HistoryManager


class TestAddAndRetrieve:
    def test_add_and_get_recent(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        hm.add("ls -la")
        recent = hm.get_recent(1)
        assert len(recent) == 1
        assert recent[0].command == "ls -la"

    def test_get_recent_order(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        hm.add("a")
        hm.add("b")
        hm.add("c")
        recent = hm.get_recent(2)
        assert [e.command for e in recent] == ["b", "c"]

    def test_persistence(self, tmp_path):
        path = str(tmp_path / "hist.jsonl")
        hm1 = HistoryManager(history_file=path)
        hm1.add("cmd1")
        hm2 = HistoryManager(history_file=path)
        recent = hm2.get_recent(1)
        assert recent[0].command == "cmd1"


class TestDeduplication:
    def test_consecutive_dedup(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        hm.add("same")
        hm.add("same")
        hm.add("same")
        assert len(hm.get_recent(10)) == 1

    def test_non_consecutive_allowed(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        hm.add("a")
        hm.add("b")
        hm.add("a")
        assert len(hm.get_recent(10)) == 3


class TestSearch:
    def test_basic_search(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        hm.add("hello world")
        hm.add("goodbye")
        results = hm.search("hello")
        assert len(results) == 1
        assert results[0].command == "hello world"

    def test_search_order_newest_first(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        hm.add("match")
        hm.add("other")
        hm.add("match")
        results = hm.search("match")
        assert [e.command for e in results] == ["match", "match"]

    def test_search_limit(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        hm.add("findme")
        hm.add("other")
        hm.add("findme")
        results = hm.search("findme", limit=1)
        assert len(results) == 1

    def test_search_case_insensitive(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        hm.add("HELLO")
        results = hm.search("hello")
        assert len(results) == 1


class TestMaxEntries:
    def test_fifo_eviction(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"), max_entries=3)
        hm.add("1")
        hm.add("2")
        hm.add("3")
        hm.add("4")
        recent = hm.get_recent(10)
        assert [e.command for e in recent] == ["2", "3", "4"]


class TestClear:
    def test_clear(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        hm.add("x")
        hm.clear()
        assert len(hm.get_recent(10)) == 0
        assert not os.path.exists(str(tmp_path / "hist.jsonl"))


class TestSanitization:
    def test_control_char_rejected(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        with pytest.raises(HistoryError):
            hm.add("cmd\x01")

    def test_null_byte_rejected(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        with pytest.raises(HistoryError):
            hm.add("cmd\x00")

    def test_non_string_rejected(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        with pytest.raises(HistoryError):
            hm.add(123)  # type: ignore[arg-type]


class TestPathTraversal:
    def test_traversal_rejected(self, tmp_path):
        with pytest.raises(HistoryError):
            HistoryManager(history_file="../outside.jsonl")


class TestThreadSafety:
    def test_concurrent_adds(self, tmp_path):
        hm = HistoryManager(history_file=str(tmp_path / "hist.jsonl"))
        errors = []

        def worker(n: int):
            try:
                for i in range(20):
                    hm.add(f"cmd-{n}-{i}")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(hm.get_recent(1000)) == 100


class TestCorruptedLine:
    def test_skips_bad_json(self, tmp_path):
        path = str(tmp_path / "hist.jsonl")
        with open(path, "w") as f:
            f.write("bad json\n")
            f.write(json.dumps({"command": "ok", "timestamp": "2024-01-01T00:00:00"}) + "\n")
        hm = HistoryManager(history_file=path)
        assert len(hm.get_recent(10)) == 1
        assert hm.get_recent(1)[0].command == "ok"
