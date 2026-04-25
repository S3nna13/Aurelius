"""Tests for CLI history store."""
from __future__ import annotations

import pytest

from src.cli.history_store import HistoryStore, HistoryEntry


class TestHistoryStore:
    def test_append_and_all(self):
        store = HistoryStore()
        store.append(HistoryEntry(command="hello"))
        assert len(store.all()) == 1

    def test_recent(self):
        store = HistoryStore()
        for i in range(5):
            store.append(HistoryEntry(command=f"cmd{i}"))
        recent = store.recent(2)
        assert len(recent) == 2
        assert recent[-1].command == "cmd4"

    def test_search(self):
        store = HistoryStore()
        store.append(HistoryEntry(command="deploy app"))
        store.append(HistoryEntry(command="train model"))
        store.append(HistoryEntry(command="deploy model"))
        results = store.search("deploy")
        assert len(results) == 2

    def test_max_entries(self):
        store = HistoryStore(max_entries=3)
        for i in range(5):
            store.append(HistoryEntry(command=f"cmd{i}"))
        assert store.count() == 3

    def test_clear(self):
        store = HistoryStore()
        store.append(HistoryEntry(command="x"))
        store.clear()
        assert store.count() == 0