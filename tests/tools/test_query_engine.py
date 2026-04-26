"""Tests for query engine."""

from __future__ import annotations

from src.tools.query_engine import QueryEngine


class TestQueryEngine:
    def test_get_nested_field(self):
        qe = QueryEngine()
        data = {"a": {"b": {"c": 42}}}
        assert qe.get(data, "a.b.c") == 42

    def test_get_missing_returns_none(self):
        qe = QueryEngine()
        assert qe.get({"a": 1}, "b") is None

    def test_get_list_index(self):
        qe = QueryEngine()
        data = {"items": [10, 20, 30]}
        assert qe.get(data, "items.1") == 20

    def test_set_value(self):
        qe = QueryEngine()
        data = {}
        qe.set(data, "x.y.z", 99)
        assert data["x"]["y"]["z"] == 99

    def test_exists(self):
        qe = QueryEngine()
        data = {"a": 1}
        assert qe.exists(data, "a") is True
        assert qe.exists(data, "b") is False
