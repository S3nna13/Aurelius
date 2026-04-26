"""Tests for CSV parser."""

from __future__ import annotations

from src.data.csv_parser import CSVParser


class TestCSVParser:
    def test_parse_integers(self):
        csv = CSVParser()
        data = "a,b\n1,2\n3,4"
        rows = csv.parse(data)
        assert rows == [{"a": 1, "b": 2}, {"a": 3, "b": 4}]

    def test_parse_strings(self):
        csv = CSVParser()
        data = "name,val\nhello,world\nfoo,bar"
        rows = csv.parse(data)
        assert rows[0]["name"] == "hello"

    def test_parse_bool(self):
        csv = CSVParser()
        data = "flag\nTrue\nFalse\nyes\nno"
        rows = csv.parse(data)
        assert rows[0]["flag"] is True
        assert rows[1]["flag"] is False

    def test_empty_cell_returns_none(self):
        csv = CSVParser()
        rows = csv.parse("a,b\n1,")
        assert rows[0]["b"] is None

    def test_custom_delimiter(self):
        csv = CSVParser(delimiter="|")
        rows = csv.parse("a|b\n1|2")
        assert rows == [{"a": 1, "b": 2}]

    def test_empty_string(self):
        csv = CSVParser()
        assert csv.parse("") == []
