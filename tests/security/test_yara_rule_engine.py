"""Unit tests for minimal YARA-like rule engine."""

import pytest

from src.security.yara_rule_engine import (
    YaraParseError,
    YaraRuleEngine,
    YaraRuleParser,
)

RULE_TEXT_AND = """
rule detect_both {
    strings:
        $a = "foo"
        $b = "bar"
    condition:
        $a and $b
}
"""


RULE_TEXT_OR = """
rule detect_any {
    strings:
        $a = "alpha"
        $b = "beta"
    condition:
        $a or $b
}
"""


def test_parse_rule_with_two_strings_and_condition():
    parser = YaraRuleParser()
    rules = parser.parse(RULE_TEXT_AND)
    assert len(rules) == 1
    assert rules[0].name == "detect_both"
    assert "a" in rules[0].strings
    assert "b" in rules[0].strings


def test_scan_and_both_present():
    eng = YaraRuleEngine()
    eng.compile(RULE_TEXT_AND)
    matches = eng.scan("foo bar baz")
    assert any(m.rule_name == "detect_both" for m in matches)


def test_scan_and_only_one_present_misses():
    eng = YaraRuleEngine()
    eng.compile(RULE_TEXT_AND)
    matches = eng.scan("only foo here")
    assert matches == []


def test_or_condition_either_present():
    eng = YaraRuleEngine()
    eng.compile(RULE_TEXT_OR)
    assert len(eng.scan("only alpha here")) == 1
    assert len(eng.scan("only beta here")) == 1


def test_not_condition():
    rule = """
    rule not_test {
        strings:
            $a = "good"
        condition:
            not $a
    }
    """
    eng = YaraRuleEngine()
    eng.compile(rule)
    assert len(eng.scan("has good things")) == 0
    assert len(eng.scan("no trigger here")) == 1


def test_hex_string_matches():
    rule = """
    rule hex_rule {
        strings:
            $h = { DE AD BE EF }
        condition:
            $h
    }
    """
    eng = YaraRuleEngine()
    eng.compile(rule)
    matches = eng.scan(b"\x01\x02\xde\xad\xbe\xef\xff")
    assert len(matches) == 1


def test_hex_wildcard():
    rule = """
    rule hex_wild {
        strings:
            $h = { DE ?? BE EF }
        condition:
            $h
    }
    """
    eng = YaraRuleEngine()
    eng.compile(rule)
    matches = eng.scan(b"\xde\xcc\xbe\xef")
    assert len(matches) == 1


def test_regex_string_matches():
    rule = """
    rule regex_rule {
        strings:
            $r = /mal[0-9]+/
        condition:
            $r
    }
    """
    eng = YaraRuleEngine()
    eng.compile(rule)
    assert len(eng.scan("detected mal1337 payload")) == 1
    assert len(eng.scan("benign text")) == 0


def test_count_condition_greater_equal():
    rule = """
    rule count_rule {
        strings:
            $a = "xyz"
        condition:
            #a >= 3
    }
    """
    eng = YaraRuleEngine()
    eng.compile(rule)
    assert len(eng.scan("xyz xyz xyz abc")) == 1
    assert len(eng.scan("xyz xyz")) == 0


def test_filesize_condition():
    rule = """
    rule small_file {
        strings:
            $a = "marker"
        condition:
            $a and filesize < 100
    }
    """
    eng = YaraRuleEngine()
    eng.compile(rule)
    assert len(eng.scan(b"marker here")) == 1
    assert len(eng.scan(b"marker " + b"x" * 200)) == 0


def test_compile_adds_to_engine():
    eng = YaraRuleEngine()
    eng.compile(RULE_TEXT_AND)
    assert any(r.name == "detect_both" for r in eng.rules)


def test_multiple_rules_in_one_parse():
    text = RULE_TEXT_AND + "\n" + RULE_TEXT_OR
    parser = YaraRuleParser()
    rules = parser.parse(text)
    assert len(rules) == 2


def test_scan_file_on_tmp_file(tmp_path):
    p = tmp_path / "sample.bin"
    p.write_bytes(b"foo and bar together")
    eng = YaraRuleEngine()
    eng.compile(RULE_TEXT_AND)
    matches = eng.scan_file(str(p))
    assert len(matches) == 1


def test_meta_fields_preserved():
    rule = """
    rule with_meta {
        meta:
            author = "aurelius"
            severity = "high"
        strings:
            $a = "x"
        condition:
            $a
    }
    """
    eng = YaraRuleEngine()
    eng.compile(rule)
    r = eng.rules[0]
    assert r.meta.get("author") == "aurelius"
    assert r.meta.get("severity") == "high"


def test_malformed_raises():
    parser = YaraRuleParser()
    with pytest.raises(YaraParseError):
        parser.parse("this is not a rule")


def test_empty_engine_returns_empty():
    eng = YaraRuleEngine()
    assert eng.scan("anything") == []


def test_determinism():
    eng1 = YaraRuleEngine()
    eng1.compile(RULE_TEXT_AND)
    eng2 = YaraRuleEngine()
    eng2.compile(RULE_TEXT_AND)
    m1 = eng1.scan("foo bar")
    m2 = eng2.scan("foo bar")
    assert [m.rule_name for m in m1] == [m.rule_name for m in m2]
