"""Unit tests for CodeAwareTokenizer."""

from __future__ import annotations

import time

import pytest

from src.retrieval.code_aware_tokenizer import (
    KEYWORDS,
    SUPPORTED_LANGUAGES,
    CodeAwareTokenizer,
)


# --- language detection ---------------------------------------------------- #


def test_detect_python():
    tok = CodeAwareTokenizer()
    assert tok.detect_language("def foo(): pass") == "python"


def test_detect_javascript():
    tok = CodeAwareTokenizer()
    assert tok.detect_language("function foo() {}") == "javascript"


def test_detect_rust():
    tok = CodeAwareTokenizer()
    assert tok.detect_language("fn foo() {}") == "rust"


def test_detect_go():
    tok = CodeAwareTokenizer()
    assert tok.detect_language("package main\nfunc Foo() {}") == "go"


def test_detect_java():
    tok = CodeAwareTokenizer()
    src = "package com.foo; public class Bar { public static void main(String[] args) {} }"
    assert tok.detect_language(src) == "java"


def test_detect_unknown_empty():
    tok = CodeAwareTokenizer()
    assert tok.detect_language("") == "unknown"


def test_detect_unknown_prose():
    tok = CodeAwareTokenizer()
    assert tok.detect_language("1234 5678 9012") == "unknown"


# --- case splitting -------------------------------------------------------- #


def test_camel_case_split():
    tok = CodeAwareTokenizer(language="python")
    out = tok.tokenize("getUserName")
    # Must include each subpart AND the whole identifier.
    assert "get" in out
    assert "user" in out
    assert "name" in out
    assert "getusername" in out


def test_pascal_case_split_with_acronym():
    tok = CodeAwareTokenizer(language="java")
    out = tok.tokenize("HTTPServer parseXMLDoc")
    assert "http" in out
    assert "server" in out
    assert "parse" in out
    assert "xml" in out
    assert "doc" in out


def test_snake_case_split():
    tok = CodeAwareTokenizer(language="python")
    out = tok.tokenize("my_func")
    assert "my" in out
    assert "func" in out
    assert "my_func" in out


def test_split_case_disabled():
    tok = CodeAwareTokenizer(language="python", split_case=False)
    out = tok.tokenize("getUserName my_func")
    assert "getusername" in out
    assert "my_func" in out
    # Sub-parts should NOT appear.
    assert "get" not in out
    assert "user" not in out
    assert "my" not in out


# --- keywords -------------------------------------------------------------- #


def test_keywords_retained_when_enabled():
    tok = CodeAwareTokenizer(language="python", keep_keywords=True)
    out = tok.tokenize("def foo(): return 1")
    assert "def" in out
    assert "return" in out
    assert "foo" in out


def test_keywords_dropped_when_disabled():
    tok = CodeAwareTokenizer(language="python", keep_keywords=False)
    out = tok.tokenize("def foo(): return 1")
    assert "def" not in out
    assert "return" not in out
    assert "foo" in out


def test_rust_keywords_set():
    assert "fn" in KEYWORDS["rust"]
    assert "let" in KEYWORDS["rust"]
    assert "pub" in KEYWORDS["rust"]


def test_go_keywords_set():
    assert "func" in KEYWORDS["go"]


# --- punctuation ----------------------------------------------------------- #


def test_punctuation_dropped():
    tok = CodeAwareTokenizer(language="python")
    out = tok.tokenize("foo(bar, baz); // !?@#")
    assert "foo" in out
    assert "bar" in out
    assert "baz" in out
    for ch in "(),;/!?@#":
        assert ch not in out


# --- dotted identifiers ---------------------------------------------------- #


def test_dotted_identifier_kept_and_split():
    tok = CodeAwareTokenizer(language="python")
    out = tok.tokenize("import os.path")
    assert "os.path" in out
    assert "os" in out
    assert "path" in out


def test_deeply_dotted_identifier():
    tok = CodeAwareTokenizer(language="java")
    out = tok.tokenize("com.example.foo.Bar")
    assert "com.example.foo.bar" in out
    assert "com" in out
    assert "example" in out
    assert "foo" in out
    assert "bar" in out


# --- min_token_len --------------------------------------------------------- #


def test_min_token_len_filter():
    tok = CodeAwareTokenizer(language="python", min_token_len=3)
    out = tok.tokenize("a ab abc abcd")
    assert "a" not in out
    assert "ab" not in out
    assert "abc" in out
    assert "abcd" in out


def test_min_token_len_one_allows_singletons():
    tok = CodeAwareTokenizer(language="python", min_token_len=1)
    out = tok.tokenize("a b c")
    assert "a" in out
    assert "b" in out
    assert "c" in out


# --- empty / misc ---------------------------------------------------------- #


def test_empty_string_returns_empty_list():
    tok = CodeAwareTokenizer()
    assert tok.tokenize("") == []


def test_dedup_first_occurrence_order():
    tok = CodeAwareTokenizer(language="python", min_token_len=1)
    out = tok.tokenize("foo bar foo baz bar")
    assert out.index("foo") < out.index("bar") < out.index("baz")
    assert out.count("foo") == 1
    assert out.count("bar") == 1


def test_determinism():
    tok = CodeAwareTokenizer()
    text = "def getUserName(self, user_id): return self.db.users.find(user_id)"
    runs = [tok.tokenize(text) for _ in range(5)]
    assert all(r == runs[0] for r in runs)


def test_unicode_safe():
    tok = CodeAwareTokenizer(language="python")
    # Accented identifier, CJK identifier. Should not raise, should lower.
    out = tok.tokenize("def café(): pass # 变量 naïve_value")
    assert "café" in out
    assert "变量" in out
    assert "naïve_value" in out
    assert "naïve" in out


def test_supported_languages_tuple():
    assert "python" in SUPPORTED_LANGUAGES
    assert "unknown" in SUPPORTED_LANGUAGES


def test_call_alias():
    tok = CodeAwareTokenizer(language="python")
    assert tok("def foo") == tok.tokenize("def foo")


def test_invalid_language_raises():
    with pytest.raises(ValueError):
        CodeAwareTokenizer(language="cobol")


def test_invalid_min_token_len_raises():
    with pytest.raises(ValueError):
        CodeAwareTokenizer(min_token_len=0)


def test_tokenize_type_error():
    tok = CodeAwareTokenizer()
    with pytest.raises(TypeError):
        tok.tokenize(123)  # type: ignore[arg-type]


# --- performance ----------------------------------------------------------- #


def test_one_megabyte_under_two_seconds():
    # Build ~1 MB of realistic-looking code text.
    snippet = (
        "def getUserName(self, user_id):\n"
        "    return self.db.users.find_by_id(user_id)\n"
        "class HTTPRequestHandler:\n"
        "    pass\n"
        "import os.path\n"
    )
    # snippet is ~150 B; repeat to ~1 MB
    text = snippet * (1_000_000 // len(snippet) + 1)
    assert len(text) >= 1_000_000

    tok = CodeAwareTokenizer(language="python")
    t0 = time.perf_counter()
    out = tok.tokenize(text)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"tokenize 1MB took {elapsed:.3f}s"
    # Sanity: dedup must collapse the output to a small vocabulary.
    assert len(out) < 500
