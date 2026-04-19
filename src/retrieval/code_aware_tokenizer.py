"""Code-aware tokenizer for retrieval over source code corpora.

This is NOT the language-model tokenizer. It is a lightweight, stdlib-only
tokenizer intended to feed BM25- and dense-retrieval pipelines with tokens
that respect the structure of source code: camelCase / snake_case splits,
dotted identifier retention, language-keyword awareness for Python,
JavaScript, Go, Rust, and Java.

Design notes
------------
- Pure stdlib (``re`` only). No foreign imports. No silent fallbacks.
- Output is lower-cased, deduplicated, and ordered by first occurrence of
  each unique token. This is what BM25/TF-IDF-style sparse retrievers
  prefer (unique term presence matters more than raw multiplicity for a
  symbol-heavy vocabulary, and duplicates would otherwise explode from
  repeated identifiers in source code).
- Identifiers with dots (``os.path``, ``foo.bar.baz``) are retained as a
  single compound token AND also emitted as their individual parts, each
  of which is further camel/snake split. This double-coverage is the
  standard trick used in source-code search engines (e.g. Sourcegraph,
  Hound) to make both ``os.path`` and ``path`` retrievable.
- Language detection is a small heuristic scoring function over curated
  keyword / sigil sets. It is intentionally coarse: ``python``,
  ``javascript``, ``go``, ``rust``, ``java``, or ``unknown``.
- Keywords are a small curated set per language; when ``keep_keywords``
  is False they are filtered out, which is useful when the downstream
  retriever would otherwise be dominated by ``def``/``class``/``return``.
"""

from __future__ import annotations

import re
from typing import Iterable

__all__ = ["CodeAwareTokenizer", "KEYWORDS", "SUPPORTED_LANGUAGES"]


# --------------------------------------------------------------------------- #
# Curated keyword sets                                                         #
# --------------------------------------------------------------------------- #
#
# Small, high-signal keyword sets per language. These are used both for
# language detection (presence-scoring) and, when ``keep_keywords=False``,
# for filtering keyword tokens out of the output.

KEYWORDS: dict[str, frozenset[str]] = {
    "python": frozenset({
        "def", "class", "import", "from", "return", "if", "elif", "else",
        "for", "while", "try", "except", "finally", "with", "as", "pass",
        "lambda", "yield", "raise", "global", "nonlocal", "async", "await",
        "self", "none", "true", "false", "and", "or", "not", "in", "is",
    }),
    "javascript": frozenset({
        "function", "var", "let", "const", "return", "if", "else", "for",
        "while", "do", "switch", "case", "break", "continue", "new", "this",
        "class", "extends", "import", "export", "default", "async", "await",
        "try", "catch", "finally", "throw", "typeof", "instanceof", "null",
        "undefined", "true", "false",
    }),
    "go": frozenset({
        "func", "package", "import", "var", "const", "type", "struct",
        "interface", "return", "if", "else", "for", "range", "switch",
        "case", "default", "break", "continue", "go", "defer", "chan",
        "map", "select", "goroutine", "nil", "true", "false",
    }),
    "rust": frozenset({
        "fn", "let", "mut", "pub", "use", "mod", "struct", "enum", "impl",
        "trait", "match", "if", "else", "for", "while", "loop", "return",
        "break", "continue", "async", "await", "move", "ref", "self",
        "super", "crate", "as", "in", "where", "dyn", "unsafe", "true",
        "false",
    }),
    "java": frozenset({
        "public", "private", "protected", "class", "interface", "extends",
        "implements", "static", "final", "abstract", "void", "return",
        "if", "else", "for", "while", "do", "switch", "case", "break",
        "continue", "new", "this", "super", "try", "catch", "finally",
        "throw", "throws", "package", "import", "null", "true", "false",
        "instanceof",
    }),
}

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "python", "javascript", "go", "rust", "java", "unknown",
)

# --------------------------------------------------------------------------- #
# Regex primitives                                                             #
# --------------------------------------------------------------------------- #

# A "raw token" is a run of word characters optionally containing dots
# between word runs (dotted identifiers). The ``re.UNICODE`` flag (default
# in Python 3 for ``\w``) makes this safe for non-ASCII identifiers.
_RAW_TOKEN_RE = re.compile(r"[\w]+(?:\.[\w]+)+|[\w]+", re.UNICODE)

# camelCase / PascalCase splitter. Handles:
#   - getUserName     -> get, User, Name
#   - HTTPServer      -> HTTP, Server
#   - parseXMLDoc     -> parse, XML, Doc
#   - User2Name       -> User, 2, Name (digits kept as their own run)
_CAMEL_SPLIT_RE = re.compile(
    r"[A-Z]+(?=[A-Z][a-z])"    # run of caps followed by Cap+lower (acronym boundary)
    r"|[A-Z]?[a-z]+"           # optional cap + lowers
    r"|[A-Z]+"                 # trailing acronym
    r"|\d+",                   # digit run
)

# Language-detection sigils: features that are hard evidence of a language
# independent of keyword overlap. Each entry contributes a fixed score.
_LANG_SIGILS: tuple[tuple[str, re.Pattern[str], int], ...] = (
    ("python",     re.compile(r"^\s*def\s+\w+\s*\("),                         3),
    ("python",     re.compile(r"^\s*class\s+\w+\s*[\(:]"),                    2),
    ("python",     re.compile(r":\s*$", re.MULTILINE),                        1),
    ("python",     re.compile(r"^\s*from\s+[\w.]+\s+import\b", re.MULTILINE), 3),
    ("javascript", re.compile(r"\bfunction\s+\w+\s*\("),                      3),
    ("javascript", re.compile(r"=>\s*[{(\w]"),                                2),
    ("javascript", re.compile(r"\b(?:const|let|var)\s+\w+\s*="),              2),
    ("go",         re.compile(r"\bfunc\s+(?:\(\w+\s+\*?\w+\)\s*)?\w+\s*\("),  3),
    ("go",         re.compile(r"^\s*package\s+\w+\s*$", re.MULTILINE),        3),
    ("go",         re.compile(r":=\s"),                                       1),
    ("rust",       re.compile(r"\bfn\s+\w+\s*[<(]"),                          3),
    ("rust",       re.compile(r"\blet\s+mut\b"),                              2),
    ("rust",       re.compile(r"\bimpl\b"),                                   2),
    ("rust",       re.compile(r"::\w"),                                       1),
    ("java",       re.compile(r"\bpublic\s+(?:static\s+)?(?:\w+\s+)+\w+\s*\("), 3),
    ("java",       re.compile(r"\bSystem\.out\.print"),                       3),
    ("java",       re.compile(r"\bpackage\s+[\w.]+\s*;"),                     3),
)


# --------------------------------------------------------------------------- #
# Tokenizer                                                                    #
# --------------------------------------------------------------------------- #


class CodeAwareTokenizer:
    """Code-aware retrieval tokenizer.

    Parameters
    ----------
    language:
        One of ``"python"``, ``"javascript"``, ``"go"``, ``"rust"``,
        ``"java"``, ``"unknown"``, or ``"auto"`` (default). When
        ``"auto"``, the tokenizer calls :meth:`detect_language` on every
        input. Otherwise the fixed language is used, which is faster and
        avoids misdetection on tiny snippets.
    keep_keywords:
        If False, drop language keywords from the output. Note that
        ``min_token_len`` is applied after keyword filtering.
    split_case:
        If False, do not split camelCase / snake_case into subparts.
        The original identifier is still emitted (lowered).
    min_token_len:
        Minimum length of an emitted token. Defaults to 2, which drops
        stray single-character fragments (``i``, ``x``) that are near-
        universal in code and add noise to BM25 statistics.
    """

    def __init__(
        self,
        language: str = "auto",
        keep_keywords: bool = True,
        split_case: bool = True,
        min_token_len: int = 2,
    ) -> None:
        if not isinstance(language, str):
            raise TypeError(f"language must be str, got {type(language).__name__}")
        if language != "auto" and language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"language must be 'auto' or one of {SUPPORTED_LANGUAGES}, "
                f"got {language!r}"
            )
        if not isinstance(keep_keywords, bool):
            raise TypeError("keep_keywords must be bool")
        if not isinstance(split_case, bool):
            raise TypeError("split_case must be bool")
        if not isinstance(min_token_len, int) or isinstance(min_token_len, bool):
            raise TypeError("min_token_len must be int")
        if min_token_len < 1:
            raise ValueError(f"min_token_len must be >= 1, got {min_token_len}")

        self.language: str = language
        self.keep_keywords: bool = keep_keywords
        self.split_case: bool = split_case
        self.min_token_len: int = min_token_len

    # -------------------------------------------------------------------- #
    # Language detection                                                    #
    # -------------------------------------------------------------------- #

    def detect_language(self, text: str) -> str:
        """Heuristically classify ``text`` as one of the supported languages.

        Scoring combines (a) sigil matches (strong, language-specific
        regexes) and (b) keyword-set membership over the raw identifier
        stream. Ties are broken by a fixed priority to keep the result
        deterministic.

        Returns ``"unknown"`` if no language accumulates a non-zero score.
        """
        if not isinstance(text, str):
            raise TypeError(f"detect_language expects str, got {type(text).__name__}")
        if not text:
            return "unknown"

        scores: dict[str, int] = {lang: 0 for lang in KEYWORDS}

        # Sigil pass.
        for lang, pat, weight in _LANG_SIGILS:
            if pat.search(text):
                scores[lang] += weight

        # Keyword pass over raw identifiers.
        idents = [m.group(0).lower() for m in re.finditer(r"[A-Za-z_][A-Za-z_0-9]*", text)]
        ident_set = set(idents)
        for lang, kwset in KEYWORDS.items():
            # Count unique keyword hits; this avoids a single repeated
            # keyword dominating the score.
            scores[lang] += len(ident_set & kwset)

        best_lang = "unknown"
        best_score = 0
        # Deterministic priority order for tie-breaking.
        priority = ("python", "javascript", "go", "rust", "java")
        for lang in priority:
            s = scores[lang]
            if s > best_score:
                best_score = s
                best_lang = lang

        return best_lang if best_score > 0 else "unknown"

    # -------------------------------------------------------------------- #
    # Tokenization                                                          #
    # -------------------------------------------------------------------- #

    def tokenize(self, text: str) -> list[str]:
        """Split ``text`` into a list of lower-cased, deduplicated tokens.

        Order is first-occurrence order. The empty string yields ``[]``.
        """
        if not isinstance(text, str):
            raise TypeError(f"tokenize expects str, got {type(text).__name__}")
        if not text:
            return []

        if self.language == "auto":
            lang = self.detect_language(text)
        else:
            lang = self.language

        keywords = KEYWORDS.get(lang, frozenset())

        seen: set[str] = set()
        out: list[str] = []

        def _emit(tok: str) -> None:
            if not tok:
                return
            low = tok.lower()
            if len(low) < self.min_token_len:
                return
            if not self.keep_keywords and low in keywords:
                return
            if low in seen:
                return
            seen.add(low)
            out.append(low)

        for m in _RAW_TOKEN_RE.finditer(text):
            raw = m.group(0)
            if "." in raw:
                # Dotted identifier: emit compound, then each part split.
                _emit(raw)
                for part in raw.split("."):
                    self._emit_identifier(part, _emit)
            else:
                self._emit_identifier(raw, _emit)

        return out

    def _emit_identifier(self, ident: str, emit) -> None:
        """Emit an identifier and, when enabled, its camel/snake splits."""
        if not ident:
            return
        # Always emit the whole identifier.
        emit(ident)
        if not self.split_case:
            return
        # snake_case split.
        if "_" in ident:
            for part in ident.split("_"):
                if not part:
                    continue
                emit(part)
                # Also run camel split on each snake part, because mixed
                # styles (my_getUserName) do occur in real corpora.
                self._emit_camel_parts(part, emit)
        else:
            self._emit_camel_parts(ident, emit)

    @staticmethod
    def _emit_camel_parts(ident: str, emit) -> None:
        """Emit camelCase / PascalCase sub-parts of a single identifier."""
        parts = _CAMEL_SPLIT_RE.findall(ident)
        # Only emit sub-parts if the split is non-trivial (>=2 pieces)
        # and at least one piece differs from the input. This avoids
        # redundant work on plain lowercase tokens.
        if len(parts) < 2:
            return
        for p in parts:
            emit(p)

    # -------------------------------------------------------------------- #
    # Introspection                                                         #
    # -------------------------------------------------------------------- #

    def __call__(self, text: str) -> list[str]:
        """Alias for :meth:`tokenize` so instances plug into BM25Retriever."""
        return self.tokenize(text)

    def __repr__(self) -> str:
        return (
            f"CodeAwareTokenizer(language={self.language!r}, "
            f"keep_keywords={self.keep_keywords}, "
            f"split_case={self.split_case}, "
            f"min_token_len={self.min_token_len})"
        )
