"""Code-aware dense embedder.

Produces a pooled dense vector from a source-code snippet by weighting
per-token embeddings according to code-specific features (identifiers,
import lines, function/class signatures, comments).

This module sits on top of :mod:`src.retrieval.code_aware_tokenizer` and
accepts a caller-supplied ``token_embed_fn`` so that the embedder itself
remains caller-agnostic: it can be driven by a hash stub for tests, by
the trained :class:`DenseEmbedder`, or by any other callable producing
fixed-size float vectors.

Design notes
------------
- Pure stdlib. No torch. ``token_embed_fn`` may wrap torch if the caller
  wants to, but this file does not import torch.
- Feature extraction is regex-based and deterministic. Python keywords
  are filtered from identifiers to keep signal high.
- Pooling is a weighted sum over unique tokens (one contribution per
  unique token string). Tokens that appear in multiple feature buckets
  take the maximum of their bucket weights (signatures > identifiers
  > imports > comments > default). This matches the intuition that a
  function name in a signature line should not be double-counted as an
  "identifier" too.
- The pooled vector is L2-normalized. Empty input yields a zero vector
  (no normalization).
"""

from __future__ import annotations

import hashlib
import keyword
import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .code_aware_tokenizer import CodeAwareTokenizer

__all__ = [
    "CodeFeatures",
    "CodeAwareEmbedder",
    "split_identifier",
    "stub_token_embed",
]


# --------------------------------------------------------------------------- #
# Constants / regexes                                                          #
# --------------------------------------------------------------------------- #

_MAX_CODE_CHARS = 200_000  # defensive truncation for very long inputs.

_IDENT_RE = re.compile(r"[^\W\d][\w]*", re.UNICODE)  # starts with letter/_, unicode ok.
_IMPORT_RE = re.compile(r"^\s*(?:import\s+\S+|from\s+\S+\s+import\s+.+)", re.MULTILINE)
_SIGNATURE_RE = re.compile(
    r"^\s*(?:def\s+\w+\s*\(.*\)|class\s+\w+.*:?)",
    re.MULTILINE,
)
_LINE_COMMENT_RE = re.compile(r"^\s*#.*$", re.MULTILINE)
_TRIPLE_STRING_RE = re.compile(r"(?:\"\"\"[\s\S]*?\"\"\"|'''[\s\S]*?''')")

_PY_KEYWORDS: frozenset[str] = frozenset(keyword.kwlist) | frozenset({
    "self", "cls", "True", "False", "None",
})


# --------------------------------------------------------------------------- #
# Identifier splitting                                                         #
# --------------------------------------------------------------------------- #


def split_identifier(name: str) -> tuple[str, ...]:
    """Split an identifier into snake_case and CamelCase sub-parts.

    The original identifier is *not* included in the returned tuple; only
    its decomposed pieces. Empty / non-string input returns ``()``.

    Examples
    --------
    >>> split_identifier("getUserName")
    ('get', 'User', 'Name')
    >>> split_identifier("HTTP_server")
    ('HTTP', 'server')
    >>> split_identifier("parseXMLDoc")
    ('parse', 'XML', 'Doc')
    """
    if not isinstance(name, str) or not name:
        return ()
    # First snake-split, then camel-split each piece.
    out: list[str] = []
    snake_parts = [p for p in name.split("_") if p]
    for part in snake_parts:
        # camel splitter: acronym run, cap+lowers, trailing acronym, digit run.
        pieces = re.findall(
            r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+",
            part,
        )
        if len(pieces) <= 1:
            out.append(part)
        else:
            out.extend(pieces)
    # Drop trivial empty results.
    return tuple(p for p in out if p)


# --------------------------------------------------------------------------- #
# Features                                                                    #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CodeFeatures:
    """Structured features extracted from a source-code snippet."""
    identifiers: tuple[str, ...]
    imports: tuple[str, ...]
    signatures: tuple[str, ...]
    comments: tuple[str, ...]
    n_tokens: int


# --------------------------------------------------------------------------- #
# Deterministic stub embedder for tests                                        #
# --------------------------------------------------------------------------- #


def stub_token_embed(token: str, d_embed: int = 384) -> list[float]:
    """Deterministic hash-seeded embedding of ``token`` into R^d_embed.

    Intended for tests. Uses SHA-256 of the token to seed a simple LCG and
    fill a vector of floats in (-1, 1). Purely deterministic; no RNG state
    is shared across calls.
    """
    if not isinstance(token, str):
        raise TypeError(f"stub_token_embed expects str, got {type(token).__name__}")
    if d_embed <= 0:
        raise ValueError(f"d_embed must be positive, got {d_embed}")
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    # Seed a 64-bit state from the first 8 bytes.
    state = int.from_bytes(digest[:8], "big") or 1
    out: list[float] = []
    # Numerical Recipes LCG constants.
    a = 6364136223846793005
    c = 1442695040888963407
    mod = 1 << 64
    for _ in range(d_embed):
        state = (a * state + c) % mod
        # Map to (-1, 1).
        out.append((state / mod) * 2.0 - 1.0)
    return out


# --------------------------------------------------------------------------- #
# Embedder                                                                    #
# --------------------------------------------------------------------------- #


class CodeAwareEmbedder:
    """Pools per-token embeddings with code-aware weight boosts.

    Parameters
    ----------
    token_embed_fn:
        Callable mapping a token string to a fixed-size list of floats of
        length ``d_embed``. The callable is invoked exactly once per
        unique token per :meth:`embed` call.
    d_embed:
        Expected dimensionality of each token embedding and of the pooled
        output vector. Token-embed-fn outputs of the wrong size raise
        ``ValueError``.
    identifier_weight, signature_weight, comment_weight, import_weight:
        Per-bucket weights applied during weighted pooling. All must be
        finite and non-negative.
    """

    def __init__(
        self,
        token_embed_fn: Callable[[str], Sequence[float]],
        d_embed: int = 384,
        identifier_weight: float = 2.0,
        signature_weight: float = 1.5,
        comment_weight: float = 0.7,
        import_weight: float = 1.2,
    ) -> None:
        if not callable(token_embed_fn):
            raise TypeError("token_embed_fn must be callable")
        if not isinstance(d_embed, int) or isinstance(d_embed, bool) or d_embed <= 0:
            raise ValueError(f"d_embed must be a positive int, got {d_embed!r}")
        for label, w in (
            ("identifier_weight", identifier_weight),
            ("signature_weight", signature_weight),
            ("comment_weight", comment_weight),
            ("import_weight", import_weight),
        ):
            if not isinstance(w, (int, float)) or isinstance(w, bool):
                raise TypeError(f"{label} must be a number, got {type(w).__name__}")
            if math.isnan(float(w)) or math.isinf(float(w)) or float(w) < 0.0:
                raise ValueError(f"{label} must be finite and >= 0, got {w}")

        self.token_embed_fn = token_embed_fn
        self.d_embed = d_embed
        self.identifier_weight = float(identifier_weight)
        self.signature_weight = float(signature_weight)
        self.comment_weight = float(comment_weight)
        self.import_weight = float(import_weight)
        self._tokenizer = CodeAwareTokenizer(language="python", min_token_len=1)

    # ------------------------------------------------------------------ #
    # Feature extraction                                                  #
    # ------------------------------------------------------------------ #

    def extract_features(self, code: str) -> CodeFeatures:
        """Extract code-aware features from ``code``.

        Returns a :class:`CodeFeatures` with deduplicated, order-preserving
        tuples for identifiers, import lines, signature lines, and
        comment strings, plus a total token count (identifier hits).
        """
        if not isinstance(code, str):
            raise TypeError(f"extract_features expects str, got {type(code).__name__}")
        if len(code) > _MAX_CODE_CHARS:
            code = code[:_MAX_CODE_CHARS]

        imports = tuple(_dedup(m.group(0).strip() for m in _IMPORT_RE.finditer(code)))
        signatures = tuple(_dedup(m.group(0).strip() for m in _SIGNATURE_RE.finditer(code)))

        comments_list: list[str] = []
        for m in _LINE_COMMENT_RE.finditer(code):
            comments_list.append(m.group(0).strip())
        for m in _TRIPLE_STRING_RE.finditer(code):
            comments_list.append(m.group(0).strip())
        comments = tuple(_dedup(comments_list))

        idents_all = [m.group(0) for m in _IDENT_RE.finditer(code)]
        n_tokens = len(idents_all)
        idents = tuple(
            _dedup(tok for tok in idents_all if tok not in _PY_KEYWORDS)
        )
        return CodeFeatures(
            identifiers=idents,
            imports=imports,
            signatures=signatures,
            comments=comments,
            n_tokens=n_tokens,
        )

    # ------------------------------------------------------------------ #
    # Embedding                                                           #
    # ------------------------------------------------------------------ #

    def embed(self, code: str) -> list[float]:
        """Embed ``code`` into an L2-normalized vector of length ``d_embed``.

        Returns an all-zero vector when ``code`` is empty or contains no
        tokens.
        """
        if not isinstance(code, str):
            raise TypeError(f"embed expects str, got {type(code).__name__}")
        zero = [0.0] * self.d_embed
        if not code:
            return zero

        features = self.extract_features(code)

        # Build per-token weight dictionary. Signature > identifier >
        # import > comment > default. Tokens that appear in multiple
        # buckets take the *max* weight (not a sum).
        weights: dict[str, float] = {}

        def _upsert(tok: str, w: float) -> None:
            if not tok:
                return
            prev = weights.get(tok)
            if prev is None or w > prev:
                weights[tok] = w

        # Default-weight pass: every identifier gets at least weight 1.0
        # (this is the "code but not special" bucket). The keyword-filter
        # from extract_features already dropped Python keywords.
        # Sub-parts from camel/snake splitting get identifier_weight,
        # which is where the "identifier boost" actually lives.
        for ident in features.identifiers:
            _upsert(ident, 1.0)
            for part in split_identifier(ident):
                _upsert(part, self.identifier_weight)

        # Import tokens.
        for line in features.imports:
            for tok in _IDENT_RE.findall(line):
                if tok in _PY_KEYWORDS:
                    continue
                _upsert(tok, self.import_weight)

        # Signature tokens (boosts identifiers appearing in def/class lines).
        for line in features.signatures:
            for tok in _IDENT_RE.findall(line):
                if tok in _PY_KEYWORDS:
                    continue
                _upsert(tok, self.signature_weight)

        # Comment tokens.
        for line in features.comments:
            for tok in _IDENT_RE.findall(line):
                if tok in _PY_KEYWORDS:
                    continue
                _upsert(tok, self.comment_weight)

        if not weights:
            return zero

        # Weighted sum pool.
        acc = [0.0] * self.d_embed
        for tok, w in weights.items():
            vec = self.token_embed_fn(tok)
            if not hasattr(vec, "__len__") or len(vec) != self.d_embed:
                raise ValueError(
                    f"token_embed_fn returned vector of length "
                    f"{len(vec) if hasattr(vec, '__len__') else '?'}, "
                    f"expected {self.d_embed}"
                )
            for i, v in enumerate(vec):
                acc[i] += w * float(v)

        # L2 normalize.
        norm = math.sqrt(sum(x * x for x in acc))
        if norm == 0.0:
            return zero
        return [x / norm for x in acc]

    def embed_batch(self, codes: list[str]) -> list[list[float]]:
        """Embed a list of code snippets; output length matches input."""
        if not isinstance(codes, list):
            raise TypeError(f"embed_batch expects list, got {type(codes).__name__}")
        return [self.embed(c) for c in codes]

    def __repr__(self) -> str:
        return (
            f"CodeAwareEmbedder(d_embed={self.d_embed}, "
            f"identifier_weight={self.identifier_weight}, "
            f"signature_weight={self.signature_weight}, "
            f"comment_weight={self.comment_weight}, "
            f"import_weight={self.import_weight})"
        )


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def _dedup(items) -> list[str]:
    """Order-preserving dedup of a string iterable."""
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out
