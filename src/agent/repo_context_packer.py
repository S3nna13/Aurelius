"""Repo-level context packer for agentic coding loops.

Given a repository root and a natural-language query, produce a bounded,
token-budgeted :class:`RepoContext` bundle consisting of:

1. An ASCII directory tree at adaptive depth.
2. A ranked set of :class:`FileSnippet` excerpts scored by BM25.
3. An imports/dependency summary for the selected files.

This surface is deliberately self-contained. The only internal import is
:mod:`src.retrieval.bm25_retriever` (the sibling BM25 index), and no
third-party libraries are used.

Typical usage::

    packer = RepoContextPacker("/path/to/repo", max_tokens=8000)
    ctx = packer.pack("what does calculate_total do?")
    prompt = ctx.tree + "\\n" + "\\n".join(s.content for s in ctx.snippets)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

from src.retrieval.bm25_retriever import BM25Retriever

_CODE_TOKEN_RE = re.compile(r"[A-Za-z][a-z0-9]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", re.UNICODE)


def _code_tokenizer(text: str) -> list[str]:
    """Code-aware tokenizer: splits on non-word chars AND on snake_/CamelCase.

    ``calculate_total`` -> ``['calculate', 'total']``; ``getHTTPServer`` ->
    ``['get', 'HTTP', 'server']``. Lowercases the final tokens for
    case-insensitive BM25 matching. Pure stdlib regex.
    """
    if not isinstance(text, str):
        raise TypeError("text must be str")
    out: list[str] = []
    for m in _CODE_TOKEN_RE.finditer(text):
        out.append(m.group(0).lower())
    return out

__all__ = ["FileSnippet", "RepoContext", "RepoContextPacker"]


@dataclass(frozen=True)
class FileSnippet:
    """A scored, line-bounded excerpt of a single source file."""

    path: str
    content: str
    score: float
    lines_selected: tuple[int, int]


@dataclass
class RepoContext:
    """Bundle of repo-derived context returned by :meth:`RepoContextPacker.pack`."""

    tree: str
    snippets: list[FileSnippet] = field(default_factory=list)
    imports_summary: dict[str, list[str]] = field(default_factory=dict)
    token_estimate: int = 0


# --------------------------------------------------------------------------- #
# Import extraction regexes (pure-regex; good enough for summary purposes).    #
# --------------------------------------------------------------------------- #

_PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([\w.]+)\s+import\s+|import\s+([\w.]+(?:\s*,\s*[\w.]+)*))",
    re.MULTILINE,
)
# JS/TS: import ... from 'x'; require('x')
_JS_IMPORT_RE = re.compile(
    r"""(?:import\s+(?:[^\n]{0,200}?\s+from\s+)?['"]([^'"]+)['"])"""
    r"""|(?:require\(\s*['"]([^'"]+)['"]\s*\))""",
    re.MULTILINE,
)
# Go: single "fmt" or grouped import blocks. Block content bounded to prevent ReDoS.
_GO_IMPORT_RE = re.compile(r"""import\s*(?:\(\s*([^)]{0,4096})\)|"([^"]+)")""")
# Rust: use a::b::c;
_RS_IMPORT_RE = re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE)
# Java: import foo.bar;
_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+);", re.MULTILINE)


def _default_token_counter(text: str) -> int:
    """Approximate token count as ``ceil(len(text)/4)``.

    Close enough for budgeting against a 4-char-per-token heuristic. Tests
    may inject a precise counter via ``token_counter``.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


def _is_probably_binary(sample: bytes) -> bool:
    """Heuristic binary check: NUL byte or a high fraction of non-text bytes."""
    if b"\x00" in sample:
        return True
    # Text chars per the standard "file" heuristic.
    text_chars = set({7, 8, 9, 10, 12, 13, 27}) | set(range(0x20, 0x7F))
    if not sample:
        return False
    nontext = sum(1 for b in sample if b not in text_chars)
    return nontext / len(sample) > 0.30


class RepoContextPacker:
    """Bounded-token repo context builder for coding agents."""

    def __init__(
        self,
        repo_root: str,
        token_counter: Callable[[str], int] | None = None,
        max_tokens: int = 8000,
        extensions: tuple[str, ...] = (
            ".py",
            ".md",
            ".ts",
            ".js",
            ".go",
            ".rs",
            ".java",
        ),
        exclude_dirs: tuple[str, ...] = (
            ".git",
            "node_modules",
            ".venv",
            "__pycache__",
            ".pytest_cache",
        ),
    ) -> None:
        if not isinstance(repo_root, str) or not repo_root:
            raise ValueError("repo_root must be a non-empty string")
        if not os.path.isdir(repo_root):
            raise ValueError(f"repo_root does not exist or is not a directory: {repo_root!r}")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise ValueError(f"max_tokens must be a positive int, got {max_tokens!r}")
        if not isinstance(extensions, tuple) or not all(
            isinstance(e, str) and e.startswith(".") for e in extensions
        ):
            raise ValueError("extensions must be a tuple of dotted strings")
        if not isinstance(exclude_dirs, tuple):
            raise ValueError("exclude_dirs must be a tuple of strings")
        if token_counter is not None and not callable(token_counter):
            raise TypeError("token_counter must be callable or None")

        self.repo_root: str = os.path.abspath(repo_root)
        self.token_counter: Callable[[str], int] = token_counter or _default_token_counter
        self.max_tokens: int = int(max_tokens)
        self.extensions: tuple[str, ...] = extensions
        self.exclude_dirs: frozenset[str] = frozenset(exclude_dirs)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def build_tree(self, max_depth: int = 3) -> str:
        """Build an ASCII directory tree up to ``max_depth`` levels deep.

        Depth 0 returns only the root label. ``exclude_dirs`` directories
        are pruned. Entries are sorted (dirs then files) for determinism.
        """
        if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 0:
            raise ValueError(f"max_depth must be a non-negative int, got {max_depth!r}")

        root = self.repo_root
        lines: list[str] = [os.path.basename(root.rstrip(os.sep)) or root + "/"]

        def _walk(path: str, prefix: str, depth: int) -> None:
            if depth >= max_depth:
                return
            try:
                entries = os.listdir(path)
            except OSError:
                return
            dirs: list[str] = []
            files: list[str] = []
            for name in entries:
                if name in self.exclude_dirs:
                    continue
                full = os.path.join(path, name)
                if os.path.isdir(full):
                    dirs.append(name)
                elif os.path.isfile(full):
                    files.append(name)
            dirs.sort()
            files.sort()
            combined = [(d, True) for d in dirs] + [(f, False) for f in files]
            n = len(combined)
            for i, (name, is_dir) in enumerate(combined):
                last = i == n - 1
                connector = "`-- " if last else "|-- "
                lines.append(f"{prefix}{connector}{name}{'/' if is_dir else ''}")
                if is_dir:
                    extension = "    " if last else "|   "
                    _walk(os.path.join(path, name), prefix + extension, depth + 1)

        _walk(root, "", 0)
        return "\n".join(lines)

    def extract_imports(self, content: str, language: str = "python") -> list[str]:
        """Extract a de-duplicated, order-preserving list of imported modules."""
        if not isinstance(content, str):
            raise TypeError("content must be a string")
        lang = language.lower()
        found: list[str] = []
        seen: set[str] = set()

        def _add(name: str | None) -> None:
            if not name:
                return
            name = name.strip()
            if not name or name in seen:
                return
            seen.add(name)
            found.append(name)

        if lang in ("python", "py"):
            for m in _PY_IMPORT_RE.finditer(content):
                if m.group(1):
                    _add(m.group(1))
                elif m.group(2):
                    for part in m.group(2).split(","):
                        _add(part.strip())
        elif lang in ("javascript", "js", "typescript", "ts"):
            for m in _JS_IMPORT_RE.finditer(content):
                _add(m.group(1) or m.group(2))
        elif lang == "go":
            for m in _GO_IMPORT_RE.finditer(content):
                block, single = m.group(1), m.group(2)
                if single:
                    _add(single)
                elif block:
                    for line in block.splitlines():
                        line = line.strip()
                        if not line or line.startswith("//"):
                            continue
                        # Strip alias and quotes.
                        parts = line.split()
                        quoted = parts[-1].strip('"')
                        _add(quoted)
        elif lang in ("rust", "rs"):
            for m in _RS_IMPORT_RE.finditer(content):
                _add(m.group(1))
        elif lang == "java":
            for m in _JAVA_IMPORT_RE.finditer(content):
                _add(m.group(1))
        else:
            # Unknown language: best-effort python-style pass.
            for m in _PY_IMPORT_RE.finditer(content):
                _add(m.group(1) or m.group(2))

        return found

    def pack(self, query: str) -> RepoContext:
        """Produce a :class:`RepoContext` bounded by ``max_tokens``."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        files = self._discover_files()
        if not files:
            return RepoContext(tree="", snippets=[], imports_summary={}, token_estimate=0)

        # Adaptive depth: shallow trees for large repos.
        if len(files) > 150:
            depth = 2
        elif len(files) > 50:
            depth = 3
        else:
            depth = 4
        tree = self.build_tree(max_depth=depth)

        # Budget accounting.
        budget = self.max_tokens
        tree_cost = self.token_counter(tree)
        # If the tree alone exceeds the budget, truncate it aggressively so we
        # still leave headroom for at least one snippet. Worst case: trim to
        # the root label only.
        if tree_cost >= budget:
            tree = os.path.basename(self.repo_root.rstrip(os.sep)) or self.repo_root
            tree_cost = self.token_counter(tree)
        remaining = max(0, budget - tree_cost)

        # Build BM25 over file contents (deterministic corpus order).
        contents: list[str] = []
        valid_paths: list[str] = []
        for path in files:
            text = self._read_text(path)
            if text is None:
                continue
            valid_paths.append(path)
            contents.append(text)

        snippets: list[FileSnippet] = []
        imports_summary: dict[str, list[str]] = {}

        if not valid_paths or not query.strip() or not _code_tokenizer(query):
            return RepoContext(
                tree=tree,
                snippets=[],
                imports_summary={},
                token_estimate=tree_cost,
            )

        retriever = BM25Retriever(tokenizer=_code_tokenizer)
        retriever.add_documents(contents)
        ranked = retriever.query(query, k=len(valid_paths))
        if not ranked:
            return RepoContext(
                tree=tree,
                snippets=[],
                imports_summary={},
                token_estimate=tree_cost,
            )

        q_tokens = set(_code_tokenizer(query))
        used = 0
        for doc_id, score in ranked:
            path = valid_paths[doc_id]
            content = contents[doc_id]
            rel = os.path.relpath(path, self.repo_root)
            snippet_text, line_range = self._select_lines(content, q_tokens, context=2)
            header = f"# {rel} (score={score:.4f}, lines {line_range[0]}-{line_range[1]})\n"
            block = header + snippet_text
            cost = self.token_counter(block)
            if cost > remaining - used:
                # Try to shrink the snippet by narrowing context further.
                snippet_text, line_range = self._select_lines(content, q_tokens, context=0)
                block = header + snippet_text
                cost = self.token_counter(block)
                if cost > remaining - used:
                    continue
            snippets.append(
                FileSnippet(
                    path=rel,
                    content=block,
                    score=float(score),
                    lines_selected=line_range,
                )
            )
            used += cost
            imports_summary[rel] = self.extract_imports(
                content, language=self._language_for(path)
            )
            if used >= remaining:
                break

        return RepoContext(
            tree=tree,
            snippets=snippets,
            imports_summary=imports_summary,
            token_estimate=tree_cost + used,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _discover_files(self) -> list[str]:
        """Deterministic DFS over the repo, honoring excludes + extensions."""
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.repo_root):
            # Prune in-place for os.walk.
            dirnames[:] = sorted(d for d in dirnames if d not in self.exclude_dirs)
            for name in sorted(filenames):
                if name.endswith(self.extensions):
                    out.append(os.path.join(dirpath, name))
        return out

    def _read_text(self, path: str) -> str | None:
        """Read a file as UTF-8 text; return ``None`` if binary or unreadable."""
        try:
            with open(path, "rb") as f:
                sample = f.read(4096)
            if _is_probably_binary(sample):
                return None
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError):
            return None

    def _language_for(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".md": "markdown",
        }.get(ext, "text")

    def _select_lines(
        self,
        content: str,
        q_tokens: set[str],
        context: int = 2,
    ) -> tuple[str, tuple[int, int]]:
        """Pick the smallest span of matching lines + surrounding context.

        Small files (<= 40 lines) are returned whole. Otherwise the first
        matching line and its neighborhood is returned. If no line matches,
        the first chunk of the file is returned as a fallback summary.
        """
        lines = content.splitlines()
        n = len(lines)
        if n == 0:
            return "", (1, 1)
        if n <= 40:
            return content, (1, n)

        matches: list[int] = []
        q_lower = q_tokens
        for i, line in enumerate(lines):
            toks = set(_code_tokenizer(line))
            if toks & q_lower:
                matches.append(i)
            if len(matches) >= 8:
                break

        if not matches:
            end = min(n, 20)
            return "\n".join(lines[:end]), (1, end)

        first = matches[0]
        last = matches[-1]
        start = max(0, first - context)
        stop = min(n, last + context + 1)
        return "\n".join(lines[start:stop]), (start + 1, stop)
