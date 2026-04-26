"""AST-aware code search for Python source files.

Inspired by repo-map in Aider (Aider-AI/aider, Apache-2.0) and
symbol indexing in SWE-agent (SWE-agent/SWE-agent, MIT);
Aurelius-native clean-room implementation. License: MIT.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

_MAX_SOURCE_LEN = 500_000
_MAX_FILENAME_LEN = 512
_MAX_RESULTS = 500
_MAX_QUERY_LEN = 1024


@dataclass
class CodeSymbol:
    """A named symbol extracted from an AST."""

    name: str
    kind: str  # "function", "class", "variable", "import"
    filename: str
    lineno: int
    col_offset: int = 0
    docstring: str = ""


@dataclass
class CodeFile:
    """Indexed Python source file."""

    filename: str
    source: str
    symbols: list[CodeSymbol] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lines = self.source.splitlines()


def extract_symbols(filename: str, source: str) -> list[CodeSymbol]:
    """Parse source with ast.parse and extract all named symbols."""
    symbols = []
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        return symbols  # non-fatal: return empty for unparseable files

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node) or ""
            symbols.append(
                CodeSymbol(
                    name=node.name,
                    kind="function",
                    filename=filename,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    docstring=doc[:256],
                )
            )
        elif isinstance(node, ast.ClassDef):
            doc = ast.get_docstring(node) or ""
            symbols.append(
                CodeSymbol(
                    name=node.name,
                    kind="class",
                    filename=filename,
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                    docstring=doc[:256],
                )
            )
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.append(
                        CodeSymbol(
                            name=target.id,
                            kind="variable",
                            filename=filename,
                            lineno=node.lineno,
                        )
                    )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                symbols.append(
                    CodeSymbol(
                        name=alias.asname or alias.name,
                        kind="import",
                        filename=filename,
                        lineno=node.lineno,
                    )
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                symbols.append(
                    CodeSymbol(
                        name=alias.asname or alias.name,
                        kind="import",
                        filename=filename,
                        lineno=node.lineno,
                    )
                )
    return symbols


class CodeSearchIndex:
    """Indexes Python source files for symbol and text search."""

    def __init__(self) -> None:
        self._files: dict[str, CodeFile] = {}  # filename -> CodeFile
        self._symbol_map: dict[str, list[CodeSymbol]] = {}  # name -> symbols

    def add_file(self, filename: str, source: str) -> int:
        """Index a source file. Returns number of symbols extracted.
        Raises ValueError for oversized inputs.
        """
        if len(filename) > _MAX_FILENAME_LEN:
            raise ValueError(f"filename exceeds {_MAX_FILENAME_LEN} chars")
        if len(source) > _MAX_SOURCE_LEN:
            raise ValueError(f"source exceeds {_MAX_SOURCE_LEN} chars")
        symbols = extract_symbols(filename, source)
        self._files[filename] = CodeFile(filename=filename, source=source, symbols=symbols)
        for sym in symbols:
            self._symbol_map.setdefault(sym.name, []).append(sym)
        return len(symbols)

    def remove_file(self, filename: str) -> None:
        """Remove a file from the index. No-op if not present."""
        if filename not in self._files:
            return
        old_symbols = self._files[filename].symbols
        for sym in old_symbols:
            if sym.name in self._symbol_map:
                self._symbol_map[sym.name] = [
                    s for s in self._symbol_map[sym.name] if s.filename != filename
                ]
                if not self._symbol_map[sym.name]:
                    del self._symbol_map[sym.name]
        del self._files[filename]

    def search_symbols(
        self, query: str, kind: str | None = None, top_k: int = 20
    ) -> list[CodeSymbol]:
        """Search symbols by name prefix or substring.
        kind: filter by "function", "class", "variable", "import", or None for all.
        """
        if len(query) > _MAX_QUERY_LEN:
            raise ValueError(f"query exceeds {_MAX_QUERY_LEN} chars")
        if top_k > _MAX_RESULTS:
            raise ValueError(f"top_k exceeds {_MAX_RESULTS}")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        query_lower = query.lower()
        results: list[CodeSymbol] = []
        for name, syms in self._symbol_map.items():
            if query_lower in name.lower():
                for sym in syms:
                    if kind is None or sym.kind == kind:
                        results.append(sym)

        # Sort: exact matches first, then prefix, then contains; then by lineno
        def sort_key(s: CodeSymbol) -> tuple[int, int]:
            n = s.name.lower()
            q = query_lower
            if n == q:
                return (0, s.lineno)
            if n.startswith(q):
                return (1, s.lineno)
            return (2, s.lineno)

        results.sort(key=sort_key)
        return results[:top_k]

    def text_search(self, pattern: str, top_k: int = 20) -> list[tuple[str, int, str]]:
        """Regex search over source lines. Returns (filename, lineno, line) tuples."""
        if len(pattern) > _MAX_QUERY_LEN:
            raise ValueError(f"pattern exceeds {_MAX_QUERY_LEN} chars")
        if top_k > _MAX_RESULTS:
            raise ValueError(f"top_k exceeds {_MAX_RESULTS}")
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise ValueError(f"invalid regex: {e}") from e
        results: list[tuple[str, int, str]] = []
        for fname, cfile in self._files.items():
            for i, line in enumerate(cfile.lines, start=1):
                if compiled.search(line):
                    results.append((fname, i, line))
                    if len(results) >= top_k:
                        return results
        return results

    def list_files(self) -> list[str]:
        return sorted(self._files.keys())

    def __len__(self) -> int:
        return len(self._files)


CODE_SEARCH_INDEX = CodeSearchIndex()
