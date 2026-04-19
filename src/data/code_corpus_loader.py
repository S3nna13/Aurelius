"""Code corpus loader for Aurelius.

Walks a directory tree of source code files, tokenizes them via a
caller-supplied (or default) tokenizer, packs the token streams into
fixed-length training sequences separated by a ``<|file_sep|>`` marker,
and optionally augments random spans with fill-in-the-middle (FIM) using
the PSM sentinel layout shared with :mod:`src.chat.fim_formatter`.

This module is intentionally distinct from
:mod:`src.data.conversation` (chat data) and
:mod:`src.data.sequence_packing` (generic packing); it is the
code-specific ingestion path used when pretraining on raw repositories.

Only Python stdlib + torch are imported. Internal ``src.chat`` imports
are used opportunistically for FIM sentinel constants; if unavailable
(import-time failure), local fallbacks take over.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import torch

try:
    from src.chat.fim_formatter import FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE
except Exception:  # pragma: no cover - defensive fallback
    FIM_PREFIX = "<fim_prefix>"
    FIM_SUFFIX = "<fim_suffix>"
    FIM_MIDDLE = "<fim_middle>"

FILE_SEP = "<|file_sep|>"

_LANGUAGE_BY_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "shell",
}

_DEFAULT_EXTS: Tuple[str, ...] = (
    ".py", ".js", ".ts", ".go", ".rs", ".java", ".c", ".cpp", ".h",
)


@dataclass
class CodeChunk:
    """One source file's tokenized representation."""

    file_path: str
    text: str
    language: str
    tokens: List[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers


def _default_tokenizer(text: str) -> List[int]:
    """Fallback tokenizer: byte-level ids in [0, 255]. Pure stdlib."""
    return list(text.encode("utf-8", errors="replace"))


def _is_binary(path: str, sniff: int = 2048) -> bool:
    """Return True if ``path`` is likely binary (NUL in first ``sniff`` bytes)."""
    try:
        with open(path, "rb") as fh:
            chunk = fh.read(sniff)
    except OSError:
        return True
    if b"\x00" in chunk:
        return True
    # Heuristic: high ratio of non-text bytes.
    if not chunk:
        return False
    text_bytes = sum(
        1 for b in chunk
        if b in (9, 10, 13) or 32 <= b < 127 or b >= 128
    )
    return (text_bytes / len(chunk)) < 0.75


_IMPORT_RE_PY = re.compile(r"^\s*(?:from\s+([\w\.]+)|import\s+([\w\.]+))", re.MULTILINE)


def _dependency_order(paths: List[str], root: str) -> List[str]:
    """Attempt a dependency-aware traversal for Python files.

    Files with no intra-repo imports come first, then files whose deps
    are already emitted. Non-python files fall back to path-sorted
    order. Always returns a stable ordering (ties broken by path).
    """
    if not paths:
        return []
    py_files = [p for p in paths if p.endswith(".py")]
    other = sorted(p for p in paths if not p.endswith(".py"))
    if not py_files:
        return other

    # Build module name -> path index.
    mod_to_path: dict[str, str] = {}
    for p in py_files:
        rel = os.path.relpath(p, root)
        mod = rel[:-3].replace(os.sep, ".")
        mod_to_path[mod] = p
        # Also index the file's basename (without .py) for shallow imports.
        base = os.path.splitext(os.path.basename(p))[0]
        mod_to_path.setdefault(base, p)

    deps: dict[str, set[str]] = {p: set() for p in py_files}
    for p in py_files:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                src = fh.read()
        except OSError:
            continue
        for m in _IMPORT_RE_PY.finditer(src):
            name = m.group(1) or m.group(2) or ""
            head = name.split(".")[0]
            target = mod_to_path.get(name) or mod_to_path.get(head)
            if target and target != p:
                deps[p].add(target)

    # Kahn-ish topological sort with deterministic tie-break.
    ordered: List[str] = []
    remaining = set(py_files)
    emitted: set[str] = set()
    while remaining:
        ready = sorted(
            p for p in remaining if deps[p].issubset(emitted)
        )
        if not ready:
            # Cycle or unresolved; break by path-sorted.
            ready = sorted(remaining)
        for p in ready:
            ordered.append(p)
            emitted.add(p)
            remaining.discard(p)
    return ordered + other


# ---------------------------------------------------------------------------
# Loader


class CodeCorpusLoader:
    """Walk a source tree, tokenize files, pack into fixed-length sequences.

    Parameters
    ----------
    tokenizer:
        Callable mapping ``str -> list[int]``. If ``None``, a byte-level
        fallback tokenizer is used.
    extensions:
        File extensions to include (case-insensitive, leading dot).
    file_sep_token_id:
        Token id inserted between consecutive files in the packed stream.
    chunk_size:
        Output sequence length. ``pack_iter`` yields tensors of shape
        ``[chunk_size]``.
    apply_fim:
        If True, each file's text has a probability ``fim_rate`` of being
        FIM-augmented (PSM layout) before tokenization.
    fim_rate:
        Per-file probability of FIM augmentation when ``apply_fim`` is
        True. ``0.0`` disables, ``1.0`` always augments.
    rng:
        Optional ``random.Random`` for deterministic FIM sampling.
    dependency_aware:
        If True, attempt dependency-ordered traversal (Python imports).
    """

    def __init__(
        self,
        tokenizer: Optional[Callable[[str], List[int]]] = None,
        extensions: Tuple[str, ...] = _DEFAULT_EXTS,
        file_sep_token_id: int = 1,
        chunk_size: int = 2048,
        apply_fim: bool = False,
        fim_rate: float = 0.5,
        rng: Optional[random.Random] = None,
        dependency_aware: bool = True,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= fim_rate <= 1.0:
            raise ValueError("fim_rate must be in [0, 1]")
        self.tokenizer = tokenizer if tokenizer is not None else _default_tokenizer
        self.extensions = tuple(e.lower() for e in extensions)
        self.file_sep_token_id = int(file_sep_token_id)
        self.chunk_size = int(chunk_size)
        self.apply_fim = bool(apply_fim)
        self.fim_rate = float(fim_rate)
        self._rng = rng if rng is not None else random.Random()
        self.dependency_aware = bool(dependency_aware)

    # ------------------------------------------------------------------ api
    def detect_language(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return _LANGUAGE_BY_EXT.get(ext, "unknown")

    def _collect_files(self, root: str) -> List[str]:
        paths: List[str] = []
        if not os.path.isdir(root):
            return paths
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip common vendor / VCS dirs.
            dirnames[:] = [
                d for d in dirnames
                if d not in {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv"}
            ]
            for name in filenames:
                if not name.lower().endswith(self.extensions):
                    continue
                full = os.path.join(dirpath, name)
                if _is_binary(full):
                    continue
                paths.append(full)
        if self.dependency_aware:
            return _dependency_order(paths, root)
        return sorted(paths)

    def _maybe_fim(self, text: str) -> str:
        """Randomly wrap a middle span of ``text`` in PSM FIM sentinels."""
        if not self.apply_fim:
            return text
        if self.fim_rate <= 0.0:
            return text
        if self.fim_rate < 1.0 and self._rng.random() >= self.fim_rate:
            return text
        if len(text) < 3:
            return text
        # Choose two cut points on character boundaries.
        a = self._rng.randint(1, len(text) - 2)
        b = self._rng.randint(a + 1, len(text) - 1)
        prefix, middle, suffix = text[:a], text[a:b], text[b:]
        # Guard: FIM sentinels must not appear in the spans.
        if any(s in text for s in (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)):
            return text
        return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"

    def walk(self, root: str) -> Iterator[CodeChunk]:
        """Yield one :class:`CodeChunk` per source file under ``root``."""
        for path in self._collect_files(root):
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            language = self.detect_language(path)
            augmented = self._maybe_fim(text)
            tokens = list(self.tokenizer(augmented))
            yield CodeChunk(
                file_path=path,
                text=text,
                language=language,
                tokens=tokens,
            )

    def pack_iter(self, root: str, batch_size: int = 1) -> Iterator[torch.Tensor]:
        """Yield packed 1-D ``LongTensor`` of length ``chunk_size``.

        Consecutive file token streams are separated by
        ``file_sep_token_id``. Trailing tokens shorter than ``chunk_size``
        are dropped (clean pretraining packing, no padding).

        ``batch_size`` is accepted for API parity; this implementation
        yields one tensor per completed chunk.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        buf: List[int] = []
        first = True
        for chunk in self.walk(root):
            if not first:
                buf.append(self.file_sep_token_id)
            first = False
            buf.extend(chunk.tokens)
            while len(buf) >= self.chunk_size:
                head = buf[: self.chunk_size]
                buf = buf[self.chunk_size:]
                yield torch.tensor(head, dtype=torch.long)


__all__ = [
    "CodeChunk",
    "CodeCorpusLoader",
    "FILE_SEP",
    "FIM_PREFIX",
    "FIM_SUFFIX",
    "FIM_MIDDLE",
]
