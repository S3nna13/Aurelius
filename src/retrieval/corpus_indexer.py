"""Corpus indexer for retrieval.

Walks a filesystem tree, skips binary files via a null-byte heuristic,
chunks each text file into fixed-size overlapping character segments,
and (optionally) builds an Okapi BM25 index over those chunks for
retrieval. Persists the chunk set to a JSON sidecar for deterministic
reload without re-walking the corpus.

Constraints: pure stdlib + :class:`src.retrieval.bm25_retriever.BM25Retriever`.
No foreign imports.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Callable

from .bm25_retriever import BM25Retriever

__all__ = ["Chunk", "CorpusIndexer"]


# Null-byte heuristic window: read this many bytes to decide binary-ness.
_BINARY_SNIFF_BYTES = 8192


@dataclass
class Chunk:
    """A fixed-size overlapping character slice of a source file.

    ``chunk_id`` is a deterministic hash of ``(source_path, start_char,
    end_char, content)`` so that re-chunking the same file produces
    stable ids across runs. ``start_char``/``end_char`` are half-open
    character offsets into the *source text* (NOT bytes).
    """

    chunk_id: str
    source_path: str
    content: str
    start_char: int
    end_char: int
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _is_binary_file(path: str) -> bool:
    """Return True if ``path`` looks like a binary file.

    Heuristic: a NUL byte in the first :data:`_BINARY_SNIFF_BYTES` is a
    strong signal. This is the same check used by ``git`` and by most
    text-centric indexers (grep, ripgrep). Cheap, no decoding required.
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(_BINARY_SNIFF_BYTES)
    except OSError:
        return True
    return b"\x00" in chunk


class CorpusIndexer:
    """Walk, chunk, and index a filesystem corpus.

    Parameters
    ----------
    chunk_size:
        Number of characters per chunk. Must be a positive int.
    chunk_overlap:
        Number of characters shared between successive chunks. Must be
        non-negative and strictly less than ``chunk_size`` (otherwise
        the sliding window would not advance).
    extensions:
        Tuple of lowercase file extensions (including the leading dot)
        to include when walking. Files with other extensions are
        skipped. Pass an empty tuple to accept everything.
    tokenizer:
        Optional callable ``str -> list[str]`` forwarded to the
        :class:`BM25Retriever` built by :meth:`build_bm25_index`. If
        ``None``, the retriever's default tokenizer is used.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        extensions: tuple = (".py", ".md", ".txt"),
        tokenizer: Callable[[str], list[str]] | None = None,
    ) -> None:
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size <= 0
        ):
            raise ValueError(
                f"chunk_size must be a positive int, got {chunk_size!r}"
            )
        if (
            not isinstance(chunk_overlap, int)
            or isinstance(chunk_overlap, bool)
            or chunk_overlap < 0
        ):
            raise ValueError(
                f"chunk_overlap must be a non-negative int, got {chunk_overlap!r}"
            )
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be < chunk_size "
                f"({chunk_size}); otherwise the window does not advance."
            )
        if not isinstance(extensions, tuple):
            raise TypeError(
                f"extensions must be a tuple, got {type(extensions).__name__}"
            )
        if tokenizer is not None and not callable(tokenizer):
            raise TypeError("tokenizer must be callable or None")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Normalize to lowercase for case-insensitive matching on all OSes.
        self.extensions = tuple(e.lower() for e in extensions)
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------ #
    # Filesystem walk                                                     #
    # ------------------------------------------------------------------ #

    def walk_files(self, root: str) -> list[str]:
        """Recursively enumerate files under ``root`` matching ``extensions``.

        Returns absolute paths sorted lexicographically for determinism.
        If ``extensions`` is an empty tuple, all files are returned.
        Does *not* filter binary files here -- that happens at chunk
        time so ``walk_files`` remains a pure directory listing.
        """
        if not isinstance(root, str):
            raise TypeError(f"root must be str, got {type(root).__name__}")
        if not os.path.isdir(root):
            raise NotADirectoryError(f"root is not a directory: {root!r}")

        matches: list[str] = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                if self.extensions:
                    _, ext = os.path.splitext(name)
                    if ext.lower() not in self.extensions:
                        continue
                matches.append(os.path.abspath(os.path.join(dirpath, name)))
        matches.sort()
        return matches

    # ------------------------------------------------------------------ #
    # Chunking                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _chunk_id(source_path: str, start: int, end: int, content: str) -> str:
        """Deterministic SHA-1 over (path, span, content).

        Includes the content so that edits to a file produce new ids
        even if spans collide; includes the span so that two distinct
        windows with identical content (possible for blank regions) are
        still distinguishable.
        """
        h = hashlib.sha1(usedforsecurity=False)
        h.update(source_path.encode("utf-8"))
        h.update(b"\x00")
        h.update(str(start).encode("ascii"))
        h.update(b"\x00")
        h.update(str(end).encode("ascii"))
        h.update(b"\x00")
        h.update(content.encode("utf-8"))
        return h.hexdigest()

    def chunk_text(self, text: str, source_path: str) -> list[Chunk]:
        """Slice ``text`` into overlapping fixed-size character chunks.

        For text shorter than ``chunk_size`` a single chunk spanning the
        full string is returned. Empty text returns ``[]``.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not isinstance(source_path, str):
            raise TypeError("source_path must be str")

        n = len(text)
        if n == 0:
            return []

        step = self.chunk_size - self.chunk_overlap  # > 0 by ctor invariant
        chunks: list[Chunk] = []

        start = 0
        idx = 0
        while start < n:
            end = min(start + self.chunk_size, n)
            content = text[start:end]
            chunks.append(
                Chunk(
                    chunk_id=self._chunk_id(source_path, start, end, content),
                    source_path=source_path,
                    content=content,
                    start_char=start,
                    end_char=end,
                    chunk_index=idx,
                    metadata={},
                )
            )
            idx += 1
            if end == n:
                break
            start += step
        return chunks

    def chunk_file(self, path: str) -> list[Chunk]:
        """Read ``path`` as UTF-8 text and chunk it.

        Returns ``[]`` for binary files (null-byte heuristic), files
        that fail to decode as UTF-8, or empty files. Non-empty decoded
        files get a ``metadata["size_bytes"]`` annotation.
        """
        if not isinstance(path, str):
            raise TypeError("path must be str")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"not a file: {path!r}")

        if _is_binary_file(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            return []

        size = os.path.getsize(path)
        chunks = self.chunk_text(text, path)
        for c in chunks:
            c.metadata["size_bytes"] = size
        return chunks

    # ------------------------------------------------------------------ #
    # Index construction                                                  #
    # ------------------------------------------------------------------ #

    def build_bm25_index(
        self, chunks: list[Chunk]
    ) -> tuple[BM25Retriever, dict[int, Chunk]]:
        """Build a BM25 retriever over ``chunks``.

        Returns the indexed retriever plus a ``{doc_id -> Chunk}`` map
        (doc_id matches the positional index of the chunk passed in).
        Raises ``ValueError`` if ``chunks`` is empty, since BM25 IDF is
        undefined without a corpus.
        """
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list[Chunk]")
        if not chunks:
            raise ValueError("build_bm25_index requires at least one chunk")

        retriever = BM25Retriever(tokenizer=self.tokenizer)
        retriever.add_documents([c.content for c in chunks])
        doc_map = {i: c for i, c in enumerate(chunks)}
        return retriever, doc_map

    # ------------------------------------------------------------------ #
    # Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save_index(self, chunks: list[Chunk], path: str) -> None:
        """Atomically serialize ``chunks`` as JSON to ``path``.

        Uses a ``tempfile`` + ``os.replace`` dance so that a crash
        mid-write cannot corrupt an existing index file; readers either
        see the old version or the new version, never a partial one.
        """
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list[Chunk]")
        if not isinstance(path, str):
            raise TypeError("path must be str")

        payload = {
            "version": 1,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "chunks": [asdict(c) for c in chunks],
        }
        directory = os.path.dirname(os.path.abspath(path)) or "."
        os.makedirs(directory, exist_ok=True)

        fd, tmp = tempfile.mkstemp(
            prefix=".corpus_index.", suffix=".json.tmp", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
        except Exception:
            # Best-effort cleanup; do not mask the original exception.
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def load_index(self, path: str) -> list[Chunk]:
        """Load a JSON index previously written by :meth:`save_index`.

        Does not attempt schema migration; if ``version`` is unknown,
        raises ``ValueError``. Chunk fields are validated structurally.
        """
        if not isinstance(path, str):
            raise TypeError("path must be str")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            raise ValueError(f"index at {path!r} is not a JSON object")
        if payload.get("version") != 1:
            raise ValueError(
                f"unsupported index version {payload.get('version')!r} at {path!r}"
            )
        raw = payload.get("chunks")
        if not isinstance(raw, list):
            raise ValueError(f"index at {path!r} missing 'chunks' list")
        out: list[Chunk] = []
        for i, d in enumerate(raw):
            if not isinstance(d, dict):
                raise ValueError(f"chunks[{i}] is not an object")
            try:
                out.append(
                    Chunk(
                        chunk_id=d["chunk_id"],
                        source_path=d["source_path"],
                        content=d["content"],
                        start_char=d["start_char"],
                        end_char=d["end_char"],
                        chunk_index=d["chunk_index"],
                        metadata=d.get("metadata", {}) or {},
                    )
                )
            except KeyError as e:
                raise ValueError(
                    f"chunks[{i}] missing required field {e.args[0]!r}"
                ) from e
        return out
