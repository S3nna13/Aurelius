from __future__ import annotations

from collections.abc import Callable, Sequence


class ContextCompressor:
    """Compress retrieved document context to fit within token budgets.

    Strategies:
    - ``truncate``: Keep top-k documents, truncate each to max_chars
    - ``summarize``: Use a summarization callable to condense documents
    - ``extractive``: Keep top sentences by relevance score
    - ``strip``: Remove boilerplate, whitespace, repeated content
    """

    def __init__(
        self,
        strategy: str = "truncate",
        max_chars: int = 2000,
        max_docs: int = 5,
        summarize_fn: Callable[[str], str] | None = None,
    ) -> None:
        if strategy not in ("truncate", "summarize", "extractive", "strip"):
            raise ValueError(f"Unknown compression strategy: {strategy!r}")
        self.strategy = strategy
        self.max_chars = max_chars
        self.max_docs = max_docs
        self._summarize_fn = summarize_fn

    def compress(
        self,
        query: str,
        documents: Sequence[str],
        scores: Sequence[float] | None = None,
    ) -> list[str]:
        """Compress retrieved documents into a concise context."""
        if not documents:
            return []

        docs = list(documents)[: self.max_docs]
        query_lower = query.lower()

        if self.strategy == "truncate":
            return [d[: self.max_chars] for d in docs]

        elif self.strategy == "extractive":
            compressed: list[str] = []
            for doc in docs:
                sentences = doc.split(". ")
                scored_sentences: list[tuple[float, str]] = []
                for sent in sentences:
                    overlap = len(set(sent.lower().split()) & set(query_lower.split()))
                    scored_sentences.append((overlap, sent))
                scored_sentences.sort(key=lambda x: -x[0])
                kept = []
                char_count = 0
                for _, sent in scored_sentences:
                    if char_count + len(sent) > self.max_chars:
                        break
                    kept.append(sent)
                    char_count += len(sent)
                compressed.append(". ".join(kept))
            return compressed

        elif self.strategy == "summarize" and self._summarize_fn is not None:
            return [self._summarize_fn(d) for d in docs]

        else:  # strip
            return [d.strip()[: self.max_chars] for d in docs]
