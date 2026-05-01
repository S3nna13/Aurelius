from __future__ import annotations

from collections.abc import Callable


class QueryRewriter:
    """Rewrite user queries for better retrieval results.

    Strategies:
    - ``expand``: Add related terms to increase recall
    - ``decompose``: Split complex queries into sub-queries
    - ``rephrase``: Improve query syntax for search engines
    - ``hyde``: Generate a hypothetical answer document for retrieval
    """

    def __init__(
        self,
        strategy: str = "expand",
        expand_fn: Callable[[str], str] | None = None,
        decompose_fn: Callable[[str], list[str]] | None = None,
        generate_fn: Callable[[str], str] | None = None,
    ) -> None:
        if strategy not in ("expand", "rephrase", "decompose", "hyde", "none"):
            raise ValueError(f"Unknown rewrite strategy: {strategy!r}")
        required_fn = {
            "expand": "expand_fn",
            "rephrase": "generate_fn",
            "decompose": "decompose_fn",
            "hyde": "generate_fn",
        }
        if strategy in required_fn:
            fn_name = required_fn[strategy]
            if locals().get(fn_name) is None:
                raise ValueError(f"Strategy {strategy!r} requires {fn_name} argument")
        self.strategy = strategy
        self._expand_fn = expand_fn
        self._decompose_fn = decompose_fn
        self._generate_fn = generate_fn

    def rewrite(self, query: str) -> str | list[str]:
        """Rewrite the query for better retrieval."""
        if self.strategy == "none":
            return query
        if self.strategy == "expand" and self._expand_fn is not None:
            return self._expand_fn(query)
        if self.strategy == "rephrase" and self._generate_fn is not None:
            return self._generate_fn(f"Rephrase this search query for better retrieval: {query}")
        if self.strategy == "decompose" and self._decompose_fn is not None:
            return self._decompose_fn(query)
        if self.strategy == "hyde" and self._generate_fn is not None:
            return self._generate_fn(f"Generate a hypothetical answer document for: {query}")
        return query

    def rewrite_multi(self, query: str) -> list[str]:
        result = self.rewrite(query)
        if isinstance(result, str):
            return [result]
        return result
