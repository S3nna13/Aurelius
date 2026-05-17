"""PowerShell-style object pipeline with chainable transforms.

This module provides a fluent pipeline API for processing collections
similar to PowerShell's object pipeline (|).

Operations:
    - filter: Keep items matching a predicate
    - map: Transform each item
    - sort: Sort by key function
    - head: Take first N items
    - tail: Take last N items
    - dedup: Remove consecutive duplicates
    - group_by: Group items by key

Example:
    >>> result = Pipeline([1, 2, 3, 4, 5]) \\
    ...     .filter(lambda x: x % 2 == 0) \\
    ...     .map(lambda x: x * 2) \\
    ...     .collect()
    >>> result
    [4, 8]
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")
K = TypeVar("K")
U = TypeVar("U")


class Pipeline(Generic[T]):  # noqa: UP046 - keep parseable on Python <3.12
    """Represents a pipeline of transformations on a sequence of items.

    Pipeline supports chaining operations like filter, map, sort, head,
    tail, dedup, and group_by in a fluent style similar to PowerShell.
    """

    def __init__(self, source: list[T]) -> None:
        """Initialize pipeline with a source list.

        Args:
            source: The initial list of items to process.
        """
        self._source = source

    def filter(self, predicate: Callable[[T], bool]) -> Pipeline[T]:
        """Keep only items where predicate returns True.

        Args:
            predicate: Function that returns True to keep an item.

        Returns:
            A new Pipeline with filtered items.

        Example:
            >>> Pipeline([1, 2, 3, 4]).filter(lambda x: x > 2).collect()
            [3, 4]
        """
        return Pipeline([item for item in self._source if predicate(item)])

    def map(self, transform: Callable[[T], U]) -> Pipeline[U]:
        """Transform each item using the provided function.

        Args:
            transform: Function to apply to each item.

        Returns:
            A new Pipeline with transformed items.

        Example:
            >>> Pipeline([1, 2, 3]).map(lambda x: x * 2).collect()
            [2, 4, 6]
        """
        return Pipeline([transform(item) for item in self._source])

    def sort(
        self,
        key: Callable[[T], Any] | None = None,
        reverse: bool = False,
    ) -> Pipeline[T]:
        """Sort items by key function.

        Args:
            key: Function to extract sort key from each item.
            reverse: If True, sort in descending order.

        Returns:
            A new Pipeline with sorted items.

        Example:
            >>> Pipeline([3, 1, 4, 1, 5]).sort().collect()
            [1, 1, 3, 4, 5]
            >>> Pipeline([3, 1, 4, 1, 5]).sort(reverse=True).collect()
            [5, 4, 3, 1, 1]
        """
        return Pipeline(sorted(self._source, key=key, reverse=reverse))

    def head(self, count: int) -> Pipeline[T]:
        """Take the first N items.

        Args:
            count: Number of items to take from the start.

        Returns:
            A new Pipeline with at most 'count' items.

        Example:
            >>> Pipeline([1, 2, 3, 4, 5]).head(3).collect()
            [1, 2, 3]
        """
        if count <= 0:
            return Pipeline([])
        return Pipeline(self._source[:count])

    def tail(self, count: int) -> Pipeline[T]:
        """Take the last N items.

        Args:
            count: Number of items to take from the end.

        Returns:
            A new Pipeline with at most 'count' items.

        Example:
            >>> Pipeline([1, 2, 3, 4, 5]).tail(3).collect()
            [3, 4, 5]
        """
        if count <= 0:
            return Pipeline([])
        return Pipeline(self._source[-count:])

    def dedup(self) -> Pipeline[T]:
        """Remove consecutive duplicate items.

        Only removes duplicates that are adjacent in the sequence.
        Use sort() first if you want to remove all duplicates.

        Returns:
            A new Pipeline with consecutive duplicates removed.

        Example:
            >>> Pipeline([1, 1, 2, 2, 2, 3, 1, 1]).dedup().collect()
            [1, 2, 3, 1]
        """
        if not self._source:
            return Pipeline([])
        result = [self._source[0]]
        for item in self._source[1:]:
            if item != result[-1]:
                result.append(item)
        return Pipeline(result)

    def group_by(self, key: Callable[[T], K]) -> dict[K, list[T]]:
        """Group items by a key function.

        Args:
            key: Function to extract grouping key from each item.

        Returns:
            A dictionary mapping keys to lists of items.

        Example:
            >>> result = Pipeline([1, 2, 3, 4, 5, 6]).group_by(lambda x: x % 2)
            >>> result == {1: [1, 3, 5], 0: [2, 4, 6]}
            True
        """
        groups: dict[K, list[T]] = {}
        for item in self._source:
            k = key(item)
            if k not in groups:
                groups[k] = []
            groups[k].append(item)
        return groups

    def collect(self) -> list[T]:
        """Execute the pipeline and return the result as a list.

        Returns:
            The final list of processed items.
        """
        return list(self._source)

    def to_pipeline(self) -> Pipeline[T]:
        """Return a new pipeline with the current items as source.

        Useful for transforming the result of a terminal operation
        back into a pipeline for further processing.

        Returns:
            A new Pipeline initialized with current items.
        """
        return Pipeline(self._source)

    def __iter__(self) -> Iterator[T]:
        """Iterate over the pipeline items.

        Yields:
            Each item in the pipeline.
        """
        return iter(self._source)

    def __len__(self) -> int:
        """Return the number of items in the pipeline.

        Returns:
            The length of the source collection.
        """
        return len(self._source)

    def __repr__(self) -> str:
        """Return string representation of the pipeline.

        Returns:
            A string showing the pipeline contents.
        """
        return f"Pipeline({self._source!r})"


def pipeline(source: list[T]) -> Pipeline[T]:  # noqa: UP047 - keep parseable on Python <3.12
    """Create a new pipeline from a source list.

    This is a convenience function that creates a Pipeline instance.

    Args:
        source: The initial list of items to process.

    Returns:
        A Pipeline instance for chaining operations.

    Example:
        >>> result = pipeline([1, 2, 3, 4, 5]) \\
        ...     .filter(lambda x: x % 2 == 0) \\
        ...     .map(lambda x: x * 2) \\
        ...     .collect()
        >>> result
        [4, 8]
    """
    return Pipeline(source)
