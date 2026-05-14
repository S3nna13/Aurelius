"""Tests for pipeline_processor module."""

import pytest

from aurelius_cli.pipeline_processor import Pipeline, pipeline


class TestPipeline:
    """Tests for Pipeline class."""

    def test_init_empty_list(self):
        """Test initializing pipeline with empty list."""
        p = Pipeline([])
        assert list(p) == []
        assert len(p) == 0

    def test_init_with_data(self):
        """Test initializing pipeline with data."""
        p = Pipeline([1, 2, 3])
        assert list(p) == [1, 2, 3]
        assert len(p) == 3

    def test_repr(self):
        """Test string representation."""
        p = Pipeline([1, 2, 3])
        assert repr(p) == "Pipeline([1, 2, 3])"

    def test_iter(self):
        """Test iteration over pipeline."""
        p = Pipeline([1, 2, 3])
        result = []
        for item in p:
            result.append(item)
        assert result == [1, 2, 3]

    def test_filter_keep_matching(self):
        """Test filter keeps items matching predicate."""
        p = Pipeline([1, 2, 3, 4, 5]).filter(lambda x: x % 2 == 0)
        assert p.collect() == [2, 4]

    def test_filter_keep_all(self):
        """Test filter with predicate that matches all."""
        p = Pipeline([1, 2, 3]).filter(lambda x: True)
        assert p.collect() == [1, 2, 3]

    def test_filter_keep_none(self):
        """Test filter with predicate that matches none."""
        p = Pipeline([1, 2, 3]).filter(lambda x: False)
        assert p.collect() == []

    def test_filter_empty_source(self):
        """Test filter on empty pipeline."""
        p = Pipeline([]).filter(lambda x: True)
        assert p.collect() == []

    def test_map_transform(self):
        """Test map transforms each item."""
        p = Pipeline([1, 2, 3]).map(lambda x: x * 2)
        assert p.collect() == [2, 4, 6]

    def test_map_empty_source(self):
        """Test map on empty pipeline."""
        p = Pipeline([]).map(lambda x: x * 2)
        assert p.collect() == []

    def test_map_type_change(self):
        """Test map can change types."""
        p = Pipeline([1, 2, 3]).map(str)
        assert p.collect() == ["1", "2", "3"]

    def test_sort_default(self):
        """Test sort with default parameters."""
        p = Pipeline([3, 1, 4, 1, 5, 9, 2, 6]).sort()
        assert p.collect() == [1, 1, 2, 3, 4, 5, 6, 9]

    def test_sort_reverse(self):
        """Test sort in reverse order."""
        p = Pipeline([3, 1, 4, 1, 5]).sort(reverse=True)
        assert p.collect() == [5, 4, 3, 1, 1]

    def test_sort_with_key(self):
        """Test sort with custom key function."""
        # Python's sort is stable, so items with same length preserve order
        p = Pipeline(["apple", "pie", "fruit"]).sort(key=len)
        result = p.collect()
        assert len(result[0]) == 3  # "pie"
        assert len(result[1]) == 5  # either "apple" or "fruit"
        assert len(result[2]) == 5

    def test_sort_reverse_with_key(self):
        """Test sort with key and reverse."""
        # Python's sort is stable, so items with same length preserve order
        p = Pipeline(["apple", "pie", "fruit"]).sort(key=len, reverse=True)
        result = p.collect()
        assert len(result[0]) == 5
        assert len(result[1]) == 5
        assert len(result[2]) == 3

    def test_sort_empty_source(self):
        """Test sort on empty pipeline."""
        p = Pipeline([]).sort()
        assert p.collect() == []

    def test_sort_stable(self):
        """Test sort is stable."""
        data = [(1, "a"), (2, "b"), (1, "c"), (2, "d")]
        p = Pipeline(data).sort(key=lambda x: x[0])
        assert p.collect() == [(1, "a"), (1, "c"), (2, "b"), (2, "d")]

    def test_head_positive_count(self):
        """Test head takes first N items."""
        p = Pipeline([1, 2, 3, 4, 5]).head(3)
        assert p.collect() == [1, 2, 3]

    def test_head_count_exceeds_length(self):
        """Test head when count exceeds list length."""
        p = Pipeline([1, 2, 3]).head(10)
        assert p.collect() == [1, 2, 3]

    def test_head_zero(self):
        """Test head with zero count."""
        p = Pipeline([1, 2, 3]).head(0)
        assert p.collect() == []

    def test_head_empty_source(self):
        """Test head on empty pipeline."""
        p = Pipeline([]).head(5)
        assert p.collect() == []

    def test_tail_positive_count(self):
        """Test tail takes last N items."""
        p = Pipeline([1, 2, 3, 4, 5]).tail(3)
        assert p.collect() == [3, 4, 5]

    def test_tail_count_exceeds_length(self):
        """Test tail when count exceeds list length."""
        p = Pipeline([1, 2, 3]).tail(10)
        assert p.collect() == [1, 2, 3]

    def test_tail_zero(self):
        """Test tail with zero count."""
        p = Pipeline([1, 2, 3]).tail(0)
        assert p.collect() == []

    def test_tail_empty_source(self):
        """Test tail on empty pipeline."""
        p = Pipeline([]).tail(5)
        assert p.collect() == []

    def test_dedup_removes_adjacent_dupes(self):
        """Test dedup removes consecutive duplicates."""
        p = Pipeline([1, 1, 1, 2, 2, 3, 3, 3]).dedup()
        assert p.collect() == [1, 2, 3]

    def test_dedup_preserves_non_adjacent_dupes(self):
        """Test dedup keeps non-adjacent duplicates."""
        p = Pipeline([1, 2, 1, 2, 1]).dedup()
        assert p.collect() == [1, 2, 1, 2, 1]

    def test_dedup_empty_source(self):
        """Test dedup on empty pipeline."""
        p = Pipeline([]).dedup()
        assert p.collect() == []

    def test_dedup_no_dupes(self):
        """Test dedup when no duplicates exist."""
        p = Pipeline([1, 2, 3, 4]).dedup()
        assert p.collect() == [1, 2, 3, 4]

    def test_dedup_single_element(self):
        """Test dedup with single element."""
        p = Pipeline([1]).dedup()
        assert p.collect() == [1]

    def test_dedup_all_same(self):
        """Test dedup when all elements are same."""
        p = Pipeline([5, 5, 5, 5]).dedup()
        assert p.collect() == [5]

    def test_group_by_basic(self):
        """Test group_by with basic grouping."""
        data = [1, 2, 3, 4, 5, 6]
        result = Pipeline(data).group_by(lambda x: x % 2)
        assert result == {1: [1, 3, 5], 0: [2, 4, 6]}

    def test_group_by_string_keys(self):
        """Test group_by with string keys."""
        data = ["apple", "banana", "apricot", "blueberry", "cherry"]
        result = Pipeline(data).group_by(lambda x: x[0])
        assert result == {
            "a": ["apple", "apricot"],
            "b": ["banana", "blueberry"],
            "c": ["cherry"],
        }

    def test_group_by_empty_source(self):
        """Test group_by on empty pipeline."""
        result = Pipeline([]).group_by(lambda x: x)
        assert result == {}

    def test_group_by_single_item(self):
        """Test group_by with single item."""
        result = Pipeline([1]).group_by(lambda x: x)
        assert result == {1: [1]}

    def test_group_by_all_same_key(self):
        """Test group_by when all items have same key."""
        result = Pipeline([1, 2, 3]).group_by(lambda x: "same")
        assert result == {"same": [1, 2, 3]}

    def test_collect_returns_list(self):
        """Test collect returns a list."""
        result = Pipeline([1, 2, 3]).collect()
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    def test_collect_empty(self):
        """Test collect on empty pipeline."""
        result = Pipeline([]).collect()
        assert result == []

    def test_to_pipeline(self):
        """Test to_pipeline creates new pipeline from current items."""
        p1 = Pipeline([1, 2, 3]).map(lambda x: x * 2)
        p2 = p1.to_pipeline().filter(lambda x: x > 2)
        assert p2.collect() == [4, 6]

    def test_to_pipeline_preserves_current(self):
        """Test to_pipeline doesn't modify original."""
        p1 = Pipeline([1, 2, 3])
        p2 = p1.to_pipeline()
        assert p1.collect() == [1, 2, 3]
        assert p2.collect() == [1, 2, 3]

    def test_chained_operations(self):
        """Test chaining multiple operations."""
        result = (
            Pipeline(range(10))
            .filter(lambda x: x % 2 == 0)  # [0, 2, 4, 6, 8]
            .map(lambda x: x * 2)  # [0, 4, 8, 12, 16]
            .sort()  # [0, 4, 8, 12, 16]
            .collect()
        )
        assert result == [0, 4, 8, 12, 16]

    def test_chained_with_head_tail(self):
        """Test chaining with head and tail."""
        result = (
            Pipeline(range(10))
            .filter(lambda x: x > 0)
            .head(7)  # [1, 2, 3, 4, 5, 6, 7]
            .tail(3)  # [5, 6, 7]
            .collect()
        )
        assert result == [5, 6, 7]

    def test_chained_filter_map_sort(self):
        """Test filter, map, sort chain."""
        result = (
            Pipeline([3, 1, 4, 1, 5, 9, 2, 6])
            .filter(lambda x: x > 2)
            .map(lambda x: x * 10)
            .sort()
            .collect()
        )
        assert result == [30, 40, 50, 60, 90]

    def test_len(self):
        """Test __len__ method."""
        p = Pipeline([1, 2, 3])
        assert len(p) == 3

    def test_len_empty(self):
        """Test __len__ on empty pipeline."""
        p = Pipeline([])
        assert len(p) == 0


class TestPipelineFunction:
    """Tests for pipeline convenience function."""

    def test_pipeline_function_creates_pipeline(self):
        """Test pipeline function creates a Pipeline instance."""
        result = pipeline([1, 2, 3])
        assert isinstance(result, Pipeline)

    def test_pipeline_function_empty_list(self):
        """Test pipeline with empty list."""
        result = pipeline([])
        assert result.collect() == []

    def test_pipeline_function_chain(self):
        """Test pipeline function with chaining."""
        result = pipeline([1, 2, 3, 4, 5]).filter(lambda x: x > 2).map(lambda x: x**2).collect()
        assert result == [9, 16, 25]


class TestPipelineEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_negative_head_count(self):
        """Test head with negative count."""
        p = Pipeline([1, 2, 3]).head(-1)
        assert p.collect() == []

    def test_negative_tail_count(self):
        """Test tail with negative count."""
        p = Pipeline([1, 2, 3]).tail(-1)
        assert p.collect() == []

    def test_sort_with_none_key(self):
        """Test sort with None key on mixed types."""
        p = Pipeline([3, None, 1, 2])
        # This may raise TypeError, which is expected behavior
        with pytest.raises(TypeError):
            p.sort()

    def test_filter_with_exception_predicate(self):
        """Test that filter preserves exceptions from predicate."""

        def bad_pred(x):
            if x == 2:
                raise ValueError("bad")
            return True

        p = Pipeline([1, 2, 3])
        with pytest.raises(ValueError, match="bad"):
            p.filter(bad_pred).collect()

    def test_map_with_exception_transform(self):
        """Test that map preserves exceptions from transform."""

        def bad_transform(x):
            if x == 2:
                raise ValueError("bad")
            return x * 2

        p = Pipeline([1, 2, 3])
        with pytest.raises(ValueError, match="bad"):
            p.map(bad_transform).collect()

    def test_group_by_with_exception_key(self):
        """Test that group_by preserves exceptions from key function."""

        def bad_key(x):
            if x == 2:
                raise ValueError("bad")
            return x

        p = Pipeline([1, 2, 3])
        with pytest.raises(ValueError, match="bad"):
            p.group_by(bad_key)

    def test_large_list_processing(self):
        """Test pipeline with large input list."""
        large_list = list(range(10000))
        result = (
            Pipeline(large_list)
            .filter(lambda x: x % 2 == 0)
            .map(lambda x: x * 2)
            .head(100)
            .collect()
        )
        assert len(result) == 100
        assert result[0] == 0
        assert result[99] == 396


class TestPipelineReuse:
    """Tests verifying pipeline operations are non-destructive."""

    def test_filter_does_not_modify_original(self):
        """Test that filter returns new pipeline, doesn't modify original."""
        original = Pipeline([1, 2, 3, 4, 5])
        _ = original.filter(lambda x: x > 3)
        assert original.collect() == [1, 2, 3, 4, 5]

    def test_map_does_not_modify_original(self):
        """Test that map returns new pipeline, doesn't modify original."""
        original = Pipeline([1, 2, 3])
        _ = original.map(lambda x: x * 10)
        assert original.collect() == [1, 2, 3]

    def test_sort_does_not_modify_original(self):
        """Test that sort returns new pipeline, doesn't modify original."""
        original = Pipeline([3, 1, 2])
        _ = original.sort()
        assert original.collect() == [3, 1, 2]

    def test_operations_return_new_pipeline(self):
        """Test that all operations return new Pipeline instances."""
        original = Pipeline([1, 2, 3])
        assert original.filter(lambda x: x > 1) is not original
        assert original.map(lambda x: x * 2) is not original
        assert original.sort() is not original
        assert original.head(2) is not original
        assert original.tail(2) is not original
        assert original.dedup() is not original


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
