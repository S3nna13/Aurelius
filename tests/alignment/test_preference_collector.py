"""Tests for src/alignment/preference_collector.py."""

import pytest
from src.alignment.preference_collector import (
    PreferenceItem,
    PreferenceCollector,
    PREFERENCE_COLLECTOR,
)


# ---------------------------------------------------------------------------
# PreferenceItem dataclass
# ---------------------------------------------------------------------------

class TestPreferenceItem:
    def test_auto_generates_id(self):
        item = PreferenceItem(prompt="p", chosen="c", rejected="r")
        assert item.id is not None
        assert len(item.id) == 8

    def test_ids_are_unique(self):
        ids = {PreferenceItem(prompt="p", chosen="c", rejected="r").id for _ in range(50)}
        assert len(ids) == 50

    def test_id_is_hex(self):
        item = PreferenceItem(prompt="p", chosen="c", rejected="r")
        int(item.id, 16)  # should not raise

    def test_default_annotator_id(self):
        item = PreferenceItem(prompt="p", chosen="c", rejected="r")
        assert item.annotator_id == "auto"

    def test_default_score_chosen(self):
        item = PreferenceItem(prompt="p", chosen="c", rejected="r")
        assert item.score_chosen == pytest.approx(1.0)

    def test_default_score_rejected(self):
        item = PreferenceItem(prompt="p", chosen="c", rejected="r")
        assert item.score_rejected == pytest.approx(0.0)

    def test_fields_stored_correctly(self):
        item = PreferenceItem(prompt="hello", chosen="good", rejected="bad", annotator_id="alice")
        assert item.prompt == "hello"
        assert item.chosen == "good"
        assert item.rejected == "bad"
        assert item.annotator_id == "alice"


# ---------------------------------------------------------------------------
# PreferenceCollector.add
# ---------------------------------------------------------------------------

class TestPreferenceCollectorAdd:
    def setup_method(self):
        self.collector = PreferenceCollector()

    def test_add_returns_preference_item(self):
        item = self.collector.add("p", "c", "r")
        assert isinstance(item, PreferenceItem)

    def test_add_stores_item(self):
        item = self.collector.add("p", "c", "r")
        assert self.collector.get(item.id) is item

    def test_add_increments_total(self):
        self.collector.add("p", "c", "r")
        self.collector.add("p2", "c2", "r2")
        assert self.collector.stats()["total"] == 2

    def test_add_with_annotator_id(self):
        item = self.collector.add("p", "c", "r", annotator_id="bob")
        assert item.annotator_id == "bob"


# ---------------------------------------------------------------------------
# PreferenceCollector.get
# ---------------------------------------------------------------------------

class TestPreferenceCollectorGet:
    def setup_method(self):
        self.collector = PreferenceCollector()

    def test_get_existing_item(self):
        item = self.collector.add("p", "c", "r")
        assert self.collector.get(item.id) is item

    def test_get_nonexistent_returns_none(self):
        assert self.collector.get("notreal") is None

    def test_get_correct_item_among_many(self):
        items = [self.collector.add(f"p{i}", f"c{i}", f"r{i}") for i in range(5)]
        assert self.collector.get(items[2].id) is items[2]


# ---------------------------------------------------------------------------
# PreferenceCollector.sample
# ---------------------------------------------------------------------------

class TestPreferenceCollectorSample:
    def setup_method(self):
        self.collector = PreferenceCollector()
        for i in range(10):
            self.collector.add(f"p{i}", f"c{i}", f"r{i}")

    def test_sample_returns_n_items(self):
        result = self.collector.sample(5)
        assert len(result) == 5

    def test_sample_all_when_n_exceeds_len(self):
        result = self.collector.sample(100)
        assert len(result) == 10

    def test_sample_exact_len(self):
        result = self.collector.sample(10)
        assert len(result) == 10

    def test_sample_reproducible_same_seed(self):
        a = self.collector.sample(5, seed=99)
        b = self.collector.sample(5, seed=99)
        assert [x.id for x in a] == [x.id for x in b]

    def test_sample_different_seeds_likely_differ(self):
        a = self.collector.sample(5, seed=1)
        b = self.collector.sample(5, seed=2)
        # Very unlikely to be identical with 10 items sampled 5
        assert [x.id for x in a] != [x.id for x in b]

    def test_sample_returns_list_of_preference_items(self):
        result = self.collector.sample(3)
        assert all(isinstance(x, PreferenceItem) for x in result)

    def test_sample_empty_collector(self):
        empty = PreferenceCollector()
        assert empty.sample(5) == []


# ---------------------------------------------------------------------------
# PreferenceCollector.to_chatml_pairs
# ---------------------------------------------------------------------------

class TestPreferenceCollectorToChatmlPairs:
    def setup_method(self):
        self.collector = PreferenceCollector()
        self.collector.add("What is AI?", "AI is...", "I don't know", annotator_id="alice")

    def test_returns_list(self):
        pairs = self.collector.to_chatml_pairs()
        assert isinstance(pairs, list)

    def test_correct_length(self):
        self.collector.add("q2", "a2", "b2")
        pairs = self.collector.to_chatml_pairs()
        assert len(pairs) == 2

    def test_has_prompt_key(self):
        pair = self.collector.to_chatml_pairs()[0]
        assert "prompt" in pair

    def test_has_chosen_key(self):
        pair = self.collector.to_chatml_pairs()[0]
        assert "chosen" in pair

    def test_has_rejected_key(self):
        pair = self.collector.to_chatml_pairs()[0]
        assert "rejected" in pair

    def test_prompt_value_is_string(self):
        pair = self.collector.to_chatml_pairs()[0]
        assert pair["prompt"] == "What is AI?"

    def test_chosen_is_list_of_dicts(self):
        pair = self.collector.to_chatml_pairs()[0]
        assert isinstance(pair["chosen"], list)
        assert pair["chosen"][0]["role"] == "assistant"
        assert pair["chosen"][0]["content"] == "AI is..."

    def test_rejected_is_list_of_dicts(self):
        pair = self.collector.to_chatml_pairs()[0]
        assert isinstance(pair["rejected"], list)
        assert pair["rejected"][0]["role"] == "assistant"
        assert pair["rejected"][0]["content"] == "I don't know"


# ---------------------------------------------------------------------------
# PreferenceCollector.filter_by_annotator
# ---------------------------------------------------------------------------

class TestPreferenceCollectorFilterByAnnotator:
    def setup_method(self):
        self.collector = PreferenceCollector()
        self.collector.add("p1", "c1", "r1", annotator_id="alice")
        self.collector.add("p2", "c2", "r2", annotator_id="bob")
        self.collector.add("p3", "c3", "r3", annotator_id="alice")

    def test_filter_returns_matching_items(self):
        result = self.collector.filter_by_annotator("alice")
        assert len(result) == 2

    def test_filter_returns_correct_annotator(self):
        result = self.collector.filter_by_annotator("alice")
        assert all(x.annotator_id == "alice" for x in result)

    def test_filter_single_annotator(self):
        result = self.collector.filter_by_annotator("bob")
        assert len(result) == 1

    def test_filter_nonexistent_annotator(self):
        result = self.collector.filter_by_annotator("nobody")
        assert result == []


# ---------------------------------------------------------------------------
# PreferenceCollector.stats
# ---------------------------------------------------------------------------

class TestPreferenceCollectorStats:
    def setup_method(self):
        self.collector = PreferenceCollector()
        self.collector.add("p1", "c1", "r1", annotator_id="alice")
        self.collector.add("p2", "c2", "r2", annotator_id="bob")

    def test_stats_has_total_key(self):
        assert "total" in self.collector.stats()

    def test_stats_total_correct(self):
        assert self.collector.stats()["total"] == 2

    def test_stats_has_annotators_key(self):
        assert "annotators" in self.collector.stats()

    def test_stats_annotators_is_list(self):
        assert isinstance(self.collector.stats()["annotators"], list)

    def test_stats_annotators_contains_all(self):
        annotators = set(self.collector.stats()["annotators"])
        assert annotators == {"alice", "bob"}

    def test_stats_has_avg_score_gap(self):
        assert "avg_score_gap" in self.collector.stats()

    def test_stats_avg_score_gap_default(self):
        # default score_chosen=1.0, score_rejected=0.0 → gap=1.0
        assert self.collector.stats()["avg_score_gap"] == pytest.approx(1.0)

    def test_stats_empty_collector(self):
        empty = PreferenceCollector()
        s = empty.stats()
        assert s["total"] == 0
        assert s["avg_score_gap"] == pytest.approx(0.0)
        assert s["annotators"] == []


# ---------------------------------------------------------------------------
# Max items eviction
# ---------------------------------------------------------------------------

class TestPreferenceCollectorMaxItems:
    def test_max_items_eviction(self):
        collector = PreferenceCollector(max_items=3)
        items = [collector.add(f"p{i}", f"c{i}", f"r{i}") for i in range(5)]
        assert collector.stats()["total"] == 3

    def test_oldest_items_evicted(self):
        collector = PreferenceCollector(max_items=3)
        items = [collector.add(f"p{i}", f"c{i}", f"r{i}") for i in range(5)]
        # First two items (0, 1) should be evicted
        assert collector.get(items[0].id) is None
        assert collector.get(items[1].id) is None

    def test_newest_items_retained(self):
        collector = PreferenceCollector(max_items=3)
        items = [collector.add(f"p{i}", f"c{i}", f"r{i}") for i in range(5)]
        # Last three items (2, 3, 4) should be retained
        assert collector.get(items[2].id) is not None
        assert collector.get(items[3].id) is not None
        assert collector.get(items[4].id) is not None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class TestPreferenceCollectorSingleton:
    def test_preference_collector_exists(self):
        assert PREFERENCE_COLLECTOR is not None

    def test_preference_collector_is_instance(self):
        assert isinstance(PREFERENCE_COLLECTOR, PreferenceCollector)
