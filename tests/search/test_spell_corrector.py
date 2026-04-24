"""Tests for src/search/spell_corrector.py  (>=28 tests)."""

from __future__ import annotations

import pytest

from src.search.spell_corrector import (
    Correction,
    SpellCorrector,
    SPELL_CORRECTOR_REGISTRY,
    _DEFAULT_DICTIONARY,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_exists(self):
        assert SPELL_CORRECTOR_REGISTRY is not None

    def test_registry_has_default_key(self):
        assert "default" in SPELL_CORRECTOR_REGISTRY

    def test_registry_default_is_class(self):
        assert SPELL_CORRECTOR_REGISTRY["default"] is SpellCorrector


# ---------------------------------------------------------------------------
# Frozen dataclass
# ---------------------------------------------------------------------------

class TestCorrectionFrozen:
    def test_correction_is_frozen(self):
        c = Correction(original="teh", corrected="the", distance=1, confidence=0.67)
        with pytest.raises((TypeError, AttributeError)):
            c.corrected = "changed"  # type: ignore[misc]

    def test_correction_fields(self):
        c = Correction(original="teh", corrected="the", distance=1, confidence=0.67)
        assert c.original == "teh"
        assert c.corrected == "the"
        assert c.distance == 1
        assert abs(c.confidence - 0.67) < 1e-6


# ---------------------------------------------------------------------------
# Default dictionary
# ---------------------------------------------------------------------------

class TestDefaultDictionary:
    def test_default_dictionary_has_100_words(self):
        assert len(_DEFAULT_DICTIONARY) >= 100

    def test_required_words_present(self):
        required = [
            "the", "and", "for", "are", "but", "not", "you", "all", "any",
            "can", "her", "was", "one", "our", "out", "day", "get", "has",
        ]
        for word in required:
            assert word in _DEFAULT_DICTIONARY


# ---------------------------------------------------------------------------
# _edit_distance
# ---------------------------------------------------------------------------

class TestEditDistance:
    def setup_method(self):
        self.sc = SpellCorrector()

    def test_identical_strings_distance_zero(self):
        assert self.sc._edit_distance("hello", "hello") == 0

    def test_empty_vs_empty(self):
        assert self.sc._edit_distance("", "") == 0

    def test_empty_vs_word(self):
        assert self.sc._edit_distance("", "abc") == 3

    def test_word_vs_empty(self):
        assert self.sc._edit_distance("abc", "") == 3

    def test_one_insertion(self):
        # "cat" -> "cats": one insertion
        assert self.sc._edit_distance("cat", "cats") == 1

    def test_one_deletion(self):
        # "cats" -> "cat": one deletion
        assert self.sc._edit_distance("cats", "cat") == 1

    def test_one_substitution(self):
        assert self.sc._edit_distance("cat", "bat") == 1

    def test_known_distance(self):
        # "kitten" -> "sitting" = 3 operations
        assert self.sc._edit_distance("kitten", "sitting") == 3


# ---------------------------------------------------------------------------
# correct
# ---------------------------------------------------------------------------

class TestCorrect:
    def setup_method(self):
        self.sc = SpellCorrector()

    def test_exact_match_confidence_1(self):
        c = self.sc.correct("the")
        assert c.confidence == 1.0

    def test_exact_match_distance_0(self):
        c = self.sc.correct("the")
        assert c.distance == 0

    def test_exact_match_corrected_equals_original_lower(self):
        c = self.sc.correct("the")
        assert c.corrected == "the"

    def test_close_word_low_distance(self):
        # "teh" is one transposition from "the"
        c = self.sc.correct("teh")
        assert c.distance <= 2
        assert c.corrected in _DEFAULT_DICTIONARY

    def test_correct_returns_correction_instance(self):
        c = self.sc.correct("helo")
        assert isinstance(c, Correction)

    def test_original_preserved(self):
        c = self.sc.correct("Teh")
        assert c.original == "Teh"

    def test_confidence_between_0_and_1(self):
        c = self.sc.correct("xzqwerty")
        assert 0.0 <= c.confidence <= 1.0

    def test_case_insensitive_matching(self):
        # "THE" should match "the" exactly
        c = self.sc.correct("THE")
        assert c.distance == 0
        assert c.confidence == 1.0


# ---------------------------------------------------------------------------
# correct_query
# ---------------------------------------------------------------------------

class TestCorrectQuery:
    def setup_method(self):
        self.sc = SpellCorrector()

    def test_correct_query_single_word(self):
        result = self.sc.correct_query("the")
        assert result == "the"

    def test_correct_query_multi_word(self):
        result = self.sc.correct_query("the and for")
        assert result == "the and for"

    def test_correct_query_returns_string(self):
        result = self.sc.correct_query("helo wrld")
        assert isinstance(result, str)

    def test_correct_query_preserves_word_count(self):
        query = "one two three"
        result = self.sc.correct_query(query)
        assert len(result.split()) == 3


# ---------------------------------------------------------------------------
# add_word
# ---------------------------------------------------------------------------

class TestAddWord:
    def test_add_word_expands_dictionary(self):
        sc = SpellCorrector(dictionary=["hello"])
        sc.add_word("world")
        assert "world" in sc._dictionary

    def test_add_word_no_duplicate(self):
        sc = SpellCorrector(dictionary=["hello"])
        sc.add_word("hello")
        assert sc._dictionary.count("hello") == 1

    def test_add_word_lowercased(self):
        sc = SpellCorrector(dictionary=[])
        sc.add_word("Python")
        assert "python" in sc._dictionary

    def test_added_word_is_used_in_correction(self):
        sc = SpellCorrector(dictionary=["apple"])
        sc.add_word("mango")
        c = sc.correct("mango")
        assert c.distance == 0
        assert c.confidence == 1.0


# ---------------------------------------------------------------------------
# suggestions
# ---------------------------------------------------------------------------

class TestSuggestions:
    def setup_method(self):
        self.sc = SpellCorrector()

    def test_suggestions_returns_list(self):
        result = self.sc.suggestions("teh")
        assert isinstance(result, list)

    def test_suggestions_length_lte_n(self):
        result = self.sc.suggestions("teh", n=3)
        assert len(result) <= 3

    def test_suggestions_sorted_by_distance_asc(self):
        result = self.sc.suggestions("teh", n=10)
        distances = [c.distance for c in result]
        assert distances == sorted(distances)

    def test_suggestions_all_correction_instances(self):
        result = self.sc.suggestions("teh")
        assert all(isinstance(c, Correction) for c in result)

    def test_suggestions_default_n_is_5(self):
        result = self.sc.suggestions("teh")
        assert len(result) <= 5

    def test_suggestions_exact_match_first(self):
        result = self.sc.suggestions("the", n=5)
        assert result[0].distance == 0
        assert result[0].corrected == "the"
