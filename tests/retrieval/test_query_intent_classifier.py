"""Tests for query_intent_classifier.py."""

from __future__ import annotations

import pytest

from src.retrieval.query_intent_classifier import IntentRule, QueryIntentClassifier


# ---------------------------------------------------------------------------
# IntentRule dataclass
# ---------------------------------------------------------------------------

class TestIntentRule:
    def test_basic_construction(self):
        r = IntentRule(intent="test", keywords=("a", "b"))
        assert r.intent == "test"
        assert r.keywords == ("a", "b")
        assert r.weight == 1.0

    def test_custom_weight(self):
        r = IntentRule(intent="x", keywords=("y",), weight=2.5)
        assert r.weight == 2.5

    def test_empty_intent_raises(self):
        with pytest.raises(ValueError):
            IntentRule(intent="", keywords=("a",))

    def test_empty_keywords_raises(self):
        with pytest.raises(ValueError):
            IntentRule(intent="x", keywords=())

    def test_non_str_keyword_raises(self):
        with pytest.raises(ValueError):
            IntentRule(intent="x", keywords=("a", 1))  # type: ignore[arg-type]

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError):
            IntentRule(intent="x", keywords=("a",), weight=0.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError):
            IntentRule(intent="x", keywords=("a",), weight=-1.0)


# ---------------------------------------------------------------------------
# classify — built-in intents
# ---------------------------------------------------------------------------

class TestClassifyBuiltinIntents:
    def setup_method(self):
        self.clf = QueryIntentClassifier()

    def test_factual_who(self):
        assert self.clf.classify("Who wrote Hamlet?") == "factual"

    def test_factual_what(self):
        assert self.clf.classify("What is the capital of France?") == "factual"

    def test_factual_when(self):
        assert self.clf.classify("When did WW2 end?") == "factual"

    def test_factual_where(self):
        assert self.clf.classify("Where is the Eiffel Tower?") == "factual"

    def test_factual_is(self):
        assert self.clf.classify("Is the sky blue?") == "factual"

    def test_analytical_why(self):
        assert self.clf.classify("Why is the sky blue?") == "analytical"

    def test_analytical_how(self):
        assert self.clf.classify("How do magnets work?") == "analytical"

    def test_analytical_compare(self):
        assert self.clf.classify("Compare Python and JavaScript") == "analytical"

    def test_analytical_difference(self):
        assert self.clf.classify("Difference between TCP and UDP") == "analytical"

    def test_creative_write(self):
        assert self.clf.classify("Write a haiku about autumn") == "creative"

    def test_creative_create(self):
        assert self.clf.classify("Create a logo description") == "creative"

    def test_creative_generate(self):
        assert self.clf.classify("Generate a story about dragons") == "creative"

    def test_procedural_steps(self):
        assert self.clf.classify("Steps to bake sourdough") == "procedural"

    def test_procedural_how_to(self):
        assert self.clf.classify("How to change a tire") == "procedural"

    def test_procedural_guide(self):
        assert self.clf.classify("Guide to setting up Kubernetes") == "procedural"

    def test_navigational_find(self):
        assert self.clf.classify("Find the PyTorch documentation") == "navigational"

    def test_navigational_download(self):
        assert self.clf.classify("Download the latest release") == "navigational"

    def test_navigational_locate(self):
        assert self.clf.classify("Locate the nearest pharmacy") == "navigational"

    def test_fallback_factual_on_no_match(self):
        # No keywords match any rule — conservative fallback.
        assert self.clf.classify("foo bar baz qux") == "factual"

    def test_case_insensitive(self):
        assert self.clf.classify("WHO INVENTED THE LIGHTBULB?") == "factual"
        assert self.clf.classify("WHY IS ICE FLOAT?") == "analytical"

    def test_punctuation_ignored(self):
        assert self.clf.classify("  write!!!  a...story?  ") == "creative"


# ---------------------------------------------------------------------------
# confidence
# ---------------------------------------------------------------------------

class TestConfidence:
    def setup_method(self):
        self.clf = QueryIntentClassifier()

    def test_range_zero_to_one(self):
        q = "What is the speed of light?"
        c = self.clf.confidence(q)
        assert 0.0 <= c <= 1.0

    def test_confidence_is_float(self):
        assert isinstance(self.clf.confidence("Why?"), float)

    def test_high_confidence_strong_match(self):
        # Single intent with many hits should dominate.
        c = self.clf.confidence("how to steps guide tutorial")
        assert c > 0.8

    def test_low_confidence_tie(self):
        # A query that hits two different intents evenly lowers confidence.
        c = self.clf.confidence("how steps")
        assert c < 1.0

    def test_zero_confidence_empty_query(self):
        assert self.clf.confidence("") == 0.0

    def test_zero_confidence_no_match(self):
        assert self.clf.confidence("xyz123!!!") == 0.0


# ---------------------------------------------------------------------------
# batch_classify
# ---------------------------------------------------------------------------

class TestBatchClassify:
    def setup_method(self):
        self.clf = QueryIntentClassifier()

    def test_empty_list(self):
        assert self.clf.batch_classify([]) == []

    def test_preserves_order(self):
        queries = [
            "Who wrote Hamlet?",
            "Why is the sky blue?",
            "Write a poem",
            "How to bake bread",
            "Find the docs",
        ]
        expected = ["factual", "analytical", "creative", "procedural", "navigational"]
        assert self.clf.batch_classify(queries) == expected

    def test_returns_list(self):
        result = self.clf.batch_classify(["What?", "Why?"])
        assert isinstance(result, list)

    def test_non_list_raises(self):
        with pytest.raises(TypeError):
            self.clf.batch_classify("not a list")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# register_rule
# ---------------------------------------------------------------------------

class TestRegisterRule:
    def setup_method(self):
        self.clf = QueryIntentClassifier()

    def test_custom_rule_overrides(self):
        self.clf.register_rule(
            IntentRule(intent="custom", keywords=("zzzzzzzz",), weight=10.0)
        )
        assert self.clf.classify("zzzzzzzz") == "custom"

    def test_register_rule_affects_confidence(self):
        self.clf.register_rule(
            IntentRule(intent="custom", keywords=("zzzzzzzz",), weight=10.0)
        )
        c = self.clf.confidence("zzzzzzzz")
        assert c > 0.9

    def test_register_invalid_type_raises(self):
        with pytest.raises(TypeError):
            self.clf.register_rule("not a rule")  # type: ignore[arg-type]

    def test_custom_rule_coexists_with_defaults(self):
        self.clf.register_rule(
            IntentRule(intent="shopping", keywords=("buy", "purchase"), weight=2.0)
        )
        assert self.clf.classify("buy a laptop") == "shopping"
        assert self.clf.classify("Who wrote Hamlet?") == "factual"


# ---------------------------------------------------------------------------
# Input validation / security
# ---------------------------------------------------------------------------

class TestInputValidation:
    def setup_method(self):
        self.clf = QueryIntentClassifier()

    def test_classify_non_str_raises(self):
        with pytest.raises(TypeError):
            self.clf.classify(123)  # type: ignore[arg-type]

    def test_confidence_non_str_raises(self):
        with pytest.raises(TypeError):
            self.clf.confidence(None)  # type: ignore[arg-type]

    def test_classify_too_long_raises(self):
        with pytest.raises(ValueError):
            self.clf.classify("a" * 50_001)

    def test_confidence_too_long_raises(self):
        with pytest.raises(ValueError):
            self.clf.confidence("x" * 100_000)

    def test_adversarial_unicode(self):
        # Should not crash on untrusted unicode input.
        result = self.clf.classify("café résumé naïve 🚀 <script>alert(1)</script>")
        assert isinstance(result, str)
