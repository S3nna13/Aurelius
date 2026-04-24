"""
tests/data/test_vocab_builder.py
Tests for src/data/vocab_builder.py  (≥28 test methods)
"""

import unittest

from src.data.vocab_builder import (
    Vocabulary,
    VocabConfig,
    VocabEntry,
    VOCAB_BUILDER_REGISTRY,
)


class TestVocabEntryDataclass(unittest.TestCase):
    """Tests for the VocabEntry frozen dataclass."""

    def test_fields_stored(self):
        entry = VocabEntry(token="hello", token_id=5, freq=10)
        self.assertEqual(entry.token, "hello")
        self.assertEqual(entry.token_id, 5)
        self.assertEqual(entry.freq, 10)

    def test_frozen_raises_on_assignment(self):
        entry = VocabEntry(token="hello", token_id=5, freq=10)
        with self.assertRaises((AttributeError, TypeError)):
            entry.token = "world"  # type: ignore[misc]

    def test_equality(self):
        e1 = VocabEntry(token="a", token_id=0, freq=1)
        e2 = VocabEntry(token="a", token_id=0, freq=1)
        self.assertEqual(e1, e2)


class TestVocabConfig(unittest.TestCase):
    """Tests for VocabConfig defaults."""

    def test_defaults(self):
        cfg = VocabConfig()
        self.assertEqual(cfg.min_freq, 2)
        self.assertEqual(cfg.max_vocab, 50_000)
        self.assertIn("<pad>", cfg.special_tokens)
        self.assertIn("<unk>", cfg.special_tokens)
        self.assertIn("<bos>", cfg.special_tokens)
        self.assertIn("<eos>", cfg.special_tokens)

    def test_special_tokens_are_independent_across_instances(self):
        c1 = VocabConfig()
        c2 = VocabConfig()
        c1.special_tokens.append("EXTRA")
        self.assertNotIn("EXTRA", c2.special_tokens)


class TestVocabularyAddText(unittest.TestCase):
    """Tests that add_text correctly increments frequencies."""

    def test_single_word_frequency(self):
        v = Vocabulary(VocabConfig(min_freq=1))
        v.add_text("hello hello hello")
        vocab = v.build()
        self.assertIn("hello", vocab)

    def test_multiple_words(self):
        v = Vocabulary(VocabConfig(min_freq=1))
        v.add_text("cat dog cat")
        vocab = v.build()
        self.assertIn("cat", vocab)
        self.assertIn("dog", vocab)

    def test_case_normalised_to_lower(self):
        v = Vocabulary(VocabConfig(min_freq=1))
        v.add_text("Hello HELLO hello")
        vocab = v.build()
        self.assertIn("hello", vocab)
        self.assertNotIn("Hello", vocab)
        self.assertNotIn("HELLO", vocab)

    def test_punctuation_captured(self):
        v = Vocabulary(VocabConfig(min_freq=1))
        v.add_text("hello, world!")
        vocab = v.build()
        self.assertIn(",", vocab)
        self.assertIn("!", vocab)

    def test_accumulates_across_multiple_calls(self):
        v = Vocabulary(VocabConfig(min_freq=2))
        v.add_text("foo")
        v.add_text("foo")
        vocab = v.build()
        self.assertIn("foo", vocab)


class TestVocabularyBuild(unittest.TestCase):
    """Tests for the build() method."""

    def _make_vocab(self, texts, min_freq=1, max_vocab=50_000, specials=None):
        if specials is None:
            specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
        cfg = VocabConfig(min_freq=min_freq, max_vocab=max_vocab, special_tokens=specials)
        v = Vocabulary(cfg)
        for t in texts:
            v.add_text(t)
        return v

    def test_special_tokens_at_start(self):
        v = self._make_vocab(["hello world"])
        vocab = v.build()
        self.assertEqual(vocab["<pad>"], 0)
        self.assertEqual(vocab["<unk>"], 1)
        self.assertEqual(vocab["<bos>"], 2)
        self.assertEqual(vocab["<eos>"], 3)

    def test_special_tokens_all_present(self):
        v = self._make_vocab(["hello"])
        vocab = v.build()
        for s in ("<pad>", "<unk>", "<bos>", "<eos>"):
            self.assertIn(s, vocab)

    def test_build_respects_min_freq(self):
        v = self._make_vocab(["apple banana apple"], min_freq=2)
        vocab = v.build()
        self.assertIn("apple", vocab)   # freq=2
        self.assertNotIn("banana", vocab)  # freq=1

    def test_build_respects_max_vocab(self):
        # 4 specials + at most 2 regular tokens
        cfg = VocabConfig(min_freq=1, max_vocab=6)
        v = Vocabulary(cfg)
        v.add_text("a b c d e f g h i j")  # 10 unique tokens
        vocab = v.build()
        self.assertLessEqual(len(vocab), 6)

    def test_build_returns_dict(self):
        v = self._make_vocab(["hello"])
        result = v.build()
        self.assertIsInstance(result, dict)

    def test_higher_freq_gets_lower_id(self):
        """More frequent words should appear before less frequent ones."""
        v = self._make_vocab(["a a a b b c"], min_freq=1)
        vocab = v.build()
        # a(3) > b(2) > c(1), so id(a) < id(b) < id(c)
        self.assertLess(vocab["a"], vocab["b"])
        self.assertLess(vocab["b"], vocab["c"])

    def test_build_idempotent(self):
        v = self._make_vocab(["hello world"], min_freq=1)
        v1 = v.build()
        v2 = v.build()
        self.assertEqual(v1, v2)


class TestVocabularyTokenToId(unittest.TestCase):
    """Tests for token_to_id."""

    def setUp(self):
        cfg = VocabConfig(min_freq=1)
        self.v = Vocabulary(cfg)
        self.v.add_text("hello world hello")
        self.v.build()

    def test_known_token_returns_valid_id(self):
        tid = self.v.token_to_id("hello")
        self.assertIsInstance(tid, int)
        self.assertGreaterEqual(tid, 0)

    def test_unknown_token_returns_1(self):
        self.assertEqual(self.v.token_to_id("zzzzzz"), 1)

    def test_special_token_pad_returns_0(self):
        self.assertEqual(self.v.token_to_id("<pad>"), 0)

    def test_special_token_unk_returns_1(self):
        self.assertEqual(self.v.token_to_id("<unk>"), 1)


class TestVocabularyIdToToken(unittest.TestCase):
    """Tests for id_to_token."""

    def setUp(self):
        cfg = VocabConfig(min_freq=1)
        self.v = Vocabulary(cfg)
        self.v.add_text("hello world")
        self.v.build()

    def test_id_zero_returns_pad(self):
        self.assertEqual(self.v.id_to_token(0), "<pad>")

    def test_id_one_returns_unk(self):
        self.assertEqual(self.v.id_to_token(1), "<unk>")

    def test_known_id_roundtrip(self):
        tid = self.v.token_to_id("hello")
        self.assertEqual(self.v.id_to_token(tid), "hello")

    def test_unknown_id_returns_unk(self):
        self.assertEqual(self.v.id_to_token(999999), "<unk>")


class TestVocabularyVocabSize(unittest.TestCase):
    """Tests for vocab_size."""

    def test_vocab_size_includes_specials(self):
        cfg = VocabConfig(min_freq=1)
        v = Vocabulary(cfg)
        v.add_text("hello world")
        v.build()
        # 4 specials + "hello" + "world"
        self.assertEqual(v.vocab_size(), 6)

    def test_vocab_size_respects_min_freq(self):
        cfg = VocabConfig(min_freq=3)
        v = Vocabulary(cfg)
        v.add_text("a a a b b c")  # only 'a' qualifies
        v.build()
        self.assertEqual(v.vocab_size(), 5)  # 4 specials + "a"


class TestVocabularyMostCommon(unittest.TestCase):
    """Tests for most_common."""

    def setUp(self):
        cfg = VocabConfig(min_freq=1)
        self.v = Vocabulary(cfg)
        self.v.add_text("apple apple apple banana banana cherry")
        self.v.build()

    def test_most_common_returns_list(self):
        result = self.v.most_common(3)
        self.assertIsInstance(result, list)

    def test_most_common_respects_n(self):
        result = self.v.most_common(2)
        self.assertLessEqual(len(result), 2)

    def test_most_common_sorted_by_freq_desc(self):
        result = self.v.most_common(3)
        freqs = [e.freq for e in result]
        self.assertEqual(freqs, sorted(freqs, reverse=True))

    def test_most_common_entries_are_vocab_entry(self):
        result = self.v.most_common(3)
        for entry in result:
            self.assertIsInstance(entry, VocabEntry)

    def test_most_common_excludes_specials(self):
        result = self.v.most_common(10)
        specials = {"<pad>", "<unk>", "<bos>", "<eos>"}
        for entry in result:
            self.assertNotIn(entry.token, specials)


class TestVocabBuilderRegistry(unittest.TestCase):
    """Tests for VOCAB_BUILDER_REGISTRY."""

    def test_registry_has_default_key(self):
        self.assertIn("default", VOCAB_BUILDER_REGISTRY)

    def test_registry_default_is_vocabulary_class(self):
        self.assertIs(VOCAB_BUILDER_REGISTRY["default"], Vocabulary)

    def test_registry_default_is_instantiable(self):
        cls = VOCAB_BUILDER_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, Vocabulary)


if __name__ == "__main__":
    unittest.main()
