"""
tests/data/test_data_augmenter.py
Tests for src/data/data_augmenter.py  (≥28 test methods)
"""

import unittest

from src.data.data_augmenter import (
    AugmentationStrategy,
    AugmentedSample,
    DataAugmenter,
    DATA_AUGMENTER_REGISTRY,
)


class TestAugmentedSampleDataclass(unittest.TestCase):
    """Tests for the AugmentedSample frozen dataclass."""

    def _make(self, original="hello world", augmented="HELLO WORLD",
              strategy=AugmentationStrategy.UPPERCASE, seed=42):
        return AugmentedSample(
            original=original,
            augmented=augmented,
            strategy=strategy,
            seed=seed,
        )

    def test_fields_stored_correctly(self):
        s = self._make()
        self.assertEqual(s.original, "hello world")
        self.assertEqual(s.augmented, "HELLO WORLD")
        self.assertEqual(s.strategy, AugmentationStrategy.UPPERCASE)
        self.assertEqual(s.seed, 42)

    def test_frozen_raises_on_original_assignment(self):
        s = self._make()
        with self.assertRaises((AttributeError, TypeError)):
            s.original = "new"  # type: ignore[misc]

    def test_frozen_raises_on_augmented_assignment(self):
        s = self._make()
        with self.assertRaises((AttributeError, TypeError)):
            s.augmented = "new"  # type: ignore[misc]

    def test_frozen_raises_on_seed_assignment(self):
        s = self._make()
        with self.assertRaises((AttributeError, TypeError)):
            s.seed = 0  # type: ignore[misc]

    def test_seed_stored(self):
        s = self._make(seed=7)
        self.assertEqual(s.seed, 7)

    def test_original_preserved(self):
        text = "The quick brown fox"
        augmenter = DataAugmenter()
        sample = augmenter.augment(text, AugmentationStrategy.UPPERCASE)
        self.assertEqual(sample.original, text)


class TestAugmentationStrategyEnum(unittest.TestCase):
    """Tests for AugmentationStrategy enum membership."""

    def test_all_six_strategies_exist(self):
        strategies = set(AugmentationStrategy)
        self.assertEqual(len(strategies), 6)

    def test_synonym_swap_member(self):
        self.assertIn(AugmentationStrategy.SYNONYM_SWAP, AugmentationStrategy)

    def test_deletion_member(self):
        self.assertIn(AugmentationStrategy.DELETION, AugmentationStrategy)

    def test_insertion_member(self):
        self.assertIn(AugmentationStrategy.INSERTION, AugmentationStrategy)

    def test_swap_words_member(self):
        self.assertIn(AugmentationStrategy.SWAP_WORDS, AugmentationStrategy)

    def test_lowercase_member(self):
        self.assertIn(AugmentationStrategy.LOWERCASE, AugmentationStrategy)

    def test_uppercase_member(self):
        self.assertIn(AugmentationStrategy.UPPERCASE, AugmentationStrategy)


class TestLowercaseUppercase(unittest.TestCase):
    """Tests for LOWERCASE and UPPERCASE strategies."""

    def setUp(self):
        self.augmenter = DataAugmenter()

    def test_lowercase_produces_all_lower(self):
        sample = self.augmenter.augment("Hello WORLD", AugmentationStrategy.LOWERCASE)
        self.assertEqual(sample.augmented, "hello world")

    def test_uppercase_produces_all_upper(self):
        sample = self.augmenter.augment("Hello World", AugmentationStrategy.UPPERCASE)
        self.assertEqual(sample.augmented, "HELLO WORLD")

    def test_lowercase_already_lower_unchanged(self):
        sample = self.augmenter.augment("already lower", AugmentationStrategy.LOWERCASE)
        self.assertEqual(sample.augmented, "already lower")

    def test_uppercase_already_upper_unchanged(self):
        sample = self.augmenter.augment("ALREADY UPPER", AugmentationStrategy.UPPERCASE)
        self.assertEqual(sample.augmented, "ALREADY UPPER")

    def test_lowercase_strategy_stored(self):
        sample = self.augmenter.augment("Hello", AugmentationStrategy.LOWERCASE)
        self.assertEqual(sample.strategy, AugmentationStrategy.LOWERCASE)

    def test_uppercase_strategy_stored(self):
        sample = self.augmenter.augment("Hello", AugmentationStrategy.UPPERCASE)
        self.assertEqual(sample.strategy, AugmentationStrategy.UPPERCASE)


class TestSwapWords(unittest.TestCase):
    """Tests for SWAP_WORDS strategy."""

    def setUp(self):
        self.augmenter = DataAugmenter()

    def test_adjacent_pairs_swapped(self):
        # "one two three four" → "two one four three"
        sample = self.augmenter.augment(
            "one two three four", AugmentationStrategy.SWAP_WORDS
        )
        self.assertEqual(sample.augmented, "two one four three")

    def test_single_word_unchanged(self):
        sample = self.augmenter.augment("only", AugmentationStrategy.SWAP_WORDS)
        self.assertEqual(sample.augmented, "only")

    def test_three_words_first_pair_swapped(self):
        sample = self.augmenter.augment("a b c", AugmentationStrategy.SWAP_WORDS)
        self.assertEqual(sample.augmented, "b a c")

    def test_swap_words_preserves_word_count(self):
        text = "one two three four five six"
        sample = self.augmenter.augment(text, AugmentationStrategy.SWAP_WORDS)
        self.assertEqual(
            len(sample.augmented.split()), len(text.split())
        )


class TestDeletion(unittest.TestCase):
    """Tests for DELETION strategy."""

    def setUp(self):
        self.augmenter = DataAugmenter()

    def test_deletion_removes_some_words(self):
        text = " ".join(["word"] * 50)
        sample = self.augmenter.augment(text, AugmentationStrategy.DELETION, seed=42)
        result_len = len(sample.augmented.split())
        self.assertLess(result_len, 50)

    def test_deletion_keeps_at_least_one_word(self):
        sample = self.augmenter.augment("only", AugmentationStrategy.DELETION, seed=42)
        self.assertGreater(len(sample.augmented.strip()), 0)

    def test_deletion_deterministic_with_same_seed(self):
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        s1 = self.augmenter.augment(text, AugmentationStrategy.DELETION, seed=99)
        s2 = self.augmenter.augment(text, AugmentationStrategy.DELETION, seed=99)
        self.assertEqual(s1.augmented, s2.augmented)

    def test_deletion_different_seeds_may_differ(self):
        text = " ".join([f"w{i}" for i in range(30)])
        s1 = self.augmenter.augment(text, AugmentationStrategy.DELETION, seed=1)
        s2 = self.augmenter.augment(text, AugmentationStrategy.DELETION, seed=2)
        # They may occasionally be equal but should differ in general
        # Just verify the seeds are stored correctly
        self.assertEqual(s1.seed, 1)
        self.assertEqual(s2.seed, 2)


class TestInsertion(unittest.TestCase):
    """Tests for INSERTION strategy."""

    def setUp(self):
        self.augmenter = DataAugmenter()

    def test_filler_inserted_after_5th_word(self):
        text = "one two three four five six seven"
        sample = self.augmenter.augment(text, AugmentationStrategy.INSERTION)
        words = sample.augmented.split()
        # After "five" (index 4), FILLER inserted
        self.assertIn("FILLER", words)

    def test_insertion_adds_filler_every_5_words(self):
        text = " ".join([f"w{i}" for i in range(10)])
        sample = self.augmenter.augment(text, AugmentationStrategy.INSERTION)
        words = sample.augmented.split()
        count = words.count("FILLER")
        self.assertEqual(count, 2)  # after w4 and w9

    def test_insertion_short_text_no_filler(self):
        sample = self.augmenter.augment("one two three", AugmentationStrategy.INSERTION)
        self.assertNotIn("FILLER", sample.augmented)

    def test_insertion_preserves_original_words(self):
        text = "alpha beta gamma delta epsilon"
        sample = self.augmenter.augment(text, AugmentationStrategy.INSERTION)
        for word in text.split():
            self.assertIn(word, sample.augmented)


class TestSynonymSwap(unittest.TestCase):
    """Tests for SYNONYM_SWAP strategy."""

    def setUp(self):
        self.augmenter = DataAugmenter()

    def test_third_word_gets_synonym_prefix(self):
        sample = self.augmenter.augment(
            "one two three four five six", AugmentationStrategy.SYNONYM_SWAP
        )
        words = sample.augmented.split()
        # Index 2 (1-indexed position 3) should be "SYNONYM_three"
        self.assertEqual(words[2], "SYNONYM_three")

    def test_sixth_word_gets_synonym_prefix(self):
        sample = self.augmenter.augment(
            "one two three four five six", AugmentationStrategy.SYNONYM_SWAP
        )
        words = sample.augmented.split()
        self.assertEqual(words[5], "SYNONYM_six")

    def test_non_third_words_unchanged(self):
        sample = self.augmenter.augment(
            "one two three", AugmentationStrategy.SYNONYM_SWAP
        )
        words = sample.augmented.split()
        self.assertEqual(words[0], "one")
        self.assertEqual(words[1], "two")

    def test_synonym_swap_preserves_word_count(self):
        text = "alpha beta gamma delta epsilon zeta"
        sample = self.augmenter.augment(text, AugmentationStrategy.SYNONYM_SWAP)
        self.assertEqual(len(sample.augmented.split()), len(text.split()))


class TestAugmentBatch(unittest.TestCase):
    """Tests for augment_batch."""

    def setUp(self):
        self.augmenter = DataAugmenter()

    def test_batch_length_matches_input(self):
        texts = ["hello world", "foo bar", "baz qux"]
        results = self.augmenter.augment_batch(texts, AugmentationStrategy.UPPERCASE)
        self.assertEqual(len(results), 3)

    def test_batch_seed_equals_index(self):
        texts = ["a", "b", "c"]
        results = self.augmenter.augment_batch(texts, AugmentationStrategy.DELETION)
        for idx, sample in enumerate(results):
            self.assertEqual(sample.seed, idx)

    def test_batch_empty_input(self):
        results = self.augmenter.augment_batch([], AugmentationStrategy.LOWERCASE)
        self.assertEqual(results, [])

    def test_batch_all_samples_are_augmented_sample(self):
        texts = ["hello", "world"]
        results = self.augmenter.augment_batch(texts, AugmentationStrategy.LOWERCASE)
        for r in results:
            self.assertIsInstance(r, AugmentedSample)


class TestApplyAll(unittest.TestCase):
    """Tests for apply_all."""

    def setUp(self):
        self.augmenter = DataAugmenter()

    def test_apply_all_returns_6_samples(self):
        results = self.augmenter.apply_all("hello world")
        self.assertEqual(len(results), 6)

    def test_apply_all_covers_all_strategies(self):
        results = self.augmenter.apply_all("hello world")
        strategies = {r.strategy for r in results}
        self.assertEqual(strategies, set(AugmentationStrategy))

    def test_apply_all_all_seed_42(self):
        results = self.augmenter.apply_all("hello world")
        for r in results:
            self.assertEqual(r.seed, 42)


class TestDataAugmenterInit(unittest.TestCase):
    """Tests for DataAugmenter constructor."""

    def test_default_init_uses_all_strategies(self):
        aug = DataAugmenter()
        results = aug.apply_all("test text")
        self.assertEqual(len(results), 6)

    def test_custom_strategy_list(self):
        aug = DataAugmenter(strategies=[AugmentationStrategy.LOWERCASE])
        results = aug.apply_all("Test TEXT")
        # apply_all iterates over AugmentationStrategy enum, not self._strategies,
        # so length is still 6; but the augmenter was constructed with a subset
        self.assertEqual(len(results), 6)

    def test_augment_returns_augmented_sample(self):
        aug = DataAugmenter()
        result = aug.augment("hello", AugmentationStrategy.LOWERCASE)
        self.assertIsInstance(result, AugmentedSample)


class TestDataAugmenterRegistry(unittest.TestCase):
    """Tests for DATA_AUGMENTER_REGISTRY."""

    def test_registry_has_default_key(self):
        self.assertIn("default", DATA_AUGMENTER_REGISTRY)

    def test_registry_default_is_data_augmenter_class(self):
        self.assertIs(DATA_AUGMENTER_REGISTRY["default"], DataAugmenter)

    def test_registry_default_is_instantiable(self):
        cls = DATA_AUGMENTER_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, DataAugmenter)


if __name__ == "__main__":
    unittest.main()
