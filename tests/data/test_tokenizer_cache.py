"""
tests/data/test_tokenizer_cache.py
Tests for src/data/tokenizer_cache.py  (≥28 test methods)
"""

import unittest

from src.data.tokenizer_cache import (
    CachedTokenization,
    TokenizerCache,
    TOKENIZER_CACHE_REGISTRY,
)


class TestCachedTokenizationDataclass(unittest.TestCase):
    """Tests for the CachedTokenization frozen dataclass."""

    def _make(self, text="hello world", tokens=None, token_strs=None, hash_key="abcd1234abcd1234"):
        if tokens is None:
            tokens = [1, 2]
        if token_strs is None:
            token_strs = ["hello", "world"]
        return CachedTokenization(text=text, tokens=tokens, token_strs=token_strs, hash_key=hash_key)

    def test_fields_stored_correctly(self):
        ct = self._make()
        self.assertEqual(ct.text, "hello world")
        self.assertEqual(ct.tokens, [1, 2])
        self.assertEqual(ct.token_strs, ["hello", "world"])
        self.assertEqual(ct.hash_key, "abcd1234abcd1234")

    def test_frozen_raises_on_text_assignment(self):
        ct = self._make()
        with self.assertRaises((AttributeError, TypeError)):
            ct.text = "new"  # type: ignore[misc]

    def test_frozen_raises_on_tokens_assignment(self):
        ct = self._make()
        with self.assertRaises((AttributeError, TypeError)):
            ct.tokens = [99]  # type: ignore[misc]

    def test_frozen_raises_on_hash_key_assignment(self):
        ct = self._make()
        with self.assertRaises((AttributeError, TypeError)):
            ct.hash_key = "x"  # type: ignore[misc]

    def test_equality_same_values(self):
        ct1 = self._make()
        ct2 = self._make()
        self.assertEqual(ct1, ct2)

    def test_equality_different_values(self):
        ct1 = self._make(text="a")
        ct2 = self._make(text="b")
        self.assertNotEqual(ct1, ct2)


class TestTokenizerCacheHash(unittest.TestCase):
    """Tests for the _hash static method."""

    def test_hash_is_16_chars(self):
        h = TokenizerCache._hash("hello")
        self.assertEqual(len(h), 16)

    def test_hash_is_hex(self):
        h = TokenizerCache._hash("hello")
        # All characters must be valid hex digits
        int(h, 16)

    def test_hash_deterministic(self):
        h1 = TokenizerCache._hash("same text")
        h2 = TokenizerCache._hash("same text")
        self.assertEqual(h1, h2)

    def test_hash_different_for_different_text(self):
        h1 = TokenizerCache._hash("apple")
        h2 = TokenizerCache._hash("orange")
        self.assertNotEqual(h1, h2)

    def test_hash_empty_string(self):
        h = TokenizerCache._hash("")
        self.assertEqual(len(h), 16)


class TestTokenizerCachePutGet(unittest.TestCase):
    """Tests for put / get round-trips."""

    def setUp(self):
        self.cache = TokenizerCache(max_size=8)

    def test_put_returns_cached_tokenization(self):
        result = self.cache.put("hello", [1, 2], ["hello", "world"])
        self.assertIsInstance(result, CachedTokenization)

    def test_put_then_get_returns_same_entry(self):
        self.cache.put("hello world", [1, 2], ["hello", "world"])
        retrieved = self.cache.get("hello world")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.tokens, [1, 2])
        self.assertEqual(retrieved.token_strs, ["hello", "world"])

    def test_get_miss_returns_none(self):
        result = self.cache.get("not in cache")
        self.assertIsNone(result)

    def test_put_stores_correct_hash_key(self):
        entry = self.cache.put("test", [99], ["test"])
        expected_key = TokenizerCache._hash("test")
        self.assertEqual(entry.hash_key, expected_key)

    def test_put_copies_token_list(self):
        tokens = [1, 2, 3]
        entry = self.cache.put("abc", tokens, ["a", "b", "c"])
        tokens.append(999)
        self.assertEqual(entry.tokens, [1, 2, 3])

    def test_put_overwrites_existing_key(self):
        self.cache.put("hello", [1], ["h"])
        self.cache.put("hello", [2], ["x"])
        retrieved = self.cache.get("hello")
        self.assertEqual(retrieved.tokens, [2])

    def test_get_updates_recency(self):
        """After get, the entry should still be retrievable."""
        self.cache.put("a", [1], ["a"])
        _ = self.cache.get("a")
        self.assertIsNotNone(self.cache.get("a"))


class TestTokenizerCacheHitRate(unittest.TestCase):
    """Tests for hit_rate tracking."""

    def setUp(self):
        self.cache = TokenizerCache()

    def test_hit_rate_zero_when_no_requests(self):
        self.assertEqual(self.cache.hit_rate(), 0.0)

    def test_hit_rate_zero_after_pure_misses(self):
        self.cache.get("missing1")
        self.cache.get("missing2")
        self.assertEqual(self.cache.hit_rate(), 0.0)

    def test_hit_rate_one_after_pure_hits(self):
        self.cache.put("text", [1], ["t"])
        self.cache.get("text")
        self.cache.get("text")
        self.assertEqual(self.cache.hit_rate(), 1.0)

    def test_hit_rate_partial(self):
        self.cache.put("yes", [1], ["y"])
        self.cache.get("yes")    # hit
        self.cache.get("no")     # miss
        self.assertAlmostEqual(self.cache.hit_rate(), 0.5)

    def test_stats_contains_expected_keys(self):
        s = self.cache.stats()
        for key in ("size", "max_size", "hits", "misses", "hit_rate"):
            self.assertIn(key, s)

    def test_stats_size_reflects_entries(self):
        self.cache.put("a", [1], ["a"])
        self.cache.put("b", [2], ["b"])
        self.assertEqual(self.cache.stats()["size"], 2)

    def test_stats_max_size(self):
        c = TokenizerCache(max_size=7)
        self.assertEqual(c.stats()["max_size"], 7)


class TestTokenizerCacheLRUEviction(unittest.TestCase):
    """Tests for LRU eviction behaviour."""

    def test_oldest_evicted_when_at_capacity(self):
        cache = TokenizerCache(max_size=3)
        cache.put("first", [1], ["f"])
        cache.put("second", [2], ["s"])
        cache.put("third", [3], ["t"])
        # Now at capacity; inserting one more should evict "first"
        cache.put("fourth", [4], ["fo"])
        self.assertIsNone(cache.get("first"))

    def test_recently_used_not_evicted(self):
        cache = TokenizerCache(max_size=3)
        cache.put("a", [1], ["a"])
        cache.put("b", [2], ["b"])
        cache.put("c", [3], ["c"])
        # Access "a" to make it recently used
        cache.get("a")
        # Insert "d" — LRU should now be "b"
        cache.put("d", [4], ["d"])
        self.assertIsNone(cache.get("b"))
        self.assertIsNotNone(cache.get("a"))

    def test_len_does_not_exceed_max_size(self):
        cache = TokenizerCache(max_size=5)
        for i in range(20):
            cache.put(f"text_{i}", [i], [str(i)])
        self.assertLessEqual(len(cache), 5)

    def test_len_correct_after_puts(self):
        cache = TokenizerCache(max_size=10)
        for i in range(4):
            cache.put(f"text_{i}", [i], [str(i)])
        self.assertEqual(len(cache), 4)


class TestTokenizerCacheInvalidateClear(unittest.TestCase):
    """Tests for invalidate and clear."""

    def setUp(self):
        self.cache = TokenizerCache()
        self.cache.put("alpha", [1], ["alpha"])
        self.cache.put("beta", [2], ["beta"])

    def test_invalidate_existing_returns_true(self):
        self.assertTrue(self.cache.invalidate("alpha"))

    def test_invalidate_existing_removes_entry(self):
        self.cache.invalidate("alpha")
        self.assertIsNone(self.cache.get("alpha"))

    def test_invalidate_nonexistent_returns_false(self):
        self.assertFalse(self.cache.invalidate("gamma"))

    def test_invalidate_does_not_affect_other_entries(self):
        self.cache.invalidate("alpha")
        self.assertIsNotNone(self.cache.get("beta"))

    def test_clear_empties_cache(self):
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)

    def test_clear_makes_get_return_none(self):
        self.cache.clear()
        self.assertIsNone(self.cache.get("alpha"))

    def test_len_after_clear_is_zero(self):
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)


class TestTokenizerCacheRegistry(unittest.TestCase):
    """Tests for TOKENIZER_CACHE_REGISTRY."""

    def test_registry_has_default_key(self):
        self.assertIn("default", TOKENIZER_CACHE_REGISTRY)

    def test_registry_default_is_tokenizer_cache_class(self):
        self.assertIs(TOKENIZER_CACHE_REGISTRY["default"], TokenizerCache)

    def test_registry_default_is_instantiable(self):
        cls = TOKENIZER_CACHE_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, TokenizerCache)


if __name__ == "__main__":
    unittest.main()
