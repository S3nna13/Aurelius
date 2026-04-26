"""Tests for attention sink streaming inference."""

import pytest
import torch

from src.inference.attention_sink import (
    SinkConfig,
    SinkKVCache,
    StreamingGenerator,
    build_sink_mask,
    compute_sink_attention_stats,
)

# ── SinkConfig tests ──────────────────────────────────────────────────


class TestSinkConfig:
    def test_defaults(self):
        cfg = SinkConfig()
        assert cfg.n_sink_tokens == 4
        assert cfg.window_size == 256
        assert cfg.eviction_policy == "fifo"

    def test_custom_values(self):
        cfg = SinkConfig(n_sink_tokens=2, window_size=64, eviction_policy="lru")
        assert cfg.n_sink_tokens == 2
        assert cfg.window_size == 64
        assert cfg.eviction_policy == "lru"

    def test_rejects_negative_sink_tokens(self):
        with pytest.raises(ValueError, match="n_sink_tokens"):
            SinkConfig(n_sink_tokens=-1)

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="window_size"):
            SinkConfig(window_size=0)

    def test_rejects_invalid_eviction_policy(self):
        with pytest.raises(ValueError, match="eviction_policy"):
            SinkConfig(eviction_policy="random")


# ── SinkKVCache tests ─────────────────────────────────────────────────


class TestSinkKVCache:
    def _make_kv(self, seq_len: int, val: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Create (batch=1, seq_len, n_kv_heads=2, head_dim=4) tensors."""
        k = torch.full((1, seq_len, 2, 4), val)
        v = torch.full((1, seq_len, 2, 4), val)
        return k, v

    def test_update_grows_cache(self):
        cache = SinkKVCache(SinkConfig(n_sink_tokens=2, window_size=8))
        cache.update(*self._make_kv(3))
        assert cache.current_length == 3

    def test_eviction_keeps_sink_plus_window(self):
        cfg = SinkConfig(n_sink_tokens=2, window_size=3)
        cache = SinkKVCache(cfg)
        # Insert 10 tokens one at a time
        for i in range(10):
            k = torch.full((1, 1, 2, 4), float(i))
            cache.update(k, k.clone())
        # Should have sink(2) + window(3) = 5
        assert cache.current_length == 5

    def test_sink_tokens_preserved_after_eviction(self):
        cfg = SinkConfig(n_sink_tokens=2, window_size=2)
        cache = SinkKVCache(cfg)
        for i in range(8):
            k = torch.full((1, 1, 2, 4), float(i))
            cache.update(k, k.clone())
        keys, _ = cache.get_kv()
        # First two positions should be sink tokens (value 0 and 1)
        assert torch.allclose(keys[0, 0, 0, 0], torch.tensor(0.0))
        assert torch.allclose(keys[0, 1, 0, 0], torch.tensor(1.0))

    def test_window_has_most_recent_tokens(self):
        cfg = SinkConfig(n_sink_tokens=1, window_size=2)
        cache = SinkKVCache(cfg)
        for i in range(6):
            k = torch.full((1, 1, 2, 4), float(i))
            cache.update(k, k.clone())
        keys, _ = cache.get_kv()
        # Window tokens should be the last 2 inserted: 4 and 5
        assert torch.allclose(keys[0, 1, 0, 0], torch.tensor(4.0))
        assert torch.allclose(keys[0, 2, 0, 0], torch.tensor(5.0))

    def test_get_kv_raises_on_empty(self):
        cache = SinkKVCache(SinkConfig())
        with pytest.raises(RuntimeError, match="empty"):
            cache.get_kv()

    def test_reset_clears_cache(self):
        cache = SinkKVCache(SinkConfig(n_sink_tokens=1, window_size=4))
        cache.update(*self._make_kv(5))
        cache.reset()
        assert cache.current_length == 0

    def test_rejects_non_4d_input(self):
        cache = SinkKVCache(SinkConfig())
        with pytest.raises(ValueError, match="4D"):
            cache.update(torch.randn(1, 2, 3), torch.randn(1, 2, 3))

    def test_lru_eviction_keeps_accessed_tokens(self):
        cfg = SinkConfig(n_sink_tokens=1, window_size=3, eviction_policy="lru")
        cache = SinkKVCache(cfg)
        # Insert 4 tokens (fills sink=1 + window=3)
        for i in range(4):
            k = torch.full((1, 1, 2, 4), float(i))
            cache.update(k, k.clone())
        # Access cache (bumps access counts)
        cache.get_kv()
        # Insert one more to trigger eviction
        k = torch.full((1, 1, 2, 4), 99.0)
        cache.update(k, k.clone())
        # Should still be sink(1) + window(3) = 4
        assert cache.current_length == 4


# ── build_sink_mask tests ─────────────────────────────────────────────


class TestBuildSinkMask:
    def test_shape(self):
        mask = build_sink_mask(n_sink=2, window_size=3, seq_len=8)
        assert mask.shape == (8, 8)

    def test_causal_property(self):
        """No position should attend to future positions."""
        mask = build_sink_mask(n_sink=2, window_size=3, seq_len=8)
        for i in range(8):
            for j in range(i + 1, 8):
                assert not mask[i, j], f"Position {i} should not attend to future position {j}"

    def test_sink_tokens_always_attended(self):
        mask = build_sink_mask(n_sink=3, window_size=2, seq_len=10)
        # Every position >= 3 should attend to sink positions 0,1,2
        for i in range(3, 10):
            for j in range(3):
                assert mask[i, j], f"Position {i} should attend to sink {j}"

    def test_window_coverage(self):
        mask = build_sink_mask(n_sink=0, window_size=3, seq_len=6)
        # Position 5 should attend to 3,4,5 (window=3) but not 0,1,2
        assert mask[5, 5]
        assert mask[5, 4]
        assert mask[5, 3]
        assert not mask[5, 2]

    def test_rejects_negative_seq_len(self):
        with pytest.raises(ValueError):
            build_sink_mask(0, 3, 0)

    def test_small_sequence(self):
        mask = build_sink_mask(n_sink=4, window_size=4, seq_len=2)
        # With seq_len=2, both tokens are sinks; causal: pos 0->0, pos 1->0,1
        assert mask[0, 0]
        assert not mask[0, 1]
        assert mask[1, 0]
        assert mask[1, 1]


# ── StreamingGenerator tests ──────────────────────────────────────────


class _DummyModel:
    """Minimal model that returns (loss, logits, pkv) tuple."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self._call_count = 0

    def __call__(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        self._call_count += 1
        logits = torch.randn(batch, seq_len, self.vocab_size)
        # Make token 42 the argmax so output is deterministic
        logits[:, :, 42] = 100.0
        # pkv: list of (K, V) per layer, single layer
        k = torch.randn(batch, seq_len, 2, 4)
        v = torch.randn(batch, seq_len, 2, 4)
        return 0.0, logits, [(k, v)]


class TestStreamingGenerator:
    def test_yields_correct_count(self):
        model = _DummyModel()
        gen = StreamingGenerator(model, SinkConfig(n_sink_tokens=2, window_size=8))
        tokens = list(gen.generate_stream(torch.randint(0, 256, (1, 5)), max_tokens=10))
        assert len(tokens) == 10

    def test_yields_integers(self):
        model = _DummyModel()
        gen = StreamingGenerator(model, SinkConfig(n_sink_tokens=1, window_size=4))
        tokens = list(gen.generate_stream(torch.randint(0, 256, (1, 3)), max_tokens=5))
        assert all(isinstance(t, int) for t in tokens)

    def test_deterministic_output(self):
        model = _DummyModel()
        gen = StreamingGenerator(model, SinkConfig(n_sink_tokens=2, window_size=8))
        tokens = list(gen.generate_stream(torch.randint(0, 256, (1, 3)), max_tokens=4))
        # Dummy model always makes token 42 highest
        assert all(t == 42 for t in tokens)


# ── compute_sink_attention_stats tests ────────────────────────────────


class TestComputeSinkAttentionStats:
    def test_all_attention_on_sinks(self):
        # 1 batch, 1 head, 4 queries, 4 keys; all attention on first 2 (sinks)
        weights = torch.zeros(1, 1, 4, 4)
        weights[:, :, :, 0] = 0.5
        weights[:, :, :, 1] = 0.5
        stats = compute_sink_attention_stats(weights, n_sink=2)
        assert abs(stats["sink_fraction"] - 1.0) < 1e-5
        assert abs(stats["window_fraction"] - 0.0) < 1e-5

    def test_even_split(self):
        weights = torch.ones(1, 1, 4, 4) / 4.0
        stats = compute_sink_attention_stats(weights, n_sink=2)
        assert abs(stats["sink_fraction"] - 0.5) < 1e-5
        assert abs(stats["window_fraction"] - 0.5) < 1e-5

    def test_rejects_non_4d(self):
        with pytest.raises(ValueError, match="4D"):
            compute_sink_attention_stats(torch.randn(4, 4), n_sink=1)

    def test_rejects_negative_n_sink(self):
        with pytest.raises(ValueError, match="n_sink"):
            compute_sink_attention_stats(torch.randn(1, 1, 4, 4), n_sink=-1)

    def test_fractions_sum_to_one(self):
        weights = torch.softmax(torch.randn(2, 4, 8, 8), dim=-1)
        stats = compute_sink_attention_stats(weights, n_sink=3)
        assert abs(stats["sink_fraction"] + stats["window_fraction"] - 1.0) < 1e-5
