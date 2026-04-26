"""Tests for src/model/local_attention.py.

Covers:
1. RingBuffer push and ordered read
2. RingBuffer overflow behaviour (oldest overwritten)
3. RingBuffer size is capped at capacity
4. RingBuffer is_full property
5. LocalWindowKVCache.update returns correctly shaped windowed tensors
6. LocalWindowAttention output shape matches input
7. Local window mask restricts attention beyond window
8. _local_causal_mask shape is (1, 1, T, T)
9. LocalAttentionGenerator.generate returns a list of ints
10. LocalAttentionGenerator.memory_usage returns required keys
"""

import torch

from src.model.config import AureliusConfig
from src.model.local_attention import (
    LocalAttentionConfig,
    LocalAttentionGenerator,
    LocalWindowAttention,
    LocalWindowKVCache,
    RingBuffer,
)
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared test config (small, fast)
# ---------------------------------------------------------------------------

TEST_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
)

LOCAL_CFG = LocalAttentionConfig(window_size=8)


# ---------------------------------------------------------------------------
# 1. test_ring_buffer_push_and_read
# ---------------------------------------------------------------------------


def test_ring_buffer_push_and_read():
    """Push 3 items; read_ordered returns them in insertion order."""
    buf = RingBuffer(capacity=5, shape=(2,))
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    c = torch.tensor([5.0, 6.0])

    buf.push(a)
    buf.push(b)
    buf.push(c)

    result = buf.read_ordered()
    assert result.shape == (3, 2)
    assert torch.allclose(result[0], a), "First element should be a"
    assert torch.allclose(result[1], b), "Second element should be b"
    assert torch.allclose(result[2], c), "Third element should be c"


# ---------------------------------------------------------------------------
# 2. test_ring_buffer_overflow
# ---------------------------------------------------------------------------


def test_ring_buffer_overflow():
    """Push capacity+2 items; oldest 2 are overwritten."""
    capacity = 4
    buf = RingBuffer(capacity=capacity, shape=(1,))

    # Push capacity + 2 items: values 0..5
    for i in range(capacity + 2):
        buf.push(torch.tensor([float(i)]))

    result = buf.read_ordered()
    assert result.shape == (capacity, 1), "Size must equal capacity after overflow"
    # Oldest surviving item is index 2 (items 0 and 1 were overwritten)
    expected = [float(i) for i in range(2, capacity + 2)]
    actual = result[:, 0].tolist()
    assert actual == expected, f"Expected {expected}, got {actual}"


# ---------------------------------------------------------------------------
# 3. test_ring_buffer_size_capped_at_capacity
# ---------------------------------------------------------------------------


def test_ring_buffer_size_capped_at_capacity():
    """size never exceeds capacity regardless of how many items are pushed."""
    capacity = 3
    buf = RingBuffer(capacity=capacity, shape=())

    for i in range(capacity * 3):
        buf.push(torch.tensor(float(i)))
        assert buf.size <= capacity, f"size {buf.size} exceeded capacity {capacity}"


# ---------------------------------------------------------------------------
# 4. test_ring_buffer_is_full
# ---------------------------------------------------------------------------


def test_ring_buffer_is_full():
    """is_full is False until capacity inserts, then True."""
    capacity = 3
    buf = RingBuffer(capacity=capacity, shape=(4,))

    for i in range(capacity):
        assert not buf.is_full, f"Should not be full after {i} inserts"
        buf.push(torch.zeros(4))

    assert buf.is_full, "Should be full after capacity inserts"

    # Still full after additional inserts
    buf.push(torch.zeros(4))
    assert buf.is_full


# ---------------------------------------------------------------------------
# 5. test_local_kv_cache_update_returns_windowed
# ---------------------------------------------------------------------------


def test_local_kv_cache_update_returns_windowed():
    """update() returns (T_window, n_heads, head_dim) tensors."""
    n_layers, n_heads, head_dim, window_size = 2, 2, 16, 4
    cache = LocalWindowKVCache(n_layers, n_heads, head_dim, window_size)

    for i in range(3):
        k = torch.randn(n_heads, head_dim)
        v = torch.randn(n_heads, head_dim)
        k_win, v_win = cache.update(layer_idx=0, new_k=k, new_v=v)

    # After 3 pushes into a window_size=4 cache, size should be 3
    assert k_win.shape == (3, n_heads, head_dim), f"Unexpected k_win shape: {k_win.shape}"
    assert v_win.shape == (3, n_heads, head_dim), f"Unexpected v_win shape: {v_win.shape}"

    # Push beyond capacity; shape should be capped at window_size
    for _ in range(window_size + 2):
        k_win, v_win = cache.update(
            0, torch.randn(n_heads, head_dim), torch.randn(n_heads, head_dim)
        )

    assert k_win.shape == (window_size, n_heads, head_dim)
    assert v_win.shape == (window_size, n_heads, head_dim)


# ---------------------------------------------------------------------------
# 6. test_local_window_attention_output_shape
# ---------------------------------------------------------------------------


def test_local_window_attention_output_shape():
    """(B, T, D) input returns (B, T, D) output."""
    layer = LocalWindowAttention(TEST_CONFIG, LOCAL_CFG)
    layer.train(False)

    B, T = 2, 16
    x = torch.randn(B, T, TEST_CONFIG.d_model)

    with torch.no_grad():
        out = layer(x)

    assert out.shape == (B, T, TEST_CONFIG.d_model), (
        f"Expected ({B}, {T}, {TEST_CONFIG.d_model}), got {out.shape}"
    )


# ---------------------------------------------------------------------------
# 7. test_local_window_mask_restricts_attention
# ---------------------------------------------------------------------------


def test_local_window_mask_restricts_attention():
    """Token at position i cannot attend beyond i - window + 1."""
    layer = LocalWindowAttention(TEST_CONFIG, LOCAL_CFG)
    seq_len = 16
    window = LOCAL_CFG.window_size  # 8

    mask = layer._local_causal_mask(seq_len, window, device=torch.device("cpu"))
    # mask: (1, 1, T, T) bool — True = attend, False = masked

    # Token at position 15 (last) should attend to 15-8+1=8 .. 15
    for j in range(8, 16):
        assert mask[0, 0, 15, j].item(), f"Token 15 should attend to token {j}"

    # Token 15 must NOT attend to token 7 (just outside window)
    assert not mask[0, 0, 15, 7].item(), "Token 15 must not attend to token 7"

    # Causal: token 5 must not attend to future token 6
    assert not mask[0, 0, 5, 6].item(), "Token 5 must not attend to future token 6"


# ---------------------------------------------------------------------------
# 8. test_local_window_causal_mask_shape
# ---------------------------------------------------------------------------


def test_local_window_causal_mask_shape():
    """_local_causal_mask returns a (1, 1, T, T) boolean tensor."""
    layer = LocalWindowAttention(TEST_CONFIG, LOCAL_CFG)
    T = 20

    mask = layer._local_causal_mask(T, LOCAL_CFG.window_size, device=torch.device("cpu"))
    assert mask.shape == (1, 1, T, T), f"Expected (1, 1, {T}, {T}), got {mask.shape}"
    assert mask.dtype == torch.bool, f"Mask should be bool, got {mask.dtype}"


# ---------------------------------------------------------------------------
# 9. test_local_attention_generator_returns_tokens
# ---------------------------------------------------------------------------


def test_local_attention_generator_returns_tokens():
    """generate() returns a list of integers of length max_new_tokens."""
    model = AureliusTransformer(TEST_CONFIG)
    local_cfg = LocalAttentionConfig(window_size=8)

    generator = LocalAttentionGenerator(
        model=model,
        local_cfg=local_cfg,
        n_layers=TEST_CONFIG.n_layers,
        n_heads=TEST_CONFIG.n_kv_heads,
        head_dim=TEST_CONFIG.head_dim,
    )

    prompt = [1, 2, 3, 4, 5]
    max_new = 10
    result = generator.generate(prompt, max_new_tokens=max_new, temperature=1.0)

    assert isinstance(result, list), "generate() must return a list"
    assert len(result) == max_new, f"Expected {max_new} tokens, got {len(result)}"
    assert all(isinstance(t, int) for t in result), "All generated tokens must be ints"


# ---------------------------------------------------------------------------
# 10. test_memory_usage_returns_dict
# ---------------------------------------------------------------------------


def test_memory_usage_returns_dict():
    """memory_usage() returns a dict with all required keys and correct types."""
    model = AureliusTransformer(TEST_CONFIG)
    local_cfg = LocalAttentionConfig(window_size=8)

    generator = LocalAttentionGenerator(
        model=model,
        local_cfg=local_cfg,
        n_layers=TEST_CONFIG.n_layers,
        n_heads=TEST_CONFIG.n_kv_heads,
        head_dim=TEST_CONFIG.head_dim,
    )

    info = generator.memory_usage()

    required_keys = {"cache_mb", "window_size", "n_layers", "tokens_in_cache"}
    assert required_keys.issubset(info.keys()), f"Missing keys: {required_keys - info.keys()}"
    assert isinstance(info["cache_mb"], float), "cache_mb must be a float"
    assert info["window_size"] == local_cfg.window_size
    assert info["n_layers"] == TEST_CONFIG.n_layers
    assert isinstance(info["tokens_in_cache"], int)
