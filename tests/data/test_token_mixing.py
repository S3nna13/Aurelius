"""Tests for src/data/token_mixing.py — uses tiny configs throughout."""

from __future__ import annotations

import random

import pytest

from src.data.token_mixing import (
    TokenMixer,
    TokenMixingConfig,
    compute_domain_distribution,
    compute_mixing_weights_from_tokens,
    create_attention_mask,
    interleave_sequences,
    normalize_weights,
    pack_sequences,
    sample_domain,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEQ_LEN = 8
EOS_ID = 0

# Small, predictable token lists (no zeros so EOS boundaries are visible)
DOMAIN_A_SEQS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
]
DOMAIN_B_SEQS = [
    [13, 14, 15, 16, 17, 18, 19, 20],
    [21, 22, 23, 24],
]


def _make_config(**kwargs) -> TokenMixingConfig:
    defaults = dict(
        domain_weights={"web": 0.6, "books": 0.4},
        buffer_size=32,
        sequence_length=SEQ_LEN,
        pack_sequences=True,
        eos_token_id=EOS_ID,
    )
    defaults.update(kwargs)
    return TokenMixingConfig(**defaults)


def _make_mixer(config: TokenMixingConfig | None = None) -> TokenMixer:
    if config is None:
        config = _make_config()
    domain_sequences = {
        "web": DOMAIN_A_SEQS,
        "books": DOMAIN_B_SEQS,
    }
    return TokenMixer(domain_sequences, config)


# ---------------------------------------------------------------------------
# 1. Config creation
# ---------------------------------------------------------------------------


def test_config_creation():
    cfg = _make_config()
    assert cfg.sequence_length == SEQ_LEN
    assert cfg.eos_token_id == EOS_ID
    assert cfg.pack_sequences is True
    assert "web" in cfg.domain_weights


# ---------------------------------------------------------------------------
# 2. normalize_weights — sums to 1
# ---------------------------------------------------------------------------


def test_normalize_weights_sums_to_one():
    w = {"a": 3.0, "b": 1.0, "c": 6.0}
    normed = normalize_weights(w)
    assert abs(sum(normed.values()) - 1.0) < 1e-9


def test_normalize_weights_correct_fractions():
    normed = normalize_weights({"x": 1.0, "y": 3.0})
    assert abs(normed["x"] - 0.25) < 1e-9
    assert abs(normed["y"] - 0.75) < 1e-9


# ---------------------------------------------------------------------------
# 3. normalize_weights raises on zero / negative weight
# ---------------------------------------------------------------------------


def test_normalize_weights_raises_on_zero():
    with pytest.raises(ValueError):
        normalize_weights({"a": 0.0, "b": 1.0})


def test_normalize_weights_raises_on_negative():
    with pytest.raises(ValueError):
        normalize_weights({"a": -1.0, "b": 2.0})


# ---------------------------------------------------------------------------
# 4. sample_domain — returns valid key
# ---------------------------------------------------------------------------


def test_sample_domain_returns_valid_key():
    weights = {"cat": 0.5, "dog": 0.3, "fish": 0.2}
    rng = random.Random(42)
    for _ in range(50):
        domain = sample_domain(weights, rng)
        assert domain in weights


def test_sample_domain_extreme_weight():
    """With essentially all weight on one domain, it should dominate."""
    weights = {"dominant": 999.0, "rare": 0.001}
    rng = random.Random(0)
    results = [sample_domain(weights, rng) for _ in range(100)]
    assert results.count("dominant") > 90


# ---------------------------------------------------------------------------
# 5. pack_sequences — output chunks have correct length
# ---------------------------------------------------------------------------


def test_pack_sequences_chunk_length():
    seqs = [[1, 2, 3], [4, 5, 6, 7], [8, 9]]
    chunks = pack_sequences(seqs, seq_len=SEQ_LEN, eos_id=EOS_ID)
    for chunk in chunks:
        assert len(chunk) == SEQ_LEN


def test_pack_sequences_no_partial_chunk():
    """The last chunk must not be shorter than seq_len (it should be dropped)."""
    seqs = [[1, 2, 3]]
    # Stream will be [1,2,3,0] — only 4 tokens, shorter than SEQ_LEN=8.
    chunks = pack_sequences(seqs, seq_len=SEQ_LEN, eos_id=EOS_ID)
    assert chunks == []


# ---------------------------------------------------------------------------
# 6. pack_sequences — inserts EOS between sequences
# ---------------------------------------------------------------------------


def test_pack_sequences_inserts_eos():
    """EOS_ID should appear at the boundary between documents in the stream."""
    # Force at least two sequences to be packed into one chunk.
    seqs = [[1, 2, 3], [4, 5, 6]]
    # Stream: [1,2,3,0,4,5,6,0] — exactly 8 tokens → one chunk
    chunks = pack_sequences(seqs, seq_len=SEQ_LEN, eos_id=EOS_ID)
    assert len(chunks) == 1
    chunk = chunks[0]
    # EOS should appear after position 2 (index 3) and at the end (index 7)
    assert chunk[3] == EOS_ID


def test_pack_sequences_empty_input():
    assert pack_sequences([], seq_len=SEQ_LEN, eos_id=EOS_ID) == []


# ---------------------------------------------------------------------------
# 7. create_attention_mask — length matches input
# ---------------------------------------------------------------------------


def test_create_attention_mask_length():
    token_ids = [1, 2, EOS_ID, 4, 5]
    mask = create_attention_mask(token_ids, EOS_ID)
    assert len(mask) == len(token_ids)


def test_create_attention_mask_all_ones():
    token_ids = [1, EOS_ID, 3, 4]
    mask = create_attention_mask(token_ids, EOS_ID)
    assert all(m == 1 for m in mask)


# ---------------------------------------------------------------------------
# 8. TokenMixer.get_batch — length equals batch_size
# ---------------------------------------------------------------------------


def test_get_batch_length_equals_batch_size():
    mixer = _make_mixer()
    rng = random.Random(1)
    batch = mixer.get_batch(batch_size=5, rng=rng)
    assert len(batch["input_ids"]) == 5
    assert len(batch["domain_labels"]) == 5
    assert len(batch["lengths"]) == 5


# ---------------------------------------------------------------------------
# 9. get_batch — domain_labels are valid domain keys
# ---------------------------------------------------------------------------


def test_get_batch_domain_labels_valid():
    cfg = _make_config()
    mixer = _make_mixer(cfg)
    rng = random.Random(7)
    batch = mixer.get_batch(batch_size=20, rng=rng)
    valid_domains = set(cfg.domain_weights.keys())
    for label in batch["domain_labels"]:
        assert label in valid_domains


# ---------------------------------------------------------------------------
# 10. get_domain_stats — keys for each domain
# ---------------------------------------------------------------------------


def test_get_domain_stats_keys():
    mixer = _make_mixer()
    stats = mixer.get_domain_stats()
    for domain in mixer.domain_sequences:
        assert domain in stats
        for field in ("n_sequences", "mean_length", "total_tokens"):
            assert field in stats[domain]


def test_get_domain_stats_values_correct():
    mixer = _make_mixer()
    stats = mixer.get_domain_stats()
    assert stats["web"]["n_sequences"] == len(DOMAIN_A_SEQS)
    expected_total = sum(len(s) for s in DOMAIN_A_SEQS)
    assert abs(stats["web"]["total_tokens"] - expected_total) < 1e-9


# ---------------------------------------------------------------------------
# 11. compute_mixing_weights_from_tokens — sums to 1
# ---------------------------------------------------------------------------


def test_compute_mixing_weights_from_tokens_sums_to_one():
    domain_sequences = {
        "web": DOMAIN_A_SEQS,
        "books": DOMAIN_B_SEQS,
    }
    weights = compute_mixing_weights_from_tokens(domain_sequences)
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_compute_mixing_weights_from_tokens_proportional():
    """Domain with more tokens should get higher weight."""
    domain_sequences = {
        "big": [[i for i in range(100)]],  # 100 tokens
        "small": [[1, 2, 3]],  # 3 tokens
    }
    weights = compute_mixing_weights_from_tokens(domain_sequences)
    assert weights["big"] > weights["small"]


# ---------------------------------------------------------------------------
# 12. interleave_sequences — combines both sequences
# ---------------------------------------------------------------------------


def test_interleave_sequences_combines_both():
    a = [10, 20, 30]
    b = [1, 2, 3, 4, 5, 6]
    result = interleave_sequences(a, b, ratio=0.5)
    # All tokens from both sequences should be present
    assert set(result) >= set(a) | set(b)


def test_interleave_sequences_length():
    a = [10, 20, 30]
    b = [1, 2, 3, 4, 5, 6]
    result = interleave_sequences(a, b, ratio=0.5)
    assert len(result) == len(a) + len(b)


# ---------------------------------------------------------------------------
# 13. compute_domain_distribution — sums to 1
# ---------------------------------------------------------------------------


def test_compute_domain_distribution_sums_to_one():
    labels = ["web", "books", "web", "code", "web"]
    dist = compute_domain_distribution(labels)
    assert abs(sum(dist.values()) - 1.0) < 1e-9


def test_compute_domain_distribution_correct_fractions():
    labels = ["a", "a", "b"]
    dist = compute_domain_distribution(labels)
    assert abs(dist["a"] - 2 / 3) < 1e-9
    assert abs(dist["b"] - 1 / 3) < 1e-9


def test_compute_domain_distribution_empty():
    assert compute_domain_distribution([]) == {}


# ---------------------------------------------------------------------------
# 14. get_batch — handles empty domain sequences gracefully
# ---------------------------------------------------------------------------


def test_get_batch_empty_sequences_graceful():
    """Mixer with all-empty domain sequences should return an empty batch."""
    domain_sequences = {"web": [], "books": []}
    cfg = _make_config()
    mixer = TokenMixer(domain_sequences, cfg)
    rng = random.Random(0)
    batch = mixer.get_batch(batch_size=4, rng=rng)
    assert batch["input_ids"] == []
    assert batch["domain_labels"] == []
    assert batch["lengths"] == []
