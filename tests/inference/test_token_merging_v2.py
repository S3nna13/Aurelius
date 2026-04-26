"""Tests for src/inference/token_merging_v2.py.

All tensors are small to keep tests fast and dependency-free (pure PyTorch).
"""

from __future__ import annotations

import torch

from src.inference.token_merging_v2 import (
    MergeConfig,
    TokenMerger,
    compute_token_similarity,
    merge_tokens,
    select_tokens_to_merge,
    unmerge_tokens,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokens(T: int = 8, d: int = 4, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(T, d)


# ===========================================================================
# 1. MergeConfig defaults
# ===========================================================================


def test_merge_config_defaults():
    cfg = MergeConfig()
    assert cfg.merge_ratio == 0.5
    assert cfg.similarity_metric == "cosine"
    assert cfg.merge_mode == "mean"
    assert cfg.n_keep_start == 1


# ===========================================================================
# 2. compute_token_similarity — shape (T, T)
# ===========================================================================


def test_compute_similarity_cosine_shape():
    tokens = _make_tokens(T=6, d=4)
    sim = compute_token_similarity(tokens, metric="cosine")
    assert sim.shape == (6, 6)


# ===========================================================================
# 3. compute_token_similarity — cosine diagonal ≈ 1.0
# ===========================================================================


def test_compute_similarity_cosine_diagonal():
    tokens = _make_tokens(T=5, d=8)
    sim = compute_token_similarity(tokens, metric="cosine")
    diag = sim.diagonal()
    assert torch.allclose(diag, torch.ones(5), atol=1e-5), (
        f"Cosine diagonal expected all-ones, got {diag}"
    )


# ===========================================================================
# 4. compute_token_similarity — l2 diagonal ≈ 0.0
# ===========================================================================


def test_compute_similarity_l2_diagonal():
    tokens = _make_tokens(T=5, d=8)
    sim = compute_token_similarity(tokens, metric="l2")
    diag = sim.diagonal()
    assert torch.allclose(diag, torch.zeros(5), atol=1e-5), (
        f"L2 diagonal expected all-zeros, got {diag}"
    )


# ===========================================================================
# 5. compute_token_similarity — dot product shape
# ===========================================================================


def test_compute_similarity_dot_shape():
    tokens = _make_tokens(T=4, d=3)
    sim = compute_token_similarity(tokens, metric="dot")
    assert sim.shape == (4, 4)


# ===========================================================================
# 6. select_tokens_to_merge — src_indices length == k
# ===========================================================================


def test_select_tokens_src_length():
    T, d = 10, 4
    merge_ratio = 0.3
    tokens = _make_tokens(T, d)
    sim = compute_token_similarity(tokens, metric="cosine")
    src, dst = select_tokens_to_merge(sim, merge_ratio=merge_ratio, n_keep_start=1)
    expected_k = int(T * merge_ratio)
    assert src.shape[0] == expected_k, f"Expected {expected_k} src indices, got {src.shape[0]}"


# ===========================================================================
# 7. select_tokens_to_merge — no duplicates in src_indices
# ===========================================================================


def test_select_tokens_no_duplicates_in_src():
    T, d = 12, 6
    tokens = _make_tokens(T, d, seed=42)
    sim = compute_token_similarity(tokens, metric="cosine")
    src, _ = select_tokens_to_merge(sim, merge_ratio=0.4, n_keep_start=1)
    assert src.numel() == src.unique().numel(), "src_indices must not have duplicates"


# ===========================================================================
# 8. select_tokens_to_merge — n_keep_start tokens never in src
# ===========================================================================


def test_select_tokens_keep_start_not_in_src():
    T, d = 10, 4
    n_keep = 3
    tokens = _make_tokens(T, d, seed=7)
    sim = compute_token_similarity(tokens, metric="cosine")
    src, _ = select_tokens_to_merge(sim, merge_ratio=0.4, n_keep_start=n_keep)
    protected = set(range(n_keep))
    for idx in src.tolist():
        assert idx not in protected, f"Protected token {idx} appeared in src_indices"


# ===========================================================================
# 9. merge_tokens — output shape (T - k, d)
# ===========================================================================


def test_merge_tokens_output_shape():
    T, d = 8, 4
    tokens = _make_tokens(T, d)
    sim = compute_token_similarity(tokens, metric="cosine")
    src, dst = select_tokens_to_merge(sim, merge_ratio=0.5, n_keep_start=1)
    k = src.shape[0]
    merged = merge_tokens(tokens, src, dst, mode="mean")
    assert merged.shape == (T - k, d), f"Expected shape ({T - k}, {d}), got {merged.shape}"


# ===========================================================================
# 10. merge_tokens — output is finite (no NaN / Inf)
# ===========================================================================


def test_merge_tokens_output_finite():
    T, d = 8, 4
    tokens = _make_tokens(T, d)
    sim = compute_token_similarity(tokens, metric="cosine")
    src, dst = select_tokens_to_merge(sim, merge_ratio=0.5, n_keep_start=1)
    merged = merge_tokens(tokens, src, dst, mode="mean")
    assert torch.isfinite(merged).all(), "merge_tokens output contains non-finite values"


# ===========================================================================
# 11. unmerge_tokens — output shape (original_len, d)
# ===========================================================================


def test_unmerge_tokens_output_shape():
    T, d = 8, 4
    tokens = _make_tokens(T, d)
    sim = compute_token_similarity(tokens, metric="cosine")
    src, dst = select_tokens_to_merge(sim, merge_ratio=0.5, n_keep_start=1)
    merged = merge_tokens(tokens, src, dst)
    restored = unmerge_tokens(merged, src, dst, original_len=T)
    assert restored.shape == (T, d), f"Expected shape ({T}, {d}), got {restored.shape}"


# ===========================================================================
# 12. TokenMerger.merge — returns reduced sequence
# ===========================================================================


def test_token_merger_merge_reduces_length():
    T, d = 10, 6
    tokens = _make_tokens(T, d)
    cfg = MergeConfig(merge_ratio=0.4, n_keep_start=1)
    merger = TokenMerger(cfg)
    merged, info = merger.merge(tokens)
    assert merged.shape[0] < T, "TokenMerger.merge should reduce sequence length"
    assert merged.shape[1] == d


# ===========================================================================
# 13. TokenMerger.unmerge — restores original length
# ===========================================================================


def test_token_merger_unmerge_restores_length():
    T, d = 10, 6
    tokens = _make_tokens(T, d)
    cfg = MergeConfig(merge_ratio=0.4, n_keep_start=1)
    merger = TokenMerger(cfg)
    merged, info = merger.merge(tokens)
    restored = merger.unmerge(merged, info)
    assert restored.shape == (T, d), f"Expected shape ({T}, {d}), got {restored.shape}"


# ===========================================================================
# 14. TokenMerger.compression_ratio — in (0, 1]
# ===========================================================================


def test_token_merger_compression_ratio_range():
    cfg = MergeConfig(merge_ratio=0.5, n_keep_start=1)
    merger = TokenMerger(cfg)
    ratio = merger.compression_ratio(original_len=20)
    assert 0.0 < ratio <= 1.0, f"compression_ratio out of range: {ratio}"


# ===========================================================================
# 15. Bonus: merge + unmerge round-trip is finite
# ===========================================================================


def test_round_trip_finite():
    T, d = 16, 8
    tokens = _make_tokens(T, d, seed=99)
    cfg = MergeConfig(merge_ratio=0.3, similarity_metric="cosine", n_keep_start=2)
    merger = TokenMerger(cfg)
    merged, info = merger.merge(tokens)
    restored = merger.unmerge(merged, info)
    assert torch.isfinite(restored).all()
    assert restored.shape == (T, d)


# ===========================================================================
# 16. Bonus: l2 metric end-to-end through TokenMerger
# ===========================================================================


def test_token_merger_l2_metric():
    T, d = 8, 4
    tokens = _make_tokens(T, d, seed=5)
    cfg = MergeConfig(merge_ratio=0.25, similarity_metric="l2", n_keep_start=1)
    merger = TokenMerger(cfg)
    merged, info = merger.merge(tokens)
    restored = merger.unmerge(merged, info)
    assert restored.shape == (T, d)


# ===========================================================================
# 17. Bonus: dot metric similarity shape
# ===========================================================================


def test_compute_similarity_dot_diagonal_values():
    """dot similarity diagonal == sum of squared elements per token."""
    tokens = _make_tokens(T=4, d=3, seed=11)
    sim = compute_token_similarity(tokens, metric="dot")
    expected_diag = (tokens * tokens).sum(dim=-1)
    assert torch.allclose(sim.diagonal(), expected_diag, atol=1e-5)


# ===========================================================================
# 18. Bonus: merge with weighted mode gives same shape as mean
# ===========================================================================


def test_merge_tokens_weighted_mode_shape():
    T, d = 8, 4
    tokens = _make_tokens(T, d)
    sim = compute_token_similarity(tokens)
    src, dst = select_tokens_to_merge(sim, merge_ratio=0.5)
    k = src.shape[0]
    merged = merge_tokens(tokens, src, dst, mode="weighted")
    assert merged.shape == (T - k, d)


# ===========================================================================
# 19. Bonus: zero merge_ratio keeps all tokens
# ===========================================================================


def test_merge_ratio_zero_keeps_all():
    T, d = 6, 4
    tokens = _make_tokens(T, d)
    cfg = MergeConfig(merge_ratio=0.0, n_keep_start=1)
    merger = TokenMerger(cfg)
    merged, info = merger.merge(tokens)
    # k is clamped to 0, so merged == tokens
    assert merged.shape[0] == T


# ===========================================================================
# 20. Bonus: compression_ratio with ratio=0 is 1.0
# ===========================================================================


def test_compression_ratio_zero_merge():
    cfg = MergeConfig(merge_ratio=0.0, n_keep_start=1)
    merger = TokenMerger(cfg)
    ratio = merger.compression_ratio(original_len=10)
    assert ratio == 1.0
