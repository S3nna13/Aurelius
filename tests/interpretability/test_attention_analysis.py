"""
Tests for src/interpretability/attention_analysis.py

Tiny configuration:
    H=4 heads, T=6 sequence length
"""

import torch

from src.interpretability.attention_analysis import (
    AttentionAnalyzer,
    AttentionConfig,
    compute_attention_entropy,
    compute_attention_sink_score,
    compute_head_agreement,
    compute_mean_attention_distance,
    find_diagonal_heads,
    find_previous_token_heads,
)

# ---------------------------------------------------------------------------
# Tiny shared constants
# ---------------------------------------------------------------------------
H = 4  # number of heads
T = 6  # sequence length

torch.manual_seed(0)


def _uniform_attn() -> torch.Tensor:
    """Uniform attention weights (H, T, T), each row sums to 1."""
    return torch.full((H, T, T), 1.0 / T)


def _random_attn() -> torch.Tensor:
    """Random valid attention weights via softmax, shape (H, T, T)."""
    return torch.softmax(torch.randn(H, T, T), dim=-1)


def _identity_attn() -> torch.Tensor:
    """
    Attention where each token attends 100% to itself (identity matrix),
    shape (H, T, T).
    """
    eye = torch.eye(T).unsqueeze(0).expand(H, T, T)
    return eye


def _prev_token_attn() -> torch.Tensor:
    """
    Attention where each token i (i > 0) attends 100% to token i-1,
    and token 0 attends uniformly (to keep rows summing to 1).
    Shape (H, T, T).
    """
    mat = torch.zeros(T, T)
    # Token 0 attends uniformly
    mat[0, :] = 1.0 / T
    # Tokens 1..T-1 attend entirely to previous token
    for i in range(1, T):
        mat[i, i - 1] = 1.0
    return mat.unsqueeze(0).expand(H, T, T)


def _sink_attn(sink: int = 0) -> torch.Tensor:
    """
    Attention where every query attends 100% to position `sink`.
    Shape (H, T, T).
    """
    mat = torch.zeros(T, T)
    mat[:, sink] = 1.0
    return mat.unsqueeze(0).expand(H, T, T)


# ---------------------------------------------------------------------------
# 1. AttentionConfig defaults
# ---------------------------------------------------------------------------


def test_attention_config_defaults():
    cfg = AttentionConfig()
    assert cfg.n_heads == 8
    assert cfg.n_layers == 12
    assert cfg.track_entropy is True
    assert cfg.track_distance is True


# ---------------------------------------------------------------------------
# 2. compute_attention_entropy — shape (H, T)
# ---------------------------------------------------------------------------


def test_compute_attention_entropy_shape():
    attn = _random_attn()
    entropy = compute_attention_entropy(attn)
    assert entropy.shape == (H, T), f"Expected ({H},{T}), got {entropy.shape}"


# ---------------------------------------------------------------------------
# 3. compute_attention_entropy — non-negative
# ---------------------------------------------------------------------------


def test_compute_attention_entropy_nonnegative():
    attn = _random_attn()
    entropy = compute_attention_entropy(attn)
    assert (entropy >= 0).all(), f"Entropy contains negative values: {entropy.min()}"


# ---------------------------------------------------------------------------
# 4. compute_attention_entropy — uniform attention has maximum entropy
# ---------------------------------------------------------------------------


def test_compute_attention_entropy_uniform_is_max():
    uniform = _uniform_attn()
    peaked = _identity_attn()
    H_uni = compute_attention_entropy(uniform)
    H_peak = compute_attention_entropy(peaked)
    # Uniform distribution maximises entropy
    assert (H_uni >= H_peak - 1e-5).all(), (
        f"Uniform entropy should be >= peaked entropy. "
        f"uniform min={H_uni.min():.4f}, peaked max={H_peak.max():.4f}"
    )


# ---------------------------------------------------------------------------
# 5. compute_mean_attention_distance — shape (H, T)
# ---------------------------------------------------------------------------


def test_compute_mean_attention_distance_shape():
    attn = _random_attn()
    dist = compute_mean_attention_distance(attn)
    assert dist.shape == (H, T), f"Expected ({H},{T}), got {dist.shape}"


# ---------------------------------------------------------------------------
# 6. compute_mean_attention_distance — non-negative
# ---------------------------------------------------------------------------


def test_compute_mean_attention_distance_nonnegative():
    attn = _random_attn()
    dist = compute_mean_attention_distance(attn)
    assert (dist >= 0).all(), f"Mean distance contains negative values: {dist.min()}"


# ---------------------------------------------------------------------------
# 7. compute_head_agreement — shape (H,)
# ---------------------------------------------------------------------------


def test_compute_head_agreement_shape():
    attn1 = _random_attn()
    attn2 = _random_attn()
    agreement = compute_head_agreement(attn1, attn2)
    assert agreement.shape == (H,), f"Expected ({H},), got {agreement.shape}"


# ---------------------------------------------------------------------------
# 8. compute_head_agreement — identical heads score ≈ 1.0
# ---------------------------------------------------------------------------


def test_compute_head_agreement_identical_is_one():
    attn = _random_attn()
    agreement = compute_head_agreement(attn, attn)
    assert torch.allclose(agreement, torch.ones(H), atol=1e-5), (
        f"Identical attention patterns should agree = 1.0, got {agreement}"
    )


# ---------------------------------------------------------------------------
# 9. find_diagonal_heads — shape (H,) bool
# ---------------------------------------------------------------------------


def test_find_diagonal_heads_shape_and_dtype():
    attn = _random_attn()
    mask = find_diagonal_heads(attn)
    assert mask.shape == (H,), f"Expected ({H},), got {mask.shape}"
    assert mask.dtype == torch.bool, f"Expected bool, got {mask.dtype}"


# ---------------------------------------------------------------------------
# 10. find_diagonal_heads — identity attention → all diagonal
# ---------------------------------------------------------------------------


def test_find_diagonal_heads_identity_all_diagonal():
    attn = _identity_attn()
    mask = find_diagonal_heads(attn, threshold=0.5)
    assert mask.all(), f"Identity attention should mark all heads as diagonal, got {mask}"


# ---------------------------------------------------------------------------
# 11. find_previous_token_heads — shape (H,) bool
# ---------------------------------------------------------------------------


def test_find_previous_token_heads_shape_and_dtype():
    attn = _random_attn()
    mask = find_previous_token_heads(attn)
    assert mask.shape == (H,), f"Expected ({H},), got {mask.shape}"
    assert mask.dtype == torch.bool, f"Expected bool, got {mask.dtype}"


# ---------------------------------------------------------------------------
# 12. find_previous_token_heads — pure prev-token attention → all True
# ---------------------------------------------------------------------------


def test_find_previous_token_heads_pure_prev_all_true():
    attn = _prev_token_attn()
    # The mean of attn[h, i, i-1] for i>0 is 1.0 for all heads
    mask = find_previous_token_heads(attn, threshold=0.3)
    assert mask.all(), f"Pure previous-token attention should flag all heads, got {mask}"


# ---------------------------------------------------------------------------
# 13. compute_attention_sink_score — shape (H,)
# ---------------------------------------------------------------------------


def test_compute_attention_sink_score_shape():
    attn = _random_attn()
    scores = compute_attention_sink_score(attn)
    assert scores.shape == (H,), f"Expected ({H},), got {scores.shape}"


# ---------------------------------------------------------------------------
# 14. compute_attention_sink_score — all-sink attention ≈ 1.0
# ---------------------------------------------------------------------------


def test_compute_attention_sink_score_all_sink_is_one():
    attn = _sink_attn(sink=0)
    scores = compute_attention_sink_score(attn, sink_position=0)
    expected = torch.ones(H)
    assert torch.allclose(scores, expected, atol=1e-5), (
        f"All-sink attention should give score ≈ 1.0, got {scores}"
    )


# ---------------------------------------------------------------------------
# 15. AttentionAnalyzer.analyze_layer — returns dict with all required keys
# ---------------------------------------------------------------------------


def test_attention_analyzer_analyze_layer_keys():
    cfg = AttentionConfig(n_heads=H, n_layers=1)
    analyzer = AttentionAnalyzer(cfg)
    attn = _random_attn()
    result = analyzer.analyze_layer(attn)
    required = {"entropy", "mean_distance", "diagonal_heads", "prev_token_heads", "sink_scores"}
    for key in required:
        assert key in result, f"Missing key in analyze_layer output: '{key}'"


# ---------------------------------------------------------------------------
# 16. AttentionAnalyzer.analyze_layer — tensor shapes are correct
# ---------------------------------------------------------------------------


def test_attention_analyzer_analyze_layer_shapes():
    cfg = AttentionConfig(n_heads=H, n_layers=1)
    analyzer = AttentionAnalyzer(cfg)
    attn = _random_attn()
    result = analyzer.analyze_layer(attn)

    assert result["entropy"].shape == (H, T), f"entropy shape {result['entropy'].shape}"
    assert result["mean_distance"].shape == (H, T), (
        f"mean_distance shape {result['mean_distance'].shape}"
    )
    assert result["diagonal_heads"].shape == (H,), (
        f"diagonal_heads shape {result['diagonal_heads'].shape}"
    )
    assert result["prev_token_heads"].shape == (H,), (
        f"prev_token_heads shape {result['prev_token_heads'].shape}"
    )
    assert result["sink_scores"].shape == (H,), f"sink_scores shape {result['sink_scores'].shape}"


# ---------------------------------------------------------------------------
# 17. AttentionAnalyzer.summarize — returns dict with all scalar keys
# ---------------------------------------------------------------------------


def test_attention_analyzer_summarize_keys_and_scalar():
    cfg = AttentionConfig(n_heads=H, n_layers=1)
    analyzer = AttentionAnalyzer(cfg)
    attn = _random_attn()
    analysis = analyzer.analyze_layer(attn)
    summary = analyzer.summarize(analysis)

    required_scalar_keys = {
        "mean_entropy",
        "mean_distance",
        "n_diagonal",
        "n_prev_token",
        "max_sink_score",
    }
    for key in required_scalar_keys:
        assert key in summary, f"Missing key in summarize output: '{key}'"
        val = summary[key]
        assert isinstance(val, (int, float)), (
            f"summary['{key}'] should be a scalar, got {type(val)}"
        )


# ---------------------------------------------------------------------------
# 18. compute_attention_entropy — concentrated attention has low entropy
# ---------------------------------------------------------------------------


def test_compute_attention_entropy_identity_is_near_zero():
    attn = _identity_attn()
    entropy = compute_attention_entropy(attn)
    # Each row is a one-hot → entropy ≈ 0
    assert (entropy < 1e-5).all(), (
        f"One-hot attention should yield ~0 entropy, max={entropy.max():.6f}"
    )


# ---------------------------------------------------------------------------
# 19. compute_mean_attention_distance — self-attention has zero distance
# ---------------------------------------------------------------------------


def test_compute_mean_attention_distance_identity_is_zero():
    attn = _identity_attn()
    dist = compute_mean_attention_distance(attn)
    # Each token attends only to itself → |i - i| = 0
    assert torch.allclose(dist, torch.zeros(H, T), atol=1e-5), (
        f"Identity attention should give zero mean distance, got max={dist.max():.6f}"
    )


# ---------------------------------------------------------------------------
# 20. AttentionAnalyzer.summarize — n_diagonal is non-negative int-like
# ---------------------------------------------------------------------------


def test_attention_analyzer_summarize_n_diagonal_nonneg():
    cfg = AttentionConfig(n_heads=H, n_layers=1)
    analyzer = AttentionAnalyzer(cfg)
    attn = _random_attn()
    analysis = analyzer.analyze_layer(attn)
    summary = analyzer.summarize(analysis)
    assert summary["n_diagonal"] >= 0, f"n_diagonal must be >= 0, got {summary['n_diagonal']}"
    assert summary["n_diagonal"] <= H, f"n_diagonal must be <= H={H}, got {summary['n_diagonal']}"
