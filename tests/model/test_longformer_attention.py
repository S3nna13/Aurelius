import torch

from src.model.longformer_attention import (
    LongformerConfig,
    LongformerSelfAttention,
    compute_attention_sparsity,
    create_global_attention_mask,
    create_local_attention_mask,
    merge_attention_masks,
)

# Common test dimensions
T = 16
B = 2
D_MODEL = 32
N_HEADS = 2
WINDOW = 4


# ── LongformerConfig ────────────────────────────────────────────────────────


def test_longformer_config_defaults():
    cfg = LongformerConfig()
    assert cfg.window_size == 64
    assert cfg.n_global_tokens == 1
    assert cfg.dilation == 1
    assert cfg.d_model == 64
    assert cfg.n_heads == 2


# ── create_local_attention_mask ─────────────────────────────────────────────


def test_local_mask_shape():
    mask = create_local_attention_mask(T, WINDOW)
    assert mask.shape == (T, T)


def test_local_mask_dtype_bool():
    mask = create_local_attention_mask(T, WINDOW)
    assert mask.dtype == torch.bool


def test_local_mask_diagonal_true():
    """Every token attends to itself (self-attention)."""
    mask = create_local_attention_mask(T, WINDOW)
    assert mask.diagonal().all()


def test_local_mask_causal_window_zero_upper_triangle_false():
    """With window_size=0 only the diagonal is True (pure causal, self only)."""
    mask = create_local_attention_mask(T, window_size=0)
    # Upper triangle (j > i) must all be False
    for i in range(T):
        for j in range(i + 1, T):
            assert not mask[i, j].item(), f"Expected False at ({i},{j})"


def test_local_mask_window_full_lower_triangular():
    """With window_size >= T the mask equals the full causal (lower triangular) mask."""
    mask = create_local_attention_mask(T, window_size=T)
    expected = torch.tril(torch.ones(T, T, dtype=torch.bool))
    assert torch.equal(mask, expected)


def test_local_mask_window_coverage():
    """Token i should be able to attend to i, i-1, ..., i-WINDOW (but not i-WINDOW-1)."""
    mask = create_local_attention_mask(T, WINDOW)
    i = 10  # pick an interior token
    # Should attend to [i-WINDOW .. i]
    for j in range(i - WINDOW, i + 1):
        assert mask[i, j].item(), f"Expected True at ({i},{j})"
    # Should NOT attend to i - WINDOW - 1
    if i - WINDOW - 1 >= 0:
        assert not mask[i, i - WINDOW - 1].item()
    # Should NOT attend to i + 1 (future)
    if i + 1 < T:
        assert not mask[i, i + 1].item()


# ── create_global_attention_mask ────────────────────────────────────────────


def test_global_mask_shape():
    mask = create_global_attention_mask(T, n_global=1)
    assert mask.shape == (T, T)


def test_global_mask_global_row_all_true():
    """The global token's row must be entirely True (it attends to all)."""
    mask = create_global_attention_mask(T, n_global=1)
    assert mask[0, :].all()


def test_global_mask_global_col_all_true():
    """All tokens must attend to the global token (its column is all True)."""
    mask = create_global_attention_mask(T, n_global=1)
    assert mask[:, 0].all()


def test_global_mask_n_global_zero():
    """With n_global=0 the mask should be entirely False."""
    mask = create_global_attention_mask(T, n_global=0)
    assert not mask.any()


# ── merge_attention_masks ───────────────────────────────────────────────────


def test_merge_masks_is_or():
    """Merged mask should equal the logical OR of the two input masks."""
    local = create_local_attention_mask(T, WINDOW)
    glob = create_global_attention_mask(T, n_global=1)
    merged = merge_attention_masks(local, glob)
    expected = local | glob
    assert torch.equal(merged, expected)


def test_merge_masks_superset_of_local():
    local = create_local_attention_mask(T, WINDOW)
    glob = create_global_attention_mask(T, n_global=1)
    merged = merge_attention_masks(local, glob)
    # Every True in local must remain True in merged
    assert (local & merged).equal(local)


# ── LongformerSelfAttention ─────────────────────────────────────────────────


def test_longformer_attention_output_shape():
    cfg = LongformerConfig(window_size=WINDOW, n_global_tokens=1, d_model=D_MODEL, n_heads=N_HEADS)
    model = LongformerSelfAttention(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.shape == (B, T, D_MODEL)


def test_longformer_attention_output_dtype():
    cfg = LongformerConfig(window_size=WINDOW, n_global_tokens=1, d_model=D_MODEL, n_heads=N_HEADS)
    model = LongformerSelfAttention(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert out.dtype == torch.float32


def test_longformer_attention_no_nan():
    cfg = LongformerConfig(window_size=WINDOW, n_global_tokens=1, d_model=D_MODEL, n_heads=N_HEADS)
    model = LongformerSelfAttention(cfg)
    x = torch.randn(B, T, D_MODEL)
    out = model(x)
    assert not torch.isnan(out).any()


# ── compute_attention_sparsity ──────────────────────────────────────────────


def test_sparsity_keys_present():
    result = compute_attention_sparsity(T, window_size=WINDOW, n_global=1)
    for key in ("total_pairs", "attended_pairs", "sparsity_ratio", "vs_full_attention"):
        assert key in result, f"Missing key: {key}"


def test_sparsity_total_pairs():
    result = compute_attention_sparsity(T, window_size=WINDOW, n_global=1)
    assert result["total_pairs"] == T * T


def test_sparsity_ratio_in_range():
    result = compute_attention_sparsity(T, window_size=WINDOW, n_global=1)
    assert 0.0 <= result["sparsity_ratio"] <= 1.0


def test_sparsity_vs_full_in_range():
    result = compute_attention_sparsity(T, window_size=WINDOW, n_global=1)
    assert 0.0 <= result["vs_full_attention"] <= 1.0


def test_sparsity_ratio_plus_vs_full_equals_one():
    result = compute_attention_sparsity(T, window_size=WINDOW, n_global=1)
    assert abs(result["sparsity_ratio"] + result["vs_full_attention"] - 1.0) < 1e-6


def test_sparsity_full_window_low_sparsity():
    """A very large window covers the full causal (lower-triangular) mask.

    The attended fraction is T*(T+1)/2 / T^2 = (T+1)/(2*T) → 0.53125 for T=16.
    A small window (e.g. 1) should yield strictly lower vs_full_attention.
    """
    result_large = compute_attention_sparsity(T, window_size=T, n_global=1)
    result_small = compute_attention_sparsity(T, window_size=1, n_global=1)
    # Larger window → more attended pairs → higher vs_full_attention
    assert result_large["vs_full_attention"] > result_small["vs_full_attention"]
    # The full causal mask has T*(T+1)/2 pairs; with global token it can only grow
    expected_min = T * (T + 1) / 2 / (T * T)
    assert result_large["vs_full_attention"] >= expected_min - 1e-9


# ── Dilation ────────────────────────────────────────────────────────────────


def test_local_mask_dilation_skips_tokens():
    """With dilation=2 only every second token within the window is attended."""
    mask = create_local_attention_mask(T, window_size=WINDOW, dilation=2)
    i = 10
    # j = i-1 is distance 1, not divisible by dilation=2 → should be False
    assert not mask[i, i - 1].item(), "dilation=2 should skip distance-1 token"
    # j = i-2 is distance 2, divisible by 2 and within window*dilation=8 → True
    assert mask[i, i - 2].item(), "dilation=2 should attend distance-2 token"
    # j = i (self) → True
    assert mask[i, i].item(), "self-attention should always be True"
