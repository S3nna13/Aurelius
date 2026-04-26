import torch

from src.model.block_sparse_attention import (
    BigBirdBlock,
    BlockSparseAttention,
    BlockSparseConfig,
    compute_block_attention,
    compute_sparsity,
    create_block_sparse_mask,
    estimate_flop_reduction,
)

# Shared small config for all tests
CFG = BlockSparseConfig(
    d_model=64,
    n_heads=2,
    head_dim=32,
    block_size=8,
    n_global_tokens=2,
    n_random_blocks=1,
    window_size=1,
    dropout=0.0,
)
T = 32  # seq_len; divisible by block_size=8
B = 2  # batch size


def _make_input():
    torch.manual_seed(0)
    return torch.randn(B, T, CFG.d_model)


# ---------------------------------------------------------------------------
# 1. Config defaults
# ---------------------------------------------------------------------------
def test_block_sparse_config_defaults():
    cfg = BlockSparseConfig()
    assert cfg.block_size == 64
    assert cfg.n_global_tokens == 4
    assert cfg.n_random_blocks == 2
    assert cfg.window_size == 1
    assert cfg.d_model == 512
    assert cfg.n_heads == 8
    assert cfg.head_dim == 64
    assert cfg.dropout == 0.0


# ---------------------------------------------------------------------------
# 2. Mask shape
# ---------------------------------------------------------------------------
def test_create_mask_shape():
    mask = create_block_sparse_mask(T, CFG)
    assert mask.shape == (T, T), f"Expected ({T}, {T}), got {mask.shape}"
    assert mask.dtype == torch.bool


# ---------------------------------------------------------------------------
# 3. Global tokens
# ---------------------------------------------------------------------------
def test_create_mask_global_tokens():
    mask = create_block_sparse_mask(T, CFG)
    n_global = CFG.n_global_tokens
    # Global rows attend everywhere
    for i in range(n_global):
        assert mask[i].all(), f"Global row {i} should be all True"
    # All tokens attend to global columns
    for j in range(n_global):
        assert mask[:, j].all(), f"Global col {j} should be all True"


# ---------------------------------------------------------------------------
# 4. Diagonal window
# ---------------------------------------------------------------------------
def test_create_mask_diagonal_window():
    mask = create_block_sparse_mask(T, CFG)
    block_size = CFG.block_size
    n_blocks = T // block_size
    window = CFG.window_size

    # For every query block, check that the blocks within ±window are attended
    for bq in range(n_blocks):
        qs = bq * block_size
        qe = qs + block_size
        for offset in range(-window, window + 1):
            bk = bq + offset
            if 0 <= bk < n_blocks:
                ks = bk * block_size
                ke = ks + block_size
                assert mask[qs:qe, ks:ke].all(), (
                    f"Block ({bq},{bk}) should be fully attended (window)"
                )


# ---------------------------------------------------------------------------
# 5. Sparsity < dense
# ---------------------------------------------------------------------------
def test_mask_sparsity_less_than_dense():
    mask = create_block_sparse_mask(T, CFG)
    sparsity = compute_sparsity(mask)
    assert 0.0 <= sparsity < 1.0, f"Sparsity should be in [0,1), got {sparsity}"
    assert sparsity > 0.0, "There should be some masked-out positions (sparsity > 0)"


# ---------------------------------------------------------------------------
# 6. compute_block_attention shape
# ---------------------------------------------------------------------------
def test_compute_block_attention_shape():
    torch.manual_seed(0)
    H, D = CFG.n_heads, CFG.head_dim
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    mask = create_block_sparse_mask(T, CFG)
    out = compute_block_attention(q, k, v, mask)
    assert out.shape == (B, H, T, D), f"Expected ({B},{H},{T},{D}), got {out.shape}"


# ---------------------------------------------------------------------------
# 7. BlockSparseAttention output shape
# ---------------------------------------------------------------------------
def test_block_sparse_attention_output_shape():
    model = BlockSparseAttention(CFG)
    x = _make_input()
    out = model(x)
    assert out.shape == (B, T, CFG.d_model), f"Expected ({B},{T},{CFG.d_model}), got {out.shape}"


# ---------------------------------------------------------------------------
# 8. BlockSparseAttention gradient flow
# ---------------------------------------------------------------------------
def test_block_sparse_attention_gradient_flow():
    model = BlockSparseAttention(CFG)
    x = _make_input().requires_grad_(True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow back to input"
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 9. BigBirdBlock output shape
# ---------------------------------------------------------------------------
def test_bigbird_block_output_shape():
    d_ff = CFG.d_model * 4
    block = BigBirdBlock(CFG, d_ff=d_ff)
    x = _make_input()
    out = block(x)
    assert out.shape == (B, T, CFG.d_model), f"Expected ({B},{T},{CFG.d_model}), got {out.shape}"


# ---------------------------------------------------------------------------
# 10. BigBirdBlock gradient flow
# ---------------------------------------------------------------------------
def test_bigbird_block_gradient_flow():
    d_ff = CFG.d_model * 4
    block = BigBirdBlock(CFG, d_ff=d_ff)
    x = _make_input().requires_grad_(True)
    out = block(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient did not flow back to input"
    assert x.grad.shape == x.shape


# ---------------------------------------------------------------------------
# 11. compute_sparsity range
# ---------------------------------------------------------------------------
def test_compute_sparsity_range():
    mask = create_block_sparse_mask(T, CFG)
    sparsity = compute_sparsity(mask)
    assert 0.0 <= sparsity <= 1.0, f"Sparsity out of [0,1]: {sparsity}"

    # All-True mask → sparsity = 0
    all_true = torch.ones(T, T, dtype=torch.bool)
    assert compute_sparsity(all_true) == 0.0

    # All-False mask → sparsity = 1
    all_false = torch.zeros(T, T, dtype=torch.bool)
    assert compute_sparsity(all_false) == 1.0


# ---------------------------------------------------------------------------
# 12. FLOP reduction > 1.0
# ---------------------------------------------------------------------------
def test_flop_reduction_positive():
    ratio = estimate_flop_reduction(T, CFG)
    assert ratio > 1.0, f"Block-sparse should attend fewer pairs than dense; got ratio={ratio}"
