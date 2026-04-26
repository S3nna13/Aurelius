import torch

from src.model.dynamic_sparse_attention import (
    DSAConfig,
    DynamicSparseAttention,
)

# Shared config
CFG = DSAConfig(d_model=64, n_heads=4, head_dim=16, top_k=8, indexer_dim=16)
B = 2
T = 16


def _make_input(seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(B, T, CFG.d_model)


# ---------------------------------------------------------------------------
# 1. DSAConfig dataclass instantiates
# ---------------------------------------------------------------------------
def test_dsaconfig_instantiates():
    cfg = DSAConfig(d_model=64, n_heads=4, head_dim=16, top_k=8, indexer_dim=16)
    assert cfg.d_model == 64
    assert cfg.n_heads == 4
    assert cfg.head_dim == 16
    assert cfg.top_k == 8
    assert cfg.indexer_dim == 16
    assert cfg.dropout == 0.0


# ---------------------------------------------------------------------------
# 2. DynamicSparseAttention instantiates
# ---------------------------------------------------------------------------
def test_dynamic_sparse_attention_instantiates():
    model = DynamicSparseAttention(CFG)
    assert isinstance(model, DynamicSparseAttention)


# ---------------------------------------------------------------------------
# 3. Forward returns shape (B, T, d_model)
# ---------------------------------------------------------------------------
def test_forward_output_shape():
    model = DynamicSparseAttention(CFG)
    x = _make_input()
    out = model(x)
    assert out.shape == (B, T, CFG.d_model), f"Expected ({B}, {T}, {CFG.d_model}), got {out.shape}"


# ---------------------------------------------------------------------------
# 4. Output is finite
# ---------------------------------------------------------------------------
def test_forward_output_finite():
    model = DynamicSparseAttention(CFG)
    x = _make_input()
    out = model(x)
    assert torch.isfinite(out).all(), "Output contains non-finite values"


# ---------------------------------------------------------------------------
# 5. _gather_kv returns two tensors of shape (B, H, T, top_k, head_dim)
# ---------------------------------------------------------------------------
def test_gather_kv_shape():
    model = DynamicSparseAttention(CFG)
    H = CFG.n_heads
    head_dim = CFG.head_dim
    top_k = CFG.top_k

    torch.manual_seed(0)
    k = torch.randn(B, H, T, head_dim)
    v = torch.randn(B, H, T, head_dim)
    indices = torch.randint(0, T, (B, H, T, top_k))

    k_sparse, v_sparse = model._gather_kv(k, v, indices)
    expected = (B, H, T, top_k, head_dim)
    assert k_sparse.shape == expected, f"k_sparse shape: {k_sparse.shape}"
    assert v_sparse.shape == expected, f"v_sparse shape: {v_sparse.shape}"


# ---------------------------------------------------------------------------
# 6. LearnedIndexer returns indices in valid range [0, T)
# ---------------------------------------------------------------------------
def test_learned_indexer_index_range():
    model = DynamicSparseAttention(CFG)
    H = CFG.n_heads
    head_dim = CFG.head_dim
    top_k = CFG.top_k

    torch.manual_seed(0)
    q = torch.randn(B, H, T, head_dim)
    k = torch.randn(B, H, T, head_dim)

    indices = model.indexer(q, k, top_k)
    assert indices.min().item() >= 0, "Index below 0"
    assert indices.max().item() < T, f"Index >= T={T}"


# ---------------------------------------------------------------------------
# 7. Indices shape is (B, H, T, top_k)
# ---------------------------------------------------------------------------
def test_learned_indexer_indices_shape():
    model = DynamicSparseAttention(CFG)
    H = CFG.n_heads
    head_dim = CFG.head_dim
    top_k = CFG.top_k

    torch.manual_seed(0)
    q = torch.randn(B, H, T, head_dim)
    k = torch.randn(B, H, T, head_dim)

    indices = model.indexer(q, k, top_k)
    assert indices.shape == (B, H, T, top_k), (
        f"Expected ({B}, {H}, {T}, {top_k}), got {indices.shape}"
    )


# ---------------------------------------------------------------------------
# 8. Gradient flows to Q projection
# ---------------------------------------------------------------------------
def test_gradient_flows_to_q_proj():
    model = DynamicSparseAttention(CFG)
    x = _make_input()
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert model.q_proj.weight.grad is not None, "No gradient on q_proj.weight"
    assert model.q_proj.weight.grad.shape == model.q_proj.weight.shape


# ---------------------------------------------------------------------------
# 9. Works with T=1 (single token)
# ---------------------------------------------------------------------------
def test_single_token():
    model = DynamicSparseAttention(CFG)
    torch.manual_seed(0)
    x = torch.randn(B, 1, CFG.d_model)
    out = model(x)
    assert out.shape == (B, 1, CFG.d_model), f"Expected ({B}, 1, {CFG.d_model}), got {out.shape}"
    assert torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# 10. warmup_mode(True) freezes base params, keeps indexer trainable
# ---------------------------------------------------------------------------
def test_warmup_mode_enabled_freezes_base():
    model = DynamicSparseAttention(CFG)
    model.warmup_mode(True)

    # Indexer parameters must still require grad
    for name, param in model.indexer.named_parameters():
        assert param.requires_grad, f"Indexer param {name} should require grad"

    # Base parameters (q/k/v/o proj) must be frozen
    for name, param in model.q_proj.named_parameters():
        assert not param.requires_grad, f"q_proj.{name} should be frozen"
    for name, param in model.k_proj.named_parameters():
        assert not param.requires_grad, f"k_proj.{name} should be frozen"
    for name, param in model.v_proj.named_parameters():
        assert not param.requires_grad, f"v_proj.{name} should be frozen"
    for name, param in model.o_proj.named_parameters():
        assert not param.requires_grad, f"o_proj.{name} should be frozen"


# ---------------------------------------------------------------------------
# 11. warmup_mode(False) unfreezes all params
# ---------------------------------------------------------------------------
def test_warmup_mode_disabled_unfreezes_all():
    model = DynamicSparseAttention(CFG)
    model.warmup_mode(True)
    model.warmup_mode(False)

    for name, param in model.named_parameters():
        assert param.requires_grad, f"Param {name} should require grad after warmup_mode(False)"


# ---------------------------------------------------------------------------
# 12. Output changes with different inputs (not constant)
# ---------------------------------------------------------------------------
def test_output_varies_with_input():
    model = DynamicSparseAttention(CFG)
    x1 = _make_input(seed=0)
    x2 = _make_input(seed=42)
    out1 = model(x1)
    out2 = model(x2)
    assert not torch.allclose(out1, out2), "Output should differ for different inputs"
