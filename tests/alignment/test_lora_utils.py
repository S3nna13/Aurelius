"""Tests for LoRA utilities: rank estimation, adapter merging, and LoRA layers."""
import torch
import torch.nn as nn
import pytest

from src.alignment.lora_utils import (
    estimate_intrinsic_rank,
    LoRAAdapterInfo,
    merge_lora_adapters,
    LoRALinear,
    MultiLoRALinear,
    analyze_lora_rank_distribution,
)

IN, OUT, RANK = 16, 32, 4


# ---------------------------------------------------------------------------
# 1. estimate_intrinsic_rank
# ---------------------------------------------------------------------------

def test_estimate_intrinsic_rank_full_rank():
    """A random full-rank matrix should require rank close to min(m, n)."""
    torch.manual_seed(0)
    W = torch.randn(OUT, IN)  # IN < OUT, so max rank = IN = 16
    r = estimate_intrinsic_rank(W, threshold=0.99)
    # A random matrix has spectrum spread across all singular values; rank should
    # be most of min(OUT, IN) to capture 99 % variance.
    assert r >= IN // 2, f"Expected high rank for random matrix, got {r}"
    assert r <= IN  # cannot exceed min dimension


def test_estimate_intrinsic_rank_low_rank():
    """A matrix with known rank r should return exactly r at 100 % threshold."""
    torch.manual_seed(42)
    true_rank = 3
    # Construct a rank-3 matrix: W = A @ B where A is (OUT, 3) and B is (3, IN)
    A = torch.randn(OUT, true_rank)
    B = torch.randn(true_rank, IN)
    W = A @ B
    r = estimate_intrinsic_rank(W, threshold=0.9999)
    assert r == true_rank, f"Expected rank {true_rank}, got {r}"


# ---------------------------------------------------------------------------
# 3. LoRAAdapterInfo
# ---------------------------------------------------------------------------

def test_lora_adapter_info_scale():
    alpha, rank = 16.0, 4
    A = torch.zeros(rank, IN)
    B = torch.zeros(OUT, rank)
    info = LoRAAdapterInfo(name="test", rank=rank, alpha=alpha, A=A, B=B)
    assert info.scale == alpha / rank


def test_lora_adapter_info_delta_weight_shape():
    torch.manual_seed(0)
    A = torch.randn(RANK, IN)
    B = torch.randn(OUT, RANK)
    info = LoRAAdapterInfo(name="test", rank=RANK, alpha=16.0, A=A, B=B)
    dw = info.delta_weight
    assert dw.shape == (OUT, IN)


# ---------------------------------------------------------------------------
# 5–6. merge_lora_adapters
# ---------------------------------------------------------------------------

def _make_adapter(name: str, seed: int) -> LoRAAdapterInfo:
    torch.manual_seed(seed)
    return LoRAAdapterInfo(
        name=name,
        rank=RANK,
        alpha=16.0,
        A=torch.randn(RANK, IN),
        B=torch.randn(OUT, RANK),
    )


def test_merge_lora_adapters_uniform():
    """Uniform merge should equal the mean of individual delta_weights."""
    a1 = _make_adapter("a1", 1)
    a2 = _make_adapter("a2", 2)
    merged = merge_lora_adapters([a1, a2])
    expected = (a1.delta_weight + a2.delta_weight) / 2
    assert torch.allclose(merged, expected, atol=1e-6), "Uniform merge mismatch"


def test_merge_lora_adapters_weighted():
    """Weighted merge: result should equal w0*dw0 + w1*dw1."""
    a1 = _make_adapter("a1", 10)
    a2 = _make_adapter("a2", 20)
    w = [0.3, 0.7]
    merged = merge_lora_adapters([a1, a2], weights=w)
    expected = w[0] * a1.delta_weight + w[1] * a2.delta_weight
    assert torch.allclose(merged, expected, atol=1e-6), "Weighted merge mismatch"


# ---------------------------------------------------------------------------
# 7–10. LoRALinear
# ---------------------------------------------------------------------------

def test_lora_linear_forward_shape():
    layer = LoRALinear(IN, OUT, rank=RANK)
    x = torch.randn(2, 8, IN)
    out = layer(x)
    assert out.shape == (2, 8, OUT)


def test_lora_linear_merge_weights_shape():
    layer = LoRALinear(IN, OUT, rank=RANK)
    merged = layer.merge_weights()
    assert merged.shape == (OUT, IN)


def test_lora_linear_base_frozen():
    layer = LoRALinear(IN, OUT, rank=RANK)
    assert not layer.weight.requires_grad, "Base weight should be frozen"


def test_lora_linear_adapters_trainable():
    layer = LoRALinear(IN, OUT, rank=RANK)
    assert layer.lora_A.requires_grad, "lora_A should require grad"
    assert layer.lora_B.requires_grad, "lora_B should require grad"


# ---------------------------------------------------------------------------
# 11. MultiLoRALinear
# ---------------------------------------------------------------------------

def test_multi_lora_linear_switch():
    """Switching the active adapter should change the forward output."""
    torch.manual_seed(0)
    layer = MultiLoRALinear(IN, OUT, rank=RANK, n_adapters=2)

    # Give the two adapters different non-zero lora_A values so outputs differ
    with torch.no_grad():
        layer.adapters[0].lora_A.fill_(0.1)   # type: ignore[union-attr]
        layer.adapters[0].lora_B.fill_(0.1)   # type: ignore[union-attr]
        layer.adapters[1].lora_A.fill_(0.5)   # type: ignore[union-attr]
        layer.adapters[1].lora_B.fill_(0.5)   # type: ignore[union-attr]

    x = torch.randn(1, 4, IN)

    layer.switch_adapter(0)
    out0 = layer(x)

    layer.switch_adapter(1)
    out1 = layer(x)

    assert not torch.allclose(out0, out1), "Switching adapter should change output"


# ---------------------------------------------------------------------------
# 12. analyze_lora_rank_distribution
# ---------------------------------------------------------------------------

def test_analyze_lora_rank_distribution_keys():
    """Should return a dict keyed by all nn.Linear module names."""
    model = nn.Sequential(
        nn.Linear(IN, OUT, bias=False),
        nn.ReLU(),
        nn.Linear(OUT, IN, bias=False),
    )
    result = analyze_lora_rank_distribution(model, threshold=0.99)

    # nn.Sequential named_modules gives "0", "1", "2" for sub-modules
    linear_names = {name for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    assert set(result.keys()) == linear_names, (
        f"Keys mismatch: got {set(result.keys())}, expected {linear_names}"
    )
    # All values should be positive integers
    for name, rank in result.items():
        assert isinstance(rank, int) and rank > 0, f"{name}: invalid rank {rank}"
