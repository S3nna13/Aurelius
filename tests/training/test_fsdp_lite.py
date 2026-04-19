"""Unit tests for src.training.fsdp_lite."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.training.fsdp_lite import FSDPLite, ShardSpec, gather_tensor, shard_tensor


def _tiny_linear(seed: int = 0) -> nn.Linear:
    torch.manual_seed(seed)
    return nn.Linear(16, 8)


def test_shard_tensor_splits_evenly():
    t = torch.arange(12, dtype=torch.float32)
    spec0 = ShardSpec(world_size=4, rank=0)
    spec1 = ShardSpec(world_size=4, rank=1)
    spec2 = ShardSpec(world_size=4, rank=2)
    spec3 = ShardSpec(world_size=4, rank=3)
    assert torch.equal(shard_tensor(t, spec0), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(shard_tensor(t, spec1), torch.tensor([3.0, 4.0, 5.0]))
    assert torch.equal(shard_tensor(t, spec2), torch.tensor([6.0, 7.0, 8.0]))
    assert torch.equal(shard_tensor(t, spec3), torch.tensor([9.0, 10.0, 11.0]))


def test_shard_tensor_indivisible_pads_with_zero():
    # numel=10, world_size=4 -> per-shard length 3, pad 2 zeros at tail of rank 3.
    t = torch.arange(10, dtype=torch.float32)
    spec3 = ShardSpec(world_size=4, rank=3)
    shard3 = shard_tensor(t, spec3)
    assert shard3.shape == (3,)
    # Last rank receives value 9 followed by two zero pads.
    assert torch.equal(shard3, torch.tensor([9.0, 0.0, 0.0]))


def test_gather_tensor_reconstructs_original():
    original = torch.randn(12)
    spec = lambda r: ShardSpec(world_size=4, rank=r)
    shards = [shard_tensor(original, spec(r)) for r in range(4)]
    gathered = gather_tensor(shards, original_numel=12, original_shape=(12,))
    assert torch.equal(gathered, original)


def test_gather_tensor_reconstructs_after_padding():
    original = torch.arange(10, dtype=torch.float32)
    spec = lambda r: ShardSpec(world_size=4, rank=r)
    shards = [shard_tensor(original, spec(r)) for r in range(4)]
    gathered = gather_tensor(shards, original_numel=10, original_shape=(10,))
    assert torch.equal(gathered, original)


def test_world_size_one_shard_tensor_is_noop():
    t = torch.randn(5, 7)
    spec = ShardSpec(world_size=1, rank=0)
    out = shard_tensor(t, spec)
    assert out is t  # documented: no-op fast path returns original object


def test_fsdp_world_size_one_matches_unwrapped():
    torch.manual_seed(0)
    lin = _tiny_linear(seed=42)
    wrapped = FSDPLite(_tiny_linear(seed=42), ShardSpec(world_size=1, rank=0))
    x = torch.randn(3, 16)
    assert torch.allclose(wrapped(x), lin(x), atol=1e-6)


def test_fsdp_world_size_two_matches_unwrapped_via_gather():
    torch.manual_seed(0)
    base = _tiny_linear(seed=7)
    ref_out = base(torch.randn(4, 16, generator=torch.Generator().manual_seed(1)))

    torch.manual_seed(0)
    wrapped = FSDPLite(_tiny_linear(seed=7), ShardSpec(world_size=2, rank=0))
    wrapped_out = wrapped(torch.randn(4, 16, generator=torch.Generator().manual_seed(1)))
    assert torch.allclose(wrapped_out, ref_out, atol=1e-6)


def test_parameters_returns_only_local_shards():
    lin = _tiny_linear(seed=1)
    wrapped = FSDPLite(lin, ShardSpec(world_size=4, rank=2))
    params = list(wrapped.parameters())
    # Linear has 2 params (weight, bias) -> 2 local shards for rank 2.
    assert len(params) == 2
    # Weight is 16*8=128 elements, 4 shards of 32 each.
    weight_shard = next(p for p in params if p.numel() == 32)
    assert weight_shard.shape == (32,)
    # Bias is 8 elements, 4 shards of 2 each.
    bias_shard = next(p for p in params if p.numel() == 2)
    assert bias_shard.shape == (2,)


def test_gradients_flow_to_local_shards():
    lin = _tiny_linear(seed=2)
    wrapped = FSDPLite(lin, ShardSpec(world_size=2, rank=0))
    x = torch.randn(5, 16)
    out = wrapped(x)
    loss = out.pow(2).sum()
    loss.backward()
    for p in wrapped.parameters():
        assert p.grad is not None
        assert p.grad.shape == p.shape
        assert torch.isfinite(p.grad).all()


def test_gather_full_param_returns_full_tensor():
    lin = _tiny_linear(seed=3)
    wrapped = FSDPLite(lin, ShardSpec(world_size=4, rank=1))
    full_w = wrapped.gather_full_param("weight")
    full_b = wrapped.gather_full_param("bias")
    assert full_w.shape == (8, 16)
    assert full_b.shape == (8,)
    assert torch.allclose(full_w, lin.weight)
    assert torch.allclose(full_b, lin.bias)


def test_invalid_shard_spec_raises():
    with pytest.raises(ValueError):
        ShardSpec(world_size=2, rank=2)
    with pytest.raises(ValueError):
        ShardSpec(world_size=2, rank=-1)
    with pytest.raises(ValueError):
        ShardSpec(world_size=0, rank=0)


def test_determinism_with_manual_seed():
    def build_and_run() -> torch.Tensor:
        torch.manual_seed(123)
        lin = nn.Linear(16, 8)
        wrapped = FSDPLite(lin, ShardSpec(world_size=2, rank=1))
        torch.manual_seed(456)
        x = torch.randn(3, 16)
        return wrapped(x)

    a = build_and_run()
    b = build_and_run()
    assert torch.equal(a, b)


def test_sequential_of_three_linears_wrapped():
    torch.manual_seed(9)
    mlp = nn.Sequential(nn.Linear(16, 12), nn.ReLU(), nn.Linear(12, 12), nn.ReLU(), nn.Linear(12, 8))
    ref_input = torch.randn(2, 16, generator=torch.Generator().manual_seed(11))
    ref_out = mlp(ref_input)

    wrapped = FSDPLite(mlp, ShardSpec(world_size=2, rank=0))
    out = wrapped(ref_input)
    assert out.shape == (2, 8)
    assert torch.allclose(out, ref_out, atol=1e-6)

    # 3 linears * 2 params each = 6 local shard params
    assert len(list(wrapped.parameters())) == 6

    loss = out.sum()
    loss.backward()
    for p in wrapped.parameters():
        assert p.grad is not None


def test_gather_tensor_empty_raises():
    with pytest.raises(ValueError):
        gather_tensor([])


def test_shard_tensor_flatten_false_dim0_split():
    t = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    spec = ShardSpec(world_size=2, rank=1, flatten=False)
    shard = shard_tensor(t, spec)
    assert shard.shape == (2, 4)
    assert torch.equal(shard, t[2:4])
