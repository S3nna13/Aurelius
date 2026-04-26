from __future__ import annotations

import tempfile

import torch
import torch.nn as nn

from src.training.checkpoint_sharder import CheckpointSharder, ShardConfig
from src.training.shard_loader import ShardLoader


def _tiny_state() -> dict[str, torch.Tensor]:
    return {
        "layer1.weight": torch.randn(4, 4),
        "layer1.bias": torch.randn(4),
        "layer2.weight": torch.randn(4, 4),
        "layer2.bias": torch.randn(4),
        "head.weight": torch.randn(4, 4),
    }


def _simple_model() -> nn.Module:
    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(4, 4, bias=True)
            self.layer2 = nn.Linear(4, 4, bias=True)
            self.head = nn.Linear(4, 4, bias=False)

        def forward(self, x):
            return self.head(self.layer2(self.layer1(x)))

    return _Net()


def _write_checkpoint(state: dict, tmp: str, max_gb: float = 100.0) -> str:
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=max_gb))
    sharder.shard(state, tmp)
    return tmp


# ---------------------------------------------------------------------------
# load_manifest
# ---------------------------------------------------------------------------


def test_load_manifest_returns_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(_tiny_state(), tmp)
        loader = ShardLoader()
        m = loader.load_manifest(tmp)
        assert m.n_shards >= 1


def test_load_manifest_key_count():
    state = _tiny_state()
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(state, tmp)
        loader = ShardLoader()
        m = loader.load_manifest(tmp)
        assert set(m.key_to_shard.keys()) == set(state.keys())


# ---------------------------------------------------------------------------
# load_shard
# ---------------------------------------------------------------------------


def test_load_shard_returns_tensors():
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(_tiny_state(), tmp)
        loader = ShardLoader()
        m = loader.load_manifest(tmp)
        shard = loader.load_shard(tmp, 0, m)
        assert all(isinstance(v, torch.Tensor) for v in shard.values())


def test_load_shard_correct_shapes():
    state = _tiny_state()
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(state, tmp)
        loader = ShardLoader()
        m = loader.load_manifest(tmp)
        shard = loader.load_shard(tmp, 0, m)
        for k, t in shard.items():
            assert t.shape == state[k].shape, f"Shape mismatch for {k}"


# ---------------------------------------------------------------------------
# load_state_dict
# ---------------------------------------------------------------------------


def test_load_state_dict_all_keys():
    state = _tiny_state()
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(state, tmp)
        loader = ShardLoader()
        merged = loader.load_state_dict(tmp)
        assert set(merged.keys()) == set(state.keys())


def test_load_state_dict_shapes():
    state = _tiny_state()
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(state, tmp)
        loader = ShardLoader()
        merged = loader.load_state_dict(tmp)
        for k, t in merged.items():
            assert t.shape == state[k].shape


def test_load_state_dict_multi_shard():
    state = _tiny_state()
    with tempfile.TemporaryDirectory() as tmp:
        _write_checkpoint(state, tmp, max_gb=1e-9)
        loader = ShardLoader()
        merged = loader.load_state_dict(tmp)
        assert set(merged.keys()) == set(state.keys())


# ---------------------------------------------------------------------------
# load_into_model
# ---------------------------------------------------------------------------


def test_load_into_model_no_missing_keys():
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
        sharder.shard(model.state_dict(), tmp)
        loader = ShardLoader(sharder)
        missing, unexpected = loader.load_into_model(model, tmp, strict=True)
        assert missing == []
        assert unexpected == []


def test_load_into_model_returns_tuple():
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
        sharder.shard(model.state_dict(), tmp)
        loader = ShardLoader(sharder)
        result = loader.load_into_model(model, tmp)
        assert isinstance(result, tuple) and len(result) == 2


# ---------------------------------------------------------------------------
# partial_load
# ---------------------------------------------------------------------------


def test_partial_load_count():
    model = _simple_model()
    state = model.state_dict()
    with tempfile.TemporaryDirectory() as tmp:
        sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
        sharder.shard(state, tmp)
        loader = ShardLoader(sharder)
        count = loader.partial_load(model, tmp, key_pattern="layer1")
        layer1_keys = [k for k in state.keys() if "layer1" in k]
        assert count == len(layer1_keys)


def test_partial_load_no_match_returns_zero():
    model = _simple_model()
    with tempfile.TemporaryDirectory() as tmp:
        sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
        sharder.shard(model.state_dict(), tmp)
        loader = ShardLoader(sharder)
        count = loader.partial_load(model, tmp, key_pattern="__no_such_key__")
        assert count == 0


def test_partial_load_head_only():
    model = _simple_model()
    state = model.state_dict()
    with tempfile.TemporaryDirectory() as tmp:
        sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
        sharder.shard(state, tmp)
        loader = ShardLoader(sharder)
        count = loader.partial_load(model, tmp, key_pattern="head")
        expected = len([k for k in state.keys() if "head" in k])
        assert count == expected
