from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.training.checkpoint_sharder import CheckpointSharder, ShardConfig, ShardManifest


def _tiny_state() -> dict[str, torch.Tensor]:
    return {
        "layer1.weight": torch.randn(4, 4),
        "layer1.bias": torch.randn(4),
        "layer2.weight": torch.randn(4, 4),
        "layer2.bias": torch.randn(4),
        "head.weight": torch.randn(4, 4),
    }


# ---------------------------------------------------------------------------
# ShardConfig
# ---------------------------------------------------------------------------


def test_shard_config_defaults():
    cfg = ShardConfig()
    assert cfg.max_shard_size_gb == pytest.approx(5.0)
    assert cfg.format == "safetensors"
    assert cfg.prefix == "model"


def test_shard_config_custom():
    cfg = ShardConfig(max_shard_size_gb=1.0, prefix="ckpt")
    assert cfg.max_shard_size_gb == pytest.approx(1.0)
    assert cfg.prefix == "ckpt"


# ---------------------------------------------------------------------------
# ShardManifest
# ---------------------------------------------------------------------------


def test_shard_manifest_fields():
    m = ShardManifest(
        total_params=100,
        total_size_bytes=400,
        n_shards=2,
        shard_files=["a.safetensors", "b.safetensors"],
        key_to_shard={"x": 0, "y": 1},
        format="safetensors",
    )
    assert m.n_shards == 2
    assert len(m.shard_files) == 2
    assert m.key_to_shard["x"] == 0


# ---------------------------------------------------------------------------
# _estimate_size
# ---------------------------------------------------------------------------


def test_estimate_size_float32():
    sharder = CheckpointSharder()
    t = torch.zeros(4, 4, dtype=torch.float32)
    assert sharder._estimate_size(t) == 16 * 4


def test_estimate_size_float16():
    sharder = CheckpointSharder()
    t = torch.zeros(4, 4, dtype=torch.float16)
    assert sharder._estimate_size(t) == 16 * 2


# ---------------------------------------------------------------------------
# split_plan
# ---------------------------------------------------------------------------


def test_split_plan_single_shard_large_limit():
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    plan = sharder.split_plan(_tiny_state())
    assert len(plan) == 1
    assert len(plan[0]) == 5


def test_split_plan_tiny_limit_many_shards():
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=1e-9))
    state = _tiny_state()
    plan = sharder.split_plan(state)
    assert len(plan) == len(state)


def test_split_plan_deterministic():
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    state = _tiny_state()
    assert sharder.split_plan(state) == sharder.split_plan(state)


def test_split_plan_all_keys_present():
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    state = _tiny_state()
    plan = sharder.split_plan(state)
    all_keys = [k for group in plan for k in group]
    assert sorted(all_keys) == sorted(state.keys())


def test_split_plan_no_duplicates():
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    plan = sharder.split_plan(_tiny_state())
    all_keys = [k for group in plan for k in group]
    assert len(all_keys) == len(set(all_keys))


# ---------------------------------------------------------------------------
# shard (file I/O)
# ---------------------------------------------------------------------------


def test_shard_creates_manifest():
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    with tempfile.TemporaryDirectory() as tmp:
        sharder.shard(_tiny_state(), tmp)
        assert (Path(tmp) / "manifest.json").exists()


def test_shard_manifest_n_shards():
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    with tempfile.TemporaryDirectory() as tmp:
        manifest = sharder.shard(_tiny_state(), tmp)
        assert manifest.n_shards == 1


def test_shard_files_exist():
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    with tempfile.TemporaryDirectory() as tmp:
        manifest = sharder.shard(_tiny_state(), tmp)
        for fname in manifest.shard_files:
            assert (Path(tmp) / fname).exists()


def test_shard_manifest_total_params():
    state = _tiny_state()
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    with tempfile.TemporaryDirectory() as tmp:
        manifest = sharder.shard(state, tmp)
        expected = sum(t.nelement() for t in state.values())
        assert manifest.total_params == expected


def test_shard_key_to_shard_coverage():
    state = _tiny_state()
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    with tempfile.TemporaryDirectory() as tmp:
        manifest = sharder.shard(state, tmp)
        assert set(manifest.key_to_shard.keys()) == set(state.keys())


# ---------------------------------------------------------------------------
# manifest_from_dir
# ---------------------------------------------------------------------------


def test_manifest_from_dir_roundtrip():
    state = _tiny_state()
    sharder = CheckpointSharder(ShardConfig(max_shard_size_gb=100.0))
    with tempfile.TemporaryDirectory() as tmp:
        written = sharder.shard(state, tmp)
        loaded = sharder.manifest_from_dir(tmp)
        assert loaded.n_shards == written.n_shards
        assert loaded.total_params == written.total_params
        assert loaded.shard_files == written.shard_files
        assert loaded.key_to_shard == written.key_to_shard


def test_manifest_from_dir_format():
    sharder = CheckpointSharder()
    with tempfile.TemporaryDirectory() as tmp:
        sharder.shard(_tiny_state(), tmp)
        m = sharder.manifest_from_dir(tmp)
        assert m.format == "safetensors"
