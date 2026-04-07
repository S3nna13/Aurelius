"""Tests for checkpoint save/load utilities."""
import json
import math
import torch
import pytest
from src.training.checkpoint import (
    save_checkpoint, load_checkpoint, load_best_checkpoint,
    list_checkpoints, CheckpointMeta,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


@pytest.fixture
def small_model():
    torch.manual_seed(0)
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    return AureliusTransformer(cfg)


def test_save_creates_files(tmp_path, small_model):
    """save_checkpoint must create model.pt and meta.json."""
    ckpt_dir = save_checkpoint(small_model, None, step=10, epoch=1, train_loss=2.5, output_dir=tmp_path)
    assert (ckpt_dir / "model.pt").exists()
    assert (ckpt_dir / "meta.json").exists()


def test_save_meta_content(tmp_path, small_model):
    """meta.json must contain correct step, loss values."""
    save_checkpoint(small_model, None, step=42, epoch=2, train_loss=1.23, output_dir=tmp_path, val_loss=1.10)
    ckpts = list(tmp_path.glob("checkpoint-*"))
    with open(ckpts[0] / "meta.json") as f:
        meta = json.load(f)
    assert meta["step"] == 42
    assert meta["epoch"] == 2
    assert abs(meta["train_loss"] - 1.23) < 1e-6
    assert abs(meta["val_loss"] - 1.10) < 1e-6


def test_load_restores_weights(tmp_path, small_model):
    """load_checkpoint must restore model weights exactly."""
    # Perturb weights to get a known state
    for p in small_model.parameters():
        p.data.fill_(0.5)

    ckpt_dir = save_checkpoint(small_model, None, step=1, epoch=0, train_loss=1.0, output_dir=tmp_path)

    # Create a fresh model with different weights
    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    fresh = AureliusTransformer(cfg)
    for p in fresh.parameters():
        p.data.fill_(0.0)

    meta = load_checkpoint(fresh, ckpt_dir)

    assert meta.step == 1
    for p in fresh.parameters():
        assert p.data.abs().max().item() == pytest.approx(0.5)


def test_best_checkpoint_tracking(tmp_path, small_model):
    """save_checkpoint must update best/ when val_loss improves."""
    save_checkpoint(small_model, None, step=10, epoch=0, train_loss=2.0, output_dir=tmp_path, val_loss=2.0)
    save_checkpoint(small_model, None, step=20, epoch=0, train_loss=1.5, output_dir=tmp_path, val_loss=1.5)
    save_checkpoint(small_model, None, step=30, epoch=0, train_loss=1.8, output_dir=tmp_path, val_loss=1.8)

    best_path = tmp_path / "best"
    assert best_path.exists()
    with open(best_path / "meta.json") as f:
        best_meta = json.load(f)
    assert abs(best_meta["val_loss"] - 1.5) < 1e-6  # step 20 was best


def test_keep_last_n(tmp_path, small_model):
    """save_checkpoint must delete old checkpoints beyond keep_last_n."""
    for step in [10, 20, 30, 40, 50]:
        save_checkpoint(small_model, None, step=step, epoch=0, train_loss=1.0, output_dir=tmp_path, keep_last_n=3)

    ckpts = sorted(tmp_path.glob("checkpoint-*"))
    assert len(ckpts) == 3
    assert ckpts[-1].name == "checkpoint-0000050"


def test_list_checkpoints(tmp_path, small_model):
    """list_checkpoints must return sorted (path, meta) tuples."""
    for step in [100, 200, 50]:
        save_checkpoint(small_model, None, step=step, epoch=0, train_loss=float(step), output_dir=tmp_path, keep_last_n=0)

    ckpts = list_checkpoints(tmp_path)
    steps = [meta.step for _, meta in ckpts]
    assert steps == sorted(steps)


def test_load_best_checkpoint(tmp_path, small_model):
    """load_best_checkpoint must load from the best/ directory."""
    save_checkpoint(small_model, None, step=5, epoch=0, train_loss=3.0, output_dir=tmp_path, val_loss=3.0)

    cfg = AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=32,
    )
    fresh = AureliusTransformer(cfg)
    meta = load_best_checkpoint(fresh, tmp_path)
    assert meta.step == 5
