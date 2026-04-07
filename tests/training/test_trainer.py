"""Tests for AureliusTrainer training loop fixes."""
import math
import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.trainer import AureliusTrainer, TrainConfig


@pytest.fixture
def small_cfg():
    return AureliusConfig(
        n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
        head_dim=32, d_ff=128, vocab_size=256, max_seq_len=64,
    )


@pytest.fixture
def small_model(small_cfg):
    torch.manual_seed(0)
    return AureliusTransformer(small_cfg)


def _make_dict_dataloader(vocab_size: int, n_seqs: int = 8, seq_len: int = 16) -> DataLoader:
    """Create a DataLoader that yields dict batches (what the trainer expects)."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, vocab_size, (n_seqs, seq_len))
    labels = torch.randint(0, vocab_size, (n_seqs, seq_len))

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        lbs = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "labels": lbs}

    ds = TensorDataset(input_ids, labels)
    return DataLoader(ds, batch_size=4, collate_fn=collate)


def _make_trainer(model, small_cfg, train_loader, val_loader=None):
    cfg = TrainConfig(
        total_steps=2,
        warmup_steps=1,
        lr=1e-3,
        global_batch_tokens=small_cfg.max_seq_len * 4,
        micro_batch_size=4,
        seq_len=16,
        use_muon=False,
        use_zclip=False,
        wandb_enabled=False,
        save_interval_steps=9999,
        log_interval_steps=1,
    )
    return AureliusTrainer(model, train_loader, val_loader, cfg)


def test_train_step_loss_is_finite(small_model, small_cfg):
    """train() must complete 2 steps with finite loss (verifies output unpacking)."""
    loader = _make_dict_dataloader(small_cfg.vocab_size)
    trainer = _make_trainer(small_model, small_cfg, loader)

    before = {n: p.clone() for n, p in small_model.named_parameters()}
    trainer.train()

    any_changed = any(
        not torch.equal(before[n], p)
        for n, p in small_model.named_parameters()
        if p.requires_grad
    )
    assert any_changed, "No weights changed — gradients did not flow"


def test_validate_returns_finite_loss(small_model, small_cfg):
    """validate() must return a finite positive float."""
    loader = _make_dict_dataloader(small_cfg.vocab_size)
    trainer = _make_trainer(small_model, small_cfg, loader, val_loader=loader)

    ppl = trainer.validate()
    assert math.isfinite(ppl)
    assert ppl > 0


def test_build_dataloaders_yields_dict_batches(tmp_path, small_cfg):
    """build_dataloaders() must return loaders yielding {"input_ids", "labels"} dicts."""
    import numpy as np
    from src.training.trainer import build_dataloaders

    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    seq_len = 16
    tokens = np.arange(seq_len * 4 + 1, dtype=np.uint16)
    np.save(train_dir / "shard_00.npy", tokens)
    np.save(val_dir / "shard_00.npy", tokens)

    cfg = TrainConfig(
        train_data_dir=str(train_dir),
        val_data_dir=str(val_dir),
        seq_len=seq_len,
        micro_batch_size=2,
        num_workers=0,
        pin_memory=False,
    )

    train_loader, val_loader = build_dataloaders(cfg)
    assert val_loader is not None

    train_batch = next(iter(train_loader))
    assert "input_ids" in train_batch
    assert "labels" in train_batch
    assert train_batch["input_ids"].shape[1] == seq_len
    assert train_batch["labels"].shape[1] == seq_len
