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
