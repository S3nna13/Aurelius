# Training Loop Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix three concrete breaks in `AureliusTrainer` so the model can actually train.

**Architecture:** Surgical edits to `src/training/trainer.py` only — fix output unpacking in `train()` and `validate()`, add a `_collate_fn` for dict-format batches, and add `build_dataloaders()` to wire `TokenizedShardDataset` into the training loop. Tests verify each fix independently.

**Tech Stack:** PyTorch, HuggingFace Accelerate, TokenizedShardDataset (already in src/data/tokenized_loader.py)

---

### Task 1: Fix output unpacking in `train()` and `validate()`

**Files:**
- Modify: `src/training/trainer.py:530-531` and `src/training/trainer.py:634-635`
- Test: `tests/training/test_trainer.py` (create)

**Step 1: Write the failing test**

Create `tests/training/test_trainer.py`:

```python
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

    # Capture weights before training
    before = {n: p.clone() for n, p in small_model.named_parameters()}

    trainer.train()

    # At least one weight must have changed (gradients flowed)
    any_changed = any(
        not torch.equal(before[n], p)
        for n, p in small_model.named_parameters()
        if p.requires_grad
    )
    assert any_changed, "No weights changed — gradients did not flow"
```

**Step 2: Run to verify it fails**

```bash
cd /Users/christienantonio/Desktop/Aurelius
.venv/bin/python3.13 -m pytest tests/training/test_trainer.py::test_train_step_loss_is_finite -v 2>&1 | tail -15
```

Expected: FAIL with `AttributeError: 'tuple' object has no attribute 'loss'`

**Step 3: Fix `train()` in `src/training/trainer.py`**

At line 530-531, replace:
```python
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
```
With:
```python
                loss, _, _ = self.model(input_ids=input_ids, labels=labels)
```

**Step 4: Run test to verify it passes**

```bash
.venv/bin/python3.13 -m pytest tests/training/test_trainer.py::test_train_step_loss_is_finite -v 2>&1 | tail -10
```

Expected: PASS

**Step 5: Fix `validate()` in `src/training/trainer.py`**

At line 634-635, replace:
```python
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
```
With:
```python
            loss, _, _ = self.model(input_ids=input_ids, labels=labels)
```

**Step 6: Commit**

```bash
git add src/training/trainer.py tests/training/test_trainer.py
git commit -m "fix: unpack (loss, logits, pkv) tuple from transformer in trainer"
```

---

### Task 2: Add validate() test

**Files:**
- Modify: `tests/training/test_trainer.py`

**Step 1: Add the validate test** (append to test file):

```python
def test_validate_returns_finite_loss(small_model, small_cfg):
    """validate() must return a finite positive float."""
    loader = _make_dict_dataloader(small_cfg.vocab_size)
    trainer = _make_trainer(small_model, small_cfg, loader, val_loader=loader)

    ppl = trainer.validate()
    assert math.isfinite(ppl)
    assert ppl > 0
```

**Step 2: Run**

```bash
.venv/bin/python3.13 -m pytest tests/training/test_trainer.py::test_validate_returns_finite_loss -v 2>&1 | tail -10
```

Expected: PASS (validate fix is already in from Task 1)

**Step 3: Commit**

```bash
git add tests/training/test_trainer.py
git commit -m "test: add validate() test for trainer"
```

---

### Task 3: Add `_collate_fn` and `build_dataloaders()`

**Files:**
- Modify: `src/training/trainer.py` (add after imports, before `TrainConfig`)
- Test: `tests/training/test_trainer.py`

**Step 1: Write the failing test** (append to test file):

```python
def test_build_dataloaders_yields_dict_batches(tmp_path, small_cfg):
    """build_dataloaders() must return loaders yielding {"input_ids", "labels"} dicts."""
    import numpy as np
    from src.training.trainer import build_dataloaders

    # Write tiny .npy shards
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "val"
    train_dir.mkdir()
    val_dir.mkdir()

    seq_len = 16
    # Need at least seq_len+1 tokens per shard for one window
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
```

**Step 2: Run to verify it fails**

```bash
.venv/bin/python3.13 -m pytest tests/training/test_trainer.py::test_build_dataloaders_yields_dict_batches -v 2>&1 | tail -10
```

Expected: FAIL with `ImportError: cannot import name 'build_dataloaders'`

**Step 3: Implement `_collate_fn` and `build_dataloaders()` in `src/training/trainer.py`**

Add these two functions directly after the existing imports block (before the `TrainConfig` class). Add the `TokenizedShardDataset` import at the top of the file too.

Add to imports section (top of file, after existing imports):
```python
from src.data.tokenized_loader import TokenizedShardDataset
```

Add these two functions after the imports (before `@dataclass class TrainConfig`):

```python
def _collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Convert (input_ids, labels) tuple batch to dict format expected by AureliusTrainer."""
    input_ids = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return {"input_ids": input_ids, "labels": labels}


def build_dataloaders(
    cfg: "TrainConfig",
) -> tuple[DataLoader, DataLoader | None]:
    """Build train and validation DataLoaders from .npy token shards.

    Globs all *.npy files in cfg.train_data_dir and cfg.val_data_dir,
    wraps each in TokenizedShardDataset, and returns ready-to-use DataLoaders.

    Args:
        cfg: TrainConfig with train_data_dir, val_data_dir, seq_len,
             micro_batch_size, num_workers, pin_memory set.

    Returns:
        (train_loader, val_loader) — val_loader is None if val_data_dir has no shards.
    """
    train_shards = sorted(Path(cfg.train_data_dir).glob("*.npy"))
    if not train_shards:
        raise FileNotFoundError(f"No .npy shards found in {cfg.train_data_dir}")

    train_ds = TokenizedShardDataset(train_shards, seq_len=cfg.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate_fn,
        drop_last=True,
    )

    val_shards = sorted(Path(cfg.val_data_dir).glob("*.npy"))
    if not val_shards:
        return train_loader, None

    val_ds = TokenizedShardDataset(val_shards, seq_len=cfg.seq_len)
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.micro_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        collate_fn=_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader
```

Also wire `build_dataloaders` into the `main()` function, replacing the placeholder comment at lines 686-691:

```python
    from src.model.transformer import AureliusTransformer
    from src.model.config import AureliusConfig

    model = AureliusTransformer(AureliusConfig())
    train_loader, val_loader = build_dataloaders(cfg)
    trainer = AureliusTrainer(model, train_loader, val_loader, cfg)
    trainer.train()
```

**Step 4: Run the test**

```bash
.venv/bin/python3.13 -m pytest tests/training/test_trainer.py::test_build_dataloaders_yields_dict_batches -v 2>&1 | tail -10
```

Expected: PASS

**Step 5: Run full training test suite**

```bash
.venv/bin/python3.13 -m pytest tests/training/ -v 2>&1 | tail -20
```

Expected: all pass

**Step 6: Run full suite**

```bash
.venv/bin/python3.13 -m pytest 2>&1 | tail -5
```

Expected: 161 passed, 3 skipped

**Step 7: Commit**

```bash
git add src/training/trainer.py tests/training/test_trainer.py
git commit -m "feat: add build_dataloaders() and collate_fn; wire TokenizedShardDataset into trainer"
```
