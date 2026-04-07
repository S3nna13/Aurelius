# Training Loop Fix Design

**Goal:** Make `AureliusTrainer` actually trainable — fix three concrete breaks introduced when the transformer's `forward()` return type changed to a tuple.

**Architecture:** Targeted surgical fixes to `src/training/trainer.py` only. No new abstractions. Add `tests/training/test_trainer.py` with three tests.

**Tech Stack:** PyTorch, HuggingFace Accelerate (infra only), TokenizedShardDataset

---

## Breaks and Fixes

### Break 1 — Output unpacking in `train()`

`trainer.py` lines ~370-371:
```python
outputs = self.model(input_ids=input_ids, labels=labels)
loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
```
`AureliusTransformer.forward()` now returns `(loss, logits, present_key_values)` — a plain tuple with no `.loss` attribute and no dict key. Both branches fail.

**Fix:** Replace with direct unpacking:
```python
loss, _, _ = self.model(input_ids=input_ids, labels=labels)
```

### Break 2 — Batch format mismatch

`trainer.py` reads:
```python
input_ids = batch["input_ids"]
labels = batch.get("labels", input_ids)
```
`TokenizedShardDataset.__getitem__` returns `(input_ids, labels)` tuples. A plain `DataLoader` wrapping this dataset yields `[Tensor, Tensor]` lists, not dicts.

**Fix:** Add a collate function:
```python
def _collate_fn(batch):
    input_ids = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return {"input_ids": input_ids, "labels": labels}
```
Pass as `collate_fn` when constructing `DataLoader`.

### Break 3 — No dataloader wiring

`trainer.py`'s `main()` has a placeholder comment instead of building dataloaders. There is no function to wire `TokenizedShardDataset` into the trainer.

**Fix:** Add `build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader | None]` to `trainer.py`:
- Glob `*.npy` shards in `cfg.train_data_dir`
- Glob `*.npy` shards in `cfg.val_data_dir` (optional — returns None if empty)
- Wrap each in `TokenizedShardDataset(shards, seq_len=cfg.seq_len)`
- Return `DataLoader` instances with `collate_fn=_collate_fn`

### Break 4 — Same unpacking bug in `validate()`

`validate()` has identical `outputs.loss` / `outputs["loss"]` pattern. Same one-line fix.

---

## Testing Strategy

`tests/training/test_trainer.py`:

1. **`test_train_step`** — small model + in-memory TensorDataset of 8 sequences, run 2 optimizer steps, assert loss is finite and at least one weight changed
2. **`test_validate`** — same setup, assert `validate()` returns finite positive float
3. **`test_build_dataloaders`** — write two tiny `.npy` shards to `tmp_path`, call `build_dataloaders()`, assert loaders yield `{"input_ids": Tensor, "labels": Tensor}` with correct shapes

All tests run on CPU with a 2-layer, 64-dim model. No Accelerate mocking — test the real code path.
