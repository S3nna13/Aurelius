#!/usr/bin/env python3
"""Train Aurelius on the Reference Models corpus (12.5M tokens)."""

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_ref")

CONFIG = AureliusConfig(
    d_model=768,
    n_layers=10,
    n_heads=12,
    n_kv_heads=4,
    head_dim=64,
    d_ff=2048,
    vocab_size=8192,
    max_seq_len=512,
    rope_theta=500_000.0,
    rms_norm_eps=1.0e-6,
    dropout=0.0,
    tie_embeddings=True,
)


class ShardDataset(Dataset):
    def __init__(self, shard_paths, seq_len, stride=None):
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.tokens = np.concatenate([np.load(p, mmap_mode="r") for p in shard_paths])
        self.num_examples = max(0, (len(self.tokens) - seq_len - 1) // self.stride + 1)

    def __len__(self):
        return self.num_examples

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.tokens[start : start + self.seq_len + 1]
        return {
            "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
            "labels": torch.tensor(chunk[1:], dtype=torch.long),
        }


def cosine_warmup(step, warmup, total, min_ratio):
    if step < warmup:
        return step / max(1, warmup)
    p = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1.0 - min_ratio) * (0.5 * (1.0 + math.cos(math.pi * p)))


def train(
    model,
    train_loader,
    val_loader,
    device,
    total_steps=1000,
    warmup=100,
    lr=3e-4,
    min_lr=3e-5,
    wd=0.1,
    max_grad_norm=1.0,
    log_interval=50,
    save_dir=Path("checkpoints/aurelius-reference"),
):
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s (%.1fM)", f"{n_params:,}", n_params / 1e6)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "norm" in name or "ln" in name or "embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)

    opt = AdamW(
        [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    sched = LambdaLR(opt, lr_lambda=lambda s: cosine_warmup(s, warmup, total_steps, min_lr / lr))

    save_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    tokens_seen = 0
    history = []
    start = time.monotonic()
    train_iter = iter(train_loader)

    while global_step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        batch_tokens = input_ids.numel()

        loss, logits, _ = model(input_ids, labels=labels)
        if loss is None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        opt.step()
        sched.step()
        opt.zero_grad()

        global_step += 1
        tokens_seen += batch_tokens
        lr_now = sched.get_last_lr()[0]
        history.append({"step": global_step, "loss": loss.item(), "lr": lr_now})

        if global_step % log_interval == 0:
            elapsed = time.monotonic() - start
            logger.info(
                "Step %4d/%d | loss=%.4f | lr=%.2e | tok=%s | tps=%.0f",
                global_step,
                total_steps,
                loss.item(),
                lr_now,
                f"{tokens_seen:,}",
                tokens_seen / elapsed,
            )

    ckpt = save_dir / "final.pt"
    torch.save(
        {
            "step": global_step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict(),
        },
        ckpt,
    )
    logger.info("Saved checkpoint: %s", ckpt)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            loss, _, _ = model(input_ids, labels=labels)
            if loss is None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            val_losses.append(loss.item())
    avg_val = sum(val_losses) / len(val_losses)
    logger.info("Val loss=%.4f | perplexity=%.2f", avg_val, math.exp(avg_val))

    report = {
        "model_params": n_params,
        "total_steps": global_step,
        "tokens_seen": tokens_seen,
        "total_time_sec": time.monotonic() - start,
        "final_train_loss": history[-1]["loss"] if history else None,
        "val_loss": avg_val,
        "val_perplexity": math.exp(avg_val),
        "history": history,
    }
    with open(save_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = AureliusTransformer(CONFIG)
    train_ds = ShardDataset(
        ["data/pretrain/reference/train_shard_000.npy"], CONFIG.max_seq_len, stride=128
    )
    val_ds = ShardDataset(
        ["data/pretrain/reference/val_shard_000.npy"], CONFIG.max_seq_len, stride=128
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, drop_last=False)

    logger.info("Train examples: %d | Val examples: %d", len(train_ds), len(val_ds))
    report = train(
        model,
        train_loader,
        val_loader,
        device,
        total_steps=1000,
        save_dir=Path("checkpoints/aurelius-reference"),
    )

    logger.info("=" * 50)
    logger.info("REFERENCE CORPUS TRAINING COMPLETE")
    logger.info(
        "Params: %.1fM | Steps: %d | Time: %.1fmin",
        report["model_params"] / 1e6,
        report["total_steps"],
        report["total_time_sec"] / 60,
    )
    logger.info(
        "Train loss: %.4f | Val perplexity: %.2f",
        report["final_train_loss"],
        report["val_perplexity"],
    )
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
