#!/usr/bin/env python3
"""Train Aurelius from scratch on the GLM-5 paper corpus."""

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
logger = logging.getLogger("train_glm5")

# Model config for GLM-5 corpus
CONFIG = AureliusConfig(
    d_model=768,
    n_layers=10,
    n_heads=12,
    n_kv_heads=4,
    head_dim=64,
    d_ff=2048,
    vocab_size=4418,  # matches trained tokenizer
    max_seq_len=512,
    rope_theta=500_000.0,
    rms_norm_eps=1.0e-6,
    dropout=0.0,
    tie_embeddings=True,
)


class TokenizedShardDataset(Dataset):
    def __init__(self, shard_path, seq_len, stride=None):
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.tokens = np.load(shard_path, mmap_mode="r")
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


def cosine_with_warmup(step, warmup_steps, total_steps, min_lr_ratio):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr_ratio + (1.0 - min_lr_ratio) * (0.5 * (1.0 + math.cos(math.pi * progress)))


def train(
    model,
    train_loader,
    val_loader,
    device,
    total_steps=500,
    warmup_steps=50,
    lr=3e-4,
    min_lr=3e-5,
    weight_decay=0.1,
    max_grad_norm=1.0,
    log_interval=25,
    save_dir=Path("checkpoints/aurelius-glm5"),
):
    model.to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s (%.1fM)", f"{n_params:,}", n_params / 1e6)

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "bias" in name or "norm" in name or "ln" in name or "embed" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = LambdaLR(
        optimizer, lr_lambda=lambda s: cosine_with_warmup(s, warmup_steps, total_steps, min_lr / lr)
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0
    tokens_seen = 0
    history = []
    start_time = time.monotonic()
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
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        global_step += 1
        tokens_seen += batch_tokens
        lr_now = scheduler.get_last_lr()[0]
        history.append({"step": global_step, "loss": loss.item(), "lr": lr_now})

        if global_step % log_interval == 0:
            elapsed = time.monotonic() - start_time
            logger.info(
                "Step %3d/%d | loss=%.4f | lr=%.2e | tok=%s | tps=%.0f",
                global_step,
                total_steps,
                loss.item(),
                lr_now,
                f"{tokens_seen:,}",
                tokens_seen / elapsed,
            )

    # Final checkpoint
    ckpt = save_dir / "final.pt"
    torch.save(
        {
            "step": global_step,
            "tokens_seen": tokens_seen,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        ckpt,
    )
    logger.info("Saved final checkpoint: %s", ckpt)

    # Validation
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
    avg_val_loss = sum(val_losses) / len(val_losses)
    val_ppl = math.exp(avg_val_loss)
    logger.info("Val loss=%.4f | perplexity=%.2f", avg_val_loss, val_ppl)

    total_time = time.monotonic() - start_time
    report = {
        "model_params": n_params,
        "total_steps": global_step,
        "tokens_seen": tokens_seen,
        "total_time_sec": total_time,
        "final_train_loss": history[-1]["loss"] if history else None,
        "val_loss": avg_val_loss,
        "val_perplexity": val_ppl,
        "history": history,
    }
    with open(save_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = AureliusTransformer(CONFIG)
    train_ds = TokenizedShardDataset(
        "data/pretrain/glm5/train_shard_000.npy", CONFIG.max_seq_len, stride=128
    )
    val_ds = TokenizedShardDataset(
        "data/pretrain/glm5/val_shard_000.npy", CONFIG.max_seq_len, stride=128
    )
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, drop_last=False)

    logger.info("Train examples: %d | Val examples: %d", len(train_ds), len(val_ds))
    report = train(
        model,
        train_loader,
        val_loader,
        device,
        total_steps=500,
        save_dir=Path("checkpoints/aurelius-glm5"),
    )

    logger.info("=" * 50)
    logger.info("GLM-5 TRAINING COMPLETE")
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
